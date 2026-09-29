"""`plan_extend` 节点：计划演进（详细设计 6.6.3 / 8.5）。

## 它是**唯一**能追加步骤的节点

详设 6.1 的回边是 `reflect -> plan_extend -> dispatch`，6.6.3 的原话是
「`plan_extend` 是循环中唯一能修改 `task_list` 的节点」。本实现照此收口：
`reflect` 只出判定，写 `task_list` / 升 `plan_revision` / 扣 `expansions_left`
三件事全在这里做。

**这三件事必须是同一个写者**：各写各的话，漂移的症状是
「`plan_deltas` 说有一步、`task_list` 里没有」——两边都看不出来。

## 两个入口，一个走模型一个不走

| 入口 | `proposed_steps` 从哪来 | 调模型 |
|---|---|---|
| `reflect`（执行阶段） | 判定里现成的（`COMPLEMENTS` 表算出来的那一句） | **否** |
| `retry_router`（审查阶段） | 由本节点生成 | 是 |

第二条是 14.3 的要求：「由 `plan_extend` 决定具体步骤，而不是由 Reviewer
直接指定 SQL」——审查只说"缺一整类信息"，落到哪几条查询是这里的事。

**第一条刻意不调模型**：`reflect` 的判定要保证"循环会不会收敛"不依赖
云模型连通性（那是四条验收标准里第③条的核心）。演进这一步跟着它一起
确定性，整条回边就只有 `retry_router` 那一支会碰模型。

## 校验失败**不报错**（6.6.3 原文）

> 不报错、不中断任务：把未通过的步骤丢弃，记录 `plan_extend_rejected`
> 事件到轨迹，然后按 SUFFICIENT 收敛。这一点是刻意的——演进失败不应该
> 让已经拿到的证据白费。

## 预算在**入口**就扣，不在"合入成功"时扣

这条是被一个真问题逼出来的：`retry_router → plan_extend → conflict → analysis
→ reviewer → retry_router` 是一条**闭合回路**。被拒的演进不追加任何步骤，
于是 `step_results` 不变，而 `_guard` 判超限读的正是它的条数——
**那道护栏永远不会触发**，循环一路烧到 LangGraph 的 `recursion_limit`，
收尾是 `INTERNAL_ERROR` + `trace_incomplete` + 五张产出表全空
（约定 102 记的那副样子）。

把扣减放在"合入成功"时也一样绕不完：审查每一轮都可能再判一次 expand，
而预算一次都没少。所以**每进一次 `plan_extend` 就扣一次**——预算约束的是
**演进的尝试**，而"尝试了但没提出合格步骤"确实是一次尝试。

代价如实记下：一次被拒的演进也会吃掉一格预算。这与"预算只在动作真的
发生时花"（约定 104）有出入，但两者不可兼得，而**绕不完的循环比记账不精确
贵得多**——前者是一次内部错误，后者只是读数时要多看一行
`plan_extend_rejected`（拒绝原因每次都记，两者分得开）。

## 模型不可用时也按"不演进"收敛，不判 FAIL

演进是**增强**不是必经路径：一次模型侧的网络抖动不该让一条证据已经
拿齐的答案作废（那是 `supervisor` 的处置，因为那里没有别的路可走）。
但也不静默——`errors` 里留一条，`plan_extend_rejected` 里写明是模型没跑成，
好把"模型坏了"与"模型提的步骤不合格"分开。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from app.agent.nodes.analysis import _number, _render
from app.agent.prompts.plan_extend import PLAN_EXTEND_PROMPT
from app.agent.schemas.plan import PlanDelta, PlanExtension, ProposedStep, TaskStep
from app.agent.schemas.review import ReviewResult
from app.agent.state import AgentState, current_question, next_step_id
from app.core.config import Settings
from app.core.errors import AgentError
from app.infrastructure.model_gateway import ModelGateway

#: 步骤摘要进 prompt 时最多留多少字。摘要是给人读的一句话，
#: 而它的正文可能是一整段模型生成的结论——全塞进去，prompt 会**掉得比
#: 证据还重**，而它要回答的只是"这一步做过什么"。
_SUMMARY_CHARS = 200

#: 归一化时去掉的字符：空白与各类标点。**保留中英文字与数字**——
#: `\w` 在 Unicode 模式下把汉字算作 word 字符，所以中文目标文本原样留下。
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def build_plan_extend_node(
    settings: Settings,
    gateway: ModelGateway,
) -> Callable[[AgentState], Any]:
    """构造 `plan_extend` 节点。依赖走闭包注入，不进 State（同 supervisor）。"""

    async def plan_extend(state: AgentState) -> dict[str, Any]:
        left = state.get("expansions_left", 0)
        if left <= 0:
            # **先判预算再干活**：没有预算时连模型都不该调——那是一次
            # 白花的钱，而结论必然是"不演进"。这一步不扣预算（它没扣得动）。
            return {"plan_extend_rejected": ["演进预算已用尽（expansions_left=0），本轮不追加计划"]}

        # 扣在入口而不是"合入成功"时——理由是**绕不完的循环**，
        # 见模块 docstring 的「预算为什么在入口扣」。下面每条出口都带上它。
        spent: dict[str, Any] = {"expansions_left": max(0, left - 1)}

        try:
            proposed, reason = await _proposed(state, gateway)
        except AgentError as exc:
            return {
                **spent,
                "errors": [exc],
                "plan_extend_rejected": [f"模型未能给出演进步骤（{exc.code.value}），按不演进收敛"],
            }

        accepted, rejected = _validate(state, _with_ids(state, proposed), settings=settings)
        if not accepted:
            return {
                **spent,
                "plan_extend_rejected": rejected or ["本次演进没有提出可加入的新步骤"],
            }

        revision = state.get("plan_revision", 0) + 1
        delta = PlanDelta(
            # id 就是详设 7.2 要的去重键 `(task_id, revision_no)`——
            # 于是 `plan_deltas` 能直接走 `merge_by_id`，不必为它再写一个 reducer。
            id=f"{state.get('task_id') or ''}:{revision}",
            revision_no=revision,
            trigger="EXTEND",
            trigger_finding_id=_trigger_finding(state),
            added_step_ids=tuple(step.id for step in accepted),
            # **恒空**：本版两个入口都只追加、不放弃待执行步骤
            # （6.6.3 允许"把尚未执行的 PENDING 标成 SKIPPED"，但
            # `reflect` 补的是一路新工具、模型补的也是新方向，都不需要
            # 撤掉谁）。字段留着是为了 `plan.updated` 事件的形状不变，
            # 并且让"以前会不会跳过步骤"在产物上答得出来。
            skipped_step_ids=(),
            reason=reason,
        )
        return {
            **spent,
            "task_list": [*(state.get("task_list") or []), *accepted],
            "plan_revision": revision,
            "plan_deltas": [delta],
            "plan_extend_rejected": rejected,
        }

    return plan_extend


async def _proposed(
    state: AgentState, gateway: ModelGateway
) -> tuple[tuple[ProposedStep, ...], str]:
    """本次演进要加入的步骤与理由，**尚未校验、也还没有 id**。

    `reflect` 送来的判定里已经有步骤时直接用——那是确定性算出来的
    （补哪一路由 `state.COMPLEMENTS` 决定），不需要也不应该再问一次模型。
    它的 `TaskStep` 在这里降成 `ProposedStep`（去掉 id），好让**两条路
    进入同一条装配线**：id 只在 `_with_ids` 那一处分配。
    """
    assessment = state.get("progress_assessment")
    if assessment is not None and assessment.decision == "EXPAND" and assessment.proposed_steps:
        return (
            tuple(
                ProposedStep(objective=step.objective, tool=step.tool)
                for step in assessment.proposed_steps
            ),
            assessment.reason,
        )

    result = await gateway.invoke_structured(
        PLAN_EXTEND_PROMPT,
        PlanExtension,
        question=current_question(state),
        plan=_render_plan(state),
        evidence=_render_evidence(state),
        reason=_review_reason(state.get("review_result")),
    )
    return tuple(result.value.steps), result.value.reason


def _with_ids(state: AgentState, steps: Sequence[ProposedStep]) -> list[TaskStep]:
    """给新步骤分配 id，组装成 `TaskStep`。**id 只在这一处产生**。

    让调用方自带 id 的话，撞号会让新步骤的结果覆盖旧步骤的
    （`merge_step_results` 就是"新值覆盖"）——而它**不报错**，
    事后也分不清"这一步跑了两次"。
    """
    taken: set[str] = set()
    assigned: list[TaskStep] = []
    for step in steps:
        assigned_id = next_step_id(state, taken=taken)
        taken.add(assigned_id)
        assigned.append(TaskStep(id=assigned_id, objective=step.objective, tool=step.tool))
    return assigned


def _validate(
    state: AgentState, steps: Sequence[TaskStep], *, settings: Settings
) -> tuple[list[TaskStep], list[str]]:
    """6.6.3 的循环专属校验里**前提已经具备**的那几条。

    ## 另外三条是结构性满足的，所以没有代码

    - **`step_id` 全局唯一，`depends_on` 引用存在且无环**：id 由 `_with_ids`
      统一分配；而 `depends_on` 在本实现里恒为空（初始计划由 supervisor
      确定性产出，两路互不依赖；模型提出的步骤也没有这个字段）。
      "引用存在且无环"因此没有可检查的对象。
    - **工具属于启用集合**：`ProposedStep.tool` 是 `StepTool` 闭集，
      未知工具在校验之前就已经是 `MODEL_OUTPUT_INVALID`。
    - **`trigger_finding_id` 有效**：它由 `_trigger_finding` 从 State 里查出来，
      不是模型写的——**指不到就留空，不编一个**。

    写成代码的话，这几条都是恒真的分支。同 Reviewer 那条「SQL 是否通过
    安全校验」：**它在架构上已被满足**，重复检查只会在两处维护同一份副本。
    """
    accepted: list[TaskStep] = []
    rejected: list[str] = []
    seen = {_dedup_key(step) for step in state.get("task_list") or []}

    for step in steps:
        key = _dedup_key(step)
        if key in seen:
            rejected.append(f"「{step.objective}」（{step.tool}）与已有步骤方向重复，丢弃")
            continue
        seen.add(key)
        accepted.append(step)

    per_round = settings.loop.max_steps_per_expansion
    if len(accepted) > per_round:
        rejected.append(f"单次演进最多 {per_round} 步，超出的 {len(accepted) - per_round} 步丢弃")
        accepted = accepted[:per_round]

    slots = settings.loop.max_total_steps - len(state.get("task_list") or [])
    if len(accepted) > slots:
        rejected.append(f"计划总步数上限 {settings.loop.max_total_steps}，放不下的步骤丢弃")
        accepted = accepted[: max(0, slots)]

    return accepted, rejected


def _dedup_key(step: TaskStep) -> tuple[str, str]:
    """去重键：`(工具, 归一化目标)`（6.6.3 第 6 条的前半）。

    6.6.3 还要求"同时对目标文本做 embedding 相似度比较"，**那半条没做**——
    它要一次向量调用，而阈值（`loop.duplicate_similarity_threshold`）
    按 6.6.3 的原文就该由真实问题集校准。登记在【后续扩展】。
    只做键比较的代价是**近义但不同措辞的目标不会被判重**；
    而本版两个入口的目标文本都来自固定的表或固定的话术，重复是字面级的。
    """
    return step.tool, _normalize(step.objective)


def _normalize(text: str) -> str:
    """去掉空白与标点，只留下字。大小写也归一。

    **不切词、不做同义替换**：那是 embedding 那半条要做的事，
    在这里做一半只会让人以为"重复判定已经做完了"。
    """
    return _NON_WORD.sub("", text).casefold()


def _trigger_finding(state: AgentState) -> str | None:
    """催生这次演进的中间发现；**找不到就返回 `None`，不编一个**。

    判据是"哪一步查空了"——那正是 `reflect` 判定 EXPAND 的理由
    （`_assess` 只在有 `empty` 结果时才提这一路）。`tool_nodes._normalize`
    给每一步都产一条 `Finding`，**包括空结果的那一步**
    （"这一步没有回答它的问题"），所以这条路一般指得到。

    ⚠️ **从审查那条路进来时通常指不到**：审查说的是"缺一整类信息"，
    那是关于**答案**的判断，不是某一步的发现——`findings` 里没有对应物。
    留空是如实的：一个指向不存在 finding 的 id 会让"每个下钻步骤都能
    反查发现"这句话变成假的，而它看起来完全正常。
    """
    results = state.get("step_results") or {}
    for finding in reversed(state.get("findings") or []):
        result = results.get(finding.step_id)
        if result is not None and result.empty:
            return finding.id
    return None


def _render_plan(state: AgentState) -> str:
    """已执行步骤的清单（给模型看的"做过什么"）。"""
    results = state.get("step_results") or {}
    lines: list[str] = []
    for step in state.get("task_list") or []:
        result = results.get(step.id)
        status = result.status.value if result is not None else "PENDING"
        outcome = ""
        if result is not None and result.summary:
            outcome = f"：{result.summary[:_SUMMARY_CHARS]}"
        lines.append(f"- {step.id} [{status}] {step.objective}（工具 {step.tool}）{outcome}")
    return "\n".join(lines) or "（还没有执行任何步骤）"


def _render_evidence(state: AgentState) -> str:
    """证据的编号渲染。

    **复用 `analysis` 的两个函数而不是另写一份**（与 `reviewer` 同一条理由）：
    重新编一套编号规则不会报错，只会让模型基于错误的对应关系做判断。
    """
    evidence = list(state.get("evidence") or [])
    if not evidence:
        return "（还没有拿到任何证据）"
    return _render(_number(evidence))


def _review_reason(review: ReviewResult | None) -> str:
    """审查那边的理由，转成一句给人（也给模型）看的话。"""
    if review is None:
        return "（没有可用的审查结论）"
    reasons = [issue.message for issue in review.issues if issue.severity != "INFO"]
    if review.missing_evidence:
        reasons.append("审查指出缺失的证据：" + "、".join(review.missing_evidence))
    if not reasons:
        reasons.append(review.reason_code or "（审查未给出具体理由）")
    return "\n".join(reasons)


__all__ = ["build_plan_extend_node"]
