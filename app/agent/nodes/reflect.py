"""`reflect` 节点：任务循环的判断点（详细设计 6.3 / 6.6.1）。

## 它是「Agent 能根据工具结果继续分析」的落点

冲刺方案 §8.1 把这条列为**必须保留**的两点之一：

> `reflect` —— 它就是「**Agent 能根据工具结果继续分析，而不是一次性生成答案**」
> 的落点。按开发总原则 4「先跑通再跑准」，**先写成确定性降级路径**
> （结果为空 / 缺维度 → 扩展一步），图结构跑通后再接模型的 EXPAND 判定。

所以这一版**不调模型**，判定全部由代码写死，判据是三条**可复现的事实**：

| 情形 | 判定 | 理由 |
|---|---|---|
| 还有 PENDING 步骤 | `CONTINUE` | 计划没跑完，继续 |
| 某一路**跑了但空**，而另一路没进过计划，且演进预算够 | `EXPAND` | 该补一路 |
| 其余 | `SUFFICIENT` | 手上的证据就是全部 |

**第二条是真正的"继续分析"**：问「Q3 为什么下滑」，supervisor 判成只调 SQL，
而 SQL 按条件查不到数据——这时 `reflect` 补一路 RAG 去找制度/报告里的解释。
它不需要模型，却是货真价实的"根据结果决定下一步"。

## 它只出判定，不改计划

判 `EXPAND` 时它把要补的那一步放进 `ProgressAssessment.proposed_steps`，
由条件边送到 `plan_extend` 去校验与合入——详设 6.6.3 的原话是
「`plan_extend` 是循环中唯一能修改 `task_list` 的节点」。

**这里曾经自己追加步骤**（`task_list` / `plan_revision` / `expansions_left`
三样一起写）。搬走的原因是两个写者会漂移：症状是
「`plan_deltas` 说有一步、`task_list` 里没有」，而两边都看不出来。
搬走之后 `plan_deltas` 才有了**确定性**的生产者——而那是循环类评测
能写成硬断言（而不是观察用例）的前提。

## 三处刻意不做的判定

- **不接模型的 `EXPAND`**：接上之后，判定就依赖云模型连通性，
  而「循环是否收敛」是四条验收标准里第③条的核心，不该由外部服务决定。
  详设 6.3 也明写「模型的判定**不直接采信**」，要过一遍代码。
  演进换到 `plan_extend` 之后这一点**没变**：那条路上收到的是现成的
  `proposed_steps`，只做校验与合入，不问模型。
- **不实现 `BLOCKED`**：它的定义是"证据缺口属于权限或数据不存在，
  工具无法补齐"——那需要区分"查不到因为没权限"与"查不到因为真没有"，
  前者要读 Tool 的错误类别（`ACCESS_DENIED`）。切片内两路工具都还没产出过
  权限拒绝（SQL 的拒查由校验器在更早处拦下），判定会落空。
  登记为【后续扩展】，字段留着。
- **不做重复步骤去重**（`LOOP__DUPLICATE_SIMILARITY_THRESHOLD`）：
  它防的是"模型反复提出目标相同的步骤"，而确定性演进每一步的
  `tool` 都不一样（补的是"另一路"），重复不了。

## 演进预算只有一次

`expansions_left` 初值是 `LOOP__MAX_EXPANSIONS`（默认 2），本实现每次
`EXPAND` 消耗 1。**补一路只需要一次**：两路都跑过之后，
再演进也没有第三路可补（Search 后置），所以实际最多演进一次。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.schemas.plan import ProgressAssessment, TaskStep
from app.agent.state import (
    AgentState,
    executed_sources,
    missing_complement,
    next_step_id,
    pending_steps,
)

# 互补关系（"sql 空则该补 rag"）**表在 `state.COMPLEMENTS`，不在这里**。
# `reviewer` 问的是同一个问题（"还有哪一路没查过"），而两处各写一份的症状是
# **"某类补证再也不发生"**——没有任何地方会报错，只是行为静默地少了一种。


def build_reflect_node() -> Callable[[AgentState], Any]:
    """构造 `reflect` 节点。**确定性，无外部依赖**。

    不收 `settings`：这一版的判定只看 State 里的事实（待执行步骤、已跑过的源、
    空结果、演进预算），一个配置项都不读。签名里留一个用不上的 `settings`
    会让人以为"某个阈值可以调"，而实际调不动。
    """

    def reflect(state: AgentState) -> dict[str, Any]:
        assessment = _assess(state)
        # **判完就完了**：`task_list` / `plan_revision` / `expansions_left`
        # 三样都交给 `plan_extend` 写（详设 6.6.3：它是唯一能改计划的节点）。
        #
        # 这里曾经自己追加步骤。搬走不是为了少一层——而是**两个写者会漂移**：
        # 各写一份"追加步骤 + 升 revision + 扣预算"，某天其中一份改了规则，
        # 症状是「`plan_deltas` 说有一步、`task_list` 里没有」，
        # 而两边都看不出来。
        #
        # **它仍然不调模型**：`_assess` 的 `decision` 只看 State 里的事实，
        # 所以"循环会不会收敛"照样不依赖云模型连通性——见模块 docstring。
        # 演进换成 `plan_extend` 之后这一点没变：那条路上收到的是现成的
        # `proposed_steps`，只做校验与合入，不问模型。
        #
        # **节点不写 `next_route`**：路由由条件边从更新后的 State 推出来
        # （`graph._after_reflect`）。两处各算一次的话，它们会在
        # 某次改动后分叉，而症状是"判定说继续、实际却去了 analysis"。
        return {
            "progress_assessment": assessment,
            # `open_questions` 整体替换（详设 7.2）：新判定覆盖旧判定，
            # 避免已解决的问题无限累积
            "open_questions": list(_open_questions(state, assessment)),
        }

    return reflect


def _assess(state: AgentState) -> ProgressAssessment:
    pending = pending_steps(state)
    if pending:
        return ProgressAssessment(
            decision="CONTINUE",
            reason=f"计划中还有 {len(pending)} 步未执行",
        )

    sources = executed_sources(state)
    results = state.get("step_results") or {}
    budget = state.get("expansions_left", 0)
    # 「跑了但空」：SQL 0 行 / RAG 无相关知识。**这是关于数据的事实**，
    # 不是失败——`tool_nodes` 已经把这两类从 `errors` 里摘出去了。
    empties = [result for result in results.values() if result.empty]

    if empties and budget > 0:
        missing = _missing_complement(sources)
        if missing is not None:
            tool, objective = missing
            return ProgressAssessment(
                decision="EXPAND",
                reason=(
                    f"{'、'.join(sorted(sources))} 未取得有用结果"
                    f"（{empties[0].error_code or '空结果'}），补一路 {tool}"
                ),
                proposed_steps=(TaskStep(id=_next_step_id(state), objective=objective, tool=tool),),
            )

    failed = [result for result in results.values() if result.status.value == "FAILED"]
    if failed and len(failed) == len(results):
        # 全部步骤都失败了：这是「查不成」而不是「没查到」，
        # 但**不判 BLOCKED**（那要区分权限与不存在，见模块 docstring），
        # 交给 analysis 在限制里说明。决策仍是 SUFFICIENT——
        # 因为再跑一遍不会有别的结果。
        return ProgressAssessment(
            decision="SUFFICIENT",
            reason=f"{len(failed)} 步全部失败，无法产出结论性证据",
        )

    return ProgressAssessment(
        decision="SUFFICIENT",
        reason=f"{len(results)} 步已执行，证据覆盖已确定的全部数据源",
    )


def _missing_complement(sources: set[str]) -> tuple[str, str] | None:
    """该补的那一路；没有可补的（或预算不足）时返回 None。

    **表在 `state.COMPLEMENTS`**：`reviewer` 问的是同一个问题，
    两处各写一份的话，"某类补证再也不发生"不会有任何症状。
    """
    return missing_complement(sources)


def _next_step_id(state: AgentState) -> str:
    """本节点提出的那一步的 id。

    **实现搬去了 `state.next_step_id`**：`retry_router` 与 `plan_extend`
    问的是同一个问题，三份各写一份的症状是某天其中一份改了规则，
    而撞号只在那一条路径上发生。
    """
    return next_step_id(state)


def _open_questions(state: AgentState, assessment: ProgressAssessment) -> tuple[str, ...]:
    """尚未回答的子问题（详设 7.1）。

    **只列真的还开着的**：某一步跑了但空，而系统不再补证（预算耗尽 / 没有
    可补的一路）时，它就是一个悬而未决的问题。列出来是为了让最终答案
    能在"限制"里说清楚——而不是让用户以为"没提到"等于"没问题"。
    """
    if assessment.decision == "EXPAND":
        # 正要补一路去回答它，不算悬空——**补的是哪一路由演进步骤说了算**，
        # 不是"结果里有没有空"。按后者判会让"补了 A 路"也把 B 路的空结果
        # 一并勾销，而 B 路其实还开着。
        proposed = {step.tool for step in assessment.proposed_steps}
        questions = [
            f"{result.step_id}：{'、'.join(sorted(proposed))} 也未能覆盖"
            + (f"（{result.error_code}）" if result.error_code else "")
            for result in (state.get("step_results") or {}).values()
            if result.empty
        ]
        return tuple(questions)
    return tuple(
        f"{result.step_id}：按当前条件未取得结果"
        + (f"（{result.error_code}）" if result.error_code else "")
        for result in (state.get("step_results") or {}).values()
        if result.empty
    )


__all__ = ["build_reflect_node"]
