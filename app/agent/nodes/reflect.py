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

## 三处刻意不做的判定

- **不接模型的 `EXPAND`**：接上之后，判定就依赖云模型连通性，
  而「循环是否收敛」是四条验收标准里第③条的核心，不该由外部服务决定。
  详设 6.3 也明写「模型的判定**不直接采信**」，要过一遍代码。
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
from app.agent.state import AgentState, executed_sources, pending_steps

#: 补哪一路：已跑过的那路 → 该补的那路。
#: **只在这里定义一次**：`_plan` 里的数据源与工具映射在 supervisor，
#: 而这里要的是"互补"，两者是不同的关系，混在一起会写出
#: "补一路但补的是同一路"这种不会报错的错。
_COMPLEMENT: dict[str, tuple[str, str]] = {
    "sql_query": ("rag_retrieve", "从企业制度与报告知识库补查该问题涉及的规定与解释"),
    "rag_retrieve": ("sql_query", "从业务数据库补查该问题涉及指标的数值"),
}


def build_reflect_node() -> Callable[[AgentState], Any]:
    """构造 `reflect` 节点。**确定性，无外部依赖**。

    不收 `settings`：这一版的判定只看 State 里的事实（待执行步骤、已跑过的源、
    空结果、演进预算），一个配置项都不读。签名里留一个用不上的 `settings`
    会让人以为"某个阈值可以调"，而实际调不动。
    """

    def reflect(state: AgentState) -> dict[str, Any]:
        assessment = _assess(state)
        update: dict[str, Any] = {
            "progress_assessment": assessment,
            # `open_questions` 整体替换（详设 7.2）：新判定覆盖旧判定，
            # 避免已解决的问题无限累积
            "open_questions": list(_open_questions(state, assessment)),
        }
        if assessment.decision == "EXPAND" and assessment.proposed_steps:
            # **`task_list` 在这里被整体替换**——详设 7.2 只允许 Planner /
            # PlanExtend / Replan 改写它，`reflect` 是这一版里 planner 的代理
            # （确定性演进），所以这是允许的两处之一。
            update["task_list"] = [*(state.get("task_list") or []), *assessment.proposed_steps]
            update["expansions_left"] = max(0, state.get("expansions_left", 0) - 1)
            update["plan_revision"] = state.get("plan_revision", 0) + 1
        # **节点不写 `next_route`**：路由由条件边从更新后的 State 推出来
        # （`graph.route_after_reflect`）。两处各算一次的话，它们会在
        # 某次改动后分叉，而症状是"判定说继续、实际却去了 analysis"。
        return update

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
    """该补的那一路；没有可补的（或预算不足）时返回 None。"""
    for tool, complement in _COMPLEMENT.items():
        if tool in sources and complement[0] not in sources:
            return complement
    return None


def _next_step_id(state: AgentState) -> str:
    """演进步骤的 id。

    **序号接着计划往下排**，不是从 1 重来：`step_results` 是按 step_id 索引的，
    重用一个已经存在的 id 会让新步骤的结果覆盖旧步骤的（`merge_step_results`
    的设计就是"新值覆盖"）。这类覆盖不报错，只会让"这一步跑了两次"
    这种事后无从分辨。
    """
    used = {step.id for step in state.get("task_list") or []}
    index = len(used) + 1
    while f"step_{index:02d}" in used:  # pragma: no cover - 正常路径不会进循环
        index += 1
    return f"step_{index:02d}"


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
            f"{result.step_id}：{'、'.join(sorted(proposed))} 也未能覆盖（{result.error_code}）"
            for result in (state.get("step_results") or {}).values()
            if result.empty
        ]
        return tuple(questions)
    return tuple(
        f"{result.step_id}：按当前条件未取得结果（{result.error_code}）"
        for result in (state.get("step_results") or {}).values()
        if result.empty
    )


__all__ = ["build_reflect_node"]
