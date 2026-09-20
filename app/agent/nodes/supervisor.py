"""`supervisor` 节点：意图识别 + 确定性计划（详细设计 8.1 / 8.2 / 8.3）。

## 它同时干了两件事，这是刻意的

详设 6.1 的图里 `supervisor → planner → dispatch` 是三段，而冲刺方案 §8.1
把最小 Graph 定为 6 个节点（没有 planner）。于是"从意图到计划"这一步
合进了 supervisor，且**是确定性的**：`required_sources` 里有哪几路就生成
几个步骤，没有模型参与第二次。

**依赖关系（`depends_on`）在这一版恒为空**：SQL 拿数字、RAG 拿解释，
两路互不依赖。详设里"先确认基准再拆维度贡献"那种依赖链要等
`plan_extend`（演进）才真正需要，而演进按冲刺方案是后置的。
`TaskStep` 的形状留着，接演进时改的是**生成方式**，不是 State 形状。

## 模型失败时**不降级为"两路都查"**

这条要写清楚，因为降级看起来更"稳"。但 FR-PLAN-002 业务规则 1 明写
「简单指标查询不得强制调用 RAG」——模型不可用时两路都调，
等于**在故障时把这条验收悄悄作废**，而它看起来完全正常（结果确实拿到了）。
更糟的是：那正是冲刺方案 §8.4 第②条「Agent 能自主选择」的演示点，
故障期间演示会得出"这个 Agent 从来都是两路都调"的结论。

因此模型失败 → 任务以 `UPSTREAM_UNAVAILABLE` 失败，如实说"没跑成"。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.prompts.supervisor import SUPERVISOR_PROMPT
from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import IntentResult, StepTool, TaskStep
from app.agent.state import AgentState, Route
from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.domain.task import TaskStatus
from app.infrastructure.model_gateway import ModelGateway

#: 数据源 → 工具。**`TaskStep.tool` 不带状态位**：状态存在 `step_results` 里，
#: 没有结果就是 PENDING。这样"重跑同一步"只需覆盖那一条，
#: 不必回头改计划本身（而计划是只允许整体替换的，见详设 7.2）。
_SOURCE_TO_TOOL: dict[str, StepTool] = {"sql": "sql_query", "rag": "rag_retrieve"}

_SOURCE_OBJECTIVE: dict[str, str] = {
    "sql": "从业务数据库查出问题涉及指标的数值",
    "rag": "从企业制度与报告知识库查出问题涉及的规定、口径或解释",
}


def build_supervisor_node(settings: Settings, gateway: ModelGateway) -> Callable[[AgentState], Any]:
    """构造 supervisor 节点。

    依赖由闭包注入而不是从 State 里取：State 是**数据**，网关是**依赖**。
    把网关放进 State 会让它随检查点被序列化，而它里面有 HTTP 客户端。
    """

    async def supervisor(state: AgentState) -> dict[str, Any]:
        question = state.get("sanitized_query") or state["user_query"]
        try:
            result = await gateway.invoke_structured(
                SUPERVISOR_PROMPT, IntentResult, question=question
            )
        except AgentError as exc:
            # **重建一次**（详设 8.3：「校验失败可让 Supervisor 重建一次，
            # 第二次失败返回 `PLAN_INVALID`」）。网关内部已经按 VALIDATION 类
            # 重试过一次，但那一次用的是**同一段 prompt**；
            # 再打一次有机会拿到另一段输出，而这一步失败会让任务直接结束。
            # 实测：演示里 6 条问题有 1 条撞上过这个——
            # 不重试的话，一次模型抖动就等于一次演示失败。
            if exc.code is not ErrorCode.MODEL_OUTPUT_INVALID:
                return _fail(exc)
            try:
                result = await gateway.invoke_structured(
                    SUPERVISOR_PROMPT, IntentResult, question=question
                )
            except AgentError as retry_exc:
                return _fail(retry_exc)

        intent = result.value
        try:
            task_list = _plan(intent)
        except AgentError as exc:
            return {
                "errors": [exc],
                "next_route": Route.FAIL,
                "execution_status": TaskStatus.FAILED,
                "final_answer": exc.message,
            }

        if not task_list:
            # CLARIFICATION / UNSUPPORTED：没有可执行步骤。
            # **产出一个 `AnalysisResult` 而不是留空**：`final` 在
            # `analysis_result is None` 时渲染的是"任务未能完成"，
            # 而"你问的年度没说清"不是失败——它是正常的澄清。
            # 复用同一条渲染路径，比在 `final` 里加一个特判干净。
            return {
                "intent": intent,
                "task_list": [],
                "errors": [],
                "analysis_result": _explain(intent),
                "next_route": Route.ANALYSIS,
            }
        return {
            "intent": intent,
            "task_list": task_list,
            "current_step_index": 0,
            "plan_revision": state.get("plan_revision", 0),
            "expansions_left": state.get("expansions_left", settings.loop.max_expansions),
            "errors": [],
            "next_route": _route_for(task_list[0]),
        }

    return supervisor


def _fail(exc: AgentError) -> dict[str, Any]:
    """意图判定失败 → 任务直接结束。

    **不降级为"两路都查"**，见模块 docstring：降级看起来更稳，
    但它会在故障期间把 FR-PLAN-002 那条验收悄悄作废，
    而结果看起来完全正常（确实拿到了数据）。
    """
    return {
        "errors": [exc],
        "next_route": Route.FAIL,
        "execution_status": TaskStatus.FAILED,
        "final_answer": f"无法理解该问题：{exc.message}",
    }


def _explain(intent: IntentResult) -> AnalysisResult:
    """CLARIFICATION / UNSUPPORTED → 面向用户的说明。

    `missing_fields` 直接进"限制"一节：用户在追问前需要知道**缺的是哪几样**，
    而这个问题文本本身说不出来（它只说明了"我缺信息"）。
    """
    if intent.intent == "UNSUPPORTED":
        answer = "这类请求无法通过企业数据分析完成。"
        limits: tuple[str, ...] = ()
    else:
        answer = intent.clarification_question or "请补充必要信息后重试。"
        limits = (f"缺少：{'、'.join(intent.missing_fields)}",) if intent.missing_fields else ()
    # `refused=False`：澄清与不支持都**不是** 11.8 的拒答——拒答说的是
    # "证据里没有你问的这件事"，而这两者是"问题缺前提"与"问题类型不归这里管"。
    # 三者的处置完全不同，混成一个布尔量会让统计上它们变成一件事。
    return AnalysisResult(direct_answer=answer, refused=False, limitations=limits)


def _plan(intent: IntentResult) -> list[TaskStep]:
    """意图 → 计划（确定性）。**同时执行详设 8.3 的计划校验。**

    校验放在这里而不是单独一个节点：计划只有这一个产生点，
    多一个节点就多一次 State 往返，而校验失败的处理（返回错误）与生成失败
    完全一样。

    Raises:
        AgentError: 计划不合法。用 `PLAN_INVALID`（19.1 的语义就是"计划语义不合法"）。
    """
    if intent.intent in ("CLARIFICATION", "UNSUPPORTED"):
        return []

    sources = [source for source in intent.required_sources if source in _SOURCE_TO_TOOL]
    unknown = [source for source in intent.required_sources if source not in _SOURCE_TO_TOOL]
    if unknown:
        # 模型给了 search 或别的名字：**报错而不是悄悄丢掉**。
        # 丢掉会让计划少一路而没人知道，最终答案是"信息不全"却没有原因。
        raise AgentError(
            ErrorCode.PLAN_INVALID,
            f"计划引用了未启用的数据源：{'、'.join(unknown)}",
            details={"required_sources": list(intent.required_sources)},
        )
    if not sources:
        # 需要数据源却没有一路：模型常见的失败形态是把 sources 留空
        # 而 intent 填成 QUERY。按"什么都不查"继续会让任务以空答案成功。
        raise AgentError(
            ErrorCode.PLAN_INVALID,
            f"意图为 {intent.intent} 但没有给出任何数据源，无法制定计划",
            details={"intent": intent.intent, "confidence": intent.confidence},
        )
    if len(sources) > 8:
        raise AgentError(
            ErrorCode.PLAN_INVALID,
            f"计划的步骤数超过上限（{len(sources)} > 8）",
            details={"sources": sources},
        )

    seen: set[str] = set()
    steps: list[TaskStep] = []
    for index, source in enumerate(sources, start=1):
        step_id = f"step_{index:02d}"
        if step_id in seen:  # pragma: no cover - 序号由 enumerate 保证唯一
            raise AgentError(ErrorCode.PLAN_INVALID, f"step_id 重复：{step_id}")
        seen.add(step_id)
        steps.append(
            TaskStep(
                id=step_id,
                objective=_SOURCE_OBJECTIVE.get(source, source),
                tool=_SOURCE_TO_TOOL[source],
                required=True,
            )
        )
    return steps


def _route_for(step: TaskStep) -> Route:
    return Route.SQL if step.tool == "sql_query" else Route.RAG


__all__ = ["build_supervisor_node"]
