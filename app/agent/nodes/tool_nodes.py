"""`sql` 与 `rag` 两个节点（详细设计 6.2 的 `dispatch → sql_* / rag_*` 合并版）。

**它们形状相同**：找出该自己跑的下一步 → 组装入参 → 调 Tool → 归一化进 State。
差别只在入参与产物，所以共用一个 `_run_step`，两个薄包装。

## 与详设的偏差：没有拆成 8 个节点

详设 6.2 把 SQL 拆成 `sql_prepare / sql_generate / sql_validate / sql_execute`
四步、RAG 拆成 `rag_rewrite / rag_retrieve / rag_rerank` 三步。**那七步都还在**，
只是活在 Tool 内部（`app/tools/sql/tool.py` 与 `app/tools/rag/tool.py`）——
冲刺方案 §8.1 把最小 Graph 定为 6 个节点，而 Tool 本来就是"把一件事做完整"的
封装（详设 9.1）。把它们提到图里，等于让 State 承担 Tool 的内部中间态
（`sql_candidate` / `schema_context`），而那正是详设 7.1 标注它们
「当前 SQL 步骤」的原因：生命周期只到这一步结束。

**这不是"少做了"**：SQL 的 12 步校验、自修复 2 次、RAG 的双路召回与门禁
一条都没少，只是它们的**重试点在图之外**。要接 Reviewer 的补证重试时，
回边落在 Tool 调用这一层，不必先把 Tool 拆开。

## 归一化：什么进 `errors`，什么不进

`FAILED` 的步骤分成两类，混在一起会让最终答案把"没查到"说成"出错了"：

- **`EMPTY_RESULT` 类**（SQL 0 行、RAG 无相关知识）：进 `step_results.empty=True`
  与 `error_code`，**不进 `errors`**。那是关于数据的事实（详设 9.4 明写
  「不算失败」），最终答案该说的是"按当前条件没有数据"；
- **其余失败**：同时进 `errors`。它们必须出现在最终答案的"限制"一节里。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from app.agent.schemas.plan import Finding, IntentResult, StepResult, StepStatus, TaskStep
from app.agent.state import AgentState, pending_steps
from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.core.ids import IdPrefix, new_id
from app.domain.task import ToolCallRecord
from app.domain.user import PermissionScope
from app.tools.base import ToolContext, ToolError, ToolResult

#: 哪一类**错误码**表示"跑了但没拿到有用东西"（关于数据的事实，不是执行出错）。
#: 另一半在 `payload["is_empty"]` 里——那是 SQL 的空结果走的路径，见 `_normalize`。
_EMPTY_RESULT_CODES: frozenset[str] = frozenset({ErrorCode.NO_RELEVANT_KNOWLEDGE.value})

#: 步骤摘要最多多少字。进 `step_results` 的只有摘要——
#: 详设 10.6 明写「SQL 结果不直接写应用日志、也不默认持久化原始行」，
#: 而 State 会被写进轨迹。
_SUMMARY_CHARS = 240


def _find_step(state: AgentState, tool: str) -> TaskStep | None:
    """找出该工具**第一个还没跑**的步骤。

    **路由函数与节点读的是同一个判据**（`pending_steps` 的顺序）。
    条件边在 LangGraph 里不能写状态，所以节点必须自己再推一次；
    两处用同一个函数保证推出来的必然是同一个步骤——
    各写一份的话，会出现在"路由到 sql 却跑 rag 的步骤"这类错位，
    而它只在两路都进过计划时才会显形。
    """
    for step in pending_steps(state):
        if step.tool == tool:
            return step
    return None


def _context(state: AgentState, step: TaskStep) -> ToolContext:
    """一步的 Tool 上下文。

    `permission_scope` **从 State 取**（由 `_run_body` 从用户记录装载）。
    取不到就抛：`PermissionScope` 的空 `region_ids` 表示**不限**（TBC-03），
    拿它当兜底等于静默放行全量数据。
    """
    scope = state.get("permission_scope")
    if not isinstance(scope, PermissionScope):  # pragma: no cover - 由 _run_body 保证
        raise RuntimeError("State 里没有 permission_scope，拒绝以不限范围执行")
    deadline = state.get("deadline_at")
    return ToolContext(
        user_id=state["user_id"],
        task_id=state["task_id"],
        step_id=step.id,
        trace_id=state.get("trace_id", ""),
        permission_scope=scope,
        deadline_at=deadline if isinstance(deadline, datetime) else None,
    )


def _normalize(state: AgentState, step: TaskStep, result: ToolResult) -> dict[str, Any]:
    """`ToolResult` → State 的增量。

    `step_results` 与 `evidence` 用 reducer 合并，因此这里**只返回增量**：
    返回整份会让 `merge_by_id` 每次都对全量去重一遍，而更大的问题是
    返回全量会让"这一步到底新增了什么"看不出来。
    """
    # **两条路都认**：RAG 的"没有相关知识"是错误码（11.8 要求显式返回它），
    # 而 SQL 的 0 行是 `SUCCEEDED` + `payload.is_empty`——详设 9.4 明写
    # 「SQL 空集不算失败」，它是关于数据的事实。只认错误码的话，
    # 一条查不到数据的 SQL 会被当成"正常有结果"，`reflect` 就不会去补 RAG，
    # 而最终答案会拿一份空结果当证据。
    payload = result.payload or {}
    empty = (result.error is not None and result.error.code in _EMPTY_RESULT_CODES) or bool(
        payload.get("is_empty")
    )
    step_result = StepResult(
        step_id=step.id,
        status=StepStatus.SUCCEEDED if result.status != "FAILED" else StepStatus.FAILED,
        summary=_clip(result.summary),
        evidence_ids=tuple(item.id for item in result.evidence),
        error_code=result.error.code if result.error else None,
        duration_ms=result.duration_ms,
        empty=empty,
    )
    update: dict[str, Any] = {
        "step_results": {step.id: step_result},
        "evidence": list(result.evidence),
        "tool_calls": _tool_calls(step, result),
    }
    if result.evidence:
        # 发现（finding）**由代码从证据里提炼一行**，不让模型写：
        # 详设 13.5 明写 `investigation_chain` 由代码组装，
        # 而 finding 是它的输入——让模型总结自己刚才查了什么，
        # 总结出来的是"看起来合理的推理"，不是实际执行的那一步。
        update["findings"] = [
            Finding(
                id=new_id(IdPrefix.FINDING),
                statement=f"{step.objective}（{len(result.evidence)} 条证据）",
                step_id=step.id,
                evidence_ids=tuple(item.id for item in result.evidence),
            )
        ]
    # 见模块 docstring：空结果是事实、其余失败是错误，两者去处不同
    if result.error is not None and not empty:
        update["errors"] = [_as_agent_error(result.error)]
    return update


def _as_agent_error(error: ToolError) -> AgentError:
    """`ToolError`（工具的结果字段）→ `AgentError`（State 的错误类型）。

    **两者是不同层的类型，必须在这里换过来**：详设 7.1 定义 `errors` 是
    `list[AgentError]`，而 `ToolError` 是 `ToolResult` 上的一个**字段模型**，
    它的 `code` 是裸字符串、`error_class` 是 9.4 的分类（另一根轴）。
    直接把 `ToolError` 放进 State，`merge_errors` 会在 `.code.value` 上炸——
    而那只在"某一步真的失败"时才发生，正常路径永远测不到。

    未知错误码**落回 `INTERNAL_ERROR` 而不是抛**：19.1 的码表是封闭的，
    工具给出表外的码说明它自己有问题，但那时任务已经出错了，
    为了报一个错再抛一个错只会把原始信息盖掉。
    """
    try:
        code = ErrorCode(error.code)
    except ValueError:
        code = ErrorCode.INTERNAL_ERROR
    return AgentError(code, error.message, details={"error_class": error.error_class})


def _tool_calls(step: TaskStep, result: ToolResult) -> list[ToolCallRecord]:
    """`ToolResult` → `agent_tool_call` 的行（16.6）。

    **一次尝试一行**，不是一次调用一行：SQL 的自修复会让同一步骤产生
    「生成 → 修复 → 执行」多行，而"修复了几次、每次为什么失败"正是
    排查 SQL 生成质量的第一手材料（`SqlAttempt` 就是为它存在的）。
    RAG 没有自修复，所以恒为一行。

    `attempts` **只在 SQL 的 payload 里**（`SqlToolResult.attempts`）。
    拿不到时退化成一行 —— 那不是"少记了"，而是"这个工具没有分步尝试"。
    """
    attempts = (result.payload or {}).get("attempts") or []
    if not attempts:
        return [
            ToolCallRecord(
                step_id=step.id,
                tool_name=str(result.tool),
                attempt_no=1,
                status=result.status,
                error_code=result.error.code if result.error else None,
                error_summary=_clip(result.error.message) if result.error else None,
                duration_ms=result.duration_ms,
                result_summary={"summary": _clip(result.summary)},
            )
        ]
    return [
        ToolCallRecord(
            step_id=step.id,
            tool_name=str(result.tool),
            # **`request_summary` 不带原始 SQL 全文**：`normalized_sql` 单独一列
            # 是有意的（10.6 与 19.4 的脱敏纪律），混进 summary 会让它
            # 出现在本该只有摘要的地方。
            attempt_no=int(item.get("attempt_no") or index),
            status=str(item.get("status") or result.status),
            normalized_sql=item.get("normalized_sql"),
            sql_fingerprint=item.get("sql_fingerprint"),
            error_code=item.get("error_code"),
            # **`error_summary` 已经是脱敏过的**（`SqlAttempt` 的约定：
            # 存的是"哪一类错、错在第几个字符"，不是模型原文或库回显）。
            # 这里不再加工，加工只会把它变成另一个东西。
            error_summary=item.get("error_summary"),
            duration_ms=item.get("duration_ms"),
            result_summary={"stage": item.get("stage")},
        )
        for index, item in enumerate(attempts, start=1)
    ]


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SUMMARY_CHARS else f"{flat[:_SUMMARY_CHARS]}…"


def build_sql_node(settings: Settings, tool: Any) -> Callable[[AgentState], Any]:
    """SQL 节点。`tool` 是 `SqlQueryTool`，用 `Any` 标注是为了不与 `tools` 层耦合。"""

    async def sql_node(state: AgentState) -> dict[str, Any]:
        step = _find_step(state, "sql_query")
        if step is None:  # pragma: no cover - 路由保证有
            return {}
        from app.tools.sql.schemas import SqlQueryArgs

        args = SqlQueryArgs(
            question=state.get("user_query") or "",
            objective=step.objective,
        )
        result = await tool.execute(args, _context(state, step))
        return _normalize(state, step, result)

    return sql_node


def build_rag_node(settings: Settings, tool: Any) -> Callable[[AgentState], Any]:
    """RAG 节点。"""

    async def rag_node(state: AgentState) -> dict[str, Any]:
        step = _find_step(state, "rag_retrieve")
        if step is None:  # pragma: no cover - 路由保证有
            return {}
        from app.tools.rag.schemas import RagQueryArgs

        args = RagQueryArgs(
            question=state.get("user_query") or "",
            objective=step.objective,
            # **生效时点取问题所问区间的起点**：问「2025 Q3」时
            # v2.0（2025-07-01 生效）才是当时有效的那一版。
            # 取 end 会把区间末尾之后才生效的版本也算进来。
            as_of=_as_of_of(state.get("intent")),
        )
        result = await tool.execute(args, _context(state, step))
        return _normalize(state, step, result)

    return rag_node


def _as_of_of(intent: IntentResult | None) -> date | None:
    """意图的时间区间 → 生效区间过滤的基准日。

    `time_range` 是**半开区间**（详设 13.1 的 `TimeRange`），起点是"问的是从哪天起"，
    正是文档生效区间过滤该比的那一天。没给时间时返回 `None`，
    即"不限时点"——那时同名制度的两个版本会同时进候选，
    这是已知行为（`ChunkFilter.effective_at` 的说明）。
    """
    if intent is None or intent.time_range is None:
        return None
    return intent.time_range.start.astimezone(UTC).date()


def deadline_for(settings: Settings, started_at: datetime) -> datetime:
    """本任务的截止时刻。**绝对时刻**，不是"剩余秒数"（见 `ToolContext` 的说明）。"""
    return started_at + timedelta(seconds=settings.task_timeout_seconds)


__all__ = ["build_rag_node", "build_sql_node", "deadline_for"]
