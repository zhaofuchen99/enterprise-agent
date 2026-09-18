"""最小 Graph 的拓扑与运行入口（冲刺方案 §8.1 / 详细设计 6.1 的裁剪版）。

```text
START → supervisor ─┬─(sql)──→ sql ──┐
                    ├─(rag)──→ rag ──┼→ reflect ─┬─(还有待执行)──→ dispatch（回到上面两路）
                    └─(analysis)─────┘           └─(收敛)──────────→ analysis → final → END
```

八个节点：`supervisor / sql / rag / reflect / conflict / analysis / reviewer / final`。
（§8.1 的最小集是六个；`conflict` 是 §8.3 第 6 项「Evidence（含简化冲突检测）」、
`reviewer` 是第 7 项「Reviewer-lite」按详设 6.1 加上的。）
**`dispatch` 不是节点，是条件边函数**——详设 6.1 里它单独成节点是因为
计划可能有多条带依赖的步骤；本版的计划是"每条数据源一步、互不依赖"，
"找下一步"就退化成一个纯函数（`route_dispatch`），
让它当节点只会多一次 State 往返。

## 与详设 6.1 的偏差清单（都记在案）

| 详设节点 | 本版 | 归属 |
|---|---|---|
| `input_guard` | 无 | 输入清洗在 API 层做（`api/schemas.py` 的长度与格式校验） |
| `planner` | 合进 `supervisor` | 确定性计划，见 `nodes/supervisor.py` |
| `dispatch` | 条件边函数 | 见上 |
| `sql_prepare/generate/validate/execute` | 合进 `sql` 节点 | 七个子步骤活在 `SqlQueryTool` 内部 |
| `rag_rewrite/retrieve/rerank` | 合进 `rag` 节点 | 同上 |
| `normalize_tool_result` | 合进 `sql`/`rag` 节点 | `nodes/tool_nodes._normalize` |
| `plan_extend` | 无（`reflect` 直接改计划） | 【后续扩展】模型的 EXPAND 判定 |
| `conflict_detect` | **`conflict` 节点（切片版）** | 只做 VALUE 一类 |
| | | 四类缺前提的理由见 `nodes/conflict.py` |
| `reviewer` | **`reviewer` 节点（第一阶段）** | 确定性检查；模型审查属 Phase 8 |
| `retry_router` | 无 | 【Phase 8】`RETRY` / `CLARIFY` 两个状态要有预算与续跑入口 |
| `clarify` | 无 | 澄清以答案文本表达，状态位见【后续扩展】 |

**这份表是面试口径的一部分**：说"做了最小 Graph"时，被问"详设里那 20 个节点呢"
要能一条条说清它们去哪了，而不是笼统地说"简化了"。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes.analysis import build_analysis_node
from app.agent.nodes.conflict import build_conflict_node
from app.agent.nodes.final import build_final_node
from app.agent.nodes.reflect import build_reflect_node
from app.agent.nodes.reviewer import build_reviewer_node
from app.agent.nodes.supervisor import build_supervisor_node
from app.agent.nodes.tool_nodes import build_rag_node, build_sql_node, deadline_for
from app.agent.schemas.plan import StepStatus
from app.agent.state import AgentState, Route, pending_steps
from app.agent.tracing import traced
from app.core.config import Settings
from app.domain.task import ReviewRecord, StepRecord, Task, TaskOutcome, TaskStatus
from app.infrastructure.model_gateway import ModelGateway
from app.infrastructure.observability import span

#: 节点名。**写成常量而不是散在字符串里**：条件边的候选列表与 `add_node`
#: 引用的是同一批名字，而两处各写一遍字符串时，改一处漏一处的报错是
#: "节点不存在"——一个只说结果不说原因的错。
#:
#: 条件边返回的是 `Route` 枚举、由 `_TARGETS` 映射到这些名字：
#: 枚举值恰好与节点名同形是巧合，不该依赖巧合。
_NODE_SUPERVISOR = "supervisor"
_NODE_SQL = "sql"
_NODE_RAG = "rag"
_NODE_REFLECT = "reflect"
_NODE_CONFLICT = "conflict"
_NODE_ANALYSIS = "analysis"
_NODE_REVIEWER = "reviewer"
_NODE_FINAL = "final"

#: `Route` → 节点名。`CLARIFY` / `FAIL` 都收敛到 `final`：
#: 澄清与失败都是"给出一个面向用户的说明"，而 6 个节点的版本里没有
#: 独立的 `clarify` / `finalize_failure` 节点（登记为【后续扩展】）。
_TARGETS: dict[Route, str] = {
    Route.SQL: _NODE_SQL,
    Route.RAG: _NODE_RAG,
    Route.ANALYSIS: _NODE_CONFLICT,
    Route.CLARIFY: _NODE_FINAL,
    Route.FAIL: _NODE_FINAL,
}


def route_dispatch(state: AgentState) -> Route:
    """`supervisor` 与 `reflect` 之后共同的出口：下一步去哪。

    **它读的是"还没跑的步骤"，不是 `next_route`**：`next_route` 是节点写的
    一个字段，而字段会被覆盖、会过期。从 `task_list` 与 `step_results`
    现推出来的结果不会——那两个字段的合并语义由 reducer 保证。

    没有待执行步骤时回 `ANALYSIS`（详设 6.3 的 `route_dispatch` 同）。
    """
    pending = pending_steps(state)
    if not pending:
        return Route.ANALYSIS
    step = pending[0]
    return Route.SQL if step.tool == "sql_query" else Route.RAG


def _after_supervisor(state: AgentState) -> str:
    """`supervisor → ?`：失败/澄清走 `final`，否则按步骤分派。"""
    if state.get("execution_status") is TaskStatus.FAILED:
        return _NODE_FINAL
    if not (state.get("task_list") or []):
        # CLARIFICATION / UNSUPPORTED：没有可执行步骤，直接去分析（会走
        # 无证据分支）还是去 final？——**去 final**：这类问题不需要"分析"，
        # 而 `analysis` 在无证据时会产出"没有检索到内容"，
        # 那句话对"你问的年度没说清"这种澄清场景是答非所问。
        return _NODE_FINAL
    return _TARGETS[route_dispatch(state)]


def _after_reflect(state: AgentState) -> str:
    """`reflect → ?`：还有待执行就继续，否则收敛到 `conflict`（再进 `analysis`）。

    **收敛的出口是 `conflict` 而不是 `analysis`**：冲突检测要的是
    "全部证据都到齐了"这个时点，而它就在 `reflect` 判 SUFFICIENT 的那一刻。
    """
    route = route_dispatch(state)
    return _TARGETS[route]


def _add_node(graph: StateGraph[AgentState], name: str, node: Any) -> None:
    """注册节点。

    **`graph` 的类型必须写全 `StateGraph[AgentState]`，不能省成 `StateGraph`。**
    省掉之后 `add_node` 的 `NodeInputT` 就推不出来了——它的实参是
    **工厂函数返回的 `Callable[[AgentState], Any]`**（一个变量），
    而 mypy 要从 `_Node[NodeInputT] | Runnable[...]` 这个联合里反解；
    把同样的函数写成内联 `async def` 就能过。这条是拿最小复现试出来的，
    写成 `Any` 或加 `type: ignore` 都能让它闭嘴，但那会把这个文件里
    唯一一处能验证"节点签名与 State 对得上"的检查也一并关掉。
    """
    # 包一层埋点（`tracing.traced`）。**赋给 `Any` 变量**：见本函数的 docstring，
    # mypy 对"工厂返回的可调用对象"解不出 `NodeInputT`，而 `traced` 的返回类型
    # 正是那样一个对象。标注成 `Any` 比再加一个 `type: ignore` 诚实——
    # 节点本身的类型在各自的工厂函数上有精确标注。
    wrapped: Any = traced(name, node)
    graph.add_node(name, wrapped)


def build_graph(
    settings: Settings,
    *,
    gateway: ModelGateway,
    sql_tool: Any,
    rag_tool: Any,
    catalog: Any | None = None,
) -> CompiledStateGraph[AgentState]:
    """组装并编译图。

    **不在这里 `compile(checkpointer=...)`**：检查点（断点续跑）需要一张
    持久化表与一套恢复语义，而本项目的任务级重试是"整任务重跑"
    （`agent_task` 的 `max_requeue_attempts`）。接检查点等于引入第二种
    重试语义，两者并存时会出"重投了一个已经跑了一半的任务"这类问题。
    登记为【后续扩展】。

    Raises:
        ValueError: 图装配错误（节点/边引用了未注册的节点名）。
            **编译期抛比运行期抛好**：路由函数返回一个不存在的节点名时，
            LangGraph 要到那个分支真的被走到才报错，而那时已经跑了一半。
    """
    graph = StateGraph(AgentState)

    _add_node(graph, _NODE_SUPERVISOR, build_supervisor_node(settings, gateway))
    _add_node(graph, _NODE_SQL, build_sql_node(settings, sql_tool))
    _add_node(graph, _NODE_RAG, build_rag_node(settings, rag_tool))
    _add_node(graph, _NODE_REFLECT, build_reflect_node())
    _add_node(graph, _NODE_CONFLICT, build_conflict_node(catalog))
    _add_node(graph, _NODE_ANALYSIS, build_analysis_node(gateway))
    _add_node(graph, _NODE_REVIEWER, build_reviewer_node())
    _add_node(graph, _NODE_FINAL, build_final_node())

    graph.add_edge(START, _NODE_SUPERVISOR)
    graph.add_conditional_edges(
        _NODE_SUPERVISOR,
        _after_supervisor,
        [_NODE_SQL, _NODE_RAG, _NODE_ANALYSIS, _NODE_FINAL],
    )
    graph.add_edge(_NODE_SQL, _NODE_REFLECT)
    graph.add_edge(_NODE_RAG, _NODE_REFLECT)
    # **回边**：`reflect → sql/rag` 就是详设 6.1 里 `reflect -> plan_extend -> dispatch`
    # 那条任务循环的回边。本版没有 `plan_extend` 节点，演进由 `reflect` 直接改写计划。
    graph.add_conditional_edges(
        _NODE_REFLECT,
        _after_reflect,
        [_NODE_SQL, _NODE_RAG, _NODE_CONFLICT],
    )
    # 详设 6.1 的顺序是 `evidence_aggregate → conflict_detect → analysis`：
    # **冲突在分析之前算好**，让模型写结论时就知道哪里对不上，
    # 而不是写完再补一段"此外还有冲突"。
    graph.add_edge(_NODE_CONFLICT, _NODE_ANALYSIS)
    # 详设 6.1 的顺序是 `analysis → reviewer → final_answer`：审查的是
    # **草稿**（`analysis_result`），而不是渲染后的 Markdown——
    # 渲染会丢掉结构（claim 与引用的对应关系），从文本反推回结构是错的方向。
    graph.add_edge(_NODE_ANALYSIS, _NODE_REVIEWER)
    graph.add_edge(_NODE_REVIEWER, _NODE_FINAL)
    graph.add_edge(_NODE_FINAL, END)
    return graph.compile()


class TaskGraph:
    """图 + 它依赖的外部资源（工具与网关）的生命周期。

    做成一个对象而不是散在 `worker.py` 里：`SqlQueryTool` 持有数据库会话工厂、
    `RagRetrieveTool` 持有向量库与模型网关，这些都要在 Worker 停机时
    按顺序关闭。所有权集中在一处，才不会出现"关了一个忘了另一个"。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        graph: CompiledStateGraph[AgentState],
        sql_tool: Any,
        rag_tool: Any,
        gateway: ModelGateway,
    ) -> None:
        self._settings = settings
        self._graph = graph
        self._sql_tool = sql_tool
        self._rag_tool = rag_tool
        self._gateway = gateway

    async def run(self, task: Task, *, permission_scope: Any) -> TaskOutcome:
        """跑一个任务，返回最终答案（Markdown）。

        `permission_scope` **由调用方传入**而不是在这里从 `task.user_id` 查：
        调用方（`_run_body`）已经持有仓储，而这一层不应该再依赖用户仓储——
        它拿到一个 `PermissionScope` 就够跑图了。
        """
        started = datetime.now(UTC)
        initial: AgentState = {
            "user_query": task.query_text,
            "sanitized_query": task.query_text,
            "user_id": task.user_id,
            "conversation_id": task.conversation_id or "",
            "task_id": task.id,
            "trace_id": task.trace_id,
            "permission_scope": permission_scope,
            "deadline_at": deadline_for(self._settings, started),
            "evidence": [],
            "errors": [],
            "step_results": {},
            "findings": [],
            "task_list": [],
            "plan_revision": 0,
        }
        with span(
            "agent.graph",
            **{"task.id": task.id, "conversation.id": task.conversation_id or ""},
        ) as current:
            result: dict[str, Any] = await self._graph.ainvoke(
                initial,
                # 递归上限：图的正常路径最多 6 个节点 + 一次演进（2 步），
                # 给 25 是留足余量又能在"回边失控"时立刻停住。
                # **不设它的话，回边写错会一直绕到进程 OOM**，
                # 而那时看到的是内存曲线而不是"图跑飞了"。
                config={"recursion_limit": 25},
            )
            current.set_attribute("agent.steps", len(result.get("step_results") or {}))
            current.set_attribute("agent.evidence", len(result.get("evidence") or []))
            assessment = result.get("progress_assessment")
            if assessment is not None:
                current.set_attribute("agent.decision", assessment.decision)
        final_state: AgentState = result  # type: ignore[assignment]
        errors = final_state.get("errors") or []
        failed = final_state.get("execution_status") is TaskStatus.FAILED
        return TaskOutcome(
            # **图判成失败时，任务级状态也要失败**（见 `TaskOutcome.failed`）。
            # 取第一条错误进 `error_code`——多条并列时第一条通常是根因。
            failed=failed,
            error_code=errors[0].code.value if failed and errors else None,
            error_message=errors[0].message if failed and errors else None,
            answer=final_state.get("final_answer"),
            intent=getattr(final_state.get("intent"), "intent", None),
            # **计划摘要落库的形态**：步骤 + 修订号 + 最终判定。
            # 不落整份 `task_list` 的 pydantic dump——那是实现细节，
            # 而这三样才是"它是怎么查出来的"要回答的问题。
            plan=_plan_digest(final_state),
            payload=final_state.get("answer_payload"),
            trace_events=tuple(final_state.get("trace_events") or ()),
            # 五张表的行，一次带出去（见 `TaskOutcome` 的说明）
            steps=_step_records(final_state),
            tool_calls=tuple(final_state.get("tool_calls") or ()),
            evidence=tuple(final_state.get("evidence") or ()),
            conflicts=tuple(final_state.get("conflicts") or ()),
            review=_review_record(final_state),
        )

    async def aclose(self) -> None:
        """按依赖顺序关闭。**工具自己的 `aclose` 不关共享的网关与向量库**
        （那是装配点的责任），所以这里显式关网关。"""
        await self._sql_tool.aclose()
        await self._rag_tool.aclose()
        await self._gateway.aclose()


def _plan_digest(state: AgentState) -> dict[str, Any]:
    """`task_list` + `step_results` + 判定 → 落 `plan_json` 的摘要（16.5）。

    **每步的成败必须一起落**：只记"计划里有哪几步"的话，读的人看不出
    "那一步其实没查到东西"——而结论的硬度正是由这个决定的。
    """
    results = state.get("step_results") or {}
    assessment = state.get("progress_assessment")
    return {
        "revision": state.get("plan_revision", 0),
        "steps": [
            {
                "id": step.id,
                "tool": step.tool,
                "objective": step.objective,
                "status": results[step.id].status.value if step.id in results else "PENDING",
                "empty": results[step.id].empty if step.id in results else False,
                "summary": results[step.id].summary if step.id in results else "",
            }
            for step in state.get("task_list") or []
        ],
        "decision": assessment.decision if assessment is not None else None,
        "reason": assessment.reason if assessment is not None else None,
    }


def _step_records(state: AgentState) -> tuple[StepRecord, ...]:
    """`task_list` + `step_results` → `agent_task_step` 的行（16.6）。

    **两个来源缺一不可**：计划给出"本来要做哪几步"，结果给出"做成了没有"。
    只落计划的话，读的人看不出"那一步其实没查到东西"。
    """
    results = state.get("step_results") or {}
    return tuple(
        StepRecord(
            step_key=step.id,
            objective=step.objective,
            tool=step.tool,
            depends_on=step.depends_on,
            required=step.required,
            status=(
                results[step.id].status.value if step.id in results else StepStatus.PENDING.value
            ),
            result_summary=(
                {"summary": results[step.id].summary, "empty": results[step.id].empty}
                if step.id in results
                else None
            ),
            # **`origin` 恒为 PLANNER**：切片内没有 plan_extend，
            # 演进由 reflect 直接追加步骤。接 Phase 7 时这里要按
            # `plan_deltas` 区分 EXTENDED，否则"哪些步骤是下钻出来的"
            # 在表里查不到——而开发流程 7.5 正是靠它做循环类评分的。
            origin="PLANNER",
            revision_no=state.get("plan_revision", 0),
        )
        for step in state.get("task_list") or []
    )


def _review_record(state: AgentState) -> ReviewRecord | None:
    review = state.get("review_result")
    if review is None:
        return None
    return ReviewRecord(
        status=review.status,
        score=review.score,
        coverage_score=review.coverage_score,
        evidence_score=review.evidence_score,
        consistency_score=review.consistency_score,
        issues=tuple(issue.model_dump(mode="json") for issue in review.issues),
        missing_evidence=review.missing_evidence,
        retry_target=review.retry_target,
        reason_code=review.reason_code,
    )


__all__ = ["TaskGraph", "build_graph", "route_dispatch"]
