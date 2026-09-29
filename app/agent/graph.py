"""最小 Graph 的拓扑与运行入口（冲刺方案 §8.1 / 详细设计 6.1 的裁剪版）。

```text
START → supervisor ─┬─(sql)──→ sql ──┐
                    ├─(rag)──→ rag ──┼→ reflect ─┬─(该补一路)──→ plan_extend ──┐
                    └─(analysis)─────┘           ├─(还有待执行)───────────────→│（回到上面两路）
                                                 └─(收敛)──→ conflict → analysis
                                                              → reviewer → retry_router
                                                                           ├─(重试)──→ 回到上面
                                                                           └─(收工)──→ final → END
```

九个节点：`supervisor / sql / rag / reflect / plan_extend / conflict /
analysis / reviewer / retry_router / final`（§8.1 的最小集是六个；
`conflict` 是 §8.3 第 6 项、`reviewer` 是第 7 项，`plan_extend` 与
`retry_router` 是详设 6.6.1 的任务循环与 14.4 的重试路由）。
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
| `plan_extend` | **`plan_extend` 节点** | 校验只做前提已具备的那几条，见 `nodes/plan_extend.py` |
| `conflict_detect` | **`conflict` 节点（切片版）** | 只做 VALUE 一类 |
| | | 四类缺前提的理由见 `nodes/conflict.py` |
| `reviewer` | **`reviewer` 节点（两阶段）** | 确定性六条 + 14.1 第二阶段的模型审查 |
| `retry_router` | **`retry_router` 节点** | 14.4 的逐字实现，五个目标都有落点 |
| `clarify` | 无 | 澄清以答案文本表达，状态位见【后续扩展】 |
| `evidence_aggregate` | 无 | 证据由 `tool_nodes._normalize` 直接汇聚，没有单独的聚合步 |

**这份表是面试口径的一部分**：说"做了最小 Graph"时，被问"详设里那 20 个节点呢"
要能一条条说清它们去哪了，而不是笼统地说"简化了"。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes.analysis import build_analysis_node
from app.agent.nodes.conflict import build_conflict_node
from app.agent.nodes.final import build_final_node
from app.agent.nodes.plan_extend import build_plan_extend_node
from app.agent.nodes.reflect import build_reflect_node
from app.agent.nodes.retry_router import build_retry_router_node
from app.agent.nodes.reviewer import build_reviewer_node
from app.agent.nodes.supervisor import build_supervisor_node
from app.agent.nodes.tool_nodes import build_rag_node, build_sql_node, deadline_for
from app.agent.schemas.plan import StepStatus
from app.agent.state import AgentState, Route, pending_steps
from app.agent.tracing import EventPublisher, traced
from app.core.config import Settings
from app.domain.memory import ConversationContext
from app.domain.task import (
    PlanRevisionRecord,
    ReviewRecord,
    StepRecord,
    Task,
    TaskOutcome,
    TaskStatus,
)
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
_NODE_PLAN_EXTEND = "plan_extend"
_NODE_CONFLICT = "conflict"
_NODE_ANALYSIS = "analysis"
_NODE_REVIEWER = "reviewer"
_NODE_RETRY_ROUTER = "retry_router"
_NODE_FINAL = "final"

#: `Route` → 节点名。`CLARIFY` / `FAIL` 都收敛到 `final`：
#: 澄清与失败都是"给出一个面向用户的说明"，而 6 个节点的版本里没有
#: 独立的 `clarify` / `finalize_failure` 节点（登记为【后续扩展】）。
_TARGETS: dict[Route, str] = {
    Route.SQL: _NODE_SQL,
    Route.RAG: _NODE_RAG,
    #: 计划演进（`retry_target=expand`）。14.3 明写"由 `plan_extend` 决定
    #: 具体步骤，而不是由 Reviewer 直接指定 SQL"。
    Route.EXPAND: _NODE_PLAN_EXTEND,
    Route.ANALYSIS: _NODE_CONFLICT,
    Route.CLARIFY: _NODE_FINAL,
    Route.FAIL: _NODE_FINAL,
    #: 作废整份计划回 supervisor（`retry_router` 的 `REPLAN`）。**只回这里，
    #: 不回别处**：`_after_supervisor` 的候选列表里有 supervisor 的出口，
    #: 而从别处进 supervisor 会让"supervisor 只跑一次"这条前提失效。
    Route.REPLAN: _NODE_SUPERVISOR,
}


#: 直线路径上的节点数（含余量）：`supervisor → sql/rag → reflect` 这一段
#: 是固定的，加上 `conflict → analysis → reviewer → retry_router → final` 的收尾。
_FIXED_NODES = 12

#: 每绕一圈回边最多多走几个节点：`plan_extend`/工具 + `reflect` +
#: `conflict` + `analysis` + `reviewer` + `retry_router`。
_NODES_PER_LOOP = 7


def _recursion_limit(settings: Settings) -> int:
    """LangGraph 的递归上限（超级步计数）。

    **从循环预算推出来，不写死。** 它原来是一个硬编码的 25，而循环那边
    现在有三种会绕圈的预算（演进、Reviewer 补证、重新规划）——写死的话，
    某天把 `LOOP__MAX_EXPANSIONS` 从 2 调到 3 就可能撞上它，而撞上它的
    产物是 `INTERNAL_ERROR` + `trace_incomplete` + 五张产出表全空
    （约定 102 记的那副样子）：**排查方向会跑到"图坏了"上，而真正的原因
    是一个配置项**。

    它仍然是**最后一道兜底**，管的是"回边写错、绕不完"；语义上的界是
    四类循环预算与 `_over_step_budget` 那道护栏。所以这里的余量给得宽：
    撞上它意味着代码有问题，不是一个正常结局。

    **不设它的话**，回边写错会一直绕到进程 OOM，而那时看到的是内存曲线。
    """
    cycles = (
        settings.loop.max_expansions
        + settings.loop.max_replans
        + settings.loop.max_reviewer_evidence
    )
    return _FIXED_NODES + _NODES_PER_LOOP * cycles


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
    """`reflect → ?`：判了 EXPAND 就去演进，还有待执行就继续，否则收敛。

    **判 EXPAND 时要先看 `progress_assessment`，不能只看 `route_dispatch`**：
    `reflect` 已经不再自己追加步骤（那是 `plan_extend` 的事），所以此刻
    `task_list` 里还没有那一步，`route_dispatch` 会直接说"没有待执行的"
    而收敛到 `conflict`——**演进就永远不发生**，而任务照跑照结束，
    只是 `plan_deltas` 恒空、`expansions_left` 恒为初值。

    **收敛的出口是 `conflict` 而不是 `analysis`**：冲突检测要的是
    "全部证据都到齐了"这个时点，而它就在 `reflect` 判 SUFFICIENT 的那一刻。
    """
    assessment = state.get("progress_assessment")
    if assessment is not None and assessment.decision == "EXPAND" and assessment.proposed_steps:
        return _NODE_PLAN_EXTEND
    return _TARGETS[route_dispatch(state)]


def _after_plan_extend(state: AgentState) -> str:
    """`plan_extend → ?`：新步骤下来就分派，被拒了就收敛。

    **两条出口只是同一个 `route_dispatch` 的两种结果**：步骤全被拒时
    没有 PENDING 步骤，它自然回 `ANALYSIS`（落到 `conflict`）——
    那正是 6.6.3 的"按 SUFFICIENT 收敛"。
    """
    return _TARGETS[route_dispatch(state)]


def _over_step_budget(state: AgentState, limit: int) -> bool:
    """总步数超限（详设 5.4 的 `max_total_steps`）——**最后一道护栏**。

    判据是**"已经跑了多少步"**而不是"绕了多少圈"：回边写错时绕圈的症状正是
    步数停不下来，而"圈数"在 State 里没有载体（`plan_revision` 只数演进，
    补证与 replan 都不加它）。`step_results` 的键是 step_id，而每次重试都
    **追加新步骤**（`retry_router._next_step_id`），所以它数得准。

    ⚠️ 在它之前唯一的护栏是 LangGraph 的 `recursion_limit`，而那条路走到底
    的产物是 `INTERNAL_ERROR` + `trace_incomplete` + 五张表全空——**一次
    "什么东西坏了"的收尾，而不是一个说得清的结论**。这条护栏的意义就是
    让"用尽了"走正常出口（受限回答），而不是让它变成一次内部错误。

    ⚠️ **它不到 `final`，到 `conflict`**：那里才有正常的收尾链
    （`conflict → analysis → reviewer → final`）。直接去 `final` 会拿到
    `analysis is None` 的失败答案——而超限不是失败，是"不再尝试新的取证路径"
    （FR-REV-002 业务规则 3 的原话）。
    """
    return len(state.get("step_results") or {}) >= limit


def _after_retry(state: AgentState) -> str:
    """`retry_router -> ?`：`retry_route` 说去哪就去哪；没有就收工去 `final`。

    **它读 `retry_route` 而不是从 State 推**：`retry_target=analysis`
    （重跑分析）与"不重试了"在 State 上长得一模一样——两者都没有 PENDING
    步骤——条件边推不出来。这条偏离要在 `nodes/retry_router.py` 的模块说明里。
    """
    route = state.get("retry_route")
    return _NODE_FINAL if route is None else _TARGETS[route]


def _guard(route: Callable[[AgentState], str], settings: Settings) -> Callable[[AgentState], str]:
    """把「总步数超限」这道护栏套在一条回边的判据上。

    **套在回边上，不套在每个节点里**：会绕圈的只有回边，而节点有八个——
    在八个地方各判一次，漏掉的那个就是绕不完的那一条（同 `tracing.traced`
    用包装器而不是每个节点里写一遍的理由）。
    """

    def guarded(state: AgentState) -> str:
        if _over_step_budget(state, settings.loop.max_total_steps):
            return _NODE_CONFLICT
        return route(state)

    return guarded


def _add_node(
    graph: StateGraph[AgentState],
    name: str,
    node: Any,
    events: EventPublisher | None = None,
) -> None:
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
    wrapped: Any = traced(name, node, events)
    graph.add_node(name, wrapped)


def build_graph(
    settings: Settings,
    *,
    gateway: ModelGateway,
    sql_tool: Any,
    rag_tool: Any,
    catalog: Any | None = None,
    events: EventPublisher | None = None,
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

    _add_node(graph, _NODE_SUPERVISOR, build_supervisor_node(settings, gateway), events)
    _add_node(graph, _NODE_SQL, build_sql_node(settings, sql_tool), events)
    _add_node(graph, _NODE_RAG, build_rag_node(settings, rag_tool), events)
    _add_node(graph, _NODE_REFLECT, build_reflect_node(), events)
    _add_node(graph, _NODE_PLAN_EXTEND, build_plan_extend_node(settings, gateway), events)
    _add_node(graph, _NODE_CONFLICT, build_conflict_node(catalog), events)
    _add_node(graph, _NODE_ANALYSIS, build_analysis_node(gateway), events)
    _add_node(graph, _NODE_REVIEWER, build_reviewer_node(gateway), events)
    _add_node(graph, _NODE_RETRY_ROUTER, build_retry_router_node(), events)
    _add_node(graph, _NODE_FINAL, build_final_node(), events)

    graph.add_edge(START, _NODE_SUPERVISOR)
    graph.add_conditional_edges(
        _NODE_SUPERVISOR,
        _after_supervisor,
        [_NODE_SQL, _NODE_RAG, _NODE_ANALYSIS, _NODE_CONFLICT, _NODE_FINAL],
    )
    graph.add_edge(_NODE_SQL, _NODE_REFLECT)
    graph.add_edge(_NODE_RAG, _NODE_REFLECT)
    # **回边**：`reflect → plan_extend → sql/rag` 就是详设 6.1 的
    # `reflect -> plan_extend -> dispatch`。三个节点都在了，形状与图一致。
    graph.add_conditional_edges(
        _NODE_REFLECT,
        _guard(_after_reflect, settings),
        [_NODE_PLAN_EXTEND, _NODE_SQL, _NODE_RAG, _NODE_CONFLICT],
    )
    # 详设 6.1 的 `plan_extend → dispatch`：`dispatch` 在本版是条件边函数
    # （`route_dispatch`），所以这里直接接它。**被拒时也走这条边**——
    # 那时没有 PENDING 步骤，`route_dispatch` 自然给 `ANALYSIS`。
    graph.add_conditional_edges(
        _NODE_PLAN_EXTEND,
        _guard(_after_plan_extend, settings),
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
    # 详设 6.6.1：`reviewer → retry_router → Tool/analysis`。
    # **`retry_router` 是节点而不是光一个条件边**：它要消耗预算、要追加步骤，
    # 而条件边是纯函数，写不了 State（见 `nodes/retry_router.py`）。
    graph.add_edge(_NODE_REVIEWER, _NODE_RETRY_ROUTER)
    graph.add_conditional_edges(
        _NODE_RETRY_ROUTER,
        _guard(_after_retry, settings),
        [
            _NODE_SQL,
            _NODE_RAG,
            _NODE_CONFLICT,
            _NODE_SUPERVISOR,
            _NODE_PLAN_EXTEND,
            _NODE_FINAL,
        ],
    )
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

    async def run(
        self,
        task: Task,
        *,
        permission_scope: Any,
        context: ConversationContext | None = None,
    ) -> TaskOutcome:
        """跑一个任务，返回最终答案（Markdown）。

        `permission_scope` 与 `context` **都由调用方传入**而不是在这里查：
        调用方（`build_task_body`）已经持有仓储，而这一层不应该再依赖它们——
        它拿到一个范围、一份会话摘要就够跑图了。

        `context` 是 FR-CHAT-003 的会话摘要（`None` = 首轮或没有历史）。
        **不做成 `memory` 节点**：详设 7.1 说它由 `memory` 写入，而 6.1/6.2 的
        节点清单里没有 `memory`；更重要的是它是**输入装配**而不是决策——
        与 `permission_scope` 同性质，混进图里会让"图跑了几步"多出一跳，
        而那一跳什么都不决定。
        """
        started = datetime.now(UTC)
        initial: AgentState = {
            # 用户原话：**只用来给人看**（落 `agent_task.query_text`）。
            "user_query": task.query_text,
            # 本轮问题：初值就是原话，supervisor 解析代词后会改写它。
            # 下游一律读 `state.current_question()`，不直接读上面那个。
            "sanitized_query": task.query_text,
            "user_id": task.user_id,
            "conversation_id": task.conversation_id or "",
            "context_summary": context,
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
            "plan_deltas": [],
        }
        with span(
            "agent.graph",
            **{"task.id": task.id, "conversation.id": task.conversation_id or ""},
        ) as current:
            result: dict[str, Any] = await self._graph.ainvoke(
                initial,
                config={"recursion_limit": _recursion_limit(self._settings)},
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
            # 六张表的行，一次带出去（见 `TaskOutcome` 的说明）
            steps=_step_records(final_state),
            revisions=_revision_records(final_state),
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
    introduced = _origin_index(state)
    return {
        "revision": state.get("plan_revision", 0),
        # 计划演进的记录一并进摘要：**"计划为什么变成这样"是读这份摘要的
        # 人要回答的问题之一**，而只有 steps 的话，看得出多了一步、
        # 看不出它为什么被加进来。同时它是 `trigger_finding_id` 唯一的出口
        # ——那个 id 在 `steps` 里只对一个 EXTENDED 步骤有意义。
        "deltas": [
            {
                "revision_no": delta.revision_no,
                "trigger": delta.trigger,
                "trigger_finding_id": delta.trigger_finding_id,
                "added_step_ids": list(delta.added_step_ids),
                "skipped_step_ids": list(delta.skipped_step_ids),
                "reason": delta.reason,
            }
            for delta in state.get("plan_deltas") or []
        ],
        "steps": [
            {
                "id": step.id,
                "tool": step.tool,
                "objective": step.objective,
                "status": results[step.id].status.value if step.id in results else "PENDING",
                "empty": results[step.id].empty if step.id in results else False,
                "summary": results[step.id].summary if step.id in results else "",
                "origin": "EXTENDED" if step.id in introduced else "PLANNER",
                "revision_no": introduced.get(step.id, 0),
            }
            for step in state.get("task_list") or []
        ],
        "decision": assessment.decision if assessment is not None else None,
        "reason": assessment.reason if assessment is not None else None,
    }


def _origin_index(state: AgentState) -> dict[str, int]:
    """`step_id` → **引入它的那一版计划号**（16.6 的 `origin` / `revision_no`）。

    判据是"这个 id 出现在哪一条 `PlanDelta.added_step_ids` 里"：
    在 → `EXTENDED`，不在 → 最初那一版（0，`PLANNER`）。

    **不能用任务的终值代替**（这里原先就是那么写的）：那样每次演进都会把
    **所有**步骤（含初始那几条）标成新版本，于是"哪些步骤是下钻出来的"
    在表里查不到——而表看起来完全正常。开发流程 7.5 的循环类评分
    正是靠这个区分。

    ## 一处如实记下的不精确：`retry_router` 补的那一步

    14.4 的重试会给计划追加一个新步骤（"补证"），而它**不在任何 delta 的
    `added_step_ids` 里**——补证不是计划演进，`plan_revision` 不升、
    `plan_deltas` 也不记。于是它落成 `PLANNER` / `revision_no=0`，
    而它既不是初始计划里的、也不是第 0 版引入的。

    16.6 给 `origin` 的三个取值（`PLANNER` / `EXTENDED` / `REPLAN`）里
    没有第四个位置可放，而**多造一个取值就是改 16.6 的枚举**——
    那是要回写文档的偏离，不该顺手做。当前取值对 7.5 要问的那个问题
    （"哪些步骤是下钻出来的"）恰好是对的：补证步骤**不该**被算作下钻。
    代价是 `revision_no` 那一列对这几行没有意义，已登记。
    """
    index: dict[str, int] = {}
    for delta in state.get("plan_deltas") or []:
        for step_id in delta.added_step_ids:
            # **先到先得**：同一个 id 不会被两条 delta 收录
            # （`plan_extend` 只分配未被占用的 id），所以这里取第一条即是
            # 它被引入的那一版。
            index.setdefault(step_id, delta.revision_no)
    return index


def _step_records(state: AgentState) -> tuple[StepRecord, ...]:
    """`task_list` + `step_results` → `agent_task_step` 的行（16.6）。

    **两个来源缺一不可**：计划给出"本来要做哪几步"，结果给出"做成了没有"。
    只落计划的话，读的人看不出"那一步其实没查到东西"。
    """
    results = state.get("step_results") or {}
    introduced = _origin_index(state)
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
            origin="EXTENDED" if step.id in introduced else "PLANNER",
            revision_no=introduced.get(step.id, 0),
        )
        for step in state.get("task_list") or []
    )


def _revision_records(state: AgentState) -> tuple[PlanRevisionRecord, ...]:
    """`plan_deltas` → `agent_plan_revision` 的行（16.6）。

    `budget_snapshot` 取**收尾时**的余额。它回答的是"这次演进之后还剩多少"，
    而"这次绕圈是不是借了别的预算"要靠与上一行的差额看——四类预算各自独立
    是 FR-REV-002 业务规则 1 的要求，而余额快照是唯一能复盘它的东西。
    """
    return tuple(
        PlanRevisionRecord(
            revision_no=delta.revision_no,
            trigger_type=delta.trigger,
            trigger_finding_id=delta.trigger_finding_id,
            added_step_ids=delta.added_step_ids,
            skipped_step_ids=delta.skipped_step_ids,
            reason=delta.reason,
            budget_snapshot={
                "expansions_left": state.get("expansions_left", 0),
                "review_retries_left": state.get("review_retries_left", 0),
                "replans_left": state.get("replans_left", 0),
            },
        )
        for delta in state.get("plan_deltas") or []
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
