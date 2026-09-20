"""最小 Graph 的端到端行为（冲刺方案 §8.1 的两条必须保留项）。

这里测的是**图跑起来之后会发生什么**，不是单个节点：路由是否真的在选、
循环是否真的会补一路、收敛是否真的会停。三条都对应 §8.4 的验收：

| 验收 | 这里的哪条用例 |
|---|---|
| ① LLM 能规划任务 | `test_simple_query_only_calls_sql`（`required_sources` 驱动计划） |
| ② Agent 能自主选择 SQL / RAG | `test_routing_follows_the_intent`（单路与双路各测一遍） |
| ③ Agent 能根据工具结果继续分析 | `test_empty_sql_triggers_a_rag_expansion` |

工具用替身：`SqlQueryTool` 与 `RagRetrieveTool` 都要连真库/向量库，
而这里要断言的判据（哪一路被调了、调了几次）与它们内部的正确性无关——
那两条链路各自有 `make sql` / `make eval-rag` 与契约测试覆盖。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from app.agent.graph import build_graph
from app.agent.schemas.plan import IntentResult
from app.core.config import Settings, get_settings
from app.core.errors import AgentError, ErrorCode
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence
from app.domain.task import Task, TaskStatus
from app.domain.user import PermissionScope, UserRole
from app.tests.fakes import FakeModelGateway
from app.tools.base import ToolContext, ToolError, ToolResult


def _evidence(index: int, source: str) -> Evidence:
    return Evidence(
        # **用真的 id 生成器**：`Evidence.id` 有格式校验（`evd_` + 22 位 Base32），
        # 替身随手编一个会被 Pydantic 挡下——而那正是它该有的行为
        id=new_id(IdPrefix.EVIDENCE),
        source_type=source,
        title=f"{source} 第 {index} 条",
        claim=f"{source} 的第 {index} 条内容",
        locator={"section_path": [f"{source} 第二章"]} if source == "DOCUMENT" else {},
        retrieved_at=datetime(2026, 9, 17, tzinfo=UTC),
        content_hash="a" * 64,
    )


class FakeTool:
    """按脚本返回的工具替身。**记录被调了几次**——那是路由断言的全部依据。

    `fail_with` 模拟"跑了但空"（SQL 0 行 / RAG 无相关知识），
    它与"执行出错"在图里的去处不同（见 `tool_nodes` 的说明），
    所以替身必须能把两者分开表达。
    """

    def __init__(
        self,
        name: str,
        *,
        source: str = "SQL",
        evidence_count: int = 2,
        empty: ErrorCode | None = None,
        fail: ErrorCode | None = None,
        empty_as_success: bool = False,
    ) -> None:
        self.name = name
        self._source = source
        self._evidence_count = evidence_count
        self._empty = empty
        self._fail = fail
        #: 模拟 **SQL 的空结果**：`SUCCEEDED` + `payload.is_empty`（详设 9.4 明写
        #: 「SQL 空集不算失败」）。它与 `empty=` 那条路（RAG 的错误码）是
        #: 两种不同的事实载体，`reflect` 两条都要认。
        self._empty_as_success = empty_as_success
        self.calls: list[ToolContext] = []

    async def execute(self, args: Any, ctx: ToolContext) -> ToolResult:
        self.calls.append(ctx)
        now = datetime.now(UTC)
        error = None
        if self._empty is not None:
            error = (self._empty.value, "EMPTY_RESULT")
        elif self._fail is not None:
            error = (self._fail.value, "INTERNAL")
        if error is not None:
            return ToolResult(
                call_id=f"tcl_{len(self.calls):022d}",
                tool=self.name,
                status="FAILED",
                started_at=now,
                finished_at=now,
                summary="未取得结果",
                # **空结果也带 payload**：`tool_nodes` 要读 `payload` 之外的
                # 字段判断"是不是空"，而真实工具（RAG）正是这么做的
                payload={"error_class": error[1]},
                evidence=[],
                error=_tool_error(error[0], error[1]),
            )
        items = (
            []
            if self._empty_as_success
            else [_evidence(i, self._source) for i in range(1, self._evidence_count + 1)]
        )
        payload: dict[str, Any] = {}
        if self._empty_as_success:
            payload["is_empty"] = True
            # 真实工具把**本次实际施加的**权限范围放在这里。替身按调用时拿到的
            # `permission_scope` 给，且照 `model_dump(mode="json")` 出来的形状
            # （列表）给——否则用例测的是替身的形状，不是代码要处理的形状。
            if ctx.permission_scope.region_ids:
                payload["data_scope"] = list(ctx.permission_scope.region_ids)
        return ToolResult(
            call_id=f"tcl_{len(self.calls):022d}",
            tool=self.name,
            status="SUCCEEDED",
            started_at=now,
            finished_at=now,
            summary="查询未命中任何数据" if self._empty_as_success else f"命中 {len(items)} 条",
            payload=payload,
            evidence=items,
        )

    async def aclose(self) -> None:
        return None


def _tool_error(code: str, error_class: str) -> ToolError:
    from app.tools.base import ToolError

    return ToolError(code=code, message="未取得结果", error_class=error_class)


def _intent(sources: Sequence[str], *, intent: str = "QUERY") -> IntentResult:
    return IntentResult(intent=intent, required_sources=tuple(sources), confidence=0.9)


def _task() -> Task:
    return Task(
        id="tsk_0000000000000000000001",
        user_id="usr_0000000000000000000001",
        conversation_id="cnv_0000000000000000000001",
        trace_id="trc_0000000000000000000001",
        query_text="测试问题",
        status=TaskStatus.QUEUED,
        queued_at=datetime(2026, 9, 17, tzinfo=UTC),
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
        updated_at=datetime(2026, 9, 17, tzinfo=UTC),
    )


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def _run(
    settings: Settings,
    intents: list[IntentResult],
    *,
    sql_tool: FakeTool,
    rag_tool: FakeTool,
    analyses: list[Any] | None = None,
) -> tuple[Any, FakeModelGateway, list[Any]]:
    """跑一次图，返回（最终 State, sql 替身, rag 替身）。

    **只给改写/意图的脚本**，分析那一步的响应按需追加：`FakeModelGateway`
    的脚本是**按调用顺序**弹出的，而图里 supervisor 与 analysis 都会调它，
    所以脚本必须按真实调用顺序排。
    """
    from app.agent.schemas.analysis import AnalysisResult

    scripts: list[Any] = []
    for intent in intents:
        scripts.append(intent)
    scripts.extend(analyses or [AnalysisResult(direct_answer="测试结论")] * len(intents))

    gateway = FakeModelGateway(responses=scripts)
    graph = build_graph(settings, gateway=gateway, sql_tool=sql_tool, rag_tool=rag_tool)
    from langgraph.graph.state import CompiledStateGraph

    assert isinstance(graph, CompiledStateGraph)
    return graph, gateway, scripts


async def _invoke(
    settings: Settings, graph: Any, *, scope: PermissionScope | None = None
) -> dict[str, Any]:
    """跑一次图，返回最终 State。

    返回 `dict[str, Any]` 而不是 `AgentState`：**断言要能读到"不该有的字段"**。
    `AgentState` 是 `total=False` 的 TypedDict，用 `state["x"]` 访问一个不在
    类型里的键会被 mypy 挡下——而"某个字段悄悄没被写"正是这里要测的一类缺陷。
    """
    from app.agent.nodes.tool_nodes import deadline_for

    task = _task()
    initial: dict[str, Any] = {
        "user_query": task.query_text,
        "sanitized_query": task.query_text,
        "user_id": task.user_id,
        "conversation_id": task.conversation_id or "",
        "task_id": task.id,
        "trace_id": task.trace_id,
        "permission_scope": scope or PermissionScope(role=UserRole.ANALYST),
        "deadline_at": deadline_for(settings, datetime.now(UTC)),
        "evidence": [],
        "errors": [],
        "step_results": {},
        "findings": [],
        "task_list": [],
        "plan_revision": 0,
    }
    result: dict[str, Any] = await graph.ainvoke(initial, config={"recursion_limit": 25})
    return result


# ---------------------------------------------------------------- ① 计划与路由


async def test_simple_query_only_calls_sql(settings: Settings) -> None:
    """FR-PLAN-002 业务规则 1：「简单指标查询**不得强制调用 RAG**」。

    这条是 §8.4 第②条验收的反面用例——只测"两路都调了"是测不出这条的：
    一个永远两路都调的 Agent 在双路问题上表现完全正常。
    """
    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert len(sql.calls) == 1
    assert rag.calls == [], "简单指标查询不应调用 RAG"
    assert [step.tool for step in state["task_list"]] == ["sql_query"]


async def test_policy_question_only_calls_rag(settings: Settings) -> None:
    """反过来的那一路，同样要测：制度问法不该去查数据库。"""
    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["rag"], intent="POLICY_QA")], sql_tool=sql, rag_tool=rag)

    await _invoke(settings, graph)

    assert rag.calls != []
    assert sql.calls == []


async def test_cross_source_query_calls_both_without_expanding(settings: Settings) -> None:
    """「数据 + 解释」两路都要，而**两路都成功时不演进**。

    演进预算是"发现缺口才用"的，不是"多查一次更保险"。
    这条用例正是它的守门人：加了演进之后它仍然只应该各调一次。
    """
    sql = FakeTool("sql_query")
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(
        settings, [_intent(["sql", "rag"], intent="CROSS_SOURCE")], sql_tool=sql, rag_tool=rag
    )

    state = await _invoke(settings, graph)

    assert len(sql.calls) == 1
    assert len(rag.calls) == 1
    assert state["progress_assessment"].decision == "SUFFICIENT"
    assert state["evidence"], "两路的证据都要进 State"


# ---------------------------------------------------------------- ③ 任务循环


async def test_empty_sql_triggers_a_rag_expansion(settings: Settings) -> None:
    """**这是「Agent 能根据工具结果继续分析」的落点**（§8.4 第③条）。

    SQL 判成只要查库，而库按条件查不到——`reflect` 据此补一路 RAG
    去找制度与报告里的解释。整条链路不需要模型判断：
    "跑过哪一路、哪一路空了、还能补什么"都是 State 里的事实。
    """
    # SQL 的空结果走 `SUCCEEDED` + `payload.is_empty`，不是错误码
    sql = FakeTool("sql_query", empty_as_success=True)
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert len(sql.calls) == 1
    assert len(rag.calls) == 1, "SQL 空了应当补一路 RAG"
    assert state["plan_revision"] == 1
    assert [step.tool for step in state["task_list"]] == ["sql_query", "rag_retrieve"]
    # 演进出来的步骤 id 接着往下排，不复用 step_01
    assert state["task_list"][1].id == "step_02"


async def test_expansion_does_not_loop_forever(settings: Settings) -> None:
    """两路都空时**不再演进**——没有第三路可补。

    这条防的是"补一路、发现它也空、再补回去"这类回边失控：
    它不会报错，只会把同一路反复查直到递归上限。
    """
    sql = FakeTool("sql_query", empty_as_success=True)
    rag = FakeTool("rag_retrieve", source="DOCUMENT", empty=ErrorCode.NO_RELEVANT_KNOWLEDGE)
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert len(sql.calls) == 1
    assert len(rag.calls) == 1
    assert state["progress_assessment"].decision == "SUFFICIENT"
    assert state["open_questions"], "两路都没结果时，未解决的问题要留下来"


async def test_empty_sql_under_a_restricted_scope_is_not_reported_as_missing_data(
    settings: Settings,
) -> None:
    """**「你看不到」不能说成「没有这个数据」。**

    两者在执行结果上完全同形——都只是零行——而处置相反：数据不存在要换数据源，
    不在授权范围内要找数据负责人放开权限。合成一句「按当前条件未取得结果」，
    用户只会去怀疑数据。

    这条信息只可能断在 `tool_nodes` 那一层：空结果按 `build_evidence` 的设计
    **不产证据**，所以证据自带的那份 `data_scope` 标注此刻是空的；
    工具返回的 `warnings` 又只是一句给人读的提示串——判定要一个字段。
    """
    sql = FakeTool("sql_query", empty_as_success=True)
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(
        settings, graph, scope=PermissionScope(role=UserRole.ANALYST, region_ids=("华东",))
    )

    limits = " ".join(state["analysis_result"].limitations)
    # 断言整句而不是"限制里有华东"：文档来源那条限制同样会带上「华东」，
    # 按关键词断言会让两个用例互相顶替。
    assert "已按数据权限限定在 华东" in limits


async def test_restricted_scope_says_document_evidence_is_unfiltered(settings: Settings) -> None:
    """受限用户从文档里读到的东西**没有经过数据权限过滤**，必须说出来。

    数据权限在详设 9.1 里就是 SQL Tool 的服务端谓词，RAG 侧没有等价物：
    同一个数字，SQL 那条路被谓词拦下，文档这条路原样给出。不说的话，
    用户会把"文档里写着"当成"我有权看"——而这是他唯一能察觉这层差异的地方。
    """
    sql = FakeTool("sql_query", source="SQL")
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql", "rag"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(
        settings, graph, scope=PermissionScope(role=UserRole.ANALYST, region_ids=("华东",))
    )

    limits = " ".join(state["analysis_result"].limitations)
    assert "文档" in limits and "未经过滤" in limits
    assert "华东" in limits, "受限范围要点名——写「部分数据」等于没说"


async def test_unfiltered_document_limitation_needs_a_restricted_user(settings: Settings) -> None:
    """不受限的用户不写这条：他本就没有"看不看得到"的问题。

    与上一条的「只在引用文档时写」共同构成两个条件——缺一个，这句话就会
    出现在它不成立的场合，而**总在出现的提示等于没有提示**。
    """
    sql = FakeTool("sql_query", source="SQL")
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql", "rag"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph, scope=PermissionScope(role=UserRole.ADMIN))

    limits = " ".join(state["analysis_result"].limitations)
    assert "未经过滤" not in limits


async def test_unfiltered_document_limitation_needs_document_evidence(settings: Settings) -> None:
    """只走 SQL 的用户不写这条：他的证据本来就过了权限谓词。"""
    sql = FakeTool("sql_query", source="SQL")
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(
        settings, graph, scope=PermissionScope(role=UserRole.ANALYST, region_ids=("华东",))
    )

    limits = " ".join(state["analysis_result"].limitations)
    assert "未经过滤" not in limits


async def test_unrestricted_scope_adds_no_permission_limitation(settings: Settings) -> None:
    """不受限时不写这条限制。

    每跑一次都带一条「可能受权限限制」，这条提示就变成噪声，真受限的那次
    反而没人看了（同 `reranker` 的 `DISABLED` 不进 `warnings` 的理由）。
    """
    sql = FakeTool("sql_query", empty_as_success=True)
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph, scope=PermissionScope(role=UserRole.ADMIN))

    limits = " ".join(state["analysis_result"].limitations)
    assert "授权范围" not in limits


async def test_hard_failure_is_reported_as_an_error_not_as_empty(settings: Settings) -> None:
    """**「查不到」与「查不成」必须分开**：前者是关于数据的事实，后者是执行出错。

    混在一起的话，最终答案会把"服务挂了"说成"公司没有这条数据"——
    一句话就把故障变成了结论。
    """
    sql = FakeTool("sql_query", fail=ErrorCode.UPSTREAM_UNAVAILABLE)
    rag = FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    # 失败**不算空**：不触发补证（换一路也补不上服务不可用）
    assert rag.calls == []
    assert state["errors"], "执行失败要进 errors"
    assert state["step_results"]["step_01"].empty is False
    assert state["step_results"]["step_01"].status.value == "FAILED"


# ---------------------------------------------------------------- 澄清与失败


async def test_clarification_gives_an_answer_not_a_failure(settings: Settings) -> None:
    """澄清是**正常路径**，不是失败。

    不区分的话，`final` 会把"你问的年度没说清"渲染成"任务未能完成"，
    而这会让用户以为系统坏了。
    """
    intent = IntentResult(
        intent="CLARIFICATION",
        missing_fields=("时间范围",),
        clarification_question="请问要看哪个季度？",
    )
    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    gateway = FakeModelGateway(responses=[intent])
    graph = build_graph(settings, gateway=gateway, sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert sql.calls == [] and rag.calls == []
    assert state["analysis_result"] is not None
    assert "哪个季度" in (state["final_answer"] or "")
    assert "时间范围" in (state["final_answer"] or "")
    assert not state["errors"]


async def test_supervisor_model_failure_fails_the_task_loudly(settings: Settings) -> None:
    """模型不可用时**报错，不降级为"两路都查"**。

    降级看起来更稳，但它会在故障期间把 FR-PLAN-002 那条验收悄悄作废，
    而结果看起来完全正常（确实拿到了数据）。
    """
    from app.tests.fakes import FakeFailure

    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    gateway = FakeModelGateway(responses=[], embedding_dim=8, failure=FakeFailure.UNAVAILABLE)
    graph = build_graph(settings, gateway=gateway, sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert sql.calls == [] and rag.calls == []
    assert state["execution_status"] is TaskStatus.FAILED
    assert state["errors"]
    assert "未能完成" in (state["final_answer"] or "")


async def test_invalid_plan_does_not_silently_drop_a_source(settings: Settings) -> None:
    """模型给出本项目没有的数据源（`search`）时报错，**不悄悄丢掉**。

    丢掉会让计划少一路而没人知道，最终答案是"信息不全"却没有原因。
    """
    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    gateway = FakeModelGateway(responses=[_intent(["sql", "search"])])
    graph = build_graph(settings, gateway=gateway, sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert sql.calls == [] and rag.calls == []
    assert state["errors"][0].code is ErrorCode.PLAN_INVALID
    assert "search" in state["errors"][0].message


# ---------------------------------------------------------------- 权限


async def test_tools_receive_the_state_permission_scope(settings: Settings) -> None:
    """工具拿到的是 State 里的范围，**不是默认值**。

    `PermissionScope()` 的空 `region_ids` 表示**不限**（TBC-03）——
    图若用默认值兜底，受限用户会拿到全量数据，而这不会有任何报错。
    """
    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)
    scope = PermissionScope(role=UserRole.ANALYST, region_ids=("华东",))

    await _invoke(settings, graph, scope=scope)

    assert sql.calls[0].permission_scope.region_ids == ("华东",)
    assert sql.calls[0].step_id == "step_01"
    assert sql.calls[0].task_id == _task().id


async def test_missing_scope_refuses_to_run(settings: Settings) -> None:
    """State 里没有范围时**拒绝执行**，而不是按不限跑。"""
    from app.agent.nodes.tool_nodes import _context
    from app.agent.schemas.plan import TaskStep

    step = TaskStep(id="step_01", objective="x", tool="sql_query")
    with pytest.raises(RuntimeError, match="permission_scope"):
        _context({"user_id": "u", "task_id": "t"}, step)


async def test_agent_error_from_a_tool_becomes_a_failed_step(settings: Settings) -> None:
    """工具抛 `AgentError` 时图不崩，落成一条失败的步骤。

    图崩溃的表现是任务 FAILED 且没有答案；而这里要的是
    "这一步没成、其余照常、限制里写清楚"。
    """
    sql = FakeTool("sql_query")
    rag = FakeTool("rag_retrieve", source="DOCUMENT")

    async def boom(*_: Any, **__: Any) -> Any:
        raise AgentError(ErrorCode.UPSTREAM_UNAVAILABLE, "数据库连不上")

    sql.execute = boom  # type: ignore[method-assign]
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    with pytest.raises(AgentError):
        await _invoke(settings, graph)


# ---------------------------------------------------------------- ④ 执行轨迹


async def test_every_node_leaves_a_started_and_a_leave_event(settings: Settings) -> None:
    """每个跑过的节点都留下一对事件，且**顺序就是执行顺序**。

    这是"每个节点都有轨迹"的结构性验证：`traced` 在 `_add_node` 里包住所有
    节点（漏包一个不会有任何症状——它在轨迹里只是"不存在"，而轨迹本来
    就不完整，看不出少了什么）。所以断言的是**节点集合与先后关系**。
    """
    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    graph, _, _ = _run(settings, [_intent(["sql"])], sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    events = state["trace_events"]
    started = [event for event in events if event.event_type == "node.started"]
    leaves = [event for event in events if event.event_type != "node.started"]
    # 简单查询的路径：supervisor → sql → reflect → conflict → analysis → reviewer → final
    # （`conflict` 与 `reviewer` 是确定性节点，同样会在轨迹里留下痕迹）
    expected = ["supervisor", "sql", "reflect", "conflict", "analysis", "reviewer", "final"]
    assert [event.node for event in started] == expected
    # **事件是成对的**：只有离开事件的话，一个卡住的节点在轨迹上
    # 表现为"什么都没发生"，与"压根没跑到"长得一样
    assert [event.node for event in leaves] == expected
    assert all(event.status == "RUNNING" for event in started)
    # 进入事件没有耗时可言——给它 0 会被读成"瞬间完成"
    assert all(event.duration_ms is None for event in started)
    assert all(event.duration_ms is not None for event in leaves)


async def test_a_node_reporting_failure_is_recorded_as_node_failed(settings: Settings) -> None:
    """supervisor 判失败时，离开事件的类型是 `node.failed` 而非 `node.completed`。

    订阅方按 `type` 分支（18.2 的事件清单就是这个粒度），折在 `status` 里
    等于要求每个订阅方自己再判一次——漏判的症状是"失败被当成完成"。
    **这条路径是"节点返回了错误"（不抛异常）**，也就是埋点管得住的那一条。
    """
    from app.tests.fakes import FakeFailure

    sql, rag = FakeTool("sql_query"), FakeTool("rag_retrieve", source="DOCUMENT")
    gateway = FakeModelGateway(responses=[], failure=FakeFailure.UNAVAILABLE)
    graph = build_graph(settings, gateway=gateway, sql_tool=sql, rag_tool=rag)

    state = await _invoke(settings, graph)

    assert state["execution_status"] is TaskStatus.FAILED
    leaves = [event for event in state["trace_events"] if event.event_type != "node.started"]
    supervisor = next(event for event in leaves if event.node == "supervisor")
    assert supervisor.event_type == "node.failed"
    assert supervisor.status == TaskStatus.FAILED.value
    # 失败之后仍要走到 `final`（由它给出一句面向用户的说明），
    # 因此记的是 failed 而不是"图在这里断了"
    assert leaves[-1].node == "final"
