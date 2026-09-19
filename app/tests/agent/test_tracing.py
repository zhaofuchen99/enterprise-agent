"""节点埋点（详细设计 16.7 / 18.3）。

**这一层的价值是"每个节点都有轨迹"这件事本身是结构性的**：包一层之后，
漏掉某个节点不可能发生；而八个节点各写一遍埋点，漏掉一处不会有任何症状——
那个节点在轨迹里就是"不存在"，而轨迹本来就不完整，看不出少了什么。
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from app.agent.schemas.plan import IntentResult, ProgressAssessment, TaskStep
from app.agent.schemas.review import ReviewResult
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.domain.events import TaskEventType
from app.domain.task import TaskStatus


async def test_traced_emits_a_pair_of_events() -> None:
    """**成对出现**：只记完成事件的话，一个卡住的节点在轨迹上表现为
    "什么都没发生"——而"进去了还没出来"与"压根没跑到"是不同的故障，
    两者在只有完成事件的轨迹上长得一样。"""

    async def node(state: AgentState) -> dict[str, object]:
        return {"final_answer": "答"}

    update = await traced("demo", node)({})

    events = update["trace_events"]
    assert [item.event_type for item in events] == ["node.started", "node.completed"]
    assert [item.status for item in events] == ["RUNNING", "SUCCEEDED"]
    # 进入事件没有耗时可言——给它 0 会被读成"瞬间完成"
    assert events[0].duration_ms is None
    assert events[1].duration_ms is not None


async def test_traced_passes_the_nodes_update_through() -> None:
    """包装不能吃掉节点自己的返回值。

    吃掉的话，症状是"图跑完了但什么都没做"——而埋点看起来完全正常。
    """

    async def node(state: AgentState) -> dict[str, object]:
        return {"final_answer": "答", "errors": []}

    update = await traced("demo", node)({})

    assert update["final_answer"] == "答"
    assert update["errors"] == []
    assert len(update["trace_events"]) == 2


async def test_traced_works_for_sync_nodes_too() -> None:
    """同步节点（`reflect` / `conflict` / `reviewer` / `final`）走同一条包装。

    **不给两类节点写两条包装路径**：那会让"某个节点忘了被包"变成一件
    从签名上看不出来的事。
    """

    def node(state: AgentState) -> dict[str, object]:
        return {"plan_revision": 1}

    update = await traced("sync_demo", node)({})

    assert update["plan_revision"] == 1
    assert len(update["trace_events"]) == 2


async def test_a_node_reporting_failure_emits_a_node_failed_event() -> None:
    """节点**返回**失败（不抛异常）时，离开事件的类型是 `node.failed`。

    订阅方按 `type` 分支（18.2 的事件清单就是这个粒度），把它折进 `status`
    等于要求每个订阅方自己再判一次——而漏判的症状是"失败被当成完成"。
    `supervisor` 在模型不可用时走的正是这条路径。
    """

    async def node(state: AgentState) -> dict[str, object]:
        return {"execution_status": TaskStatus.FAILED}

    update = await traced("demo", node)({})
    events = update["trace_events"]

    assert [item.event_type for item in events] == ["node.started", "node.failed"]
    assert events[1].status == TaskStatus.FAILED.value


async def test_a_node_returning_nothing_still_gets_a_completed_event() -> None:
    """节点返回 `None` / `{}` 时按"成功且没改动"记。

    显式节点普遍不写 `execution_status`（只有 `supervisor` 会写），
    因此默认值必须是成功——默认成未知或失败会让整条正常轨迹全是异常事件。
    """

    async def node(state: AgentState) -> None:
        return None

    update = await traced("demo", node)({})

    assert [item.event_type for item in update["trace_events"]] == [
        "node.started",
        "node.completed",
    ]
    assert update["trace_events"][1].status == "SUCCEEDED"


async def test_traced_records_the_duration_of_a_slow_node() -> None:
    """耗时是真的量的——它是"哪一步慢"的唯一线索。"""

    async def slow(state: AgentState) -> dict[str, object]:
        await asyncio.sleep(0.05)
        return {}

    update = await traced("slow", slow)({})

    assert update["trace_events"][1].duration_ms is not None
    assert update["trace_events"][1].duration_ms >= 40


class _Recorder:
    """记录发布出去的事件（形状与 `EventBus.publish` 一致）。

    **不 import `EventBus`**：`agent/tracing.py` 只依赖一个结构化的 `publish`，
    替身照做即可——这正是那个 Protocol 存在的意义（与 `PromptSource` 同理）。
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict[str, object]]] = []

    async def publish(
        self,
        *,
        task_id: str,
        trace_id: str,
        event_type: TaskEventType,
        data: dict[str, object] | None = None,
        node: str | None = None,
        step_id: str | None = None,
    ) -> None:
        self.events.append((event_type.value, node, data or {}))

    @property
    def types(self) -> list[str]:
        return [event_type for event_type, _node, _data in self.events]


def _state(**extra: object) -> AgentState:
    """一个最小的 State。**显式 cast**：`AgentState` 是 `total=False` 的 TypedDict，
    而这里要按需塞进被测节点关心的那几个字段，逐个标注类型只会把夹具写得比用例还长。"""
    base: dict[str, Any] = {"task_id": "tsk_1", "trace_id": "trc_1", **extra}
    return cast(AgentState, base)


async def test_node_lifecycle_events_are_published_in_pairs() -> None:
    """`node.started` 与 `node.completed` **成对**出现，且包裹着节点自身的执行。

    只发完成事件的轨迹在客户端看来与"这个节点压根没跑"一样——
    而"进去了还没出来"（卡住）与"没跑到"是不同的故障。
    """
    recorder = _Recorder()

    async def node(state: AgentState) -> dict[str, object]:
        return {}

    await traced("sql", node, recorder)(_state())

    assert recorder.types == ["node.started", "node.completed"]
    assert recorder.events[0][1] == "sql"
    assert "duration_ms" in recorder.events[1][2]


async def test_a_failing_node_publishes_node_failed() -> None:
    """失败换的是**事件类型**，不只换 status（订阅方按 `type` 分支）。"""
    recorder = _Recorder()

    async def node(state: AgentState) -> dict[str, object]:
        return {"execution_status": TaskStatus.FAILED}

    await traced("supervisor", node, recorder)(_state())

    assert recorder.types == ["node.started", "node.failed"]


async def test_plan_created_is_derived_from_the_supervisor_update() -> None:
    """`plan.created`：首次成计划时报一次（**按 State 字段判，不按节点名**）。

    判据是"这次更新里有没有 `task_list` / `plan_revision`"——
    它们是 Pydantic 校验过的对象，比"哪个节点返回了它"稳定。
    """
    recorder = _Recorder()
    plan = (TaskStep(id="step_01", objective="查销售额", tool="sql_query"),)

    async def node(state: AgentState) -> dict[str, object]:
        return {"task_list": list(plan), "plan_revision": 0}

    await traced("supervisor", node, recorder)(_state())

    assert recorder.types == ["node.started", "node.completed", "plan.created"]
    assert recorder.events[-1][2] == {"step_count": 1, "tool_types": ["sql_query"]}


async def test_progress_assessed_and_plan_updated_are_derived_from_reflect() -> None:
    """`reflect` 一次产出两条：判定（`progress.assessed`）与演进（`plan.updated`）。

    **18.2 明写这两个不允许省略**——它们是"任务循环对用户可见"的载体。
    演进靠 `plan_revision` 递增识别（`supervisor` 写的是当前值）。
    """
    recorder = _Recorder()
    existing = TaskStep(id="step_01", objective="查销售额", tool="sql_query")
    added = TaskStep(id="step_02", objective="补查制度", tool="rag_retrieve")

    async def node(state: AgentState) -> dict[str, object]:
        return {
            "task_list": [existing, added],
            "plan_revision": 1,
            "progress_assessment": ProgressAssessment(
                decision="EXPAND", reason="sql_query 未取得有用结果，补一路 rag_retrieve"
            ),
            "open_questions": ["step_01：按当前条件未取得结果"],
        }

    await traced("reflect", node, recorder)(_state(task_list=[existing], plan_revision=0))

    assert recorder.types == [
        "node.started",
        "node.completed",
        "plan.updated",
        "progress.assessed",
    ]
    plan_updated = recorder.events[2][2]
    assert plan_updated["revision_no"] == 1
    assert plan_updated["added_steps"] == ["step_02"]
    assert plan_updated["skipped_steps"] == []
    # 切片内没有模型的 EXPAND 判定，触发结论为空——**字段留着**，
    # 客户端的 schema 不该因为一个后置能力而变形
    assert plan_updated["trigger_finding_id"] is None
    assert recorder.events[3][2] == {
        "decision": "EXPAND",
        "finding_statement": "sql_query 未取得有用结果，补一路 rag_retrieve",
        "open_question_count": 1,
    }


async def test_review_completed_is_derived_from_the_reviewer_update() -> None:
    recorder = _Recorder()
    review = ReviewResult(status="PASS", score=90, issues=())

    async def node(state: AgentState) -> dict[str, object]:
        return {"review_result": review}

    await traced("reviewer", node, recorder)(_state())

    assert recorder.types == ["node.started", "node.completed", "review.completed"]
    assert recorder.events[-1][2] == {"status": "PASS", "score": 90, "issue_count": 0}


async def test_clarification_required_is_derived_from_the_supervisor_intent() -> None:
    """澄清：没有可执行计划 + 判成 `CLARIFICATION` → 报"需要用户补充输入"。"""
    recorder = _Recorder()

    async def node(state: AgentState) -> dict[str, object]:
        return {
            "intent": IntentResult(
                intent="CLARIFICATION",
                required_sources=(),
                confidence=0.4,
                missing_fields=("time_range",),
                clarification_question="您问的是哪个季度？",
            ),
            "task_list": [],
            "plan_revision": 0,
        }

    await traced("supervisor", node, recorder)(_state())

    assert recorder.types[-1] == "clarification.required"
    assert recorder.events[-1][2] == {
        "question": "您问的是哪个季度？",
        "missing_fields": ["time_range"],
    }


async def test_without_a_bus_nothing_is_published() -> None:
    """不传总线时行为与加事件之前完全一致（单元测试与不关心事件的装配路径）。

    **它不影响状态内的轨迹**：`trace_events` 照样成对产出。
    """

    async def node(state: AgentState) -> dict[str, object]:
        return {}

    update = await traced("sql", node)(_state())

    assert [item.event_type for item in update["trace_events"]] == [
        "node.started",
        "node.completed",
    ]


async def test_a_publish_failure_does_not_break_the_node() -> None:
    """事件发布失败只记日志，**不让任务跟着倒**。

    埋点与事件流是"关于执行的事后信息"，而 Redis 抖一下不该把一次正常的分析
    判成失败——那是拿可观测性换可用性，方向反了。
    """

    class _Broken(_Recorder):
        async def publish(self, **_kwargs: object) -> None:
            raise RuntimeError("Redis 挂了")

    async def node(state: AgentState) -> dict[str, object]:
        return {"final_answer": "答"}

    update = await traced("sql", node, _Broken())(_state())

    assert update["final_answer"] == "答"
    assert len(update["trace_events"]) == 2


async def test_a_raising_node_does_not_swallow_the_exception() -> None:
    """节点抛异常时**原样抛出**，不把它变成一条"失败的轨迹"。

    包一层很容易顺手 `except Exception` 再返回一个错误更新——那会把
    "图中止"变成"节点返回了个错误"，两者的处置完全不同（前者任务 FAILED，
    后者由 `reflect` 决定要不要补证）。

    ⚠️ **抛出前的那条事件其实留不下来**（LangGraph 不写中止节点的更新），
    这一点写在 `tracing.py` 的模块 docstring 里，不在这里假装它管用。
    """

    async def boom(state: AgentState) -> dict[str, object]:
        raise RuntimeError("炸了")

    with pytest.raises(RuntimeError, match="炸了"):
        await traced("boom", boom)({})
