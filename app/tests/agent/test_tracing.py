"""节点埋点（详细设计 16.7 / 18.3）。

**这一层的价值是"每个节点都有轨迹"这件事本身是结构性的**：包一层之后，
漏掉某个节点不可能发生；而八个节点各写一遍埋点，漏掉一处不会有任何症状——
那个节点在轨迹里就是"不存在"，而轨迹本来就不完整，看不出少了什么。
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent.state import AgentState
from app.agent.tracing import traced
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
