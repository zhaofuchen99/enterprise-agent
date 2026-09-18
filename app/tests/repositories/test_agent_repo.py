"""执行产出仓储的契约（详细设计 16.6 / 16.7）。

**同一份断言跑两个实现**（同 `test_task_repo.py` 的规矩）。

这里最要紧的不是"能读能写"，而是**重投时整体替换**：
这几张表都没有天然唯一键，再写一遍若变成追加，就会造出两批并存的行，
而**从数据上分不出哪批属于最后一次执行**——复盘时看到的是两份互相矛盾的
证据清单，且没有任何地方报错。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest

from app.domain.evidence import (
    Conflict,
    ConflictResolution,
    ConflictSeverity,
    ConflictType,
    Evidence,
    TimeRange,
)
from app.domain.task import ReviewRecord, StepRecord, ToolCallRecord
from app.domain.trace import NodeTrace
from app.repositories.agent_repo import (
    AgentArtifactRepository,
    InMemoryAgentArtifactRepository,
    SqlAgentArtifactRepository,
)

_TASK = "tsk_0000000000000000000001"


@pytest.fixture(
    params=[
        pytest.param("memory", id="memory"),
        # 连真实 MySQL 的变体。标记写在 param 上而不是用例上——
        # 否则内存变体会被一起排除掉，`make test` 就什么都测不到了。
        pytest.param("sql", id="sql", marks=pytest.mark.integration),
    ]
)
async def make_repo(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[[], AgentArtifactRepository]]:
    if request.param == "memory":
        yield InMemoryAgentArtifactRepository
        return

    from app.tests.db import sql_sessions

    async with sql_sessions() as sessions:
        yield lambda: SqlAgentArtifactRepository(sessions)


def _evidence(claim: str = "net_sales=100") -> Evidence:
    # **每条证据一个独立的 id**：`agent_evidence` 的主键就是它，
    # 一批里撞了会让**整批插入失败**（先清后写在同一个事务里）。
    # 第一版这里硬编码了一个固定 id，于是"写两次"的用例第二次炸在唯一键上——
    # 那是夹具的问题，但也说明这个约束是真的。
    from app.core.ids import IdPrefix, new_id

    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title="结果第 1 行",
        claim=claim,
        locator={"call_id": "tcl_0000000000000000000001", "sql_fingerprint": "f" * 64},
        event_time=TimeRange(
            start=datetime(2025, 7, 1, tzinfo=UTC), end=datetime(2025, 10, 1, tzinfo=UTC)
        ),
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        metric_code="net_sales",
        scope={"region": "华东"},
        reliability="HIGH",
        content_hash="a" * 64,
    )


def _step() -> StepRecord:
    return StepRecord(
        id="tsk_0000000000000000000002",
        step_key="step_01",
        objective="查数",
        tool="sql_query",
        status="SUCCEEDED",
        result_summary={"summary": "查询返回 1 行", "empty": False},
    )


def _tool_call() -> ToolCallRecord:
    return ToolCallRecord(
        id="tcl_0000000000000000000001",
        step_id="step_01",
        tool_name="sql_query",
        attempt_no=1,
        status="SUCCEEDED",
        normalized_sql="SELECT 1",
        sql_fingerprint="f" * 64,
        duration_ms=12,
    )


def _conflict() -> Conflict:
    return Conflict(
        id="cft_0000000000000000000001",
        type=ConflictType.VALUE,
        evidence_ids=("evd_0000000000000000000001", "evd_0000000000000000000002"),
        severity=ConflictSeverity.WARNING,
        description="两个来源的净销售额相差 1.4%",
        detected_difference={"delta": 1.0},
        possible_explanations=("口径不同",),
        resolution=ConflictResolution.UNRESOLVED,
    )


def _review() -> ReviewRecord:
    return ReviewRecord(
        id="evd_0000000000000000000009",
        status="PASS",
        score=100,
        evidence_score=100,
        issues=({"code": "UNVERIFIED_HYPOTHESIS", "severity": "INFO", "message": "含推测"},),
        reason_code="GROUNDED",
    )


async def test_save_then_read_evidence_round_trips(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """证据能写进去、能读回来，且**关键字段一个不差**。

    读回来的是 `Evidence` 而不是行对象：17.2 的任务详情与 Phase 9 的冲突检测
    都要用它，而它们不认识 SQLAlchemy 的行。
    """
    repo = make_repo()
    await repo.save(_TASK, evidence=[_evidence()])

    stored = await repo.list_evidence(_TASK)

    assert len(stored) == 1
    item = stored[0]
    assert item.source_type == "SQL"
    assert item.claim == "net_sales=100"
    assert item.metric_code == "net_sales"
    assert item.scope == {"region": "华东"}
    assert item.reliability == "HIGH"
    assert item.content_hash == "a" * 64
    assert item.event_time is not None
    assert item.event_time.start == datetime(2025, 7, 1, tzinfo=UTC)


async def test_saving_twice_replaces_instead_of_appending(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """**重投时整体替换**，不是追加。

    这几张表没有天然唯一键，追加会让两批行并存，而**从数据上分不出
    哪批属于最后一次执行**——复盘时看到的是两份互相矛盾的证据清单，
    且没有任何地方报错。
    """
    repo = make_repo()
    await repo.save(_TASK, evidence=[_evidence("第一次"), _evidence("第二次")])
    await repo.save(_TASK, evidence=[_evidence("重投之后")])

    stored = await repo.list_evidence(_TASK)

    assert len(stored) == 1
    assert stored[0].claim == "重投之后"


async def test_all_five_tables_are_written_in_one_call(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """五张表一次写齐。

    **分五次写会让它们有机会不一致**（证据落了、冲突没落），
    而那种不一致不会报错，只会让复盘看到的图景缺一块。
    """
    repo = make_repo()
    await repo.save(
        _TASK,
        steps=[_step()],
        tool_calls=[_tool_call()],
        evidence=[_evidence()],
        conflicts=[_conflict()],
        review=_review(),
    )

    # 内存实现直接看属性；SQL 实现只能通过证据与"不抛异常"来验证——
    # 那五张表各自的查询入口属 Phase 9 的任务详情，这里只保证写入路径通。
    assert await repo.list_evidence(_TASK)
    if isinstance(repo, InMemoryAgentArtifactRepository):
        assert repo.steps[_TASK] == [_step()]
        assert repo.tool_calls[_TASK] == [_tool_call()]
        assert repo.conflicts[_TASK] == [_conflict()]
        assert repo.reviews[_TASK] == _review()


async def test_saving_without_evidence_still_clears_the_previous_batch(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """第二次执行没有产出证据时，**上一批要被清掉**。

    不清的话，重投后表里留着上一次的证据，而任务的答案是这一版的——
    两者对不上，且看起来完全正常。
    """
    repo = make_repo()
    await repo.save(_TASK, evidence=[_evidence("上一批")])
    await repo.save(_TASK)

    assert await repo.list_evidence(_TASK) == []


def _trace(node: str, event_type: str, *, duration: int | None = None) -> NodeTrace:
    return NodeTrace(
        node=node,
        event_type=event_type,
        status="RUNNING" if duration is None else "SUCCEEDED",
        duration_ms=duration,
        created_at=datetime(2026, 9, 18, tzinfo=UTC),
    )


async def test_trace_events_keep_the_execution_order(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """**顺序是语义的一部分**：18.3 说 `(task_id, sequence)` 的唯一索引
    就是顺序保证本身，重放时按它排出来的就是真实执行顺序。

    `sequence` 由写入侧按列表顺序分配——图的节点是串行的（见 `domain/trace.py`）。
    """
    repo = make_repo()
    await repo.save(
        _TASK,
        trace_id="trc_0000000000000000000001",
        trace_events=[
            _trace("supervisor", "node.started"),
            _trace("supervisor", "node.completed", duration=2631),
            _trace("sql", "node.started"),
            _trace("sql", "node.completed", duration=1568),
        ],
    )

    rows = await repo.list_trace_events(_TASK)

    assert [item.sequence for item in rows] == [1, 2, 3, 4]
    assert [item.node for item in rows] == ["supervisor", "supervisor", "sql", "sql"]
    assert rows[1].duration_ms == 2631
    assert rows[0].duration_ms is None
    # **时刻取事件自己的，不取落库那一刻**：八条事件都写成同一个时间之后，
    # 时间轴就只剩顺序、没有间隔了——而顺序已经由 `sequence` 表达
    assert rows[0].created_at == datetime(2026, 9, 18, tzinfo=UTC)


async def test_after_sequence_is_strictly_greater(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """`after_sequence` 的语义是「我已经有的最后一条」，因此是**严格大于**。

    用 `>=` 会让客户端每次重连都重复拿到同一条——而重复的事件在
    "按序号去重"的客户端上表现为"最后一条永远处理两遍"。
    """
    repo = make_repo()
    await repo.save(
        _TASK,
        trace_id="trc_0000000000000000000001",
        trace_events=[_trace(f"n{i}", "node.started") for i in range(1, 6)],
    )

    rows = await repo.list_trace_events(_TASK, after_sequence=3)

    assert [item.sequence for item in rows] == [4, 5]
    assert len(await repo.list_trace_events(_TASK, after_sequence=5)) == 0
    assert len(await repo.list_trace_events(_TASK, limit=2)) == 2


async def test_saving_twice_replaces_the_trace_too(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """轨迹也整体替换——与另外五张表同一语义。

    追加的话，重投后表里会有两次执行的轨迹**首尾相接**，
    而 `sequence` 会在中间跳回 1，客户端按序号去重时行为未定义。
    """
    repo = make_repo()
    await repo.save(_TASK, trace_id="trc_1", trace_events=[_trace("a", "node.started")])
    await repo.save(
        _TASK,
        trace_id="trc_1",
        trace_events=[_trace("b", "node.started"), _trace("b", "node.completed", duration=1)],
    )

    rows = await repo.list_trace_events(_TASK)

    assert [item.node for item in rows] == ["b", "b"]
    assert [item.sequence for item in rows] == [1, 2]


async def test_evidence_of_another_task_is_untouched(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """替换只作用于**这一个任务**——清表时按 `task_id` 过滤，不是全表删。"""
    repo = make_repo()
    other = "tsk_0000000000000000000009"
    await repo.save(other, evidence=[_evidence("另一个任务")])
    await repo.save(_TASK, evidence=[_evidence()])

    assert len(await repo.list_evidence(other)) == 1
