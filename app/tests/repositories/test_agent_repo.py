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
    return Evidence(
        id="evd_0000000000000000000001",
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


async def test_evidence_of_another_task_is_untouched(
    make_repo: Callable[[], AgentArtifactRepository],
) -> None:
    """替换只作用于**这一个任务**——清表时按 `task_id` 过滤，不是全表删。"""
    repo = make_repo()
    other = "tsk_0000000000000000000009"
    await repo.save(other, evidence=[_evidence("另一个任务")])
    await repo.save(_TASK, evidence=[_evidence()])

    assert len(await repo.list_evidence(other)) == 1
