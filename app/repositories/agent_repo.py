"""一次任务的执行产出落库（详细设计 16.6 与 16.7 的四张表）。

## 为什么这几张表合成一个仓储

`agent_task_step` / `agent_tool_call` / `agent_evidence` / `agent_conflict` /
`agent_review` 各自是独立的表，但它们有同一个生命周期：**都只在一次任务
结束时写一次，且必须一起写**。拆成五个仓储会让调用点从"一次执行落一次库"
变成五行互不相干的写语句——而它们的不一致（证据落了、冲突没落）
不会有任何报错，只会让复盘时看到的图景缺一块。

## 为什么之前一直没写，以及为什么现在要写

16.5 的 `agent_task.plan_json` / `result_json` 两列已经能把整份产出装下来，
所以 17.2 的任务详情一直有内容。**但那两列解决不了两类问题**：

- **按内容查**：`idx_tool_call_sql_fingerprint` 存在的意义是回答"同一批烂 SQL
  是不是反复出现"，而在 JSON 列里查等于全表扫；
- **表的权威性**：14.2 的 `agent_review`、13.3 的 `agent_conflict` 是设计文档
  指定的落点。它们长期为空，会让"证据落在哪张表"这个问题没有答案——
  而那是**可追溯**这个卖点最容易被问到的一处。

## 重投时**整体替换**，而不是追加

一次任务可能因为重投（`max_requeue_attempts`）或重试跑第二遍。
这几张表都没有天然唯一键（`agent_task_step` 有 `(task_id, step_key)`，
但那是"计划内唯一"），所以再写一遍会造出两批并存的行，
而**从数据上分不出哪批属于最后那次执行**。

因此 `save` 先按 `task_id` 删掉上一批、再写这一批——
任务的产出是"当前这一版答案"的支撑，而不是一份跨重投的审计流水。
⚠️ **代价如实记下**：跨重投的历史轨迹（上几次是怎么失败的）会丢。
真要留就得给这几张表加 `attempt_no` 或一张执行流水表，属后续扩展。

## 只认识 `domain/` 的对象

`repositories/` 在依赖链的底端（`api → services → agent/tools → repositories`），
**不能 import `agent/schemas`**。所以"计划里的一步"（`TaskStep`）与
"这一步的结果"（`StepResult`）由调用方合成 `StepRecord` 再传进来。
这个约束是好事：它逼着仓储的入参只描述**行**，不描述业务。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.evidence import Conflict, Evidence
from app.domain.task import ReviewRecord, StepRecord, ToolCallRecord
from app.infrastructure.db import session_scope
from app.infrastructure.models.evidence import AgentConflict, AgentEvidence, AgentReview
from app.infrastructure.models.task import AgentTaskStep, AgentToolCall


class AgentArtifactRepository:
    """执行产出的读写。两个实现共用一个 `save` 契约。"""

    async def save(
        self,
        task_id: str,
        *,
        steps: Sequence[StepRecord] = (),
        tool_calls: Sequence[ToolCallRecord] = (),
        evidence: Sequence[Evidence] = (),
        conflicts: Sequence[Conflict] = (),
        review: ReviewRecord | None = None,
    ) -> None:
        """整体替换一个任务的执行产出。见模块 docstring 的说明。"""
        raise NotImplementedError

    async def list_evidence(self, task_id: str) -> list[Evidence]:
        """按任务取回证据。**它是"这条结论依据什么"的查询入口**。"""
        raise NotImplementedError


class SqlAgentArtifactRepository(AgentArtifactRepository):
    """MySQL 实现。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def save(
        self,
        task_id: str,
        *,
        steps: Sequence[StepRecord] = (),
        tool_calls: Sequence[ToolCallRecord] = (),
        evidence: Sequence[Evidence] = (),
        conflicts: Sequence[Conflict] = (),
        review: ReviewRecord | None = None,
    ) -> None:
        now = datetime.now(UTC).replace(tzinfo=None)
        async with session_scope(self._sessions) as session:
            # 先清后写，且**同一个事务**：中途失败时要么是上一批、要么是这一批，
            # 不会出现"证据删了、新的还没写进去"——那会让复盘看到的图景
            # 比任何一批单独存在时都更误导。
            await _clear(session, task_id)
            if steps:
                session.add_all([_step_row(task_id, item, now) for item in steps])
            if tool_calls:
                session.add_all([_tool_call_row(task_id, item, now) for item in tool_calls])
            if evidence:
                session.add_all([_evidence_row(task_id, item, now) for item in evidence])
            if conflicts:
                session.add_all([_conflict_row(task_id, item, now) for item in conflicts])
            if review is not None:
                session.add(_review_row(task_id, review, now))

    async def list_evidence(self, task_id: str) -> list[Evidence]:
        async with session_scope(self._sessions) as session:
            rows = (
                (
                    await session.execute(
                        select(AgentEvidence)
                        .where(AgentEvidence.task_id == task_id)
                        .order_by(AgentEvidence.created_at, AgentEvidence.id)
                    )
                )
                .scalars()
                .all()
            )
        return [_evidence_of(row) for row in rows]


class InMemoryAgentArtifactRepository(AgentArtifactRepository):
    """进程内实现，供不连 MySQL 的单元测试使用。

    它也**整体替换**——与 SQL 实现同一语义。少这一点的话，
    "重投不会留下两批"这条断言在两个实现上会给出不同结论。
    """

    def __init__(self) -> None:
        self.steps: dict[str, list[StepRecord]] = {}
        self.tool_calls: dict[str, list[ToolCallRecord]] = {}
        self.evidence: dict[str, list[Evidence]] = {}
        self.conflicts: dict[str, list[Conflict]] = {}
        self.reviews: dict[str, ReviewRecord] = {}

    async def save(
        self,
        task_id: str,
        *,
        steps: Sequence[StepRecord] = (),
        tool_calls: Sequence[ToolCallRecord] = (),
        evidence: Sequence[Evidence] = (),
        conflicts: Sequence[Conflict] = (),
        review: ReviewRecord | None = None,
    ) -> None:
        self.steps[task_id] = list(steps)
        self.tool_calls[task_id] = list(tool_calls)
        self.evidence[task_id] = list(evidence)
        self.conflicts[task_id] = list(conflicts)
        if review is not None:
            self.reviews[task_id] = review

    async def list_evidence(self, task_id: str) -> list[Evidence]:
        return list(self.evidence.get(task_id, ()))


# ------------------------------------------------------------------ 行构造


async def _clear(session: AsyncSession, task_id: str) -> None:
    for model in (AgentTaskStep, AgentToolCall, AgentEvidence, AgentConflict, AgentReview):
        await session.execute(delete(model).where(model.task_id == task_id))


def _step_row(task_id: str, item: StepRecord, now: datetime) -> AgentTaskStep:
    return AgentTaskStep(
        id=item.id,
        task_id=task_id,
        step_key=item.step_key,
        objective=item.objective,
        tool=item.tool,
        depends_on_json=list(item.depends_on),
        required=item.required,
        status=item.status,
        attempt_count=item.attempt_count,
        result_summary_json=item.result_summary,
        origin=item.origin,
        revision_no=item.revision_no,
    )


def _tool_call_row(task_id: str, item: ToolCallRecord, now: datetime) -> AgentToolCall:
    return AgentToolCall(
        id=item.id,
        task_id=task_id,
        step_id=item.step_id,
        tool_name=item.tool_name,
        attempt_no=item.attempt_no,
        request_summary_json=item.request_summary,
        normalized_sql=item.normalized_sql,
        sql_fingerprint=item.sql_fingerprint,
        result_summary_json=item.result_summary,
        status=item.status,
        error_code=item.error_code,
        error_summary=item.error_summary,
        duration_ms=item.duration_ms,
        created_at=now,
    )


def _evidence_row(task_id: str, item: Evidence, now: datetime) -> AgentEvidence:
    return AgentEvidence(
        id=item.id,
        task_id=task_id,
        # `locator` 里的 `call_id` 就是产生它的工具调用（`tools/sql/evidence.py`
        # 与 `tools/rag/evidence.py` 都写了这个键）。**取不到就是 NULL**——
        # 不是所有证据都来自工具调用，而编一个外键关联比空着更糟。
        tool_call_id=str(item.locator.get("call_id") or "") or None,
        source_type=item.source_type,
        title=item.title,
        claim=item.claim,
        locator_json=item.locator,
        event_time_start=item.event_time.start.replace(tzinfo=None) if item.event_time else None,
        event_time_end=item.event_time.end.replace(tzinfo=None) if item.event_time else None,
        metric_code=item.metric_code,
        definition_version=item.definition_version,
        scope_json=dict(item.scope),
        reliability=item.reliability,
        content_hash=item.content_hash,
        created_at=now,
    )


def _conflict_row(task_id: str, item: Conflict, now: datetime) -> AgentConflict:
    return AgentConflict(
        id=item.id,
        task_id=task_id,
        type=item.type.value,
        severity=item.severity.value,
        evidence_ids_json=[str(value) for value in item.evidence_ids],
        description=item.description,
        difference_json=dict(item.detected_difference),
        explanations_json=list(item.possible_explanations),
        resolution=item.resolution.value,
        selected_basis=item.selected_basis,
        created_at=now,
    )


def _review_row(task_id: str, item: ReviewRecord, now: datetime) -> AgentReview:
    return AgentReview(
        id=item.id,
        task_id=task_id,
        round_no=item.round_no,
        status=item.status,
        score=item.score,
        coverage_score=item.coverage_score,
        evidence_score=item.evidence_score,
        consistency_score=item.consistency_score,
        issues_json=[dict(issue) for issue in item.issues],
        missing_evidence_json=list(item.missing_evidence),
        retry_target=item.retry_target,
        reason_code=item.reason_code,
        created_at=now,
    )


def _evidence_of(row: AgentEvidence) -> Evidence:
    """读回来。**与 `_evidence_row` 逐字段对称**——不对称的话，
    "落库的证据"与"答案里引用的证据"会给出两个不同的对象，
    而它们本该是同一条。"""
    payload: dict[str, object] = {
        "id": row.id,
        "source_type": row.source_type,
        "title": row.title or "",
        "claim": row.claim,
        "locator": dict(row.locator_json or {}),
        "metric_code": row.metric_code,
        "definition_version": row.definition_version,
        "scope": dict(row.scope_json or {}),
        "reliability": row.reliability or "MEDIUM",
        "content_hash": row.content_hash or "0" * 64,
    }
    if row.event_time_start is not None and row.event_time_end is not None:
        payload["event_time"] = {
            "start": row.event_time_start.replace(tzinfo=UTC),
            "end": row.event_time_end.replace(tzinfo=UTC),
        }
    return Evidence.model_validate(payload)


__all__ = [
    "AgentArtifactRepository",
    "InMemoryAgentArtifactRepository",
    "SqlAgentArtifactRepository",
]
