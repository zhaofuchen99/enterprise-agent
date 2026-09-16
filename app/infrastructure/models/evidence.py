"""证据、冲突、审查与轨迹（详细设计 16.7）。

`agent_trace_event` 是事件流的**权威来源**——Redis Stream 只负责快，
它被 `MAXLEN` 裁剪后，重连补齐必须回到这张表（详细设计 18.3）。
因此 `(task_id, sequence)` 的唯一约束不是「防重复」的保险，而是顺序保证本身。
"""

from __future__ import annotations

from sqlalchemy import Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base
from app.infrastructure.models.common import (
    json_list_opt,
    json_obj_opt,
    ulid_pk,
    ulid_ref,
    ulid_ref_opt,
    utc_dt,
    utc_dt_opt,
)


class AgentEvidence(Base):
    """统一证据（16.7）。

    两个来源（SQL 结果行、RAG 文档片段）汇聚到同一张表，
    使 Analysis 与 Reviewer 不必区分证据来自哪条链路——
    这正是「双源」能被一致对待的前提。
    """

    __tablename__ = "agent_evidence"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    #: 产生该证据的工具调用。非工具来源（如用户输入）可空。
    tool_call_id: Mapped[ulid_ref_opt]
    #: SQL / DOCUMENT / WEB —— 取值按详细设计 13.1 的 `Evidence.source_type`，
    #: 不是「SQL / RAG / SEARCH」。RAG 产出的证据按 13.1 记为 `DOCUMENT`：
    #: `locator` 与 `reliability` 都是按「文档」的语义定义的（document_id /
    #: version / chunk_id）。表结构未变，`String(16)` 两者都装得下。
    source_type: Mapped[str] = mapped_column(String(16))
    title: Mapped[str | None] = mapped_column(String(255))
    #: 该证据支撑的具体陈述。这是「答案有证据」里被引用的那一句。
    claim: Mapped[str] = mapped_column(Text)
    #: 定位信息：SQL 为库/表/行号，RAG 为 document_id / section_path / page_no
    locator_json: Mapped[json_obj_opt]
    event_time_start: Mapped[utc_dt_opt]
    event_time_end: Mapped[utc_dt_opt]
    metric_code: Mapped[str | None] = mapped_column(String(64))
    #: 指标口径版本（16.8 的 embedding_version 之外，证据还要记口径版本）
    definition_version: Mapped[str | None] = mapped_column(String(32))
    #: 范围口径：区域/产品/渠道的取值集合，SCOPE 冲突的比对依据
    scope_json: Mapped[json_obj_opt]
    #: 可靠性等级。外部来源默认低于内部业务库与已发布制度（详细设计 12.3）。
    reliability: Mapped[str | None] = mapped_column(String(16))
    content_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[utc_dt]

    __table_args__ = (
        Index("idx_evidence_task_source", "task_id", "source_type"),
        Index("idx_evidence_content_hash", "content_hash"),
    )


class AgentConflict(Base):
    """多源冲突（16.7）。

    `evidence_ids_json` 指向参与冲突的证据。冲突**只描述差异，不擅自裁决**：
    取舍依据记在 `selected_basis`，供 Reviewer 与用户复核。
    """

    __tablename__ = "agent_conflict"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    #: TIME / SCOPE / DEFINITION / VALUE / SOURCE（详细设计 13.3）
    type: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(16))
    evidence_ids_json: Mapped[json_list_opt]
    description: Mapped[str] = mapped_column(Text)
    #: 差异的量化描述（差值、比值、口径差异点）
    difference_json: Mapped[json_obj_opt]
    #: 可能造成差异的解释（含税/未税、统计截止日…）
    explanations_json: Mapped[json_list_opt]
    resolution: Mapped[str | None] = mapped_column(String(32))
    selected_basis: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[utc_dt]

    __table_args__ = (Index("idx_conflict_task_severity", "task_id", "severity"),)


class AgentReview(Base):
    """Reviewer 记录（16.7）。

    一轮审查一行，`(task_id, round_no)` 唯一使「审了几轮」可直接计数，
    与 FR-REV-002 的重试上限对得上。
    """

    __tablename__ = "agent_review"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    round_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))
    score: Mapped[float | None] = mapped_column(Float)
    coverage_score: Mapped[float | None] = mapped_column(Float)
    evidence_score: Mapped[float | None] = mapped_column(Float)
    consistency_score: Mapped[float | None] = mapped_column(Float)
    issues_json: Mapped[json_list_opt]
    missing_evidence_json: Mapped[json_list_opt]
    #: 需要补证时指向的重试目标（步骤或节点）
    retry_target: Mapped[str | None] = mapped_column(String(64))
    reason_code: Mapped[str | None] = mapped_column(String(64))
    #: prompt 与模型版本随审查记录落库，否则改完 prompt 无法归因
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    model_name: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[utc_dt]

    __table_args__ = (UniqueConstraint("task_id", "round_no", name="uk_review_round"),)


class AgentTraceEvent(Base):
    """节点和工具轨迹（16.7 / FR-TRACE-001）。

    `payload_json` **不得包含 prompt 全文、模型原始输出或工具原始行数据**，
    只留哈希、版本与摘要（19.4 脱敏纪律）。
    """

    __tablename__ = "agent_trace_event"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    trace_id: Mapped[str] = mapped_column(String(26))
    #: 单任务内单调递增，先持久化再投递（详细设计 18.4）
    sequence: Mapped[int] = mapped_column(Integer)
    node: Mapped[str | None] = mapped_column(String(64))
    tool: Mapped[str | None] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str | None] = mapped_column(String(32))
    payload_json: Mapped[json_obj_opt]
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[utc_dt]

    __table_args__ = (
        # 顺序保证来自这条唯一约束，不来自 Redis（18.3）
        UniqueConstraint("task_id", "sequence", name="uk_trace_event_sequence"),
    )
