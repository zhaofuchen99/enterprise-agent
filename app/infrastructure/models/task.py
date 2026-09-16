"""任务、步骤、计划演进、发现与工具调用（详细设计 16.5 / 16.6）。

这五张表是「任务循环」能力的持久化形态：
`agent_plan_revision` 是「Agent 为什么改变了下一步」的**唯一权威记录**，
`agent_finding` 是演进步骤的触发依据，两者缺一，FR-PLAN-004 就无法回溯。
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
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


class AgentTask(Base):
    """任务状态、计划和最终结果（16.5）。

    这张表同时承载三种查询模式，索引不是「多建几个保险」而是各自对应一条路径：
    用户翻自己的任务、孤儿回收扫 `RUNNING`、补偿扫描扫 `QUEUED`。
    """

    __tablename__ = "agent_task"

    id: Mapped[ulid_pk]
    #: 澄清续跑或用户重试的来源任务（详细设计 15.2）
    parent_task_id: Mapped[ulid_ref_opt]
    conversation_id: Mapped[ulid_ref]
    user_id: Mapped[ulid_ref]
    #: 跨 API / 队列 / Worker 贯通的追踪 ID（唯一）
    trace_id: Mapped[str] = mapped_column(String(26))
    #: 同用户范围内可空唯一。MySQL 唯一索引允许多个 NULL，正是这里要的语义。
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    query_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    intent: Mapped[str | None] = mapped_column(String(32))
    #: 领取该任务的 Worker 标识，NULL 表示尚未领取
    worker_id: Mapped[str | None] = mapped_column(String(64))
    #: Worker 最后心跳时间，孤儿任务回收的判据
    heartbeat_at: Mapped[utc_dt_opt]
    #: 入队时间。与 started_at 之差即排队等待（NFR-P-08）
    queued_at: Mapped[utc_dt_opt]
    plan_json: Mapped[json_obj_opt]
    result_json: Mapped[json_obj_opt]
    final_answer_md: Mapped[str | None] = mapped_column(MEDIUMTEXT)
    error_code: Mapped[str | None] = mapped_column(String(64))
    #: 只落安全摘要，不落原始异常文本（可能含 SQL 与参数）
    error_message: Mapped[str | None] = mapped_column(Text)
    #: 轨迹写入失败时置位（FR-TRACE-001 异常情况），不得影响主任务
    trace_incomplete: Mapped[bool] = mapped_column(Boolean)
    started_at: Mapped[utc_dt_opt]
    finished_at: Mapped[utc_dt_opt]
    created_at: Mapped[utc_dt]
    updated_at: Mapped[utc_dt]

    __table_args__ = (
        UniqueConstraint("trace_id", name="uk_task_trace"),
        UniqueConstraint("user_id", "idempotency_key", name="uk_task_user_idempotency"),
        Index("idx_task_user_created", "user_id", "created_at"),
        Index("idx_task_status_updated", "status", "updated_at"),
        # 供孤儿任务回收扫描：按 (状态, 心跳时间) 定位超时未续期的 RUNNING 任务
        Index("idx_task_status_heartbeat", "status", "heartbeat_at"),
        Index("idx_task_worker", "worker_id"),
    )


class AgentTaskStep(Base):
    """子任务状态，含来源与下钻路径（16.6）。

    `origin` 与 `trigger_finding_id` 是任务循环的追溯入口：
    评测要求「每个 `origin=EXTENDED` 的步骤可反查到 `finding_id`」（开发流程 7.5）。
    """

    __tablename__ = "agent_task_step"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    #: 计划内稳定标识。同一任务内唯一——重新规划产生的步骤要换新的 step_key。
    step_key: Mapped[str] = mapped_column(String(64))
    objective: Mapped[str] = mapped_column(Text)
    tool: Mapped[str | None] = mapped_column(String(64))
    #: 依赖的其他步骤 step_key 列表
    depends_on_json: Mapped[json_list_opt]
    required: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(32))
    attempt_count: Mapped[int] = mapped_column(Integer)
    result_summary_json: Mapped[json_obj_opt]
    started_at: Mapped[utc_dt_opt]
    finished_at: Mapped[utc_dt_opt]
    #: PLANNER / EXTENDED / REPLAN
    origin: Mapped[str] = mapped_column(String(16))
    trigger_finding_id: Mapped[ulid_ref_opt]
    #: 自适应下钻路径（详细设计 4.4）
    drilldown_path_json: Mapped[json_list_opt]
    revision_no: Mapped[int] = mapped_column(Integer)
    #: 该步被放弃时的原因，对应 agent_plan_revision.skipped_step_ids_json
    skipped_reason: Mapped[str | None] = mapped_column(String(255))

    __table_args__ = (
        UniqueConstraint("task_id", "step_key", name="uk_task_step_key"),
        # 按计划版本查看演进历史
        Index("idx_task_step_revision", "task_id", "revision_no"),
    )


class AgentPlanRevision(Base):
    """计划演进与重新规划记录（16.6）。

    `plan_deltas` 从这张表加载——它不落模型原始输出，只落经校验的结构化字段。
    """

    __tablename__ = "agent_plan_revision"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    revision_no: Mapped[int] = mapped_column(Integer)
    #: INITIAL / EXTEND / REPLAN
    trigger_type: Mapped[str] = mapped_column(String(16))
    trigger_finding_id: Mapped[ulid_ref_opt]
    added_step_ids_json: Mapped[json_list_opt]
    skipped_step_ids_json: Mapped[json_list_opt]
    reason: Mapped[str | None] = mapped_column(Text)
    #: 本次变更时的剩余预算，用于复盘「预算是否被借用」
    budget_snapshot_json: Mapped[json_obj_opt]
    created_at: Mapped[utc_dt]

    __table_args__ = (
        # revision_no 必须连续无跳号
        UniqueConstraint("task_id", "revision_no", name="uk_plan_revision_no"),
    )


class AgentFinding(Base):
    """中间发现及其引出的问题（16.6）。

    **不保存模型原始输出文本**，只保存经校验的结构化字段。
    """

    __tablename__ = "agent_finding"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    #: 产生该发现的步骤
    step_id: Mapped[ulid_ref_opt]
    statement: Mapped[str] = mapped_column(Text)
    #: 被本次发现排除的假设
    ruled_out_json: Mapped[json_list_opt]
    evidence_ids_json: Mapped[json_list_opt]
    kind: Mapped[str] = mapped_column(String(32))
    #: 该发现对应的进度判定（CONTINUE / EXTEND / AGGREGATE / CLARIFY / FAIL）
    decision: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[utc_dt]

    __table_args__ = (Index("idx_finding_task_created", "task_id", "created_at"),)


class AgentToolCall(Base):
    """Tool 调用与 SQL 执行摘要（16.6）。

    **结果只保存行数、列名、统计摘要和 hash，不保存全部原始行**（19.4 脱敏纪律）。
    `normalized_sql` 是规范化后的 SQL 文本；绑定参数另行脱敏，不落在这里。
    """

    __tablename__ = "agent_tool_call"

    id: Mapped[ulid_pk]
    task_id: Mapped[ulid_ref]
    step_id: Mapped[ulid_ref_opt]
    tool_name: Mapped[str] = mapped_column(String(64))
    #: 同一 (step, tool) 的第几次尝试，对应自修复与重试
    attempt_no: Mapped[int] = mapped_column(Integer)
    request_summary_json: Mapped[json_obj_opt]
    normalized_sql: Mapped[str | None] = mapped_column(Text)
    #: 规范化 SQL 的指纹，用于「同一批烂 SQL 反复出现」的归因
    sql_fingerprint: Mapped[str | None] = mapped_column(String(64))
    result_summary_json: Mapped[json_obj_opt]
    status: Mapped[str] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_summary: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[utc_dt]

    __table_args__ = (
        Index("idx_tool_call_task_created", "task_id", "created_at"),
        Index("idx_tool_call_sql_fingerprint", "sql_fingerprint"),
    )
