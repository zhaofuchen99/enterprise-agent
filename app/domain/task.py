"""任务领域模型（对应详细设计 16.5 的 agent_task 表）。

字段与表结构一一对应，字段名刻意与列名保持一致，使 Phase 2 接入 MySQL 时
不需要再做一层映射（少一层映射就少一处能写错的地方）。

未纳入的列：`plan_json` / `result_json` 是执行期中间产物，
对外只暴露**已校验**的结构化字段，不把裸 JSON 直接回给客户端（开发流程 5.2）。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from app.core.errors import ErrorCode


class TaskStatus(StrEnum):
    """任务状态（详细设计 2.2）。

    `CANCEL_REQUESTED` 只在详细设计 17.5 出现，需求规格 FR-CHAT-002 未列出。
    以详细设计为准：它区分了「用户已请求取消」与「Worker 已确认取消」，
    缺少它就无法表达「取消请求已记录但尚未生效」这个中间态。
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_CLARIFICATION = "WAITING_CLARIFICATION"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

#: 占用用户并发配额的状态。
#:
#: 取「一切非终态」而不是「仅 QUEUED/RUNNING」：详细设计 19.3 规定
#: 「任务结束（任意终态）时必须递减」，即释放配额的时机就是进入终态，
#: 在此之前都算占用。这样 WAITING_CLARIFICATION 也不会被用来绕过并发限制。
ACTIVE_STATUSES: frozenset[TaskStatus] = frozenset(TaskStatus) - TERMINAL_STATUSES


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    conversation_id: str
    trace_id: str
    query_text: str
    status: TaskStatus = TaskStatus.QUEUED
    parent_task_id: str | None = None
    #: 同用户范围内唯一（16.5 的 uk_task_user_idempotency）
    idempotency_key: str | None = None
    intent: str | None = None
    worker_id: str | None = None
    heartbeat_at: datetime | None = None
    final_answer_md: str | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    trace_incomplete: bool = False
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
