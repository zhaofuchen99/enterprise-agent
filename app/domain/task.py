"""任务领域模型（对应详细设计 16.5 的 agent_task 表）。

字段与表结构一一对应，字段名刻意与列名保持一致，使 Phase 2 接入 MySQL 时
不需要再做一层映射（少一层映射就少一处能写错的地方）。

未纳入的列：`plan_json` / `result_json` 是执行期中间产物，
对外只暴露**已校验**的结构化字段，不把裸 JSON 直接回给客户端（开发流程 5.2）。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

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
    #: 已校验的计划摘要（16.5）。**审阅一份答案时第一个要看的东西**：
    #: "它查了哪几步、每步成没成"比答案本身更能说明结论有多硬。
    plan_json: dict[str, Any] | None = None
    #: 结构化结果（16.5）。审查结论、证据、冲突、限制都在这里——
    #: 它们是**可追溯性的载体**，只留在进程内等于没记。
    result_json: dict[str, Any] | None = None
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


class TaskOutcome(BaseModel):
    """任务体的产出（16.5 的 `plan_json` / `result_json` + 答案）。

    **放在 `domain/` 而不是 `services/` 或 `agent/`**：两边都要用它——
    `agent.runner` 产出它、`services.task_runner` 收下并落库——
    而依赖方向是单向的（`services → agent`），放进任一侧都会让另一侧反向依赖。

    三个字段分开而不是塞进一个 `payload`：`answer` 要落 `final_answer_md`
    （MEDIUMTEXT，给人读），`plan` 与 `payload` 落两个 JSON 列（给程序读）。
    合成一个的话，"这条任务的答案"与"这条任务的结构化结果"就开始互相污染。
    """

    model_config = ConfigDict(frozen=True)

    answer: str | None = None
    #: Supervisor 判出的意图（8.1 的 `IntentResult.intent`）。
    #: 落 `agent_task.intent`——16.5 有这个列，而它回答的是
    #: "这条任务是当查询问的、还是当制度问的"，排查误答时第一个要看的东西。
    intent: str | None = None
    plan: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
