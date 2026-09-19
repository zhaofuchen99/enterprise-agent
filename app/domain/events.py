"""任务事件模型（详细设计 18.1 的事件模型 + 18.2 的事件清单）。

## 它为什么在 `domain/` 而不是 `services/`

事件的**生产者是图里的节点**（`app/agent/**`），而依赖方向是
`api → services → agent / domain / tools → …`——agent 不能反向 import services。
把它放在 `services/event_bus.py` 里，`agent/tracing.py` 就只有两条路：
向上依赖（破坏分层），或者自己再造一份事件类型（两份清单必然漂移，
而症状是"某个事件名只有一边有"）。

放这里之后，两边都向下依赖它：`services/event_bus.py` 提供 Redis 适配器，
`agent/tracing.py` 只依赖这个模型与一个 `publish` 形状的 Protocol。

## `TaskEventType` 只列**真的有生产者**的取值

没有生产者的枚举项会变成「看起来已经支持、实际永远收不到」的假契约，
读代码的人无法分辨。每条都注明了谁发它。18.2 里仍未落地的三类各有缺前提，
见 `services/event_bus.py` 的说明（`answer.delta` / `task.retrying` /
`plan.updated` 的 `trigger_finding_id`），已登记在 CLAUDE.md。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskEventType(StrEnum):
    """18.2 的事件清单。注释里写的是**生产者**。"""

    # ---- Worker 侧（TaskRunner）------------------------------------------
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    #: 18.2 缺这一项（17.5 要求取消可用，但清单里没有能表达"Worker 已确认取消"
    #: 的事件），已作为文档缺口登记
    TASK_CANCELLED = "task.cancelled"

    # ---- Worker 侧（图里的节点）------------------------------------------
    #: 计划通过校验（`supervisor`）。data: step_count、tool_types
    PLAN_CREATED = "plan.created"
    #: 任务循环的判定完成（`reflect`）。data: decision、finding_statement、
    #: open_question_count。**18.2 明写它（连同 plan.updated）不允许省略**：
    #: 「本系统任务循环能力对用户可见的载体」
    PROGRESS_ASSESSED = "progress.assessed"
    #: 计划被演进（`reflect` 追加了步骤）。data: revision_no、added_steps、
    #: skipped_steps、trigger_finding_id、reason。**生产者是 `reflect` 而不是
    #: 18.2 写的 `plan_extend`**：切片内没有 `plan_extend` 节点，演进由
    #: `reflect` 直接改计划（见 `agent/graph.py` 的偏差清单）
    PLAN_UPDATED = "plan.updated"
    #: 进入/离开一个可见节点（`agent/tracing.py` 的包装器）。
    #: 离开时带 status 与 duration_ms；进入时 duration_ms 为空（给它 0 会被读成"瞬间完成"）
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    #: 节点**返回**了失败（不抛异常）。抛出异常的节点在此留下痕迹的通道是
    #: `trace_incomplete`，见 `agent/tracing.py` 的能力边界说明
    NODE_FAILED = "node.failed"
    #: 审查完成（`reviewer`）。data: status、score、issue_count
    REVIEW_COMPLETED = "review.completed"
    #: 需要用户补充输入（`supervisor` 判 CLARIFICATION）。
    #: data: question、missing_fields
    CLARIFICATION_REQUIRED = "clarification.required"

    # ---- API 侧（SSE 端点自己发，不经事件总线）----------------------------
    #: 空闲时的心跳（18.2）。**不带 `id:` 帧头**：心跳不是任务事件，
    #: 让它推进客户端的 Last-Event-ID 会把续传游标带到一条不存在的事件上
    HEARTBEAT = "heartbeat"
    #: 流终止（18.2）。**由订阅端发出，不由 Worker 发**（18.2 原文）
    DONE = "done"
    #: 任务已结束、而 Redis 流已被清理时补发的当前状态快照。
    #: ⚠️ **18.2 的清单里没有这一类**，是本实现补的（已回写登记）：
    #: 18.4 要求"事件已被清理时发送 snapshot 并标记 replay_lost=true"，
    #: 却没定义它叫什么、data 是什么。这里定成事件名 `snapshot`，
    #: data = 任务当前状态 + `replay_lost: true`
    SNAPSHOT = "snapshot"


class TaskEvent(BaseModel):
    """18.1 的事件模型。字段名与顺序都对齐文档，不另起名字。

    `event_id` 与 `sequence` 的默认值是空的，因为它们在**写入流之前无从得知**：
    序号来自 Redis 分配的 Stream ID。调用方永远不该自己构造一个带这两个字段的
    事件——用 `publish` 拿到补全后的对象，用 `read` 拿到完整的事件。
    """

    #: Redis Stream ID，同时用作 SSE 的 Last-Event-ID
    event_id: str = Field(default="", description="由事件总线在写入后回填")
    sequence: int = Field(default=0, description="单任务内单调递增，客户端据此去重排序")
    type: TaskEventType
    timestamp: datetime
    task_id: str
    trace_id: str
    node: str | None = None
    step_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


__all__ = ["TaskEvent", "TaskEventType"]
