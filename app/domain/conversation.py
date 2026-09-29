"""会话与消息领域模型（对应详细设计 16.4 的 agent_conversation / agent_message 表）。

Phase 1 只需要「创建或复用会话」这一层语义：会话归属校验 + 最近消息时间。
多轮上下文（FR-CHAT-003 的 10 轮窗口与结构化摘要）**不往这里塞**——
那是「给模型的裁剪视图」，与「会话里存了什么」是两件事，
住在 `app/domain/memory.py`。本模块只管**存了什么**。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    title: str | None = None
    last_message_at: datetime
    created_at: datetime
    updated_at: datetime


class MessageRole(StrEnum):
    """详设 16.4 定死了这三个取值，**不给第四个位置**。

    多造一个（比如 `TOOL`）就等于在消息表里放工具输出，
    而消息表是**用户看得见的东西**——工具原始返回里有未脱敏的业务行数据。
    """

    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM_NOTICE = "SYSTEM_NOTICE"


class MessageStatus(StrEnum):
    """消息状态。

    ⚠️ **详设 16.4 只列了这一列的名字，没有定义取值**（`agent_message.status` 是
    `NOT NULL`）——本实现定为 `SENT` 并把这条回写详细设计。定在这里而不是留空，
    是因为它 `NOT NULL`：不给值第一次插入就 `IntegrityError`，
    而写入失败沿约定 50 是被吞掉的，症状是「消息表一直空着、没有任何报错」。
    """

    SENT = "SENT"


class Message(BaseModel):
    """一条用户可见的消息（16.4 的 `agent_message`）。

    **`task_id` 可空**：`SYSTEM_NOTICE` 不一定由某个任务产生。
    对 `USER` / `ASSISTANT` 而言它是**这一轮的口径从哪来**的挂钩——
    上一轮解析出的指标与区域记在 `agent_task.result_json` 上，
    多轮上下文靠这个键把「谁问了什么」与「解析成了什么」接起来。

    ⚠️ **`content` 里绝不能放模型思维链**（16.4 的硬规定）：
    这张表是给人看的，也是会进下一次 prompt 的历史。
    """

    model_config = ConfigDict(frozen=True)

    id: str
    conversation_id: str
    task_id: str | None = None
    role: MessageRole
    content: str
    status: MessageStatus = MessageStatus.SENT
    #: token 用量。**v1 恒为空**：网关的用量统计还没接进这条链路，
    #: 填一个估算值会让「这条消息花了多少」在产物上看起来是被计量过的。
    token_count: int | None = None
    created_at: datetime
