"""会话领域模型（对应详细设计 16.4 的 agent_conversation 表）。

Phase 1 只需要「创建或复用会话」这一层语义：会话归属校验 + 最近消息时间。
多轮上下文构建（FR-CHAT-003 的 10 轮窗口与结构化摘要）属 Phase 6，
届时在 `ConversationContext` 里实现，不往这里塞。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Conversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    title: str | None = None
    last_message_at: datetime
    created_at: datetime
    updated_at: datetime
