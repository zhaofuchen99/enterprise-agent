"""会话仓储。Phase 2 换 MySQL 实现（详细设计 16.4 的 agent_conversation）。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from app.domain.conversation import Conversation


class ConversationRepository(Protocol):
    async def add(self, conversation: Conversation) -> None: ...

    async def get(self, conversation_id: str) -> Conversation | None: ...

    async def touch(self, conversation_id: str, at: datetime) -> None:
        """更新 `last_message_at` 与 `updated_at`，供会话列表按时间排序。"""
        ...


class InMemoryConversationRepository:
    def __init__(self) -> None:
        self._by_id: dict[str, Conversation] = {}

    async def add(self, conversation: Conversation) -> None:
        if conversation.id in self._by_id:
            raise ValueError(f"会话重复：{conversation.id}")
        self._by_id[conversation.id] = conversation

    async def get(self, conversation_id: str) -> Conversation | None:
        return self._by_id.get(conversation_id)

    async def touch(self, conversation_id: str, at: datetime) -> None:
        conversation = self._by_id.get(conversation_id)
        if conversation is None:
            # 与真实数据库 UPDATE 影响 0 行的行为保持一致：不报错，交给调用方决定
            return
        self._by_id[conversation_id] = conversation.model_copy(
            update={"last_message_at": at, "updated_at": at}
        )
