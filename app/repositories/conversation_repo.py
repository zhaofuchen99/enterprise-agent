"""会话仓储（详细设计 16.4 的 `agent_conversation`）。

`InMemoryConversationRepository` 供单元测试使用；`SqlConversationRepository`
是 Phase 2 起的生产实现。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.conversation import Conversation
from app.infrastructure.db import session_scope
from app.infrastructure.models.identity import AgentConversation
from app.repositories._mapping import (
    conversation_from_row,
    conversation_to_row_values,
    to_db_time,
)


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


class SqlConversationRepository:
    """MySQL 实现。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, conversation: Conversation) -> None:
        async with session_scope(self._sessions) as session:
            session.add(AgentConversation(**conversation_to_row_values(conversation)))

    async def get(self, conversation_id: str) -> Conversation | None:
        async with session_scope(self._sessions) as session:
            row = await session.get(AgentConversation, conversation_id)
            return conversation_from_row(row) if row is not None else None

    async def touch(self, conversation_id: str, at: datetime) -> None:
        """更新 `last_message_at` 与 `updated_at`。

        会话不存在时**静默返回**，与内存实现一致：调用方在「创建或复用会话」
        的路径上，此刻会话可能已被清理，这不是错误。
        """
        async with session_scope(self._sessions) as session:
            await session.execute(
                update(AgentConversation)
                .where(AgentConversation.id == conversation_id)
                .values(last_message_at=to_db_time(at), updated_at=to_db_time(at))
            )
