"""消息仓储（详细设计 16.4 的 `agent_message`）。

这张表从 Phase 2 建好起一直是空的（跟 `agent_task_step` 那五张表同一处境），
FR-CHAT-003 的多轮上下文是它的第一个消费者——而写入方来自 FR-CHAT-001
的处理流程「**保存用户消息**」。

## 为什么没有一个 `list_by_conversation` 式的"给会话列表用"的方法

会话列表接口属【后续扩展】（登记在 CLAUDE.md）。现在加一个没有调用方的方法，
它就没有契约测试能钉住它的行为（"按时间倒序还是正序"这种问题只有真的被用
才会暴露），而它看起来是"已经支持的"。
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.conversation import Message, MessageRole
from app.infrastructure.db import session_scope
from app.infrastructure.models.identity import AgentMessage
from app.repositories._mapping import message_from_row, message_to_row_values


class MessageRepository(Protocol):
    async def add(self, message: Message) -> None: ...

    async def list_recent(self, conversation_id: str, *, limit: int) -> list[Message]:
        """按**时间正序**返回最近 `limit` 条消息。

        **顺序是契约的一部分**：对话历史的顺序反了，渲染出来的上下文
        会让模型把因果读反（"我说了这个"变成"它先说了这个"），
        而产物上完全看不出来——所以这一条由契约测试两个实现同跑钉住。
        """
        ...

    async def replace_for_task(
        self, task_id: str, *, role: MessageRole, message: Message | None
    ) -> None:
        """把某个任务的某种角色的消息**整体替换**成一条（`None` 表示删掉）。

        照约定 49 的先例：这几张表都没有天然唯一键，而任务**可能被执行两次**
        （图跑完、收尾提交前进程死掉，孤儿回收会重跑）。纯 `add` 会写出
        第二条助手消息，于是下一轮的上下文里**上一轮出现两次**——
        而两条内容还一模一样，看起来完全正常。
        """
        ...


class InMemoryMessageRepository:
    def __init__(self) -> None:
        self._by_id: dict[str, Message] = {}

    async def add(self, message: Message) -> None:
        if message.id in self._by_id:
            raise ValueError(f"消息重复：{message.id}")
        self._by_id[message.id] = message

    async def list_recent(self, conversation_id: str, *, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        matching = [
            message
            for message in self._by_id.values()
            if message.conversation_id == conversation_id
        ]
        # 与 SQL 侧同一套排序键（含 id 作为次级键），否则两个实现在
        # 「同一毫秒两条消息」时给出不同顺序，而那种不一致只在真库上出现。
        matching.sort(key=lambda item: (item.created_at, item.id))
        return matching[-limit:]

    async def replace_for_task(
        self, task_id: str, *, role: MessageRole, message: Message | None
    ) -> None:
        for key in [key for key, item in self._by_id.items() if _matches(item, task_id, role)]:
            del self._by_id[key]
        if message is not None:
            await self.add(message)


def _matches(message: Message, task_id: str, role: MessageRole) -> bool:
    return message.task_id == task_id and message.role is role


class SqlMessageRepository:
    """MySQL 实现。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, message: Message) -> None:
        async with session_scope(self._sessions) as session:
            session.add(AgentMessage(**message_to_row_values(message)))

    async def list_recent(self, conversation_id: str, *, limit: int) -> list[Message]:
        if limit <= 0:
            return []
        async with session_scope(self._sessions) as session:
            # `LIMIT` 作用在**倒序**上（要的是最近 N 条，不是最早 N 条），
            # 再在 Python 里翻回时间正序——写成子查询也能对，但这里读起来
            # 与「取最近 N 条再按时间排」这句需求是同一句话。
            #
            # **次级键 `id` 不是装饰**：`created_at` 是 `DATETIME(3)`，
            # 同一毫秒内的两条消息只靠它排序时顺序由存储引擎决定，
            # 而顺序不确定的对话历史是最难查的一类 bug。
            # `id` 是 ULID 派生（前缀 + 时间戳 + 随机，时间戳在高位），
            # 所以按它排等价于按写入时间排。
            result = await session.execute(
                select(AgentMessage)
                .where(AgentMessage.conversation_id == conversation_id)
                .where(AgentMessage.deleted_at.is_(None))
                .order_by(AgentMessage.created_at.desc(), AgentMessage.id.desc())
                .limit(limit)
            )
            rows = list(result.scalars().all())
        rows.reverse()
        return [message_from_row(row) for row in rows]

    async def replace_for_task(
        self, task_id: str, *, role: MessageRole, message: Message | None
    ) -> None:
        # 删除与插入在**同一个** `session_scope` 里：分两次提交的话，
        # 中间那一刻这一轮的消息是消失的，而另一个进程正好在读上下文。
        async with session_scope(self._sessions) as session:
            await session.execute(
                delete(AgentMessage)
                .where(AgentMessage.task_id == task_id)
                .where(AgentMessage.role == role.value)
            )
            if message is not None:
                session.add(AgentMessage(**message_to_row_values(message)))


__all__ = [
    "InMemoryMessageRepository",
    "MessageRepository",
    "SqlMessageRepository",
]
