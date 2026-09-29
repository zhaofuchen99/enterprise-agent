"""`TaskService.create_task` 的**消息写入时机**。

这个文件只为一件事存在：用户消息写在 `tasks.add` **成功之后**。
这个位置是承重的，而它写错的症状在别处看不见——

| 写在哪 | 症状 |
|---|---|
| 放在 `tasks.add` 之前 | 幂等竞争的**败者**也写了一条 → 同一句话出现两次 |
| 放在 `try` 里 | 同上（败者在 `DuplicateTaskError` 之前已经写过） |
| 放在 `_dispatch` 之后 | 投递失败时消息已经写了，其实无害；但 `_replay` 分支会漏写 |

代价是它离"看起来很自然的写法"（写在构造 `Task` 之后）只差几行，
所以下面三条用例是**位置断言**，不是功能断言。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import fakeredis.aioredis
import pytest

from app.core.config import Settings
from app.core.ids import IdPrefix, deterministic_id, new_trace_id
from app.domain.conversation import Conversation, MessageRole
from app.domain.user import User, UserRole
from app.repositories.conversation_repo import InMemoryConversationRepository
from app.repositories.message_repo import InMemoryMessageRepository
from app.repositories.task_repo import (
    DuplicateTaskError,
    InMemoryTaskRepository,
    TaskRepository,
)
from app.services.conversation_service import ConversationService
from app.services.task_runner import TaskRunner
from app.services.task_service import TaskService
from app.tests.fakes import FakeJobQueue

_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _settings() -> Settings:
    return Settings(
        model_provider="p",
        model_name="m",
        model_api_key="k",
        embedding_model="e",
        embedding_api_key="k",
        database_url_agent="mysql+asyncmy://a@localhost/a",
        database_url_business_ro="mysql+asyncmy://a@localhost/b",
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
        jwt_secret="x" * 32,
    )


def _user() -> User:
    return User(
        id=deterministic_id(IdPrefix.USER, "analyst"),
        username="analyst",
        display_name="analyst",
        role=UserRole.ANALYST,
        password_hash="x",
    )


class _FailingAddRepository(InMemoryTaskRepository):
    """`add` 必失败——**但失败发生在写入点之前**（`DuplicateTaskError` 是
    唯一能让 `create_task` 在写入点前返回的异常，且 `idempotency_key` 为空时
    它会原样抛给调用方）。"""

    async def add(self, task: Any) -> None:
        raise DuplicateTaskError("task_id 重复")


async def _service(
    *,
    tasks: TaskRepository | None = None,
    conversations: InMemoryConversationRepository | None = None,
) -> tuple[TaskService, InMemoryMessageRepository]:
    settings = _settings()
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    messages = InMemoryMessageRepository()
    task_repo = tasks if tasks is not None else InMemoryTaskRepository()
    # `events` 要真的事件总线（它写 Redis Stream），所以这里用 fakeredis
    # 而不是再造一个替身——造替身会让"事件发没发"变成测替身的行为。
    from app.services.event_bus import RedisStreamEventBus

    runner = TaskRunner(
        tasks=task_repo,
        queue=FakeJobQueue(),
        events=RedisStreamEventBus(redis, settings),
        settings=settings,
        redis=redis,
        artifacts=_artifacts(),
        conversations=ConversationService(messages),
    )
    service = TaskService(
        tasks=task_repo,
        conversations=conversations or InMemoryConversationRepository(),
        settings=settings,
        runner=runner,
        conversation_service=ConversationService(messages),
    )
    return service, messages


def _artifacts() -> Any:
    from app.repositories.agent_repo import InMemoryAgentArtifactRepository

    return InMemoryAgentArtifactRepository()


async def _create(
    service: TaskService, *, message: str = "华东销售额为什么下降", key: str | None = None
) -> Any:
    return await service.create_task(
        user=_user(),
        message=message,
        conversation_id=None,
        idempotency_key=key,
        trace_id=new_trace_id(),
    )


async def test_a_new_task_records_exactly_one_user_message() -> None:
    service, messages = await _service()

    result = await _create(service)

    stored = await messages.list_recent(result.task.conversation_id, limit=10)
    assert [item.role for item in stored] == [MessageRole.USER]
    assert stored[0].content == "华东销售额为什么下降"
    assert stored[0].task_id == result.task.id


async def test_a_replayed_request_does_not_record_a_second_message() -> None:
    """同幂等键重放 → 会话里仍然只有**一条**用户消息。

    重放走的是 `_replay` 分支（在 `tasks.add` 之前 return），
    写入点在它之后，所以不会重复写。
    """
    service, messages = await _service()
    first = await _create(service, key="key-1")

    second = await _create(service, key="key-1")

    assert second.created is False
    assert second.task.id == first.task.id
    stored = await messages.list_recent(first.task.conversation_id, limit=10)
    assert len(stored) == 1


async def test_a_failed_add_records_no_message() -> None:
    """`tasks.add` 抛异常 → 一条消息都不写。

    这条钉的是**写入点在 `try/except` 之后**：写在里面的话，
    并发同键的败者在拿到 `DuplicateTaskError` 之前就已经写过一条了。

    会话**先建好并显式传进去**（而不是让服务自己建）：失败的路径上拿不到
    服务内部生成的 `conversation_id`，没有它就没法读回消息——
    而"读不回来"与"没写进去"是两件事，用例必须能分辨。
    """
    conversations = InMemoryConversationRepository()
    conversation = Conversation(
        id="cnv_0000000001AAAAAAAAAAAA",
        user_id=_user().id,
        title=None,
        last_message_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )
    await conversations.add(conversation)
    service, messages = await _service(tasks=_FailingAddRepository(), conversations=conversations)

    with pytest.raises(DuplicateTaskError):
        await service.create_task(
            user=_user(),
            message="华东销售额为什么下降",
            conversation_id=conversation.id,
            idempotency_key=None,
            trace_id=new_trace_id(),
        )

    assert await messages.list_recent(conversation.id, limit=10) == []
