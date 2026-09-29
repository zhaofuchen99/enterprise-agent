"""消息仓储契约（16.4 的 `agent_message`）。

**同一份断言跑两个实现**，理由同 `test_task_repo.py`：只测一边，
另一边会悄悄烂掉，而"内存跑得通、SQL 报错"这类差异恰恰只在换实现时才暴露。

这张表从 Phase 2 建好起一直是空的——本切片之前**没有任何写入方**，
所以下面的往返用例是它的第一次执行验证（此前连列名写错都不会有人发现）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.conversation import Message, MessageRole
from app.repositories.message_repo import (
    InMemoryMessageRepository,
    MessageRepository,
    SqlMessageRepository,
)

_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _message(
    index: int,
    *,
    conversation_id: str = "cnv_0000000001AAAAAAAAAAAA",
    role: MessageRole = MessageRole.USER,
    task_id: str | None = None,
    created_at: datetime = _NOW,
    content: str = "华东销售额为什么下降",
) -> Message:
    return Message(
        id=f"msg_{index:022d}",
        conversation_id=conversation_id,
        task_id=task_id,
        role=role,
        content=content,
        created_at=created_at,
    )


@pytest.fixture(
    params=[
        pytest.param("memory", id="memory"),
        # 标记写在 param 上：写在用例上会把内存变体一起排除掉。
        pytest.param("sql", id="sql", marks=pytest.mark.integration),
    ]
)
async def make_repo(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[[], MessageRepository]]:
    if request.param == "memory":
        yield InMemoryMessageRepository
        return

    from app.tests.db import sql_sessions

    async with sql_sessions() as sessions:
        yield lambda: SqlMessageRepository(sessions)


# ------------------------------------------------------------------ 往返
async def test_add_then_list_round_trips_every_field(
    make_repo: Callable[[], MessageRepository],
) -> None:
    repo = make_repo()
    message = _message(1, role=MessageRole.ASSISTANT, task_id="tsk_1")
    await repo.add(message)

    loaded = await repo.list_recent(message.conversation_id, limit=10)

    assert loaded == [message]


async def test_list_recent_is_chronological(
    make_repo: Callable[[], MessageRepository],
) -> None:
    """**时间正序**是契约的一部分。

    反了的话，渲染出来的上下文会让人（与模型）把因果读反——
    "我说了这个、它才答那个"变成"它先说了、我才问"——
    而产物上完全看不出来（上下文里就是几句正常的话）。
    """
    repo = make_repo()
    for index in range(4):
        await repo.add(_message(index, created_at=_NOW.replace(minute=index)))

    loaded = await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10)

    assert [item.created_at.minute for item in loaded] == [0, 1, 2, 3]


async def test_list_recent_returns_the_latest_ones(
    make_repo: Callable[[], MessageRepository],
) -> None:
    """`limit` 取的是**最近 N 条**，不是最早 N 条。"""
    repo = make_repo()
    for index in range(5):
        await repo.add(_message(index, created_at=_NOW.replace(minute=index)))

    loaded = await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=2)

    assert [item.created_at.minute for item in loaded] == [3, 4]


async def test_same_instant_messages_are_ordered_by_id(
    make_repo: Callable[[], MessageRepository],
) -> None:
    """同一毫秒的两条按 `id` 定序。

    `created_at` 是 `DATETIME(3)`，而 `id` 是 ULID 派生（时间戳在高位）——
    只按时间排的话，这两条的先后由存储引擎决定，而**顺序不确定的对话历史**
    是最难查的一类 bug（本地复现不了，线上偶发）。
    """
    repo = make_repo()
    await repo.add(_message(2))
    await repo.add(_message(1))

    loaded = await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10)

    # `msg_...0001` 在 `msg_...0002` 之前——id 升序与写入顺序相反，
    # 所以这条断言只有"按 id 排"才成立，"按插入顺序"会失败。
    assert [item.id for item in loaded] == [_message(1).id, _message(2).id]


async def test_another_conversation_is_not_returned(
    make_repo: Callable[[], MessageRepository],
) -> None:
    repo = make_repo()
    await repo.add(_message(1))
    await repo.add(_message(2, conversation_id="cnv_0000000002AAAAAAAAAAAA"))

    loaded = await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10)

    assert [item.id for item in loaded] == [_message(1).id]


async def test_zero_limit_reads_nothing(make_repo: Callable[[], MessageRepository]) -> None:
    """`limit=0` 返回空——**不查库、也不当成"不限"**。

    当成"不限"的话，`max_turns` 被配成 0 的那天会把整个会话倒进 prompt。
    """
    repo = make_repo()
    await repo.add(_message(1))

    assert await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=0) == []


# ------------------------------------------------------------ 任务的替换语义
async def test_replace_for_task_is_idempotent(
    make_repo: Callable[[], MessageRepository],
) -> None:
    """同一个任务写两次助手消息 → 仍然只有一条。

    任务**可能被执行两次**（图跑完、收尾提交前进程死掉，孤儿回收会重跑）。
    纯追加会造出两条一模一样的助手消息，而下一轮的上下文里上一轮出现两次——
    看起来完全正常，只是多占了一倍的预算。
    """
    repo = make_repo()
    await repo.add(_message(1, task_id="tsk_1"))

    for _ in range(2):
        await repo.replace_for_task(
            "tsk_1",
            role=MessageRole.ASSISTANT,
            message=_message(9, role=MessageRole.ASSISTANT, task_id="tsk_1"),
        )

    loaded = await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10)
    assert len(loaded) == 2  # 一条 USER + 一条 ASSISTANT
    assert loaded[-1].id == _message(9).id


async def test_replace_for_task_with_none_removes_it(
    make_repo: Callable[[], MessageRepository],
) -> None:
    """`message=None` 是**删掉**：一轮失败留下的半条消息，
    下一轮会被当成"上一轮答过"。"""
    repo = make_repo()
    await repo.add(_message(1, role=MessageRole.ASSISTANT, task_id="tsk_1"))

    await repo.replace_for_task("tsk_1", role=MessageRole.ASSISTANT, message=None)

    assert await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10) == []


async def test_replace_for_task_leaves_other_roles_and_tasks_alone(
    make_repo: Callable[[], MessageRepository],
) -> None:
    """只动「同一个任务 + 同一种角色」那一条。"""
    repo = make_repo()
    await repo.add(_message(1, task_id="tsk_1"))  # 别的任务的 USER
    await repo.add(_message(2, task_id="tsk_2"))  # 同任务的 USER
    await repo.add(_message(3, role=MessageRole.ASSISTANT, task_id="tsk_2"))

    await repo.replace_for_task(
        "tsk_2",
        role=MessageRole.ASSISTANT,
        message=_message(4, role=MessageRole.ASSISTANT, task_id="tsk_2"),
    )

    loaded = await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10)
    assert [item.id for item in loaded] == [_message(1).id, _message(2).id, _message(4).id]


# -------------------------------------------------------- 软删除（仅 SQL 侧）
@pytest.mark.integration
async def test_soft_deleted_messages_are_not_returned() -> None:
    """`deleted_at` 非空的读不出来。

    **只跑 SQL 变体**：内存实现里根本没有"软删除"这个概念，硬把它塞进去
    等于给替身加一个只存在于测试里的语义。

    这条过滤当前没有生产者（`make cleanup` 还是占位实现），所以它**只有
    这一条断言**——没有它的话，哪天有人把 `deleted_at.is_(None)` 删掉，
    要等到 180 天清理真的落地才会发现，而那时症状是"会话列表里冒出已删消息"。
    """
    from app.tests.db import sql_sessions

    async with sql_sessions() as sessions:
        await _seed_one_deleted(sessions)
        repo = SqlMessageRepository(sessions)

        assert await repo.list_recent("cnv_0000000001AAAAAAAAAAAA", limit=10) == []


async def _seed_one_deleted(sessions: async_sessionmaker[AsyncSession]) -> None:
    repo = SqlMessageRepository(sessions)
    await repo.add(_message(1))
    async with sessions() as session:
        await session.execute(
            text("UPDATE agent_message SET deleted_at = :now WHERE id = :id"),
            {"now": _NOW, "id": _message(1).id},
        )
        await session.commit()
