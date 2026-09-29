"""`ConversationService`：消息写什么、以及**写不进去时怎么办**。

最后两条（吞异常 + 留下可检索的信号）是这个文件存在的理由。反过来做
——把落库失败抛出去——一次**已经跑完并产出答案**的分析会因为一条消息没写上
而被判 `FAILED`（约定 50 说得很清楚：它不该把成功的分析变成失败）。
但吞掉必须留痕，否则症状是"下一轮的代词解析悄悄没有了历史"：
任务全绿、答案看着正常、数字是错的（丢了上一轮的区域）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from app.domain.conversation import Message, MessageRole
from app.domain.task import Task, TaskOutcome
from app.repositories.message_repo import InMemoryMessageRepository
from app.services.conversation_service import ConversationService

_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _task() -> Task:
    return Task(
        id="tsk_0000000001AAAAAAAAAAAA",
        user_id="usr_1",
        conversation_id="cnv_0000000001AAAAAAAAAAAA",
        trace_id="trc_0000000001AAAAAAAAAAAA",
        query_text="那Q2呢？",
        queued_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _service(
    repo: InMemoryMessageRepository | None = None,
) -> tuple[ConversationService, InMemoryMessageRepository]:
    messages = repo or InMemoryMessageRepository()
    return ConversationService(messages, clock=lambda: _NOW), messages


class _ExplodingRepository(InMemoryMessageRepository):
    """两种写入都失败。用来验证"吞掉但留痕"。"""

    async def add(self, message: Message) -> None:
        raise RuntimeError("MySQL 连接断了")

    async def replace_for_task(
        self, task_id: str, *, role: MessageRole, message: Message | None
    ) -> None:
        raise RuntimeError("MySQL 连接断了")


async def test_a_user_message_carries_the_task_it_belongs_to() -> None:
    """`task_id` 不是可选的装饰：下一轮要靠它把"问的是什么"与
    "解析成了什么口径"接起来（`agent_task.result_json` 按它读）。"""
    service, messages = _service()
    task = _task()

    await service.record_user_message(task)

    stored = await messages.list_recent(task.conversation_id, limit=10)
    assert len(stored) == 1
    assert stored[0].role is MessageRole.USER
    assert stored[0].content == "那Q2呢？"
    assert stored[0].task_id == task.id


async def test_an_assistant_message_replaces_the_previous_one() -> None:
    """同一个任务写两次 → 只有一条（孤儿回收重跑时不会重复）。"""
    service, messages = _service()
    task = _task()

    for answer in ("第一次的答案", "第二次的答案"):
        await service.record_assistant_message(task, TaskOutcome(answer=answer))

    stored = await messages.list_recent(task.conversation_id, limit=10)
    assert len(stored) == 1
    assert stored[0].content == "第二次的答案"


async def test_a_task_without_an_answer_leaves_no_assistant_message() -> None:
    """失败的轮次不留半条消息——留着的话，下一轮会把它当成"上一轮答过"。"""
    service, messages = _service()
    task = _task()
    await service.record_assistant_message(task, TaskOutcome(answer="先写一条"))

    await service.record_assistant_message(task, TaskOutcome(answer=None, failed=True))

    assert await messages.list_recent(task.conversation_id, limit=10) == []


async def test_a_write_failure_is_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    """写不进去**不抛**——一次成功的分析不该因为一条消息没写上而变成失败。"""
    service, _ = _service(_ExplodingRepository())

    with caplog.at_level(logging.ERROR):
        await service.record_user_message(_task())
        await service.record_assistant_message(_task(), TaskOutcome(answer="答案"))


async def test_a_write_failure_is_greppable(caplog: pytest.LogCaptureFixture) -> None:
    """吞掉的那一刻要留下一个**能 grep 到的事件名**。

    只 `logger.exception` 一句是不够的：它的症状（下一轮的代词解析没有上下文）
    出现在**另一条任务**上，而那条任务日志里一片正常。排查的人需要一个
    能直接捞出来的名字，`memory.record_failed` 就是它。
    """
    service, _ = _service(_ExplodingRepository())

    with caplog.at_level(logging.ERROR):
        await service.record_user_message(_task())

    assert any(record.msg == "memory.record_failed" for record in caplog.records)
    failure = next(r for r in caplog.records if r.msg == "memory.record_failed")
    # `extra` 里的字段在 `LogRecord` 上是动态属性，`getattr` 是唯一的读法。
    # **要断言它们存在**：没有 task_id 的那条日志只能说明"有人写消息失败了"，
    # 说明不了"是哪条任务的哪一轮"——而排查的人手上只有后者的线索。
    assert getattr(failure, "task_id", None) == _task().id
    assert getattr(failure, "conversation_id", None) == _task().conversation_id
