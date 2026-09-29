"""多轮上下文的**真实 SQL 读路径**（前置：`make up`）。

单元测试跑的是内存替身，而这一组补的是替身挡在外面的两件事：

1. `list_recent_by_conversation` 是**本次新写的 SQL**（`agent_task` 上没有
   `conversation_id` 索引，排序键是 `queued_at` + `id`）——它此前从未被执行过；
2. `agent_message` 从 Phase 2 建好起**没有任何写入方**，这份映射
   （`message_to_row_values` / `message_from_row`）也没有被执行过。

两条都是"列名写错也不会有人发现"的那一类。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.agent.runner import load_conversation_context
from app.core.config import Settings, get_settings
from app.core.ids import new_trace_id
from app.domain.conversation import Conversation, MessageRole
from app.domain.memory import TurnResolution, resolution_to_payload
from app.domain.task import Task, TaskOutcome, TaskStatus
from app.repositories.conversation_repo import SqlConversationRepository
from app.repositories.message_repo import SqlMessageRepository
from app.repositories.task_repo import SqlTaskRepository, TaskPatch
from app.services.conversation_service import ConversationService
from app.tests.db import sql_sessions

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _settings() -> Settings:
    return get_settings()


async def test_a_recorded_pair_comes_back_as_a_turn() -> None:
    """写一轮（用户消息 + 助手消息 + 任务口径）→ 读回来是一轮对话。

    这条贯通的是**四段**：仓储写入 → `agent_message` 映射 →
    `agent_task.result_json` → `load_conversation_context` 的归纳。
    中间任何一段漏了（比如口径没进 payload），这里都会看到"有历史但没口径"。
    """
    async with sql_sessions() as sessions:
        messages = SqlMessageRepository(sessions)
        tasks = SqlTaskRepository(sessions)
        conversations = SqlConversationRepository(sessions)
        service = ConversationService(messages, clock=lambda: _NOW)

        user_id = "usr_0000000001AAAAAAAAAAAA"
        conversation = Conversation(
            id="cnv_0000000001AAAAAAAAAAAA",
            user_id=user_id,
            title="华东销售额",
            last_message_at=_NOW,
            created_at=_NOW,
            updated_at=_NOW,
        )
        await conversations.add(conversation)

        first = Task(
            id="tsk_0000000001AAAAAAAAAAAA",
            user_id=user_id,
            conversation_id=conversation.id,
            trace_id=new_trace_id(),
            query_text="2025年华东Q3销售额多少",
            status=TaskStatus.QUEUED,
            queued_at=_NOW,
            created_at=_NOW,
            updated_at=_NOW,
        )
        await tasks.add(first)
        await service.record_user_message(first)
        # 口径由任务持有（`final._payload` 的产物）：这里直接写出那个形状，
        # 因为要验的是**读路径**能不能把它读回来
        await tasks.update(
            first.id,
            TaskPatch(
                result_json=resolution_to_payload(
                    TurnResolution(
                        entities={"metric": "净销售额", "region": "华东", "period": "2025-Q3"}
                    )
                )
            ),
            at=_NOW,
        )
        await service.record_assistant_message(
            first, TaskOutcome(answer="2025年Q3华东净销售额为 1.1 亿元")
        )

        # 第二轮：同一会话的新任务，加载上下文
        second = first.model_copy(
            update={"id": "tsk_0000000002AAAAAAAAAAAA", "trace_id": new_trace_id()}
        )
        context = await load_conversation_context(
            second, messages=messages, tasks=tasks, settings=_settings()
        )

    assert context is not None
    assert context.confirmed_entities == {
        "metric": "净销售额",
        "region": "华东",
        "period": "2025-Q3",
    }
    assert [item.content for item in context.recent_messages] == [
        "2025年华东Q3销售额多少",
        "2025年Q3华东净销售额为 1.1 亿元",
    ]
    assert [item.role for item in context.recent_messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


async def test_a_fresh_conversation_has_no_context() -> None:
    """空会话返回 `None` ——**不是**一个空对象。

    `None` 的语义是"这一轮不必带上下文"，而空对象是"历史确实是空的"。
    两者渲染出来一样，但 `load_conversation_context` 的调用方靠 `None`
    判"首轮"，把它统一成空对象会让那个判断永远为假。
    """
    async with sql_sessions() as sessions:
        task = Task(
            id="tsk_0000000001AAAAAAAAAAAA",
            user_id="usr_0000000001AAAAAAAAAAAA",
            conversation_id="cnv_0000000001AAAAAAAAAAAA",
            trace_id=new_trace_id(),
            query_text="首个问题",
            status=TaskStatus.QUEUED,
            queued_at=_NOW,
            created_at=_NOW,
            updated_at=_NOW,
        )

        context = await load_conversation_context(
            task,
            messages=SqlMessageRepository(sessions),
            tasks=SqlTaskRepository(sessions),
            settings=_settings(),
        )

    assert context is None
