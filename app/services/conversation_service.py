"""会话消息的**写入**（FR-CHAT-001 的「保存用户消息」+ 助手答复）。

## 为什么这个服务只有写、没有读

读那一半落在 `app/agent/runner.py` 的输入装配处（`load_conversation_context`），
因为**上下文是「图的输入」**：它必须与 `permission_scope` 在同一处装好、
一起交给 `TaskGraph.run`。放到这里就得让 agent 反向 import services
（违反依赖方向，约定 69），或者为它单立一个 Protocol——
而那个 Protocol 描述的事情只是"读两个仓储再调两个纯函数"。

两边共用的是 `app/domain/memory.py` 的纯函数（`group_turns` /
`build_conversation_context`），**「怎么归纳历史」只有一份实现**。

## 消息为什么值得存

不是为了当前这一轮——上下文其实可以从 `agent_task` 全部反推出来。
存它的理由有三条，缺一条它就该被砍掉：

1. **FR-CHAT-001 的处理流程明写「保存用户消息」**（需求规格 263），
   而 `agent_message` 是冲刺方案 §2 点名要建的 10 张表之一；
2. 它是**用户可见的消息流**，而 `agent_task` 是执行记录——任务表里没有
   `SYSTEM_NOTICE` 的位置，也没法表达"一条与任务无关的提示"；
3. 会话列表/历史回看接口（已登记）只能建在它上面。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from app.core.ids import new_message_id
from app.domain.conversation import Message, MessageRole
from app.domain.task import Task, TaskOutcome
from app.repositories.message_repo import MessageRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """记一轮对话。**所有写入失败都被吞掉**（见 `_record`）。"""

    def __init__(
        self,
        messages: MessageRepository,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._messages = messages
        self._clock = clock or (lambda: datetime.now(UTC))

    async def record_user_message(self, task: Task) -> None:
        """用户在 `POST /chat` 里问的那句话。

        调用方必须在 `tasks.add` **成功之后**调它（`TaskService.create_task`
        的位置注释说明了为什么）：放在之前的话，两个并发同幂等键的请求中
        输掉的那个在拿到 `DuplicateTaskError` 之前已经写过一条了。
        """
        await self._record(
            Message(
                id=new_message_id(),
                conversation_id=task.conversation_id,
                task_id=task.id,
                role=MessageRole.USER,
                content=task.query_text,
                created_at=self._clock(),
            ),
            task=task,
        )

    async def record_assistant_message(self, task: Task, outcome: TaskOutcome) -> None:
        """助手给出的答案。

        **失败 / 取消时删掉而不是留空**（`message=None` 走替换语义）：
        一轮失败留下的半条消息，下一轮会被当成"上一轮答过"。
        真正的失败信息在任务详情里（`error_code`），那里才是它的位置。

        **不用 `add` 用替换**：同一个任务可能被执行两次（图跑完、收尾提交前
        进程死掉 → 孤儿回收重跑），续上一条会造出两条一模一样的助手消息，
        症状是下一轮的上下文里上一轮出现两次——而它看起来完全正常。
        """
        message = None
        answer = (outcome.answer or "").strip()
        if answer:
            message = Message(
                id=new_message_id(),
                conversation_id=task.conversation_id,
                task_id=task.id,
                role=MessageRole.ASSISTANT,
                content=answer,
                created_at=self._clock(),
            )
        await self._guarded(
            task,
            role=MessageRole.ASSISTANT.value,
            action=lambda: self._messages.replace_for_task(
                task.id, role=MessageRole.ASSISTANT, message=message
            ),
        )

    async def _record(self, message: Message, *, task: Task) -> None:
        await self._guarded(
            task, role=message.role.value, action=lambda: self._messages.add(message)
        )

    async def _guarded(
        self, task: Task, *, role: str, action: Callable[[], Awaitable[None]]
    ) -> None:
        """执行写入，**失败只记日志**。

        沿用约定 50（产出落库失败不该把一次成功的分析变成任务失败）。
        但这里多一步：用**独立的事件名** `memory.record_failed` 记一条 error。

        理由是它的症状极其隐蔽：消息写不进去 → 下一轮的代词解析没有上下文 →
        「那 Q2 呢？」被答成 **Q2 的全公司数字**（丢掉了上一轮的"华东"）。
        任务全绿、答案看着正常、数字是错的，而且第二轮**无从知道自己少了什么**。
        所以"吞掉"必须留下一个能 grep 到的痕迹，而不是只靠一句 `logger.exception`。
        """
        try:
            await action()
        except Exception:
            _log_failure(task, role=role)


def _log_failure(task: Task, *, role: str) -> None:
    logger.error(
        "memory.record_failed",
        extra={
            "task_id": task.id,
            "conversation_id": task.conversation_id,
            "role": role,
        },
    )


__all__ = ["ConversationService"]
