"""任务创建与查询。

Phase 1 只负责「把任务登记下来」，**不启动 LangGraph**（开发流程 6.2 施工项 5）。
入队与执行在 Phase 1.5 接入：按详细设计 17.1 的事务边界，
必须先写库成功、再投递队列，否则 Worker 可能在记录可见前就领到任务；
写库成功而入队失败的补偿由 Worker 侧的队列扫描负责
（【后续扩展】登记于 CLAUDE.md，Phase 1.5 实现）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.core.ids import new_conversation_id, new_task_id
from app.domain.conversation import Conversation
from app.domain.task import Task
from app.domain.user import User, UserRole
from app.repositories.conversation_repo import ConversationRepository
from app.repositories.task_repo import DuplicateTaskError, TaskRepository

#: 会话标题取问题开头，够列表页展示即可
_TITLE_MAX_LENGTH = 60


@dataclass(frozen=True, slots=True)
class CreateTaskResult:
    task: Task
    #: False 表示命中幂等、返回的是既有任务，调用方据此决定要不要记账
    created: bool


class TaskService:
    def __init__(
        self,
        *,
        tasks: TaskRepository,
        conversations: ConversationRepository,
        settings: Settings,
    ) -> None:
        self._tasks = tasks
        self._conversations = conversations
        self._settings = settings

    async def create_task(
        self,
        *,
        user: User,
        message: str,
        conversation_id: str | None,
        idempotency_key: str | None,
        trace_id: str,
    ) -> CreateTaskResult:
        """登记一个新任务，或返回幂等命中的既有任务。

        **幂等检查排在配额检查之前**：重放一个已创建的任务并没有占用新的执行能力，
        不应该再消耗一次配额，更不应该因为用户当前已满额就把重放请求判成 429。

        `trace_id` 由调用方传入**本次请求的 trace_id**，而不是在这里新生成一个：
        详细设计 19.4.1 要求「从 /api/agent/chat 到最终 final_answer 的完整链路
        属于同一个 trace」。若任务另起一个 trace，API 侧的 HTTP span 与 Worker
        侧的节点 span 会落在两条链上，Phase 11 的跨进程验收必然失败。
        """
        if idempotency_key is not None:
            existing = await self._tasks.find_by_idempotency_key(user.id, idempotency_key)
            if existing is not None:
                return self._replay(existing, message=message, conversation_id=conversation_id)

        await self._ensure_quota(user)

        now = datetime.now(UTC)
        conversation = await self._resolve_conversation(
            user=user, conversation_id=conversation_id, message=message, now=now
        )
        task = Task(
            id=new_task_id(),
            user_id=user.id,
            conversation_id=conversation.id,
            trace_id=trace_id,
            query_text=message,
            idempotency_key=idempotency_key,
            queued_at=now,
            created_at=now,
            updated_at=now,
        )
        try:
            await self._tasks.add(task)
        except DuplicateTaskError:
            # 并发下另一个请求抢先写入了同一个幂等键。
            # 换成 MySQL 后唯一索引 uk_task_user_idempotency 会以同样的方式在这里报错，
            # 所以这段处理逻辑不需要随 Phase 2 改写。
            if idempotency_key is None:
                raise
            existing = await self._tasks.find_by_idempotency_key(user.id, idempotency_key)
            if existing is None:
                raise
            return self._replay(existing, message=message, conversation_id=conversation_id)

        await self._conversations.touch(conversation.id, now)
        return CreateTaskResult(task=task, created=True)

    async def get_task(self, *, user: User, task_id: str) -> Task:
        """按 task_id 取任务，并做归属校验。

        不区分「不存在」与「不属于你」会更好，但 FR-CHAT-002 明确要求
        前者 404、后者 403，以及 ADMIN 可跨用户查看（详细设计 7.2 的鉴权列），
        因此按需求实现。
        """
        task = await self._tasks.get(task_id)
        if task is None:
            raise AgentError(ErrorCode.TASK_NOT_FOUND, "任务不存在")
        if task.user_id != user.id and user.role is not UserRole.ADMIN:
            raise AgentError(ErrorCode.ACCESS_DENIED, "无权访问该任务")
        return task

    async def _ensure_quota(self, user: User) -> None:
        limit = self._settings.max_running_tasks_per_user
        if await self._tasks.count_active(user.id) >= limit:
            raise AgentError(
                ErrorCode.RATE_LIMITED,
                f"同时运行的任务已达上限（{limit} 个），请等待已有任务结束",
            )

    async def _resolve_conversation(
        self, *, user: User, conversation_id: str | None, message: str, now: datetime
    ) -> Conversation:
        """FR-CHAT-001：创建或复用会话。"""
        if conversation_id is None:
            conversation = Conversation(
                id=new_conversation_id(),
                user_id=user.id,
                title=message[:_TITLE_MAX_LENGTH],
                last_message_at=now,
                created_at=now,
                updated_at=now,
            )
            await self._conversations.add(conversation)
            return conversation

        existing = await self._conversations.get(conversation_id)
        if existing is None:
            raise AgentError(ErrorCode.INVALID_ARGUMENT, "指定的会话不存在")
        if existing.user_id != user.id:
            # FR-CHAT-001 异常情况：会话不属于当前用户返回 403
            raise AgentError(ErrorCode.ACCESS_DENIED, "无权使用该会话")
        return existing

    def _replay(
        self, existing: Task, *, message: str, conversation_id: str | None
    ) -> CreateTaskResult:
        """同一个幂等键只允许对应同一个请求。

        判据直接用任务行里已有的字段，不额外存「请求指纹」列——
        agent_task 没有这一列（16.5），而 `query_text` 与 `conversation_id`
        本来就唯一确定了这次请求。客户端没传 conversation_id 时不做比对：
        那次请求的会话是服务端新建的，重放时本来就不可能知道它的 ID。
        """
        same_conversation = conversation_id is None or conversation_id == existing.conversation_id
        if existing.query_text != message or not same_conversation:
            raise AgentError(ErrorCode.TASK_CONFLICT, "该 Idempotency-Key 已用于另一个请求")
        return CreateTaskResult(task=existing, created=False)
