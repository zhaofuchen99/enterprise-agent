"""任务仓储（详细设计 16.5 的 `agent_task`）。

两个实现，同一份协议：

| 实现 | 用途 |
|---|---|
| `InMemoryTaskRepository` | 单元测试与纯内存场景，无外部依赖 |
| `SqlTaskRepository` | 生产实现，MySQL 权威存储（Phase 2 起） |

`RedisTaskRepository` 是 Phase 1.5 的过渡实现，已于 Phase 2 删除——
连同 `RedisKey` 里它的 5 个索引键。**保留其中的三个**：
`task_cancel` / `task_requeue_count` / `task_heartbeat`，它们不是任务存储，
而是跨进程信号，Phase 7 之前仍有用途。

**为什么 `update` 收的是 `TaskPatch` 而不是整份 `Task`**：Worker 每 10 秒续一次心跳，
若用「读出来、改字段、整份写回」，它与 API 侧的取消请求会互相覆盖——
Worker 手里那份 `Task` 是几秒前读的，写回时会把 `CANCEL_REQUESTED` 抹掉，
表现为「点了取消但任务照跑」。字段级的定向更新没有这个窗口：
心跳只动 `heartbeat_at`，取消只动 `status`，两者物理上不重叠。

**`only_if_status` 在 MySQL 实现里靠行锁而非乐观重试**：`SELECT ... FOR UPDATE`
锁住该行后再判状态，使「读-判-写」整体原子。不这么做的话，两个 Worker 同时领到
同一个任务（重投、补偿扫描都可能造出这种局面）会双双开跑，同一个问题被算两遍、
两次 SQL 两次模型调用，是实打实的钱。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.task import ACTIVE_STATUSES, Task, TaskStatus
from app.infrastructure.db import session_scope
from app.infrastructure.models.task import AgentTask
from app.repositories._mapping import task_from_row, task_to_row_values, to_db_time

#: `TaskPatch` 里需要时区转换的字段。其余字段（状态、标识、文本）原样写。
_PATCH_TIME_FIELDS: frozenset[str] = frozenset({"heartbeat_at", "started_at", "finished_at"})


class DuplicateTaskError(Exception):
    """唯一约束冲突：task_id 重复，或 (user_id, idempotency_key) 已存在。"""


class TaskPatch(BaseModel):
    """一次状态流转要改的字段。**未列出的字段一律不动**。

    用 Pydantic 而不是 `**kwargs: Any`：字段名写错时 `**kwargs` 会静默丢弃
    （你得到的是一个「更新成功了但什么都没改」的结果），而模型会直接拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    status: TaskStatus | None = None
    worker_id: str | None = None
    heartbeat_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    final_answer_md: str | None = None
    #: 计划摘要与结构化结果（16.5）。**JSON 列不需要时区转换**，
    #: 所以它们不在 `_PATCH_TIME_FIELDS` 里——那一份只管 datetime。
    plan_json: dict[str, Any] | None = None
    result_json: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    intent: str | None = None


class TaskRepository(Protocol):
    async def add(self, task: Task) -> None: ...

    async def get(self, task_id: str) -> Task | None: ...

    async def update(
        self,
        task_id: str,
        patch: TaskPatch,
        *,
        at: datetime,
        only_if_status: TaskStatus | None = None,
    ) -> Task | None:
        """定向更新并返回更新后的任务；任务不存在或前置状态不符时返回 None。

        `only_if_status` 是**领取互斥**的实现手段：QUEUED -> RUNNING 必须
        「只有当前确实是 QUEUED 才写得进去」。
        """
        ...

    async def touch_heartbeat(self, task_id: str, at: datetime) -> None: ...

    async def find_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Task | None: ...

    async def count_active(self, user_id: str) -> int:
        """统计占用并发配额的任务数（终态不计，见 `ACTIVE_STATUSES`）。"""
        ...

    async def list_stale_queued(self, before: datetime, *, limit: int) -> list[Task]:
        """`queued_at` 早于 `before` 且仍处于 QUEUED 的任务（详细设计 17.1 补偿扫描）。"""
        ...

    async def list_stale_running(self, heartbeat_before: datetime, *, limit: int) -> list[Task]:
        """心跳早于 `heartbeat_before` 且仍处于 RUNNING 的任务（孤儿回收）。"""
        ...


# ------------------------------------------------------------------ 内存实现
class InMemoryTaskRepository:
    """内存实现。**三条唯一约束都要强制**，与 `agent_task` 的索引一一对应。

    少一条就会出现「内存实现跑得通、SQL 实现报 1062」这类只在换实现时
    才暴露的差异，而契约测试的意义恰恰是让这种差异在测试里就暴露出来。
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Task] = {}
        self._idempotency: dict[tuple[str, str], str] = {}
        #: 对应 `uk_task_trace`。trace_id 撞车意味着调用方生成了重复的追踪 ID。
        self._by_trace: dict[str, str] = {}

    async def add(self, task: Task) -> None:
        if task.id in self._by_id:
            raise DuplicateTaskError(f"task_id 重复：{task.id}")
        if task.trace_id in self._by_trace:
            raise DuplicateTaskError(f"trace_id 重复：{task.trace_id}")
        if task.idempotency_key is not None:
            index_key = (task.user_id, task.idempotency_key)
            if index_key in self._idempotency:
                raise DuplicateTaskError(f"幂等键重复：{task.idempotency_key}")
            self._idempotency[index_key] = task.id
        self._by_trace[task.trace_id] = task.id
        self._by_id[task.id] = task

    async def get(self, task_id: str) -> Task | None:
        return self._by_id.get(task_id)

    async def update(
        self,
        task_id: str,
        patch: TaskPatch,
        *,
        at: datetime,
        only_if_status: TaskStatus | None = None,
    ) -> Task | None:
        task = self._by_id.get(task_id)
        if task is None:
            return None
        # 内存实现里所有方法内部都没有 await 点，因此这里读-判-写之间
        # 不会被别的协程插队，等价于 SQL 实现的行锁。
        if only_if_status is not None and task.status is not only_if_status:
            return None
        updated = task.model_copy(update={**patch.model_dump(exclude_unset=True), "updated_at": at})
        self._by_id[task_id] = updated
        return updated

    async def touch_heartbeat(self, task_id: str, at: datetime) -> None:
        await self.update(task_id, TaskPatch(heartbeat_at=at), at=at)

    async def find_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Task | None:
        task_id = self._idempotency.get((user_id, idempotency_key))
        return self._by_id.get(task_id) if task_id else None

    async def count_active(self, user_id: str) -> int:
        return sum(
            1
            for task in self._by_id.values()
            if task.user_id == user_id and task.status in ACTIVE_STATUSES
        )

    async def list_stale_queued(self, before: datetime, *, limit: int) -> list[Task]:
        stale = [
            task
            for task in self._by_id.values()
            if task.status is TaskStatus.QUEUED and task.queued_at < before
        ]
        return sorted(stale, key=lambda t: t.queued_at)[:limit]

    async def list_stale_running(self, heartbeat_before: datetime, *, limit: int) -> list[Task]:
        stale = [
            task
            for task in self._by_id.values()
            if task.status is TaskStatus.RUNNING
            and (task.heartbeat_at or task.started_at or task.queued_at) < heartbeat_before
        ]
        return sorted(stale, key=lambda t: t.heartbeat_at or t.queued_at)[:limit]


# ------------------------------------------------------------------- SQL 实现
class SqlTaskRepository:
    """MySQL 实现。`agent_task` 是任务状态的**唯一权威**（原则 9）。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, task: Task) -> None:
        async with session_scope(self._sessions) as session:
            session.add(AgentTask(**task_to_row_values(task)))
            try:
                # flush 而不是等 commit：唯一约束冲突要在**这里**变成
                # `DuplicateTaskError` 交给调用方（幂等命中要返回既有任务，
                # 不是 500）。放到 commit 时才炸，异常就跑到 session_scope 外面了。
                await session.flush()
            except IntegrityError as exc:
                raise _duplicate_error(exc, task) from exc

    async def get(self, task_id: str) -> Task | None:
        async with session_scope(self._sessions) as session:
            row = await session.get(AgentTask, task_id)
            return task_from_row(row) if row is not None else None

    async def update(
        self,
        task_id: str,
        patch: TaskPatch,
        *,
        at: datetime,
        only_if_status: TaskStatus | None = None,
    ) -> Task | None:
        # `exclude_none`：TaskPatch 的语义是「设置给定且非空的字段」，
        # 显式传 None 等同于不改。与 Phase 1.5 的 Redis 实现保持一致。
        changed = patch.model_dump(exclude_none=True)
        async with session_scope(self._sessions) as session:
            # FOR UPDATE：锁住该行，使「读-判-写」整体原子。
            # 不用 rowcount 判断是否命中——MySQL 在「新值与旧值相同」时
            # 报 rowcount=0，那会把一次成功的幂等更新误判成状态不符。
            row = await session.get(AgentTask, task_id, with_for_update=True)
            if row is None:
                return None
            if only_if_status is not None and row.status != only_if_status.value:
                return None
            for name, value in changed.items():
                if name == "status":
                    value = TaskStatus(value).value
                elif name in _PATCH_TIME_FIELDS:
                    value = to_db_time(value)
                setattr(row, name, value)
            row.updated_at = to_db_time(at)
            await session.flush()
            return task_from_row(row)

    async def touch_heartbeat(self, task_id: str, at: datetime) -> None:
        """续心跳。**单条 UPDATE，不先读**。

        心跳每 10 秒一次，走「先 SELECT FOR UPDATE 再写」会让它变成两条语句
        外加一次行锁等待。任务被回收后再续一次心跳是无害的，因此不需要判存在。
        """
        moment = to_db_time(at)
        async with session_scope(self._sessions) as session:
            await session.execute(
                update(AgentTask)
                .where(AgentTask.id == task_id)
                .values(heartbeat_at=moment, updated_at=moment)
            )

    async def find_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Task | None:
        # 防御空键：`column == None` 会被编译成 `IS NULL`，从而匹配上
        # 所有**没有**幂等键的任务，把无关任务当成幂等命中返回。
        if not idempotency_key:
            return None
        stmt = (
            select(AgentTask)
            .where(
                AgentTask.user_id == user_id,
                AgentTask.idempotency_key == idempotency_key,
            )
            .limit(1)
        )
        async with session_scope(self._sessions) as session:
            row = (await session.execute(stmt)).scalar_one_or_none()
            return task_from_row(row) if row is not None else None

    async def count_active(self, user_id: str) -> int:
        stmt = (
            select(func.count())
            .select_from(AgentTask)
            .where(
                AgentTask.user_id == user_id,
                AgentTask.status.in_([status.value for status in ACTIVE_STATUSES]),
            )
        )
        async with session_scope(self._sessions) as session:
            return int((await session.execute(stmt)).scalar_one())

    async def list_stale_queued(self, before: datetime, *, limit: int) -> list[Task]:
        stmt = (
            select(AgentTask)
            .where(
                AgentTask.status == TaskStatus.QUEUED.value,
                AgentTask.queued_at < to_db_time(before),
            )
            .order_by(AgentTask.queued_at)
            .limit(limit)
        )
        return await self._fetch(stmt)

    async def list_stale_running(self, heartbeat_before: datetime, *, limit: int) -> list[Task]:
        # COALESCE 而不是只比 heartbeat_at：领取时就写了心跳，但历史数据或
        # 异常路径下可能为空，那时退到 started_at / queued_at，
        # 否则任务永远不会被判为孤儿，一直停在 RUNNING 占着并发配额。
        last_seen = func.coalesce(AgentTask.heartbeat_at, AgentTask.started_at, AgentTask.queued_at)
        stmt = (
            select(AgentTask)
            .where(
                AgentTask.status == TaskStatus.RUNNING.value,
                last_seen < to_db_time(heartbeat_before),
            )
            .order_by(last_seen)
            .limit(limit)
        )
        return await self._fetch(stmt)

    async def _fetch(self, stmt: Any) -> list[Task]:
        async with session_scope(self._sessions) as session:
            rows: Sequence[AgentTask] = (await session.execute(stmt)).scalars().all()
            return [task_from_row(row) for row in rows]

    async def delete(self, task_id: str) -> None:
        """物理删除。供 `make cleanup` 按保留期清理（详细设计 16.12）。"""
        async with session_scope(self._sessions) as session:
            await session.execute(delete(AgentTask).where(AgentTask.id == task_id))


def _duplicate_error(exc: IntegrityError, task: Task) -> DuplicateTaskError:
    """把 MySQL 的 1062 映射成可区分的领域异常。

    三种冲突的**处置方式完全不同**，所以必须分开报，不能笼统说「重复」：

    | 约束 | 含义 | 调用方该做什么 |
    |---|---|---|
    | `uk_task_user_idempotency` | 幂等命中 | 返回既有任务，不是错误 |
    | `uk_task_trace` | trace_id 撞车 | 调用方生成错了 ID，是真错误 |
    | `pk_agent_task` | task_id 重复 | 同上 |

    把 trace 撞车报成「task_id 重复」会让人往 task_id 生成器上查，
    而问题其实在 trace_id——错误信息指错方向比没有信息更费时间。
    靠驱动错误文本里的约束名判断；拿不到名字时按 task_id 处理（最常见）。
    """
    detail = str(getattr(exc, "orig", exc))
    if "uk_task_user_idempotency" in detail:
        return DuplicateTaskError(f"幂等键重复：{task.idempotency_key}")
    if "uk_task_trace" in detail:
        return DuplicateTaskError(f"trace_id 重复：{task.trace_id}")
    return DuplicateTaskError(f"task_id 重复：{task.id}")
