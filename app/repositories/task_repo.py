"""任务仓储。Phase 2 换成 MySQL 实现（详细设计 16.5 的 agent_task）。

三个实现，同一份协议：

| 实现 | 用途 |
|---|---|
| `InMemoryTaskRepository` | 单元测试与纯内存场景，无外部依赖 |
| `RedisTaskRepository` | **Phase 1.5 的生产实现（临时）**，让 API 与 Worker 跨进程共享任务状态 |

**为什么 `update` 收的是 `TaskPatch` 而不是整份 `Task`**：Worker 每 10 秒续一次心跳，
若用「读出来、改字段、整份写回」，它与 API 侧的取消请求会互相覆盖——
Worker 手里那份 `Task` 是几秒前读的，写回时会把 `CANCEL_REQUESTED` 抹掉，
表现为「点了取消但任务照跑」。字段级的定向更新没有这个窗口：
心跳只动 `heartbeat_at`，取消只动 `status`，两者物理上不重叠。

**Phase 1.5 的存储形态**：`task:{id}:record` 是一个 Redis HASH，
外加三个 ZSET 索引（active / queued / running）。Phase 2 接 MySQL 时，
这份实现整份删除，`TaskPatch` 会平移成 `UPDATE agent_task SET <patch 里的字段>`，
协议与调用方都不用动。
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, cast

import redis.asyncio as aioredis
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.domain.task import ACTIVE_STATUSES, TERMINAL_STATUSES, Task, TaskStatus
from app.infrastructure.redis import RedisKey, register_scripts


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
        「只有当前确实是 QUEUED 才写得进去」。少了这一层，两个 Worker
        同时领到同一个任务（重投、补偿扫描都可能造出这种局面）时会双双开跑，
        同一个问题被算两遍、两次 SQL 两次模型调用，是实打实的钱。
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
    def __init__(self) -> None:
        self._by_id: dict[str, Task] = {}
        self._idempotency: dict[tuple[str, str], str] = {}

    async def add(self, task: Task) -> None:
        if task.id in self._by_id:
            raise DuplicateTaskError(f"task_id 重复：{task.id}")
        if task.idempotency_key is not None:
            index_key = (task.user_id, task.idempotency_key)
            if index_key in self._idempotency:
                raise DuplicateTaskError(f"幂等键重复：{task.idempotency_key}")
            self._idempotency[index_key] = task.id
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
        # 不会被别的协程插队，等价于 Redis 实现的 CAS 脚本。
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


# ------------------------------------------------------------------ Redis 实现
class RedisTaskRepository:
    """Phase 1.5 的临时实现。见模块 docstring 的说明。"""

    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._scripts = register_scripts(redis)
        self._heartbeat_ttl = settings.worker.heartbeat_ttl_seconds

    async def add(self, task: Task) -> None:
        idem_key = (
            RedisKey.idempotency_index(task.user_id, task.idempotency_key)
            if task.idempotency_key
            else ""
        )
        fields: list[str] = []
        for name, value in _hash_fields(task).items():
            fields.extend((name, value))
        result = await self._scripts["add_task"](
            keys=[
                RedisKey.task_record(task.id),
                idem_key,
                RedisKey.active_tasks(task.user_id),
                RedisKey.queued_tasks(),
                RedisKey.running_tasks(),
            ],
            args=[
                task.id,
                task.queued_at.timestamp(),
                _flag(task.status is TaskStatus.QUEUED),
                _flag(task.status is TaskStatus.RUNNING),
                _flag(task.status in ACTIVE_STATUSES),
                *fields,
            ],
        )
        if int(result) == 0:
            raise DuplicateTaskError(f"task_id 重复：{task.id}")
        if int(result) == 1:
            raise DuplicateTaskError(f"幂等键重复：{task.idempotency_key}")

    async def get(self, task_id: str) -> Task | None:
        # cast：redis-py 把好几个命令的返回类型标成 `Awaitable[T] | T`
        # （同步/异步客户端共用一份签名），mypy 因此拒绝直接 await。
        raw = await cast(
            "Awaitable[dict[str, Any]]", self._redis.hgetall(RedisKey.task_record(task_id))
        )
        if not raw:
            return None
        return Task.model_validate(_decode_hash(raw))

    async def update(
        self,
        task_id: str,
        patch: TaskPatch,
        *,
        at: datetime,
        only_if_status: TaskStatus | None = None,
    ) -> Task | None:
        task = await self.get(task_id)
        if task is None:
            return None
        if only_if_status is not None and task.status is not only_if_status:
            return None

        # `exclude_none`：TaskPatch 的语义是「设置给定且非空的字段」，
        # 显式传 None 等同于不改。这样字段一旦写进去就只可能被下一次流转覆盖，
        # 不会出现把 "None" 字符串写进 HASH、再被 Pydantic 当成时间解析的坑。
        changed = patch.model_dump(exclude_none=True)
        fields = {name: _encode_scalar(value) for name, value in changed.items()}
        fields["updated_at"] = _encode_scalar(at)
        updated = task.model_copy(update={**changed, "updated_at": at})

        if only_if_status is not None:
            flat: list[str] = []
            for name, value in fields.items():
                flat.extend((name, value))
            result = await self._scripts["cas_update"](
                keys=[RedisKey.task_record(task_id)], args=[only_if_status.value, *flat]
            )
            if int(result) != 2:
                # 状态在这两步之间被改掉了（比如刚刚被取消），放弃本次流转
                return None
        else:
            await cast(
                "Awaitable[Any]",
                self._redis.hset(RedisKey.task_record(task_id), mapping=fields),
            )

        # 状态流转要同步索引，否则补偿扫描与孤儿回收会看到过期的任务集合。
        # 放在写入之后：查询侧会以记录里的状态再过滤一次，索引短暂滞后无害。
        await self._apply_index_moves(task, updated, at)
        if updated.status in TERMINAL_STATUSES:
            # 4.4：取消键在任务结束后删除。留着它会误导后续的取消请求判断。
            await self._redis.unlink(RedisKey.task_cancel(task_id))
        return updated

    async def touch_heartbeat(self, task_id: str, at: datetime) -> None:
        """续心跳。**刻意不检查任务是否存在**：调用方是 Worker 的任务循环，
        任务被回收后再续一次心跳是无害的，而每次心跳都先读一次记录会让
        心跳路径变成读-写两条命令，正是我们要避开的形态。
        """
        key = RedisKey.task_record(task_id)
        pipeline = self._redis.pipeline(transaction=True)
        pipeline.hset(
            key, mapping={"heartbeat_at": _encode_scalar(at), "updated_at": _encode_scalar(at)}
        )
        pipeline.zadd(RedisKey.running_tasks(), {task_id: at.timestamp()})
        # 详细设计 4.4 的心跳键：TTL 到点即自动消失，是「Worker 是否还活着」
        # 最廉价的一个信号，Phase 7 的节点内检查会读它。
        pipeline.set(RedisKey.task_heartbeat(task_id), at.isoformat(), ex=self._heartbeat_ttl)
        await pipeline.execute()

    async def find_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Task | None:
        task_id = await self._redis.get(RedisKey.idempotency_index(user_id, idempotency_key))
        if not task_id:
            return None
        return await self.get(_text(task_id))

    async def count_active(self, user_id: str) -> int:
        count = await cast("Awaitable[int]", self._redis.zcard(RedisKey.active_tasks(user_id)))
        return int(count)

    async def list_stale_queued(self, before: datetime, *, limit: int) -> list[Task]:
        ids = await self._redis.zrangebyscore(
            RedisKey.queued_tasks(), min=0, max=before.timestamp(), start=0, num=limit
        )
        return await self._load_matching(ids, status=TaskStatus.QUEUED)

    async def list_stale_running(self, heartbeat_before: datetime, *, limit: int) -> list[Task]:
        ids = await self._redis.zrangebyscore(
            RedisKey.running_tasks(), min=0, max=heartbeat_before.timestamp(), start=0, num=limit
        )
        return await self._load_matching(ids, status=TaskStatus.RUNNING)

    # ------------------------------------------------------------ 内部
    async def _load_matching(self, ids: list[Any], *, status: TaskStatus) -> list[Task]:
        tasks: list[Task] = []
        for raw_id in ids:
            task_id = _text(raw_id)
            task = await self.get(task_id)
            if task is None:
                # 索引里有一条记录不存在，说明它被别处删了。顺手把索引清掉，
                # 否则这条幽灵会永远占着每次扫描的名额。
                await self._redis.zrem(RedisKey.queued_tasks(), task_id)
                await self._redis.zrem(RedisKey.running_tasks(), task_id)
                continue
            # 索引会滞后于状态（比如刚被 Worker 领取），因此**以记录里的状态为准**。
            # 少了这一层过滤，补偿扫描会把正在跑的任务重新投递一遍。
            if task.status is status:
                tasks.append(task)
        return tasks

    async def _apply_index_moves(self, before: Task, after: Task, at: datetime) -> None:
        """状态变化引起的索引维护。

        与 `update` 不在同一个事务里：索引先于记录更新，最坏情况是索引里
        多出/少了一条而记录是准的，而两种查询（`list_stale_*`）都会
        **以记录里的状态为准**再过滤一遍，因此索引短暂不一致不会造成误判。
        反过来（记录先改、索引滞后）同样有那层过滤兜住。
        """
        if after.status is before.status:
            return

        redis = self._redis
        if after.status in TERMINAL_STATUSES:
            # 进终态：释放并发配额并清掉所有任务级键。
            # 19.3 要求「任务结束（任意终态）时必须递减」——漏掉这一步，
            # 用户的并发配额会被泄漏的计数永久占住，表现为「再也创建不了任务」。
            await redis.zrem(RedisKey.active_tasks(after.user_id), after.id)
            await redis.zrem(RedisKey.queued_tasks(), after.id)
            await redis.zrem(RedisKey.running_tasks(), after.id)
            await redis.delete(
                RedisKey.task_heartbeat(after.id), RedisKey.task_requeue_count(after.id)
            )
            return
        if after.status is TaskStatus.QUEUED:
            await redis.zadd(RedisKey.queued_tasks(), {after.id: at.timestamp()})
        elif after.status is TaskStatus.RUNNING:
            await redis.zrem(RedisKey.queued_tasks(), after.id)
            # **领取时就要进 running 索引**，不能等第一次心跳。
            # 只靠 touch_heartbeat 建索引的话，Worker 在领取之后、首次心跳之前
            # 挂掉的任务永远不在索引里，孤儿回收扫不到它，任务会一直停在
            # RUNNING——恰好是孤儿回收要解决的那个场景。
            await redis.zadd(
                RedisKey.running_tasks(), {after.id: (after.heartbeat_at or at).timestamp()}
            )


def _hash_fields(task: Task) -> dict[str, str]:
    """`Task` -> HASH 字段。**None 一律不写**。

    写空串代替 None 会让 `model_validate` 拿到 `""` 去解析 datetime 而报错；
    省略字段则让 Pydantic 用模型默认值（正是 None），语义一致且省空间。
    """
    return {
        name: _encode_scalar(value)
        for name, value in task.model_dump(mode="json").items()
        if value is not None
    }


def _encode_scalar(value: Any) -> str:
    """HASH 只能存字符串。bool 存成 `"true"/"false"`，Pydantic 能反向解析。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        # 不能直接用 str()：StrEnum 的 __str__ 恰好返回 value，
        # 但普通 Enum 会返回 "TaskStatus.QUEUED"，存进去就再也解析不回来了。
        return str(value.value)
    return str(value)


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _decode_hash(raw: dict[Any, Any]) -> dict[str, str]:
    return {_text(key): _text(value) for key, value in raw.items()}


def _text(value: Any) -> str:
    """`decode_responses` 可能是 False（比如某些测试替身），两种都要能处理。"""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)
