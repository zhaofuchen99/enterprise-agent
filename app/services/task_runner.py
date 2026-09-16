"""任务投递、领取、心跳、取消与回收（开发流程 6.3 施工项 6）。

这个模块是「跨进程执行模型」的落点。Phase 1 里任务只是被登记下来，
从本阶段起它真的会被 Worker 领取并执行。

**四件事的边界，分开看才不会混**：

| 动作 | 触发方 | 关键约束 |
|---|---|---|
| 投递 `enqueue` | API | 先写库再投递（17.1）；投递失败不算请求失败 |
| 领取 `claim` | Worker | QUEUED -> RUNNING 必须原子（CAS），否则同一任务被算两遍 |
| 心跳 `heartbeat` | Worker | 只动 `heartbeat_at` 一个字段，不与取消请求抢同一字段 |
| 回收 `reclaim_orphans` | 任意实例的定时任务 | 心跳过期的 RUNNING 任务转 FAILED，释放并发配额 |
| 补偿 `reconcile_queue` | 任意实例的定时任务 | 写库成功但没进队列的任务重投，超过次数转 FAILED |

**任务体在本阶段是空实现**（见 `_run_body`）。Phase 7 会用 LangGraph 替换它，
其余部分——投递、领取、心跳、回收——都不需要再改。这正是把执行模型放在
所有业务逻辑之前做的理由。
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import redis.asyncio as aioredis

from app.core.config import Settings
from app.core.errors import DEFAULT_RETRYABLE, ErrorCode
from app.domain.task import Task, TaskStatus
from app.infrastructure.logging import bind_context
from app.infrastructure.observability import capture_trace_context, span
from app.infrastructure.queue import JobQueue
from app.infrastructure.redis import RedisKey, register_scripts
from app.repositories.task_repo import TaskPatch, TaskRepository
from app.services.event_bus import EventBus, TaskEventType

logger = logging.getLogger(__name__)

#: 单次扫描最多处理多少个任务。设上限是为了让扫描本身有界——
#: 一次性处理积压的十万条任务会把 Worker 卡死，而扫描是定时跑的，
#: 这一轮做不完下一轮继续做。
_SCAN_BATCH = 100


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """补偿扫描结果。带数量而不是只返回 bool：这是定时任务，
    「这一轮重投了 37 个」和「这一轮什么都没做」是完全不同的信号，
    只返回成功与否等于把可观测性丢掉了。"""

    scanned: int
    requeued: int
    failed: int


@dataclass(frozen=True, slots=True)
class ReclaimReport:
    scanned: int
    reclaimed: int
    #: 没抢到锁时为 True——多实例下这是常态，不是异常
    skipped: bool = False


class TaskRunner:
    def __init__(
        self,
        *,
        tasks: TaskRepository,
        queue: JobQueue,
        events: EventBus,
        settings: Settings,
        redis: aioredis.Redis,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._tasks = tasks
        self._queue = queue
        self._events = events
        self._settings = settings
        self._redis = redis
        self._scripts = register_scripts(redis)
        self._tuning = settings.redis_tuning
        self._worker_tuning = settings.worker
        #: 时间可注入：回收与补偿的用例要验证「超过 N 秒之后会怎样」，
        #: 真 sleep 会让这类用例又慢又不稳定。
        self._clock = clock or (lambda: datetime.now(UTC))

    # ------------------------------------------------------------------ 投递
    async def enqueue(self, task: Task) -> bool:
        """投递任务。**返回 False 不代表请求失败**——任务已经在库里，
        只是没进队列，补偿扫描会把它捡回来（详细设计 17.1）。"""
        carrier = capture_trace_context(task.trace_id)
        return await self._queue.enqueue(
            task_id=task.id, trace_id=task.trace_id, trace_context=carrier
        )

    # ------------------------------------------------------------------ 领取
    async def claim(self, task_id: str, *, worker_id: str) -> Task | None:
        """领取任务。返回 None 表示「不该执行」，原因有三，均已记日志。

        17.5 第 3 条要求「QUEUED 任务尚未被领取时，取消请求由 Worker 领取时
        首先读取并直接转 CANCELLED，不执行任何节点」，所以取消检查排在领取之前。
        """
        task = await self._tasks.get(task_id)
        if task is None:
            logger.warning("任务不存在，跳过", extra={"task_id": task_id})
            return None

        if task.status is TaskStatus.CANCEL_REQUESTED or await self.is_cancel_requested(task_id):
            await self._finish_cancelled(task, worker_id=worker_id)
            return None

        now = self._clock()
        claimed = await self._tasks.update(
            task_id,
            TaskPatch(
                status=TaskStatus.RUNNING,
                worker_id=worker_id,
                started_at=now,
                heartbeat_at=now,
            ),
            at=now,
            only_if_status=TaskStatus.QUEUED,
        )
        if claimed is None:
            # 状态已不是 QUEUED：要么被别的实例抢先领走，要么刚被取消。
            # 这里不能报错——多实例下这是正常竞争结果。
            logger.info("任务已被其他实例领取或状态已变更", extra={"task_id": task_id})
        return claimed

    # ------------------------------------------------------------------ 心跳
    async def heartbeat(self, task_id: str) -> None:
        """续心跳。**两处都写，各有分工**：

        - `agent_task.heartbeat_at` 是**权威**（原则 9）。孤儿回收按它判死，
          因为它是唯一能被 MySQL 重建的判断依据。
        - Redis 的 `task:{id}:heartbeat` 是跨进程快通道，带 TTL，
          「键还在」即「Worker 还活着」，比查库便宜得多。

        Phase 2 之前这两处都写在任务仓储里；仓储换成 MySQL 之后 Redis 那一半
        上移到本方法——**心跳的所有者是 Worker 而不是存储层**，
        让 SQL 仓储去写 Redis 只会把两个基础设施耦在一起。
        """
        now = self._clock()
        await self._tasks.touch_heartbeat(task_id, now)
        await self._redis.set(
            RedisKey.task_heartbeat(task_id),
            now.isoformat(),
            ex=self._worker_tuning.heartbeat_ttl_seconds,
        )

    @asynccontextmanager
    async def heartbeat_while(self, task_id: str) -> AsyncIterator[None]:
        """任务执行期间持续续心跳。

        心跳跑在独立协程里而不是任务体内部：任务体将来是 LangGraph，
        节点之间没有插入心跳的位置；而且一旦某个节点阻塞，
        把心跳写在任务体里就等于**心跳跟着一起停**——那会让一个卡住的
        健康任务被误判成孤儿回收掉。
        """
        stop = asyncio.Event()
        beats = asyncio.create_task(self._beat_until(task_id, stop))
        try:
            yield
        finally:
            stop.set()
            await beats

    async def _beat_until(self, task_id: str, stop: asyncio.Event) -> None:
        interval = self._worker_tuning.heartbeat_interval_seconds
        while not stop.is_set():
            try:
                # wait_for(stop) 而不是 sleep：停机时不必等满一个心跳周期
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except TimeoutError:
                pass
            try:
                await self.heartbeat(task_id)
            except Exception as exc:
                # **刻意捕获所有异常**：心跳写不进去不该让任务失败。
                # 真正判死的是回收扫描，而它读的是库里的 heartbeat_at——
                # 心跳漏几拍的最坏后果是被误判成孤儿，那也比
                # 「数据库抖了一下，一个正在正常执行的任务直接失败」轻。
                #
                # 捕获范围从 RedisError 放宽到 Exception 是 Phase 2 的连带影响：
                # 心跳写入从纯 Redis 变成「MySQL 权威 + Redis 快通道」，
                # 数据库异常（SQLAlchemyError）原先不在这条路径上。
                logger.warning("心跳写入失败：%s", type(exc).__name__, extra={"task_id": task_id})

    # ------------------------------------------------------------------ 取消
    async def is_cancel_requested(self, task_id: str) -> bool:
        return bool(await self._redis.exists(RedisKey.task_cancel(task_id)))

    async def signal_cancel(self, task_id: str) -> None:
        """写取消信号。

        TTL 取任务超时（详细设计 17.5）：取消信号必须活得比任务本身久，
        否则一个超时未清理的任务会带着已消失的取消标记继续跑完。
        """
        await self._redis.set(
            RedisKey.task_cancel(task_id),
            self._clock().isoformat(),
            ex=self._settings.task_timeout_seconds,
        )

    # ------------------------------------------------------------------ 执行
    async def execute(self, task_id: str, *, worker_id: str) -> Task | None:
        """领取并执行一个任务。Phase 7 之后任务体换成 LangGraph，其余不变。"""
        task = await self.claim(task_id, worker_id=worker_id)
        if task is None:
            return None

        bind_context(task_id=task.id, conversation_id=task.conversation_id, trace_id=task.trace_id)
        await self._events.publish(
            task_id=task.id,
            trace_id=task.trace_id,
            event_type=TaskEventType.TASK_STARTED,
            data={"status": TaskStatus.RUNNING.value},
        )

        try:
            async with self.heartbeat_while(task.id):
                with span("worker.task_body", task_id=task.id, worker_id=worker_id):
                    answer = await self._run_body(task)
        except asyncio.CancelledError:
            # 停机中断：不要把 CancelledError 吞成任务失败，交给 arq 处理重投。
            # 任务停在 RUNNING，由孤儿回收接手——这正是回收存在的意义。
            raise
        except Exception as exc:
            logger.exception("任务执行失败", extra={"task_id": task.id})
            return await self._finish_failed(
                task,
                code=ErrorCode.INTERNAL_ERROR,
                message="任务执行过程中出现内部错误",
                detail=type(exc).__name__,
            )

        if await self.is_cancel_requested(task.id):
            # 任务体跑完才发现取消（底层调用不可中断，17.5 第 5 条）
            return await self._finish_cancelled(task, worker_id=worker_id)

        return await self._finish_succeeded(task, answer=answer)

    async def _run_body(self, task: Task) -> str | None:
        """任务体。

        【后续扩展】Phase 7 用 LangGraph 替换本函数（登记于 CLAUDE.md）。
        在此之前它是空实现，因此任务会以 `SUCCEEDED` 且 `final_answer_md`
        为 None 结束——**这是本阶段的预期行为**，不是 bug：
        Phase 1.5 验证的是执行链路（投递/领取/心跳/写回/事件），不是分析能力。
        """
        logger.info("任务体为空实现，直接完成", extra={"task_id": task.id})
        return None

    # ------------------------------------------------------------------ 收尾
    async def _finish_succeeded(self, task: Task, *, answer: str | None) -> Task | None:
        updated = await self._transition(
            task,
            TaskPatch(
                status=TaskStatus.SUCCEEDED,
                finished_at=self._clock(),
                final_answer_md=answer,
            ),
            expected=TaskStatus.RUNNING,
        )
        if updated is not None:
            await self._emit_final(
                updated,
                TaskEventType.TASK_COMPLETED,
                {"answer_available": answer is not None, "evidence_count": 0},
            )
        return updated

    async def _finish_failed(
        self, task: Task, *, code: ErrorCode, message: str, detail: str | None = None
    ) -> Task | None:
        updated = await self._transition(
            task,
            TaskPatch(
                status=TaskStatus.FAILED,
                finished_at=self._clock(),
                error_code=code.value,
                error_message=message,
            ),
            expected=TaskStatus.RUNNING,
        )
        if updated is not None:
            logger.warning(
                "任务失败：%s", message, extra={"task_id": task.id, "error_code": code.value}
            )
            await self._emit_final(
                updated,
                TaskEventType.TASK_FAILED,
                {
                    "code": code.value,
                    "safe_message": message,
                    # 事件的 retryable 取错误码自带的语义，不在这里另判一套
                    "retryable": DEFAULT_RETRYABLE.get(code, False),
                    "detail": detail,
                },
            )
        return updated

    async def _finish_cancelled(self, task: Task, *, worker_id: str) -> Task | None:
        now = self._clock()
        updated = await self._tasks.update(
            task.id,
            TaskPatch(
                status=TaskStatus.CANCELLED,
                finished_at=now,
                worker_id=worker_id,
            ),
            at=now,
        )
        if updated is not None:
            logger.info("任务已取消", extra={"task_id": task.id, "status": updated.status.value})
            await self._emit_final(
                updated, TaskEventType.TASK_CANCELLED, {"final_status": updated.status.value}
            )
        return updated

    async def _transition(
        self, task: Task, patch: TaskPatch, *, expected: TaskStatus
    ) -> Task | None:
        """带前置状态的流转。返回 None 说明状态已被别的路径改掉，
        此时**不能**再发终态事件——同一任务的终态只能有一个。"""
        return await self._tasks.update(task.id, patch, at=self._clock(), only_if_status=expected)

    async def _emit_final(
        self, task: Task, event_type: TaskEventType, data: dict[str, object]
    ) -> None:
        await self._events.publish(
            task_id=task.id, trace_id=task.trace_id, event_type=event_type, data=dict(data)
        )
        # 任务结束后给事件流设保留期（详细设计 4.4：保留 1 小时）
        await self._events.finish(task.id)
        # 4.4：心跳键在任务结束时删除。它带 TTL，留着不会永久泄漏，
        # 但会在 TTL 内让一个**已经结束**的任务继续「证明自己活着」——
        # 任何以「心跳键存在」为判据的逻辑（Phase 7 的节点内检查）都会误判。
        # 一并删掉重投计数，理由相同：它的生命周期就是这一轮任务。
        #
        # 这三处清理原先在 `RedisTaskRepository._apply_index_moves` 里，
        # 随该实现一起删除；现在放在这里，因为**终态的唯一出口是本方法**
        # （成功/失败/取消三条路径都经过它）。
        await self._redis.delete(
            RedisKey.task_heartbeat(task.id), RedisKey.task_requeue_count(task.id)
        )

    # ------------------------------------------------------------------ 回收
    async def reclaim_orphans(self) -> ReclaimReport:
        """回收心跳过期的 RUNNING 任务。

        判据取记录里的 `heartbeat_at` 而不是 Redis 心跳键是否存在：
        `agent_task.heartbeat_at` 是权威字段（原则 9），键只是快通道。
        回收结果置 FAILED 而不是重新入队——重跑一个可能已经产生副作用的
        任务，风险高于让用户重发一次。
        """
        now = self._clock()
        cutoff = now - timedelta(seconds=self._worker_tuning.heartbeat_ttl_seconds)

        async with self._exclusive("orphan_reclaim") as acquired:
            if not acquired:
                return ReclaimReport(scanned=0, reclaimed=0, skipped=True)

            stale = await self._tasks.list_stale_running(cutoff, limit=_SCAN_BATCH)
            reclaimed = 0
            for task in stale:
                outcome = await self._task_failed_as_orphan(task, now)
                reclaimed += 1 if outcome is not None else 0
            if reclaimed:
                logger.warning("回收了 %d 个孤儿任务", reclaimed, extra={"status": "RECLAIMED"})
            return ReclaimReport(scanned=len(stale), reclaimed=reclaimed)

    async def _task_failed_as_orphan(self, task: Task, now: datetime) -> Task | None:
        updated = await self._tasks.update(
            task.id,
            TaskPatch(
                status=TaskStatus.FAILED,
                finished_at=now,
                error_code=ErrorCode.WORKER_INTERRUPTED.value,
                error_message="执行该任务的 Worker 已失联，任务被回收",
            ),
            at=now,
            only_if_status=TaskStatus.RUNNING,
        )
        if updated is None:
            return None
        await self._emit_final(
            updated,
            TaskEventType.TASK_FAILED,
            {
                "code": ErrorCode.WORKER_INTERRUPTED.value,
                "safe_message": "执行该任务的 Worker 已失联，任务被回收",
                "retryable": True,
            },
        )
        return updated

    # ------------------------------------------------------------------ 补偿
    async def reconcile_queue(self) -> ReconcileReport:
        """把「写库成功但没进队列」的 QUEUED 任务重新投递（详细设计 17.1）。

        **先问队列里有没有，再决定重投**：队列积压时一批任务会长时间停在
        QUEUED，它们不是投递失败，只是还没轮到。少了这一步判断，
        补偿扫描会把正在正常排队的任务全部重投一遍并计入失败次数。
        """
        now = self._clock()
        cutoff = now - timedelta(seconds=self._worker_tuning.queue_reconcile_seconds)
        stale = await self._tasks.list_stale_queued(cutoff, limit=_SCAN_BATCH)

        requeued = 0
        failed = 0
        for task in stale:
            try:
                if await self._queue.is_pending(task.id):
                    continue
            except (aioredis.RedisError, OSError) as exc:
                # 队列查不动时**什么都不做**：既不能判定它丢了，
                # 也不该把一次 Redis 抖动记成投递失败。
                logger.warning("队列状态查询失败，本轮跳过：%s", type(exc).__name__)
                continue

            attempts = await self._bump_requeue(task.id)
            if attempts > self._worker_tuning.max_requeue_attempts:
                if await self._task_failed_as_undeliverable(task, now) is not None:
                    failed += 1
                continue
            if await self.enqueue(task):
                requeued += 1
                logger.warning("任务重新投递（第 %d 次）", attempts, extra={"task_id": task.id})
        return ReconcileReport(scanned=len(stale), requeued=requeued, failed=failed)

    async def _bump_requeue(self, task_id: str) -> int:
        # TTL 取任务超时的 4 倍：计数键必须活得比任何一次重投周期都久，
        # 否则计数被清掉后「最多重投 2 次」会变成无限重投。
        ttl = self._settings.task_timeout_seconds * 4
        return int(
            await self._scripts["increment_with_ttl"](
                keys=[RedisKey.task_requeue_count(task_id)], args=[ttl]
            )
        )

    async def _task_failed_as_undeliverable(self, task: Task, now: datetime) -> Task | None:
        updated = await self._tasks.update(
            task.id,
            TaskPatch(
                status=TaskStatus.FAILED,
                finished_at=now,
                error_code=ErrorCode.ENQUEUE_FAILED.value,
                error_message="任务多次投递失败，请稍后重试",
            ),
            at=now,
            only_if_status=TaskStatus.QUEUED,
        )
        if updated is None:
            return None
        logger.error("任务投递失败次数超限，已置为 FAILED", extra={"task_id": task.id})
        await self._emit_final(
            updated,
            TaskEventType.TASK_FAILED,
            {
                "code": ErrorCode.ENQUEUE_FAILED.value,
                "safe_message": "任务多次投递失败，请稍后重试",
                "retryable": True,
            },
        )
        return updated

    # ------------------------------------------------------------------ 互斥
    @asynccontextmanager
    async def _exclusive(self, purpose: str) -> AsyncIterator[bool]:
        """`lock:{purpose}` 互斥（详细设计 4.4）。

        拿不到锁不是错误，是**正常的**——多实例部署时每一轮扫描只该有一个实例做。
        正确性并不依赖这把锁（回收本身有 CAS 保护），它省的是重复扫描。
        """
        key = RedisKey.lock(purpose)
        token = secrets.token_hex(8)
        # 先置 None：set 抛异常时 `acquired` 从未被赋值，
        # 下面的 finally 会以一个 UnboundLocalError 把原始异常盖掉
        acquired: str | None = None
        try:
            acquired = await self._redis.set(key, token, nx=True, ex=self._tuning.lock_ttl_seconds)
        except (aioredis.RedisError, OSError) as exc:
            logger.warning("获取锁失败，跳过本轮：%s", type(exc).__name__)
            yield False
            return
        try:
            yield bool(acquired)
        finally:
            if acquired:
                try:
                    await self._scripts["release_lock"](keys=[key], args=[token])
                except (aioredis.RedisError, OSError):
                    # 释放失败只影响下一轮的互斥性，锁会自己过期，不值得中断
                    logger.warning("释放锁失败，等待 TTL 自动过期：%s", purpose)
