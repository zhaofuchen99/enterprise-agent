"""需要**真实 Redis** 的用例（开发流程 6.3 验收命令 3：空任务闭环）。

单元测试里 arq 只能换成替身——它需要真实连接才能 `enqueue_job`。
被替身挡在外面的恰是「任务到底有没有进 `q:agent`、Worker 能不能领到」
这两件事，而它们是本阶段门禁的核心。因此这一组必须打真靶。

前置：`make up`（Redis 在宿主 6381 端口，见 CLAUDE.md 的本机环境事实）。

用 arq 的 `burst=True` 跑一次真实的「投递 -> 领取 -> 执行 -> 写回 -> 发事件」，
而不是手工调用 `run_agent_task`：后者跳过了队列本身，
而队列正是这一阶段唯一新增的、最容易配错的东西（队列名、任务名、
序列化方式三者只要有一处对不上，任务就会静静地躺在那里没有人领）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis
from arq.worker import Worker

from app.core.config import get_settings
from app.domain.task import Task, TaskStatus
from app.infrastructure.queue import TASK_JOB_NAME, ArqJobQueue
from app.infrastructure.redis import RedisKey, create_client
from app.repositories.task_repo import RedisTaskRepository, TaskPatch
from app.services.event_bus import RedisStreamEventBus, TaskEventType
from app.worker import WorkerSettings, on_shutdown, on_startup

pytestmark = pytest.mark.integration


@pytest.fixture
async def redis() -> AsyncIterator[aioredis.Redis]:
    """连真实 Redis。用 db 15 与开发数据隔离（conftest 的 REDIS_URL 已指定）。"""
    client = create_client(get_settings())
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


def _task(task_id: str = "tsk_integration_00000000001") -> Task:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return Task(
        id=task_id,
        user_id="usr_int",
        conversation_id="cnv_int",
        trace_id="trc_int",
        query_text="华东销售额为什么下降",
        queued_at=now,
        created_at=now,
        updated_at=now,
    )


async def test_job_can_be_enqueued_and_found_in_the_documented_queue(redis: aioredis.Redis) -> None:
    """投递后任务要出现在详细设计 4.4 写明的 `q:agent` 里。

    arq 的默认队列名是 `arq:queue`，与本项目文档不一致。这条用例钉住
    「文档里写的那个键真的存在」——否则按文档排查的人会以为链路没跑起来。
    """
    queue = await ArqJobQueue.create(get_settings())
    try:
        assert await queue.enqueue(task_id="tsk_a", trace_id="trc_a", trace_context={}) is True
        assert await redis.zcard(RedisKey.queue()) == 1
        assert await queue.is_pending("tsk_a") is True
    finally:
        await queue.aclose()


async def test_enqueue_is_idempotent_per_task(redis: aioredis.Redis) -> None:
    """同一个 task_id 重复投递是幂等的（补偿扫描会反复投同一个任务）。

    没有这层去重，「重投」就变成「同一个任务被两个 Worker 同时领取」，
    而重复执行在 LLM 与 SQL 场景下都是要花钱的。
    """
    queue = await ArqJobQueue.create(get_settings())
    try:
        await queue.enqueue(task_id="tsk_a", trace_id="trc_a", trace_context={})

        assert await queue.enqueue(task_id="tsk_a", trace_id="trc_a", trace_context={}) is False
        assert await redis.zcard(RedisKey.queue()) == 1
    finally:
        await queue.aclose()


async def test_full_loop_through_a_real_worker(redis: aioredis.Redis) -> None:
    """门禁：任务被投递 -> Worker 领取 -> 执行 -> 写回状态 -> 发布事件。

    这是开发流程 6.3 验收命令 3 的可执行版本。
    """
    settings = get_settings()
    repo = RedisTaskRepository(redis, settings)
    task = _task()
    await repo.add(task)

    queue = await ArqJobQueue.create(settings)
    assert await queue.enqueue(task_id=task.id, trace_id=task.trace_id, trace_context={})
    await queue.aclose()

    worker = Worker(
        functions=WorkerSettings.functions,
        queue_name=RedisKey.queue(),
        redis_settings=WorkerSettings.redis_settings,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        burst=True,  # 处理完队列里现有的任务就退出，不会挂住用例
        max_jobs=1,
        job_timeout=30,
    )
    await worker.async_run()

    finished = await repo.get(task.id)
    assert finished is not None
    # Phase 1.5 的任务体是空实现，因此 answer 为 None 是预期结果
    assert finished.status is TaskStatus.SUCCEEDED
    assert finished.worker_id  # 领取时写下了是哪个 Worker 跑的

    events = await RedisStreamEventBus(redis, settings).read(task.id)
    assert [event.type for event in events] == [
        TaskEventType.TASK_STARTED,
        TaskEventType.TASK_COMPLETED,
    ]

    # 跑完的作业不再是「等待中」——补偿扫描据此判断任务是不是真的丢了。
    # 若把 complete 也算作还在队列里，一个空跑一轮的任务会永远卡在 QUEUED，
    # 连失败都不会失败。
    queue = await ArqJobQueue.create(settings)
    try:
        assert await queue.is_pending(task.id) is False
    finally:
        await queue.aclose()


async def test_worker_leaves_no_running_task_behind(redis: aioredis.Redis) -> None:
    """门禁：Worker 独立启停后不留下停在 RUNNING 的任务。"""
    settings = get_settings()
    repo = RedisTaskRepository(redis, settings)
    task = _task("tsk_integration_00000000002")
    await repo.add(task)

    queue = await ArqJobQueue.create(settings)
    await queue.enqueue(task_id=task.id, trace_id=task.trace_id, trace_context={})
    await queue.aclose()

    worker = Worker(
        functions=WorkerSettings.functions,
        queue_name=RedisKey.queue(),
        redis_settings=WorkerSettings.redis_settings,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        burst=True,
        max_jobs=1,
    )
    await worker.async_run()

    left_running = await repo.list_stale_running(
        __import__("datetime").datetime.now(__import__("datetime").UTC), limit=10
    )
    assert left_running == []


async def test_two_api_instances_share_one_rate_limit_quota(redis: aioredis.Redis) -> None:
    """门禁：多 API 实例行为一致——两个实例合计放行数不超过配置上限。

    这是 Phase 1.5 相对 Phase 1 的**核心改进**，也只有在真实 Redis 上
    才验证得了「两个进程共享同一份计数」。
    """
    from app.services.rate_limit import RedisFixedWindowLimiter

    settings = get_settings()
    instance_a = RedisFixedWindowLimiter(redis=redis, settings=settings)
    instance_b = RedisFixedWindowLimiter(redis=redis, settings=settings)

    allowed = 0
    for _ in range(6):
        allowed += int(
            (
                await instance_a.check(
                    scope="task:create", subject="usr_x", limit=3, window_seconds=60
                )
            ).allowed
        )
        allowed += int(
            (
                await instance_b.check(
                    scope="task:create", subject="usr_x", limit=3, window_seconds=60
                )
            ).allowed
        )

    assert allowed == 3


async def test_orphan_reclamation_picks_up_a_dead_worker_task(redis: aioredis.Redis) -> None:
    """模拟「Worker 被 kill -9」：任务停在 RUNNING，心跳过期后必须被回收。

    这一条补的是优雅停机覆盖不到的那一半——SIGKILL 走不到 on_shutdown，
    只能靠心跳超时兜住。
    """
    from datetime import UTC, datetime, timedelta

    settings = get_settings()
    repo = RedisTaskRepository(redis, settings)
    task = _task("tsk_integration_00000000003")
    await repo.add(task)
    await repo.update(
        task.id, TaskPatch(status=TaskStatus.RUNNING), at=datetime.now(UTC) - timedelta(hours=1)
    )

    from app.services.task_runner import TaskRunner

    runner = TaskRunner(
        tasks=repo,
        queue=await ArqJobQueue.create(settings),
        events=RedisStreamEventBus(redis, settings),
        settings=settings,
        redis=redis,
    )
    report = await runner.reclaim_orphans()
    await runner._queue.aclose()

    assert report.reclaimed == 1
    reclaimed = await repo.get(task.id)
    assert reclaimed is not None
    assert reclaimed.status is TaskStatus.FAILED
    assert await repo.count_active("usr_int") == 0


async def test_worker_settings_job_name_matches_the_dispatch_constant() -> None:
    """任务名与队列名两处写错都不会报错，只会「入队了但没人执行」，因此显式钉住。"""
    assert WorkerSettings.queue_name == RedisKey.queue()
    registered = {func.__name__ for func in WorkerSettings.functions}
    assert registered == {TASK_JOB_NAME}
