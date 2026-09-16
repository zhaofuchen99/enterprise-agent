"""TaskRunner：投递 / 领取 / 心跳 / 取消 / 回收 / 补偿（开发流程 6.3 施工项 6）。

这一层是 Phase 1.5 的验收核心，因此用 fakeredis 跑**真实的仓储与事件流**，
只把 arq 的队列换成替身（它需要真实连接）。断言分两类：

- 任务记录最终落到哪个状态（写回是否发生、是否只发生一次）
- 事件流上出现了什么（SSE 在 Phase 10 要消费的正是这些）

时间是注入的，除了心跳那一组——心跳循环本身就在和真实时间打交道，
用假时钟测「心跳有没有真的发生」等于没测。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import pytest

from app.core.config import Settings
from app.core.errors import ErrorCode
from app.domain.task import Task, TaskStatus
from app.infrastructure.redis import RedisKey
from app.repositories.task_repo import RedisTaskRepository, TaskPatch, TaskRepository
from app.services.event_bus import RedisStreamEventBus, TaskEventType
from app.services.task_runner import TaskRunner
from app.tests.fakes import FakeJobQueue

_NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)


def _settings(**worker: object) -> Settings:
    return Settings(
        model_provider="p",
        model_name="m",
        model_api_key="k",
        embedding_model="e",
        embedding_api_key="k",
        database_url_agent="mysql+asyncmy://a@localhost/a",
        database_url_business_ro="mysql+asyncmy://a@localhost/b",
        redis_url="redis://localhost:6379/0",
        milvus_uri="http://localhost:19530",
        jwt_secret="x" * 32,
        worker=worker,
    )


def _task(
    task_id: str = "tsk_0000000001AAAAAAAAAAAA",
    *,
    status: TaskStatus = TaskStatus.QUEUED,
    queued_at: datetime = _NOW,
) -> Task:
    return Task(
        id=task_id,
        user_id="usr_1",
        conversation_id="cnv_0000000001AAAAAAAAAAAA",
        trace_id="trc_0000000001AAAAAAAAAAAA",
        query_text="华东销售额为什么下降",
        status=status,
        queued_at=queued_at,
        created_at=queued_at,
        updated_at=queued_at,
    )


class _Clock:
    def __init__(self, now: datetime = _NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


class ExplodingRunner(TaskRunner):
    """任务体抛异常，用来验证失败路径。"""

    async def _run_body(self, task: Task) -> str | None:
        raise RuntimeError("模型超时了")


class AnsweringRunner(TaskRunner):
    """任务体正常返回答案。默认实现返回 None，这里给一个非空值，
    以便区分「跑完了但没答案」与「跑完并拿到了答案」。"""

    async def _run_body(self, task: Task) -> str | None:
        return "# 分析结果"


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


def build_runner(
    redis: fakeredis.aioredis.FakeRedis,
    clock: Callable[[], datetime],
    *,
    queue: FakeJobQueue | None = None,
    runner_cls: type[TaskRunner] = TaskRunner,
    **worker: object,
) -> tuple[TaskRunner, TaskRepository, FakeJobQueue, RedisStreamEventBus]:
    settings = _settings(**worker)
    repo = RedisTaskRepository(redis, settings)
    events = RedisStreamEventBus(redis, settings)
    job_queue = queue or FakeJobQueue()
    runner = runner_cls(
        tasks=repo, queue=job_queue, events=events, settings=settings, redis=redis, clock=clock
    )
    return runner, repo, job_queue, events


# ------------------------------------------------------------------ 投递
async def test_enqueue_puts_the_job_with_trace_context(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """投递时要把 trace 上下文序列化进载荷，否则跨进程链路断开（19.4.1）。"""
    runner, _, queue, _ = build_runner(redis, clock)

    assert await runner.enqueue(_task()) is True

    assert len(queue.jobs) == 1
    job = queue.jobs[0]
    assert job.task_id == "tsk_0000000001AAAAAAAAAAAA"
    assert job.trace_id == "trc_0000000001AAAAAAAAAAAA"
    # 没有活动的 span 时 carrier 可能为空，但类型必须是可 JSON 序列化的 dict
    assert isinstance(job.trace_context, dict)


# ------------------------------------------------------------------ 领取
async def test_claim_moves_task_to_running(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, _, _ = build_runner(redis, clock)
    await repo.add(_task())

    claimed = await runner.claim("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert claimed is not None
    assert claimed.status is TaskStatus.RUNNING
    assert claimed.worker_id == "w1"
    assert claimed.started_at == _NOW
    assert claimed.heartbeat_at == _NOW


async def test_second_worker_cannot_claim_the_same_task(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """领取互斥。两个 Worker 同时开跑会把同一个问题算两遍。"""
    runner, repo, _, _ = build_runner(redis, clock)
    await repo.add(_task())

    first = await runner.claim("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")
    second = await runner.claim("tsk_0000000001AAAAAAAAAAAA", worker_id="w2")

    assert first is not None
    assert second is None
    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None and loaded.worker_id == "w1"


async def test_claim_of_unknown_task_is_a_noop(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, _, _, _ = build_runner(redis, clock)
    assert await runner.claim("tsk_不存在", worker_id="w1") is None


async def test_claim_honours_a_pending_cancel_before_starting(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """17.5 第 3 条：尚未被领取的取消请求，在领取时直接转 CANCELLED，不执行任何节点。"""
    runner, repo, _, events = build_runner(redis, clock)
    await repo.add(_task())
    await repo.update(
        "tsk_0000000001AAAAAAAAAAAA",
        TaskPatch(status=TaskStatus.CANCEL_REQUESTED),
        at=_NOW,
    )

    claimed = await runner.claim("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert claimed is None
    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None and loaded.status is TaskStatus.CANCELLED
    assert await repo.count_active("usr_1") == 0
    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert [event.type for event in published] == [TaskEventType.TASK_CANCELLED]


async def test_cancel_signal_without_status_change_also_stops_the_task(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """Redis 取消键是给另一个实例上的 Worker 的信号，单独存在时也必须生效。"""
    runner, repo, _, _ = build_runner(redis, clock)
    await repo.add(_task())
    await runner.signal_cancel("tsk_0000000001AAAAAAAAAAAA")

    assert await runner.claim("tsk_0000000001AAAAAAAAAAAA", worker_id="w1") is None
    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None and loaded.status is TaskStatus.CANCELLED


async def test_cancel_signal_expires_with_the_task_timeout(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """取消键必须活得比任务本身久（17.5），否则超时未清理的任务会「越跑越精神」。"""
    runner, _, _, _ = build_runner(redis, clock)
    await runner.signal_cancel("tsk_x")

    ttl = await redis.ttl("task:tsk_x:cancel")
    assert 0 < ttl <= 600


# ------------------------------------------------------------------ 执行
async def test_execute_writes_success_and_publishes_events(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, _, events = build_runner(redis, clock, runner_cls=AnsweringRunner)
    await repo.add(_task())

    finished = await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert finished is not None
    assert finished.status is TaskStatus.SUCCEEDED
    assert finished.final_answer_md == "# 分析结果"
    assert await repo.count_active("usr_1") == 0

    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert [event.type for event in published] == [
        TaskEventType.TASK_STARTED,
        TaskEventType.TASK_COMPLETED,
    ]
    assert published[-1].data["answer_available"] is True


async def test_empty_body_completes_without_an_answer(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """Phase 1.5 的任务体是空实现，因此任务会 SUCCEEDED 但 answer 为 None。

    **这是本阶段的预期行为**：验证的是执行链路，不是分析能力。
    事件里的 `answer_available` 如实报 False，不编造一个答案。
    """
    runner, repo, _, events = build_runner(redis, clock)
    await repo.add(_task())

    finished = await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert finished is not None
    assert finished.status is TaskStatus.SUCCEEDED
    assert finished.final_answer_md is None
    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert published[-1].data["answer_available"] is False


async def test_execute_marks_failed_when_the_body_raises(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, _, events = build_runner(redis, clock, runner_cls=ExplodingRunner)
    await repo.add(_task())

    finished = await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert finished is not None
    assert finished.status is TaskStatus.FAILED
    assert finished.error_code == ErrorCode.INTERNAL_ERROR.value
    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert published[-1].type is TaskEventType.TASK_FAILED
    assert published[-1].data["code"] == ErrorCode.INTERNAL_ERROR.value
    # 失败也要释放并发配额，否则用户再也创建不了任务
    assert await repo.count_active("usr_1") == 0


async def test_execute_cancelled_mid_flight_is_recorded_as_cancelled(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """17.5 第 5 条：底层调用不可中断时，等它返回后再转 CANCELLED。"""

    class CancellingRunner(TaskRunner):
        async def _run_body(self, task: Task) -> str | None:
            await self.signal_cancel(task.id)
            return "答案已经算出来了"

    runner, repo, _, events = build_runner(redis, clock, runner_cls=CancellingRunner)
    await repo.add(_task())

    finished = await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert finished is not None
    assert finished.status is TaskStatus.CANCELLED
    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert published[-1].type is TaskEventType.TASK_CANCELLED


async def test_execute_skips_a_task_that_was_already_claimed(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, _, events = build_runner(redis, clock)
    await repo.add(_task(status=TaskStatus.RUNNING))

    assert await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w2") is None
    assert await events.read("tsk_0000000001AAAAAAAAAAAA") == []


async def test_events_stream_gets_a_retention_ttl(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """4.4：任务结束后事件流保留 1 小时，不删除——重连的客户端还要靠它补齐。"""
    runner, repo, _, _ = build_runner(redis, clock)
    await repo.add(_task())

    await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    ttl = await redis.ttl("task:tsk_0000000001AAAAAAAAAAAA:events")
    assert 0 < ttl <= 3600
    assert await redis.exists("task:tsk_0000000001AAAAAAAAAAAA:events") == 1


# ------------------------------------------------------------------ 心跳
async def test_heartbeat_runs_while_the_body_is_working(
    redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """心跳必须与任务体**并发**跑：写在任务体里的话，某个节点一阻塞心跳就跟着停，
    一个健康但慢的任务会被判成孤儿回收掉。

    这一组用真实时钟（心跳循环本来就与真实时间打交道），
    因此把周期压到 1 秒，让用例整体在 2 秒内结束。
    """
    #: 任务体运行期间观察心跳键是否存在。**必须在体内看**：
    #: 任务一进终态，心跳键就会被清掉（4.4「任务结束后删除」），
    #: 事后再查只会看到它已经消失，从而把「写过了」误判成「没写」。
    observed: list[int] = []

    class SlowRunner(TaskRunner):
        async def _run_body(self, task: Task) -> str | None:
            await asyncio.sleep(1.3)
            observed.append(await self._redis.exists(RedisKey.task_heartbeat(task.id)))
            return None

    runner, repo, _, _ = build_runner(
        redis,
        # 这里必须用真实时钟：_Clock 是冻结的，心跳写进去的时间会和领取时刻
        # 一模一样，用例就分不出「跳过了」和「跳了但时间没动」。
        lambda: datetime.now(UTC),
        runner_cls=SlowRunner,
        heartbeat_interval_seconds=1,
        heartbeat_ttl_seconds=2,
    )
    await repo.add(_task())

    finished = await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    assert finished is not None
    assert finished.started_at is not None
    assert finished.heartbeat_at is not None
    # 领取时写过一次心跳，任务体跑期间又续了至少一拍，因此一定晚于开始时间
    assert finished.heartbeat_at > finished.started_at
    # 心跳键在任务运行期间是存在的（4.4）
    assert observed == [1]
    # 任务进终态后清掉，避免一个已结束的任务持续「证明自己活着」
    assert await redis.exists("task:tsk_0000000001AAAAAAAAAAAA:heartbeat") == 0


async def test_heartbeat_stops_after_the_body_returns(
    redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """任务结束后不该再有心跳——否则一个已结束的任务会持续「证明自己活着」。"""
    runner, repo, _, _ = build_runner(
        redis,
        lambda: datetime.now(UTC),
        heartbeat_interval_seconds=1,
        heartbeat_ttl_seconds=2,
    )
    await repo.add(_task())
    await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")
    before = (await repo.get("tsk_0000000001AAAAAAAAAAAA")).heartbeat_at  # type: ignore[union-attr]

    await asyncio.sleep(1.2)

    after = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert after is not None and after.heartbeat_at == before


# ------------------------------------------------------------------ 孤儿回收
async def test_reclaim_fails_tasks_whose_worker_disappeared(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, _, events = build_runner(redis, clock)
    await repo.add(_task(status=TaskStatus.RUNNING))
    # 心跳停在 5 分钟前，远超 TTL
    clock.advance(minutes=5)

    report = await runner.reclaim_orphans()

    assert (report.scanned, report.reclaimed) == (1, 1)
    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None
    assert loaded.status is TaskStatus.FAILED
    assert loaded.error_code == ErrorCode.WORKER_INTERRUPTED.value
    # 回收的**主要目的**是释放并发配额：不释放的话用户被永久限流
    assert await repo.count_active("usr_1") == 0
    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert published[-1].data["retryable"] is True


async def test_reclaim_leaves_healthy_tasks_alone(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, _, _ = build_runner(redis, clock)
    await repo.add(_task(status=TaskStatus.RUNNING))
    await repo.touch_heartbeat("tsk_0000000001AAAAAAAAAAAA", _NOW)
    clock.advance(seconds=5)

    report = await runner.reclaim_orphans()

    assert report.reclaimed == 0
    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None and loaded.status is TaskStatus.RUNNING


async def test_reclaim_skips_when_another_instance_holds_the_lock(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """多实例下每轮扫描只该有一个实例做。拿不到锁是常态，不是错误。"""
    runner, repo, _, _ = build_runner(redis, clock)
    await repo.add(_task(status=TaskStatus.RUNNING))
    await redis.set("lock:orphan_reclaim", "另一个实例")
    clock.advance(minutes=5)

    report = await runner.reclaim_orphans()

    assert report.skipped is True
    assert report.reclaimed == 0
    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None and loaded.status is TaskStatus.RUNNING


async def test_reclaim_releases_the_lock_afterwards(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """不释放的话下次回收要等到锁 TTL 过期，孤儿任务的悬挂时间会凭空变长。"""
    runner, _, _, _ = build_runner(redis, clock)

    await runner.reclaim_orphans()

    assert await redis.exists("lock:orphan_reclaim") == 0


# ------------------------------------------------------------------ 队列补偿
async def test_reconcile_requeues_a_lost_job(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """写库成功但没进队列（详细设计 17.1 的补偿场景）。"""
    runner, repo, queue, _ = build_runner(redis, clock)
    await repo.add(_task(queued_at=_NOW - timedelta(minutes=5)))
    clock.advance(minutes=5)

    report = await runner.reconcile_queue()

    assert (report.scanned, report.requeued, report.failed) == (1, 1, 0)
    assert [job.task_id for job in queue.jobs] == ["tsk_0000000001AAAAAAAAAAAA"]


async def test_reconcile_skips_a_job_that_is_merely_waiting_in_line(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """队列积压 ≠ 投递失败。

    少了这一步判断，补偿扫描会把一批正常排队的任务全部重投并计入失败次数，
    队列一堵用户的任务就全变 FAILED——比原问题严重得多。
    """
    runner, repo, queue, _ = build_runner(redis, clock)
    await repo.add(_task(queued_at=_NOW - timedelta(minutes=5)))
    queue.present.add("tsk_0000000001AAAAAAAAAAAA")
    clock.advance(minutes=5)

    report = await runner.reconcile_queue()

    assert (report.requeued, report.failed) == (0, 0)
    assert queue.jobs == []


async def test_reconcile_gives_up_after_max_attempts(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """超过重投上限即置 FAILED + ENQUEUE_FAILED，不能无限重投下去。

    队列置为「投递总是失败」：否则第一次重投成功后任务就真的在队列里了，
    后续扫描会（正确地）跳过它，永远走不到放弃那一步。
    """
    runner, repo, _, events = build_runner(
        redis, clock, queue=FakeJobQueue(fail_with=True), max_requeue_attempts=2
    )
    await repo.add(_task(queued_at=_NOW - timedelta(minutes=5)))

    for _ in range(3):
        clock.advance(minutes=1)
        await runner.reconcile_queue()

    loaded = await repo.get("tsk_0000000001AAAAAAAAAAAA")
    assert loaded is not None
    assert loaded.status is TaskStatus.FAILED
    assert loaded.error_code == ErrorCode.ENQUEUE_FAILED.value
    assert await repo.count_active("usr_1") == 0
    published = await events.read("tsk_0000000001AAAAAAAAAAAA")
    assert published[-1].type is TaskEventType.TASK_FAILED


async def test_reconcile_ignores_tasks_that_are_not_queued_any_more(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    runner, repo, queue, _ = build_runner(redis, clock)
    await repo.add(_task(status=TaskStatus.RUNNING, queued_at=_NOW - timedelta(minutes=5)))
    clock.advance(minutes=5)

    report = await runner.reconcile_queue()

    assert report.scanned == 0
    assert queue.jobs == []


async def test_published_events_carry_increasing_sequences(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """事件里的 `sequence` 必须真的递增，客户端靠它排序去重（18.4）。

    这条用例是针对一个真实写错过的版本：事件的标识（event_id / sequence）
    由 Redis 分配的 Stream ID 派生，而最初实现把**派生之前**的那份
    序列化进了流，于是所有事件的 sequence 都是 0——
    发布方拿到的是正确的对象，消费方读到的却全是 0，两边不一致。
    """
    runner, repo, _, events = build_runner(redis, clock, runner_cls=AnsweringRunner)
    await repo.add(_task())

    published = await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")
    assert published is not None

    from_stream = await events.read("tsk_0000000001AAAAAAAAAAAA")

    assert [e.type for e in from_stream] == [
        TaskEventType.TASK_STARTED,
        TaskEventType.TASK_COMPLETED,
    ]
    sequences = [e.sequence for e in from_stream]
    assert sequences == sorted(sequences) and len(set(sequences)) == 2
    # 流里读出来的必须与发布时返回的是同一份内容
    assert sequences[0] > 0
    assert [e.event_id for e in from_stream] == sorted(e.event_id for e in from_stream)


async def test_events_can_be_read_incrementally_with_last_event_id(
    redis: fakeredis.aioredis.FakeRedis, clock: _Clock
) -> None:
    """断线重连：按 Last-Event-ID 只取新增的那部分（详细设计 17.3）。"""
    runner, repo, _, events = build_runner(redis, clock, runner_cls=AnsweringRunner)
    await repo.add(_task())
    await runner.execute("tsk_0000000001AAAAAAAAAAAA", worker_id="w1")

    all_events = await events.read("tsk_0000000001AAAAAAAAAAAA")
    tail = await events.read("tsk_0000000001AAAAAAAAAAAA", after_id=all_events[0].event_id)

    assert [e.type for e in tail] == [TaskEventType.TASK_COMPLETED]
    assert tail[0].sequence > all_events[0].sequence
