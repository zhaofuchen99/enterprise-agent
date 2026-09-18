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
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.domain.task import Task, TaskStatus
from app.infrastructure.queue import TASK_JOB_NAME, ArqJobQueue
from app.infrastructure.redis import RedisKey, create_client
from app.repositories.task_repo import SqlTaskRepository, TaskPatch
from app.services.event_bus import RedisStreamEventBus, TaskEventType
from app.tests.db import sql_sessions
from app.worker import WorkerSettings, on_shutdown, on_startup

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
async def _vocab_snapshot() -> AsyncIterator[None]:
    """给测试库准备一份**最小词表快照**。

    **为什么需要它**：这组用例起的是**真 Worker**，而 Worker 的 `on_startup`
    在 Phase 6 起要装配整张图（含 RAG 工具 → 词表快照）。而 `agent_test` 库的
    `rag_vocab` 是空的，版本是 `0-0`，快照自然不存在——
    于是 Worker 起不来，用例失败在"文件不存在"上，指向的是存储而不是装配逻辑。

    **不改成"快照缺失时降级"**：产品行为是「起不来」，那是对的
    （`load_vocabulary` 明写不降级，见 `tools/rag/vocabulary.py`）。
    测试要准备的是它运行所需的环境，不是让产品宽容。

    快照与开发库的那份独立：版本取自测试库的 `rag_vocab`（空 → `0-0`），
    因此不会覆盖 `data/storage` 下开发用的那一份。
    """
    from app.infrastructure.storage import build_object_storage
    from app.tools.rag.tokenizer import Vocabulary
    from app.tools.rag.vocabulary import export_snapshot

    storage = build_object_storage(get_settings())
    # 空词表：图中没有 RAG 步骤时会真的用到它（`build_rag_retrieve_tool` 只装载，
    # 不检索），而有 RAG 步骤的用例不在这组里
    await export_snapshot(storage, Vocabulary({}, document_count=1), version="0-0")
    yield


def _demo_user_id() -> str:
    """演示账号里 analyst 的 ID（由用户名确定性派生，见 `seed_demo_users`）。"""
    from app.repositories.user_repo import seed_demo_users

    return next(user.id for user in seed_demo_users(get_settings()) if user.username == "analyst")


@pytest.fixture
async def redis() -> AsyncIterator[aioredis.Redis]:
    """连真实 Redis。用 db 15 与开发数据隔离（conftest 的 REDIS_URL 已指定）。"""
    client = create_client(get_settings())
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
async def sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """连真实 MySQL（`agent_test` 库）、清空数据并播种演示用户。

    与 Worker 内部那份仓储连的是**同一个库**——两边都从
    `get_settings().database_url_agent` 取连接串，这正是要验证的事：
    测试里写的任务，真 Worker 得能领到。

    **顺带播种演示用户**：Phase 6 起 `agent/runner.py` 会先查用户再跑图，
    查不到就拒绝执行（`agent_task` 没有数据范围字段，而 `PermissionScope`
    的空 `region_ids` 表示**不限**，拿它兜底等于让已删除用户的任务拿到全量数据）。
    **播种必须在这里、不能在另一个夹具里**：`sql_sessions` 每次进入都清表，
    分开写的那个夹具会被这里的清表抹掉——实测就是这个症状：
    用户播了，跑的时候查不到。

    做成独立的夹具（而不是塞进 `repo`）是因为**执行产出要按同一份会话读**：
    `sql_sessions` 每次进入都会清表，用例里再进一次会把刚落的数据抹掉，
    因此只能从这一个入口分发给所有仓储。
    """
    from app.repositories.user_repo import SqlUserRepository, seed_demo_users

    async with sql_sessions() as factory:
        # 播种在清表**之后**（`sql_sessions` 进入时已清）
        await SqlUserRepository(factory).upsert_demo_users(seed_demo_users(get_settings()))
        yield factory


@pytest.fixture
def repo(sessions: async_sessionmaker[AsyncSession]) -> SqlTaskRepository:
    return SqlTaskRepository(sessions)


def _task(task_id: str = "tsk_0000000000000000000001") -> Task:
    """造一个任务。

    **ID 必须严格是 26 字符**（`前缀(4) + 22`，详细设计 16.1）。
    这里原先写的是 `tsk_integration_00000000001`（27 字符），在 Redis 实现下
    跑得好好的——Redis 不校验长度；换成 `CHAR(26)` 之后数据库直接拒绝
    （1406 Data too long）。这不是测试的噪音，而是**数据库替我们抓住了
    一个一直违反 ID 规范的测试夹具**。
    """
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    suffix = task_id.removeprefix("tsk_")
    return Task(
        id=task_id,
        user_id=_demo_user_id(),
        conversation_id="cnv_0000000000000000000001",
        trace_id=f"trc_{suffix}",
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


async def test_full_loop_through_a_real_worker(
    redis: aioredis.Redis,
    repo: SqlTaskRepository,
    sessions: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """门禁：任务被投递 -> Worker 领取 -> 执行 -> 写回状态 -> 发布事件。

    这是开发流程 6.3 验收命令 3 的可执行版本。Phase 6 起任务体是真的图，
    因此这一条也是**唯一一条把真实 Worker、真实图、真实工具、真实库串起来**的
    端到端用例——执行产出（含轨迹）确实落进 MySQL 这件事只有这里能验。

    **模型换成替身，工具与数据库保持真实**：`app/tests/conftest.py` 把
    `MODEL_BASE_URL` 钉在不可解析的 `model.invalid`（刻意的——测试不该打
    真实服务、也不该花真钱），所以真模型在这一组里只可能失败。
    替身返回三段脚本：①「只查 SQL」的意图 → ② 一条真实可跑的候选 SQL →
    ③ 分析结论。于是 SQL 走的是**真校验器、真只读执行、真业务库**，
    而模型那一段是确定性的。
    """
    from sqlalchemy import select

    from app.agent.schemas.analysis import AnalysisResult
    from app.agent.schemas.plan import IntentResult
    from app.infrastructure.models.evidence import AgentTraceEvent
    from app.repositories.agent_repo import SqlAgentArtifactRepository
    from app.tests.fakes import FakeModelGateway
    from app.tools.sql.schemas import SqlCandidate

    scripts = [
        IntentResult(intent="QUERY", required_sources=("sql",), confidence=0.9),
        SqlCandidate(
            sql=(
                "SELECT SUM(net_amount) AS net_sales FROM fact_sales_order_item "
                "WHERE order_date >= :start AND order_date < :end"
            ),
            parameters={"start": "2025-07-01", "end": "2025-10-01"},
            selected_tables=("fact_sales_order_item",),
            selected_columns=("net_amount", "order_date"),
            expected_columns=("net_sales",),
            explanation="按季度汇总净销售额",
        ),
        AnalysisResult(direct_answer="2025 年 Q3 的净销售额已从销售事实表取得。"),
    ]
    monkeypatch.setattr(
        "app.worker.build_model_gateway", lambda _settings: FakeModelGateway(scripts)
    )

    settings = get_settings()
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
    assert finished.status is TaskStatus.SUCCEEDED, finished.error_message
    assert finished.worker_id  # 领取时写下了是哪个 Worker 跑的
    # 跑完了、产出也落了，因此轨迹是完整的（见 `TaskPatch.trace_incomplete`）
    assert finished.trace_incomplete is False

    events = await RedisStreamEventBus(redis, settings).read(task.id)
    assert [event.type for event in events] == [
        TaskEventType.TASK_STARTED,
        TaskEventType.TASK_COMPLETED,
    ]

    # 执行产出真的落了库：轨迹按执行顺序取回，且与任务同 trace_id。
    # **这一条只能在真库上验**：单元测试里的仓储是内存替身，
    # 它不会把"API 与 Worker 的装配分家"这类问题暴露出来。
    artifacts = SqlAgentArtifactRepository(sessions)
    trace = await artifacts.list_trace_events(task.id)
    assert [row.sequence for row in trace] == list(range(1, len(trace) + 1))
    started = [row.node for row in trace if row.event_type == "node.started"]
    # `trace_id` 是**任务级**属性，落库时由收尾路径带上（`NodeTrace` 上不带它）。
    # 单查这一列是因为它在领域对象上读不回来——SSE 侧要靠它把事件与链路对上，
    # 而写错/写空不会有任何报错。
    async with sessions() as session:
        trace_ids = set(
            (
                await session.execute(
                    select(AgentTraceEvent.trace_id).where(AgentTraceEvent.task_id == task.id)
                )
            ).scalars()
        )
    assert trace_ids == {task.trace_id}
    assert started[0] == "supervisor"
    # 简单查询的路径：supervisor → sql → reflect → conflict → analysis → reviewer → final
    assert started == ["supervisor", "sql", "reflect", "conflict", "analysis", "reviewer", "final"]
    # **每个进入都有一个离开**：卡住的节点在只有离开事件的轨迹上看不出来
    assert [row.node for row in trace if row.event_type != "node.started"] == started
    assert all(row.duration_ms is not None for row in trace if row.event_type != "node.started")

    # SQL 真的执行了：证据带着结果落进 `agent_evidence`，
    # 而它同时是"答案里那条引用点得开"的依据
    evidence = await artifacts.list_evidence(task.id)
    assert [item.source_type for item in evidence] == ["SQL"]
    assert "net_sales=" in evidence[0].claim

    # 跑完的作业不再是「等待中」——补偿扫描据此判断任务是不是真的丢了。
    # 若把 complete 也算作还在队列里，一个空跑一轮的任务会永远卡在 QUEUED，
    # 连失败都不会失败。
    queue = await ArqJobQueue.create(settings)
    try:
        assert await queue.is_pending(task.id) is False
    finally:
        await queue.aclose()


async def test_worker_leaves_no_running_task_behind(
    redis: aioredis.Redis, repo: SqlTaskRepository
) -> None:
    """门禁：Worker 独立启停后不留下停在 RUNNING 的任务。"""
    settings = get_settings()
    task = _task("tsk_0000000000000000000002")
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


async def test_orphan_reclamation_picks_up_a_dead_worker_task(
    redis: aioredis.Redis, repo: SqlTaskRepository
) -> None:
    """模拟「Worker 被 kill -9」：任务停在 RUNNING，心跳过期后必须被回收。

    这一条补的是优雅停机覆盖不到的那一半——SIGKILL 走不到 on_shutdown，
    只能靠心跳超时兜住。
    """
    from datetime import UTC, datetime, timedelta

    settings = get_settings()
    task = _task("tsk_0000000000000000000003")
    await repo.add(task)

    # **必须连 heartbeat_at 一起回拨**，只改 status 不足以模拟「Worker 被 kill」：
    # 真实路径上 `claim` 会把 heartbeat_at 与 started_at 一起写上，
    # 判死依据（16.5 的 `idx_task_status_heartbeat`）读的正是 heartbeat_at。
    # 只设 status 的话，SQL 实现的
    # `COALESCE(heartbeat_at, started_at, queued_at)` 会退回到刚落库的
    # queued_at，于是任务「看起来很新」，回收扫描自然扫不到——
    # 这就是这条用例在 Redis 实现下能过、换 MySQL 后失败的原因。
    dead_at = datetime.now(UTC) - timedelta(hours=1)
    await repo.update(
        task.id,
        TaskPatch(status=TaskStatus.RUNNING, heartbeat_at=dead_at, started_at=dead_at),
        at=dead_at,
    )

    from app.repositories.agent_repo import InMemoryAgentArtifactRepository
    from app.services.task_runner import TaskRunner

    runner = TaskRunner(
        tasks=repo,
        queue=await ArqJobQueue.create(settings),
        events=RedisStreamEventBus(redis, settings),
        settings=settings,
        redis=redis,
        artifacts=InMemoryAgentArtifactRepository(),
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
