"""Worker 进程入口（arq）。

**分层硬约束（开发流程 5.2）**：本模块及 `app/agent/`、`app/tools/` 下任何模块，
不得 import `fastapi`。破坏后 Worker 无法独立扩缩容。
该约束由 `scripts/check_layering.py` 在 CI 中强制。

**注意约束的实质而不只是字面**：`app/infrastructure/observability.py`
刻意不 import FastAPI 的自动埋点，就是为了让本模块能安全地引用它——
检查器只看本文件的 import 行，看不出「引了一个引了 fastapi 的模块」。
FastAPI 埋点因此留在 `app/main.py` 里单独调用。

本文件只做三件事：装配依赖、注册任务函数、注册定时任务。
投递/领取/心跳/回收的逻辑全在 `services/task_runner.py`，
这样 Phase 7 把任务体换成 LangGraph 时，改动面只有那个函数。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any, ClassVar

import redis.asyncio as aioredis
from arq import cron
from arq.connections import RedisSettings
from arq.worker import Worker
from sqlalchemy.ext.asyncio import AsyncEngine

from app.agent.graph import TaskGraph, build_graph
from app.agent.runner import build_task_body
from app.core.config import Settings, get_settings
from app.domain.task import Task, TaskOutcome
from app.infrastructure.cache import VersionedCache
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import (
    SERVICE_WORKER,
    bind_context,
    clear_context,
    setup_logging,
)
from app.infrastructure.model_gateway import ModelGateway, build_model_gateway
from app.infrastructure.observability import (
    business_trace_id_from_context,
    restore_trace_context,
    setup_error_tracking,
    setup_observability,
)
from app.infrastructure.queue import TASK_JOB_NAME, ArqJobQueue
from app.infrastructure.redis import RedisKey, create_client
from app.infrastructure.storage import build_object_storage
from app.infrastructure.vector_store import build_vector_store
from app.repositories import build_sql_repositories
from app.repositories.user_repo import SqlUserRepository
from app.repositories.vocab_repo import SqlVocabRepository
from app.services.event_bus import RedisStreamEventBus
from app.services.task_runner import TaskBody, TaskRunner
from app.tools.rag.tool import build_rag_retrieve_tool
from app.tools.sql.tool import build_sql_query_tool

logger = logging.getLogger(__name__)


async def run_agent_task(
    ctx: dict[str, Any],
    task_id: str,
    trace_id: str = "",
    trace_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """执行一个分析任务。**任务体本身在 `TaskRunner.execute` 里**。

    `trace_context` 是 API 侧在投递时序列化的 OTel 上下文（含 traceparent）。
    恢复它，API 的 HTTP span 与这里的 span 才属于同一条 trace（详细设计 19.4.1）。
    """
    runner: TaskRunner = ctx["runner"]
    worker_id: str = ctx["worker_id"]

    # baggage 里的业务 trace_id 优先：它来自 API 侧那个请求，
    # 比队列参数更可靠——参数可能是重投时补的，baggage 一定跟着原始请求走。
    effective_trace_id = business_trace_id_from_context() or trace_id
    with restore_trace_context(trace_context or {}):
        bind_context(trace_id=effective_trace_id or None)
        try:
            task = await runner.execute(task_id, worker_id=worker_id)
        finally:
            clear_context()

    if task is None:
        # 没执行不代表出错：已被领取、已取消都是正常结果（见 TaskRunner.claim）
        return {"task_id": task_id, "executed": False}
    return {"task_id": task_id, "executed": True, "status": task.status.value}


async def reclaim_orphans(ctx: dict[str, Any]) -> dict[str, Any]:
    """定时任务：回收心跳过期的 RUNNING 任务。"""
    runner: TaskRunner = ctx["runner"]
    report = await runner.reclaim_orphans()
    return asdict(report)


async def reconcile_queue(ctx: dict[str, Any]) -> dict[str, Any]:
    """定时任务：重新投递「写库成功但没进队列」的任务（详细设计 17.1）。"""
    runner: TaskRunner = ctx["runner"]
    report = await runner.reconcile_queue()
    return asdict(report)


def _worker_id() -> str:
    """Worker 标识。带主机名与 pid：排查孤儿任务时要能定位到**哪个**进程死了。"""
    return f"{socket.gethostname()}:{os.getpid()}"


def _build_runner(
    settings: Settings,
    redis: aioredis.Redis,
    queue: ArqJobQueue,
    engine: AsyncEngine,
    *,
    body: TaskBody,
    events: RedisStreamEventBus,
) -> TaskRunner:
    """装配 Worker 侧的任务仓储与任务体。

    与 `app/main.py` 走**同一份** SQL 装配（`build_sql_repositories`），
    仓储全部从那里取——不共用一个函数的话，API 与 Worker 的仓储实现
    可能在某次改动中分家：任务仓储分家的症状是「任务建得出来但 Worker 领不到」，
    执行产出仓储分家的症状是「任务跑完了但 `/trace` 查出来是空的」。
    后者更隐蔽，因为两条链路都"成功"了。
    """
    repos = build_sql_repositories(create_session_factory(engine))
    return TaskRunner(
        tasks=repos.tasks,
        queue=queue,
        # **与图是同一个实例**（见 `_build_task_graph` 的形参）：Worker 发
        # 任务级事件、节点发节点级事件，两者必须落在同一条流上——
        # 各建一个也能跑（Redis 键是同一个），但"谁在发"就说不清了
        events=events,
        settings=settings,
        redis=redis,
        artifacts=repos.artifacts,
        body=body,
    )


async def _build_task_graph(
    settings: Settings,
    redis: aioredis.Redis,
    gateway: ModelGateway,
    engine: AsyncEngine,
    *,
    events: RedisStreamEventBus,
) -> tuple[TaskGraph, Callable[[Task], Awaitable[TaskOutcome]]]:
    """装配最小 Graph 与它的任务体。

    **只在 Worker 进程里装配**：`app/main.py`（API 进程）不得 import
    `agent.graph` / `tools.*`——分层检查器 L1 拦的就是这件事（API 进程加载
    LangGraph 与全部 Tool 会让启动变慢、内存翻倍）。这个函数在 `worker.py`
    里，正是那条约束的落点。

    **装配失败就让 Worker 起不来**（`on_startup` 里不 catch）：向量库、
    对象存储、词表快照缺一个的报错是"跑任务时连不上"，而那时已经在处理
    真实请求了——半可用的 Worker 比起不来的 Worker 难查得多。

    词表从**快照**装载（`build_rag_retrieve_tool` 内部做），所以这是
    async 的：快照缺失时会以"先跑 make vocab"报错，而不是退化成一个
    查不到东西的检索器。
    """
    sessions = create_session_factory(engine)
    storage = build_object_storage(settings)
    vector_store = build_vector_store(settings)
    sql_tool = build_sql_query_tool(settings, gateway)
    rag_tool = await build_rag_retrieve_tool(
        settings,
        gateway,
        vector_store=vector_store,
        storage=storage,
        vocab=SqlVocabRepository(sessions),
        cache=VersionedCache(redis, default_ttl_seconds=settings.rag.vocab_cache_ttl_seconds),
    )
    graph = TaskGraph(
        settings=settings,
        graph=build_graph(
            settings,
            gateway=gateway,
            sql_tool=sql_tool,
            rag_tool=rag_tool,
            # 指标目录是"文档表头 → metric_code"的桥，没有它冲突检测整条跳过
            catalog=sql_tool.catalog,
            # 节点级事件（`node.started` / `progress.assessed` / …）经它发出。
            # **任务级事件（`task.started` 等）由 TaskRunner 发**，
            # 两者是同一条流上的两类事件，客户端不必知道谁发的
            events=events,
        ),
        sql_tool=sql_tool,
        rag_tool=rag_tool,
        gateway=gateway,
    )
    return graph, build_task_body(graph, SqlUserRepository(sessions))


async def on_startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    setup_logging(SERVICE_WORKER, settings.log_level)
    setup_observability(settings, service_name=SERVICE_WORKER)
    setup_error_tracking(settings)

    # 用自建连接而不是复用 arq 的 ctx["redis"]：arq 的连接池是否
    # decode_responses 由它的内部实现决定，而本项目的仓储与事件总线
    # 都按「取出来就是 str」写。多几条连接换掉这个隐含耦合是划算的。
    redis = create_client(settings)
    queue = await ArqJobQueue.create(settings)

    # 数据库连接池归本进程所有，退出时在 on_shutdown 里释放。
    # 与 API 进程各持一个池：两者是不同的进程，共用一个池在语言层面就不成立。
    engine = create_engine(settings)

    # 模型网关持有两个 httpx 连接池（chat 与 embedding 各一个），
    # 与 db_engine 同样归本进程所有，退出时在 on_shutdown 里释放。
    # Phase 4 的 SQL Tool 是它的第一个消费者。
    gateway = build_model_gateway(settings)

    ctx["redis"] = redis
    ctx["queue"] = queue
    ctx["db_engine"] = engine
    ctx["model_gateway"] = gateway
    ctx["worker_id"] = _worker_id()
    # 事件总线**只建一份**，同时交给图（节点级事件）与 TaskRunner（任务级事件）：
    # 17.3 的 SSE 端点在 API 进程里订阅的就是这条流，所以两条路径必须落到同一个
    # Redis 键上的同一份实现，否则"为什么流里的顺序怪怪的"会变成查不清的问题
    events = RedisStreamEventBus(redis, settings)
    ctx["event_bus"] = events
    # 任务体（最小 Graph）在**启动时**装配好：它的依赖里有两样会在启动时
    # 真的连一次（Qdrant 的 collection 检查、词表快照的读取），
    # 放到第一次跑任务时才连等于把那两类故障推迟到有真实请求的时候。
    task_graph, body = await _build_task_graph(settings, redis, gateway, engine, events=events)
    ctx["task_graph"] = task_graph
    ctx["runner"] = _build_runner(settings, redis, queue, engine, body=body, events=events)
    logger.info("worker 启动完成", extra={"status": ctx["worker_id"]})


async def on_shutdown(ctx: dict[str, Any]) -> None:
    """优雅停机。

    arq 在收到停机信号后会停止领取新任务并等待**在跑的任务**结束，
    因此这里只负责释放连接。注意：强制停机（SIGKILL / 容器被 kill -9）
    走到不了这里，那种情况下的 RUNNING 任务由孤儿回收接手——
    两条路径都必须存在，少一条就会留下永远停在 RUNNING 的任务。
    """
    logger.info("worker 正在退出，等待在跑任务结束", extra={"status": ctx.get("worker_id")})
    queue: ArqJobQueue | None = ctx.get("queue")
    if queue is not None:
        await queue.aclose()
    redis: aioredis.Redis | None = ctx.get("redis")
    if redis is not None:
        await redis.aclose()
    engine: AsyncEngine | None = ctx.get("db_engine")
    if engine is not None:
        await engine.dispose()
    # 先关图（它会关掉两个工具），再关网关：工具持有网关的引用，
    # 顺序反了会在已关闭的网关上发请求，而那个报错指向的是"连接被关闭"
    # 而不是"生命周期写错了"。
    task_graph: TaskGraph | None = ctx.get("task_graph")
    if task_graph is not None:
        await task_graph.aclose()
    else:
        gateway = ctx.get("model_gateway")
        if gateway is not None:
            await gateway.aclose()
    clear_context()


def _schedule(coro: Any, *, name: str, seconds: int) -> Any:
    """按周期生成一个定时任务。

    arq 的 cron 用「在第几秒 / 第几分钟触发」表达周期，因此要把间隔展开成集合。
    分两种写法而不是拼一个 dict 再展开：间隔一旦大于等于 60 秒，
    `set(range(0, 60, seconds))` 就是**空集**，定时任务会静默地永不触发——
    这种错在运行时不报任何警，只表现为「回收从来没跑过」。
    """
    if seconds < 60:
        return cron(coro, name=name, second=set(range(0, 60, seconds)), max_tries=1, timeout=30)
    return cron(
        coro, name=name, minute=set(range(0, 60, max(1, seconds // 60))), max_tries=1, timeout=30
    )


def _cron_jobs(settings: Settings) -> list[Any]:
    return [
        # 孤儿回收要在心跳过期之后尽快跑，因此与心跳 TTL 同量级
        _schedule(
            reclaim_orphans,
            name="reclaim_orphans",
            seconds=settings.worker.orphan_scan_interval_seconds,
        ),
        _schedule(
            reconcile_queue,
            name="reconcile_queue",
            seconds=settings.worker.queue_reconcile_seconds,
        ),
    ]


#: 模块级单例：`arq` 在导入本模块时就读取 `WorkerSettings` 的类属性，
#: 而 `get_settings()` 是 lru_cache 的，这里读一次与运行时读到的是同一份。
_settings = get_settings()


class WorkerSettings:
    """`arq app.worker.WorkerSettings` 的入口。"""

    functions: ClassVar[list[Any]] = [run_agent_task]
    cron_jobs: ClassVar[list[Any]] = _cron_jobs(_settings)
    #: 必须与投递端指定的队列名一致（详细设计 4.4 的 `q:agent`）。
    #: arq 的默认名是 `arq:queue`，两边不一致的话任务会投进一个没人监听的队列。
    queue_name: ClassVar[str] = RedisKey.queue()
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings: ClassVar[RedisSettings] = RedisSettings.from_dsn(_settings.redis_url)
    #: 任务超时与 loop.max_expansions 联动（开发流程 5.4）
    job_timeout: ClassVar[int] = _settings.task_timeout_seconds
    max_jobs: ClassVar[int] = _settings.worker.concurrency
    keep_result: ClassVar[int] = 3600
    #: arq 默认（0）意味着收到停机信号立刻取消在跑的任务。给一小段收尾时间，
    #: 但**不取 task_timeout**：那会让 `Ctrl-C` 之后开发机要等十分钟才退出。
    #: 超过这段时间仍未结束的任务被中断，由孤儿回收接手——两条路径都在。
    job_completion_wait: ClassVar[int] = _settings.worker.shutdown_grace_seconds


#: arq 按 `functions` 里的函数名查找任务，投递端用的 `TASK_JOB_NAME` 必须
#: 与函数名一致。不一致时不会有任何报错，只会表现为「任务入队了但永远没人执行」，
#: 因此这里在导入期就把它变成一次显式失败。
assert run_agent_task.__name__ == TASK_JOB_NAME, (
    f"任务函数名 {run_agent_task.__name__} 与投递端常量 {TASK_JOB_NAME} 不一致"
)

#: 重启退避的上限。Redis 长时间不可用时持续以最大间隔重试，
#: 既不会放弃，也不会把日志刷爆。
_MAX_RESTART_BACKOFF_SECONDS = 30.0


async def run_forever() -> None:
    """Worker 主循环：**依赖抖动时重启自己，而不是让进程死掉**。

    为什么需要这一层：arq 的轮询循环（`_poll_iteration`）没有任何异常保护，
    Redis 一断，`zrangebyscore` 抛出的 `ConnectionError` 会一路冒到主循环外，
    进程直接退出。之后即使 Redis 恢复，也**没有任何人来消费队列**——
    任务全部堆在 `q:agent` 里，而健康检查、日志、监控都看不出异常
    （进程没了，不是进程坏了）。这正是「Redis 不可用时降级而非不可用」
    要挡住的那类失败。

    停机信号（SIGTERM/SIGINT）走 `CancelledError`，**原样上抛**：
    那是正常退出，不该被当成故障重启。
    """
    backoff = 1.0
    while True:
        ctx: dict[str, Any] = {}
        worker = Worker(
            functions=WorkerSettings.functions,
            cron_jobs=WorkerSettings.cron_jobs,
            queue_name=WorkerSettings.queue_name,
            redis_settings=WorkerSettings.redis_settings,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            ctx=ctx,
            max_jobs=WorkerSettings.max_jobs,
            job_timeout=WorkerSettings.job_timeout,
            keep_result=WorkerSettings.keep_result,
            job_completion_wait=WorkerSettings.job_completion_wait,
        )
        try:
            await worker.async_run()
            return
        except asyncio.CancelledError:
            raise
        except (aioredis.RedisError, OSError) as exc:
            logger.warning(
                "Worker 因依赖不可用而中断，%.0f 秒后重启：%s",
                backoff,
                type(exc).__name__,
                extra={"status": "RESTARTING"},
            )
            # 尽力释放上一轮的连接。Redis 已经不可用时 aclose 也会失败，
            # 那没关系——进程要重建连接池，旧连接会被操作系统回收。
            with contextlib.suppress(Exception):
                await on_shutdown(ctx)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_RESTART_BACKOFF_SECONDS)


if __name__ == "__main__":
    # `make worker` 走这里而不是 arq 的命令行入口：命令行入口没有重启外壳，
    # Redis 一断 Worker 就永久消失（见 run_forever 的说明）。
    asyncio.run(run_forever())
