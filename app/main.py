"""API 进程入口。

**分层硬约束（开发流程 5.2）**：本模块及 `app/api/` 下任何模块，
不得 import `agent.graph`、`agent.nodes.*`、`tools.*`。
破坏后 API 进程会加载 LangGraph 与全部 Tool，启动变慢、内存翻倍。
该约束由 `scripts/check_layering.py` 在 CI 中强制。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api import auth, chat, tasks
from app.api.errors import register_exception_handlers
from app.api.middleware import TraceContextMiddleware
from app.core.config import Settings, get_settings
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import SERVICE_API, clear_context, setup_logging
from app.infrastructure.model_gateway import ModelGateway, build_model_gateway
from app.infrastructure.observability import setup_error_tracking, setup_observability
from app.infrastructure.queue import ArqJobQueue, JobQueue
from app.infrastructure.redis import create_client
from app.repositories import Repositories, build_sql_repositories
from app.repositories.agent_repo import SqlAgentArtifactRepository
from app.services.auth_service import AuthService
from app.services.event_bus import RedisStreamEventBus
from app.services.rate_limit import RedisFixedWindowLimiter
from app.services.task_runner import TaskRunner
from app.services.task_service import TaskService

#: OpenAPI 的接口分组（开发流程 6.2 施工项 8）
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "认证", "description": "本地账号登录与令牌签发（TBC-08 决议）"},
    {"name": "任务", "description": "分析任务的创建、查询与取消"},
    {"name": "health", "description": "存活与就绪探针"},
]

OPENAPI_DESCRIPTION = """\
把自然语言问题转化为可追溯的数据结论的多 Agent 系统。

**统一响应外壳**：所有接口（含错误）都返回
`{code, message, data, trace_id, retryable}`。
`code` 为 `OK` / `ACCEPTED` 表示成功，其余取值见详细设计 19.1 的错误码表。

**追踪**：每个请求都会生成 `trc_` 前缀的 trace_id，
响应体的 `trace_id` 字段与 `X-Trace-Id` 响应头一致，可直接用于日志检索。
"""


def wire_dependencies(
    app: FastAPI,
    settings: Settings,
    *,
    queue: JobQueue,
    gateway: ModelGateway,
    repositories: Repositories | None = None,
) -> None:
    """装配服务依赖。

    **所有替换点都收敛在这一个函数里**，这是那些 `Protocol` 存在的唯一理由。
    Phase 2 的状态：

    - 用户 / 会话 / 任务仓储 → **MySQL**（`agent_task` 是任务状态的唯一权威）
    - 限流器 → Redis + Lua，带本地降级
    - 队列 / 事件总线 → arq / Redis Stream
    - 取消信号 / 重投计数 / 心跳快通道 → 仍在 Redis（是跨进程信号，不是任务存储）

    **`queue`、`gateway` 与 `repositories` 都从外部传入**，理由相同：它们的
    生命周期不属于本函数。`queue` 在 API 进程里是 arq 连接池、在测试里是替身；
    仓储在进程里连 MySQL、在单元测试里是内存实现；模型网关持有两个 httpx
    连接池，**必须由 lifespan 成对创建与释放**。在这里 new 死一个，
    测试要么连不上库，要么观察不到投递行为，要么留下没人关的连接池。

    `gateway` 目前**还没有业务消费者**——Phase 4 的 SQL Tool 是第一个。
    现在接进来是为了让「模型调用可被替身替换」这条门禁在 Phase 3 就有实证，
    而不是等到 Phase 4 才发现替换点没留。

    默认值 `repositories=None` 时才构造 MySQL 实现——**单元测试必须显式
    注入内存实现**，因为「测试跑不跑得起来」取决于开发机上有没有起 MySQL，
    是不能接受的：`make test` 的契约是「不依赖任何外部组件」。
    """
    redis: aioredis.Redis = app.state.redis
    repos = repositories if repositories is not None else build_sql_repositories(app.state.sessions)

    runner = TaskRunner(
        tasks=repos.tasks,
        queue=queue,
        events=RedisStreamEventBus(redis, settings),
        settings=settings,
        redis=redis,
        # **API 进程也要装配它**，虽然它自己不跑任务：`TaskRunner` 的构造签名
        # 对两条进程是同一份，差别只在 body（API 不传）。让 API 传 None
        # 会诱使 `_persist_artifacts` 长出"仓储不存在就跳过"的分支，
        # 而那条分支在 Worker 里永远为假——测试不出来的死代码。
        artifacts=SqlAgentArtifactRepository(app.state.sessions),
    )

    app.state.repositories = repos
    app.state.user_repo = repos.users
    app.state.conversation_repo = repos.conversations
    app.state.task_repo = repos.tasks
    app.state.task_runner = runner
    app.state.job_queue = queue
    app.state.model_gateway = gateway
    app.state.rate_limiter = RedisFixedWindowLimiter(redis=redis, settings=settings)
    app.state.auth_service = AuthService(repos.users, settings)
    app.state.task_service = TaskService(
        tasks=repos.tasks, conversations=repos.conversations, settings=settings, runner=runner
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    setup_logging(SERVICE_API, settings.log_level)
    setup_observability(settings, service_name=SERVICE_API)
    setup_error_tracking(settings)

    redis = create_client(settings)
    app.state.redis = redis

    # 连接池在 lifespan 内建、在退出时释放。不放进 wire_dependencies：
    # 那里是**同步**装配函数，而建 engine 会解析驱动并可能触发连接，
    # 把它塞进去会让「装配」这一步变得可能失败，测试也无法只替换仓储。
    db_engine = create_engine(settings)
    app.state.db_engine = db_engine
    app.state.sessions = create_session_factory(db_engine)

    queue = await ArqJobQueue.create(settings)
    # 模型网关持有两个 httpx 连接池（chat 与 embedding 各一个），
    # 归本进程所有，退出时在 finally 里释放。
    gateway = build_model_gateway(settings)
    wire_dependencies(app, settings, queue=queue, gateway=gateway)
    try:
        yield
    finally:
        clear_context()
        await gateway.aclose()
        await queue.aclose()
        await redis.aclose()
        await db_engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    # 不指定 default_response_class：FastAPI 现在直接用 Pydantic 的 Rust 核心
    # 序列化到 JSON 字节，比挂 orjson 响应类更快，且 ORJSONResponse 已废弃。
    app = FastAPI(
        title="企业智能数据分析与决策 Agent",
        version="0.1.0",
        description=OPENAPI_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    # 中间件必须在注册异常处理器之前挂上：异常处理器要靠它写进 scope 的 trace_id
    app.add_middleware(TraceContextMiddleware)
    register_exception_handlers(app)

    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(tasks.router)

    # FastAPI 自动埋点在本文件里调用，**不能放进 infrastructure/observability.py**：
    # 那个模块会被 app/worker.py 引用，一旦它 import 了 instrumentation-fastapi，
    # Worker 进程就会跟着把 fastapi 加载进内存，L2「Worker 不得依赖 Web 框架」
    # 在实质上失效，而分层检查器只看 worker.py 自己的 import，查不出来。
    if settings.otel_enabled:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)

    @app.get("/health/live", tags=["health"], summary="存活探针")
    async def health_live() -> dict[str, str]:
        """进程是否还在。不检查任何外部依赖。"""
        return {"status": "ok", "service": SERVICE_API}

    @app.get("/health/ready", tags=["health"], summary="就绪探针")
    async def health_ready(request: Request) -> Response:
        """关键外部依赖是否可用。

        **两个依赖的处理刻意不同**：

        - **Redis 不可用返回 200 + `degraded`，不是 503。** 这不是放松要求：
          Phase 1.5 已经实现了降级路径（限流退回本地令牌桶、投递失败由补偿扫描
          兜底），API 在 Redis 挂掉时仍能对外服务。此时报 not_ready 会让编排层
          摘掉这个实例，而摘掉它恰恰是最不该做的事——降级路径本来就是为了
          「Redis 挂了也要撑住」而存在的。
        - **MySQL 不可用返回 503。** Phase 2 起它是任务、会话、用户的**唯一权威**，
          没有任何降级路径：库连不上时登录、建任务、查状态全部失败，
          此时如实报 not_ready 让编排层摘掉实例才是对的。

        Milvus 的就绪检查随 Phase 5 一并接入。
        """
        redis_client: aioredis.Redis = request.app.state.redis
        sessions: async_sessionmaker[AsyncSession] = request.app.state.sessions

        try:
            async with sessions() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "failed": [f"mysql({type(exc).__name__})"],
                    "checked": ["mysql"],
                },
            )

        degraded: list[str] = []
        try:
            await redis_client.ping()
        except Exception as exc:
            degraded.append(f"redis({type(exc).__name__})")

        if degraded:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "degraded",
                    "degraded": degraded,
                    "checked": ["mysql", "redis"],
                },
            )
        return JSONResponse(
            status_code=200, content={"status": "ready", "checked": ["mysql", "redis"]}
        )

    return app


app = create_app()
