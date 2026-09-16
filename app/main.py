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

from app.api import auth, chat, tasks
from app.api.errors import register_exception_handlers
from app.api.middleware import TraceContextMiddleware
from app.core.config import Settings, get_settings
from app.infrastructure.logging import SERVICE_API, clear_context, setup_logging
from app.infrastructure.observability import setup_error_tracking, setup_observability
from app.infrastructure.queue import ArqJobQueue, JobQueue
from app.infrastructure.redis import create_client
from app.repositories.conversation_repo import InMemoryConversationRepository
from app.repositories.task_repo import RedisTaskRepository
from app.repositories.user_repo import InMemoryUserRepository, seed_demo_users
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


def wire_dependencies(app: FastAPI, settings: Settings, *, queue: JobQueue) -> None:
    """装配服务依赖。

    **所有替换点都收敛在这一个函数里**，这是那些 `Protocol` 存在的唯一理由。
    Phase 1.5 的状态：

    - 任务仓储 → Redis（临时实现，Phase 2 换 MySQL）
    - 限流器 → Redis + Lua，带本地降级
    - 队列 / 事件总线 → arq / Redis Stream

    - 用户与会话仓储 → **仍是进程内占位**，Phase 2 换 MySQL。

    `queue` 从外部传入而不是在这里构造：API 进程用 arq 连接池，
    测试用替身，两者的生命周期归属不同（前者属于 lifespan，后者属于夹具），
    在这里 new 一个会让测试无法观察到投递行为。
    """
    redis: aioredis.Redis = app.state.redis

    user_repo = InMemoryUserRepository(seed_demo_users(settings))
    conversation_repo = InMemoryConversationRepository()
    task_repo = RedisTaskRepository(redis, settings)

    runner = TaskRunner(
        tasks=task_repo,
        queue=queue,
        events=RedisStreamEventBus(redis, settings),
        settings=settings,
        redis=redis,
    )

    app.state.user_repo = user_repo
    app.state.conversation_repo = conversation_repo
    app.state.task_repo = task_repo
    app.state.task_runner = runner
    app.state.job_queue = queue
    app.state.rate_limiter = RedisFixedWindowLimiter(redis=redis, settings=settings)
    app.state.auth_service = AuthService(user_repo, settings)
    app.state.task_service = TaskService(
        tasks=task_repo, conversations=conversation_repo, settings=settings, runner=runner
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    setup_logging(SERVICE_API, settings.log_level)
    setup_observability(settings, service_name=SERVICE_API)
    setup_error_tracking(settings)

    redis = create_client(settings)
    app.state.redis = redis
    queue = await ArqJobQueue.create(settings)
    wire_dependencies(app, settings, queue=queue)
    try:
        yield
    finally:
        clear_context()
        await queue.aclose()
        await redis.aclose()


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

        **Redis 不可用返回 200 + `degraded`，不是 503。** 这不是放松要求：
        Phase 1.5 已经实现了降级路径（限流退回本地令牌桶、投递失败由补偿扫描
        兜底），API 在 Redis 挂掉时仍能对外服务。此时报 not_ready 会让编排层
        摘掉这个实例，而摘掉它恰恰是最不该做的事——降级路径本来就是为了
        「Redis 挂了也要撑住」而存在的。真正不可恢复的依赖（Phase 2 的 MySQL）
        届时按 critical 处理，返回 503。

        MySQL / Milvus 的就绪检查随 Phase 2 / Phase 5 一并接入。
        """
        redis_client: aioredis.Redis = request.app.state.redis
        failures: list[str] = []
        try:
            await redis_client.ping()
        except Exception as exc:
            failures.append(f"redis({type(exc).__name__})")

        if failures:
            return JSONResponse(
                status_code=200,
                content={"status": "degraded", "degraded": failures, "checked": ["redis"]},
            )
        return JSONResponse(status_code=200, content={"status": "ready", "checked": ["redis"]})

    return app


app = create_app()
