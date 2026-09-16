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
from app.infrastructure.logging import clear_context, setup_logging
from app.repositories.conversation_repo import InMemoryConversationRepository
from app.repositories.task_repo import InMemoryTaskRepository
from app.repositories.user_repo import InMemoryUserRepository, seed_demo_users
from app.services.auth_service import AuthService
from app.services.rate_limit import InProcessTokenBucket
from app.services.task_service import TaskService

#: OpenAPI 的接口分组（开发流程 6.2 施工项 8）
OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "认证", "description": "本地账号登录与令牌签发（TBC-08 决议）"},
    {"name": "任务", "description": "分析任务的创建与查询"},
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


def wire_dependencies(app: FastAPI, settings: Settings) -> None:
    """装配服务依赖。

    Phase 1 全部是**进程内占位实现**（重启即丢，多实例互不可见）。
    Phase 1.5 接 Redis 限流与任务队列、Phase 2 接 MySQL 仓储时，
    **只改这个函数**——服务层、路由层与 `Protocol` 定义都不用动。
    把替换面收敛到一处，是这些 `Protocol` 存在的唯一理由。
    """
    user_repo = InMemoryUserRepository(seed_demo_users(settings))
    conversation_repo = InMemoryConversationRepository()
    task_repo = InMemoryTaskRepository()

    app.state.user_repo = user_repo
    app.state.conversation_repo = conversation_repo
    app.state.task_repo = task_repo
    app.state.rate_limiter = InProcessTokenBucket()
    app.state.auth_service = AuthService(user_repo, settings)
    app.state.task_service = TaskService(
        tasks=task_repo, conversations=conversation_repo, settings=settings
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    setup_logging(settings.otel_service_name, settings.log_level)

    # redis-py 未给 from_url 标注类型，只能就地豁免
    redis = aioredis.from_url(  # type: ignore[no-untyped-call]
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    app.state.redis = redis
    wire_dependencies(app, settings)
    try:
        yield
    finally:
        clear_context()
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

    @app.get("/health/live", tags=["health"], summary="存活探针")
    async def health_live() -> dict[str, str]:
        """进程是否还在。不检查任何外部依赖。"""
        return {"status": "ok", "service": settings.otel_service_name}

    @app.get("/health/ready", tags=["health"], summary="就绪探针")
    async def health_ready(request: Request) -> Response:
        """关键外部依赖是否可用。

        Phase 0 只校验 Redis（本阶段唯一被 API 进程依赖的外部组件）；
        MySQL / Milvus 的就绪检查随 Phase 2 / Phase 5 一并接入。
        """
        redis_client: aioredis.Redis = request.app.state.redis
        try:
            await redis_client.ping()
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "failed": ["redis"], "detail": type(exc).__name__},
            )
        return JSONResponse(status_code=200, content={"status": "ready", "checked": ["redis"]})

    return app


app = create_app()
