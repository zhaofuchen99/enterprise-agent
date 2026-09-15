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

from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.infrastructure.logging import clear_context, setup_logging

_redis: aioredis.Redis | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _redis

    settings: Settings = get_settings()
    setup_logging(settings.otel_service_name, settings.log_level)
    # redis-py 未给 from_url 标注类型，只能就地豁免
    _redis = aioredis.from_url(  # type: ignore[no-untyped-call]
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    app.state.redis = _redis
    try:
        yield
    finally:
        clear_context()
        await _redis.aclose()
        _redis = None


def create_app() -> FastAPI:
    settings = get_settings()
    # 不指定 default_response_class：FastAPI 现在直接用 Pydantic 的 Rust 核心
    # 序列化到 JSON 字节，比挂 orjson 响应类更快，且 ORJSONResponse 已废弃。
    app = FastAPI(
        title="企业智能数据分析与决策 Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(AgentError)
    async def _agent_error_handler(_: Request, exc: AgentError) -> JSONResponse:
        """AgentError -> HTTP 映射。用户可见信息只有 code/message，
        堆栈与 details 只进日志（详细设计 19.1）。"""
        return JSONResponse(
            status_code=exc.http_status,
            content={"code": exc.code.value, "message": exc.message, "retryable": exc.retryable},
        )

    @app.get("/health/live", tags=["health"])
    async def health_live() -> dict[str, str]:
        """存活探针：进程是否还在。不检查任何外部依赖。"""
        return {"status": "ok", "service": settings.otel_service_name}

    @app.get("/health/ready", tags=["health"])
    async def health_ready(request: Request) -> Response:
        """就绪探针：关键外部依赖是否可用。

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
