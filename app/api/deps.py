"""依赖注入装配点。

**换实现只改这里**：Phase 1 的仓储与限流器都是进程内占位实现，
Phase 1.5 / Phase 2 换成 Redis 与 MySQL 时，只需要把 `app/main.py` 里的装配
换成新实现，本文件与路由层不动。这是 `Protocol` + 依赖注入存在的唯一理由，
不是为了让目录看起来更整齐。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.errors import AgentError, ErrorCode
from app.domain.user import User
from app.services.auth_service import AuthService
from app.services.rate_limit import RateLimiter
from app.services.task_service import TaskService

#: `auto_error=False`：HTTPBearer 默认在缺少 Authorization 头时抛 403，
#: 而需求规格 7.3 规定「未认证或令牌失效」必须是 401。开启自动报错就拿不到正确语义，
#: 因此关掉它、统一由下面自己抛。
_bearer = HTTPBearer(auto_error=False, description="Bearer <access_token>")


def get_trace_id(request: Request) -> str:
    """本次请求的 trace_id。

    取不到就直接报错而不是生成一个新的：那会让响应里的 trace_id
    与日志、Trace 里的对不上，排查时比没有 ID 更误导人。
    """
    trace_id = getattr(request.state, "trace_id", None)
    if not isinstance(trace_id, str) or not trace_id:
        raise AgentError(ErrorCode.INTERNAL_ERROR, "追踪上下文缺失，请检查中间件装配")
    return trace_id


def get_auth_service(request: Request) -> AuthService:
    service: AuthService = request.app.state.auth_service
    return service


def get_task_service(request: Request) -> TaskService:
    service: TaskService = request.app.state.task_service
    return service


def get_rate_limiter(request: Request) -> RateLimiter:
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


SettingsDep = Annotated[Settings, Depends(get_settings)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
RateLimiterDep = Annotated[RateLimiter, Depends(get_rate_limiter)]
TraceIdDep = Annotated[str, Depends(get_trace_id)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    auth: AuthServiceDep,
) -> User:
    if credentials is None:
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "缺少访问令牌")
    return await auth.authenticate(credentials.credentials)


CurrentUser = Annotated[User, Depends(get_current_user)]


def rate_limit_dependency(
    scope: str, limit_of: Callable[[Settings], int], window_seconds: int
) -> Callable[..., Awaitable[None]]:
    """生成一个限流依赖。

    `limit_of` 取配置而不是直接取整数，是为了让依赖在**请求时**读配置——
    测试里改配置不需要重建路由。

    限流排在认证之后（登录用户才有 subject），因此鉴权失败永远是 401 而不是 429。
    """

    async def dependency(
        request: Request,
        response: Response,
        user: CurrentUser,
        settings: SettingsDep,
        limiter: RateLimiterDep,
    ) -> None:
        result = await limiter.check(
            scope=scope,
            subject=user.id,
            limit=limit_of(settings),
            window_seconds=window_seconds,
        )
        headers = result.headers()
        response.headers.update(headers)
        # 抛异常时 FastAPI 不再合并注入式 Response 的头部，
        # 因此把头部同时暂存在 request.state，由全局异常处理器补回（见 api/errors.py）
        request.state.rate_limit_headers = headers

        if not result.allowed:
            raise AgentError(
                ErrorCode.RATE_LIMITED,
                f"请求过于频繁，请 {result.reset_after_seconds} 秒后重试",
            )

    return dependency


#: 创建任务 10 次/分钟、状态查询 120 次/分钟（详细设计 19.3）
require_create_task_rate_limit = rate_limit_dependency(
    "task:create", lambda s: s.rate_limit_create_per_minute, 60
)
require_task_status_rate_limit = rate_limit_dependency(
    "task:status", lambda s: s.rate_limit_status_per_minute, 60
)
