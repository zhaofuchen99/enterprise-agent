"""依赖注入装配点。

**换实现只改这里**：Phase 1 的仓储与限流器都是进程内占位实现，
Phase 1.5 / Phase 2 换成 Redis 与 MySQL 时，只需要把 `app/main.py` 里的装配
换成新实现，本文件与路由层不动。这是 `Protocol` + 依赖注入存在的唯一理由，
不是为了让目录看起来更整齐。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.errors import AgentError, ErrorCode
from app.domain.user import User
from app.repositories.agent_repo import AgentArtifactRepository
from app.repositories.user_repo import UserRepository
from app.services.auth_service import AuthService
from app.services.event_bus import EventBus
from app.services.rate_limit import RateLimiter
from app.services.stream_token import StreamTokenService
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


def get_artifacts(request: Request) -> AgentArtifactRepository:
    """执行产出仓储（16.6 / 16.7）。

    17.4 的轨迹接口直接读表而不是读 `result_json`——**那张表是权威重放来源**
    （18.3），而 `result_json` 只是给任务详情的一个快照。
    """
    artifacts: AgentArtifactRepository = request.app.state.artifacts
    return artifacts


ArtifactsDep = Annotated[AgentArtifactRepository, Depends(get_artifacts)]


def get_event_bus(request: Request) -> EventBus:
    """任务事件总线（18.1）。**与 TaskRunner 手里那份是同一个对象**（见 main.py）。

    17.3 的 SSE 端点读它；它与 Worker 写的是同一条 Redis Stream，
    因此订阅请求落在哪个 API 实例都一样（FR-SSE-001 业务规则 4）。
    """
    bus: EventBus = request.app.state.event_bus
    return bus


def get_stream_tokens(request: Request) -> StreamTokenService:
    service: StreamTokenService = request.app.state.stream_tokens
    return service


EventBusDep = Annotated[EventBus, Depends(get_event_bus)]
StreamTokenDep = Annotated[StreamTokenService, Depends(get_stream_tokens)]
TraceIdDep = Annotated[str, Depends(get_trace_id)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    auth: AuthServiceDep,
) -> User:
    if credentials is None:
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "缺少访问令牌")
    return await auth.authenticate(credentials.credentials)


CurrentUser = Annotated[User, Depends(get_current_user)]


@dataclass(frozen=True, slots=True)
class StreamPrincipal:
    """SSE 的凭据持有者。**两条来路**（详设 17.3.1）。

    Attributes:
        user: 解析出来的用户。
        token_task_id: 走订阅令牌时的绑定任务；走请求头时为 None。
            调用方**必须**拿它跟路径里的 task_id 比对——令牌本身就是
            "只对这一条流有效"的承诺，不比对等于把承诺丢了。
    """

    user: User
    token_task_id: str | None = None


async def get_stream_principal(
    request: Request,
    auth: AuthServiceDep,
    tokens: StreamTokenDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> StreamPrincipal:
    """SSE 的鉴权：**优先请求头，其次查询参数里的订阅令牌**。

    浏览器只能用后者（`EventSource` 设不了请求头）；而命令行、SSE 客户端
    这类能带头的调用方走前者更直接——两条都留，是因为"只支持 token 查询参数"
    会逼着所有非浏览器调用方多跑一趟换令牌。

    **查询参数里只接受订阅令牌**：它短时效、一次性、绑定任务；
    把 Access Token 放查询参数是详设 17.3.1 明令禁止的（会进访问日志）。
    因此这里不校验"token 参数长得像不像 access token"——它压根不进那条分支。
    """
    if credentials is not None:
        return StreamPrincipal(user=await auth.authenticate(credentials.credentials))

    raw = request.query_params.get("token")
    if not raw:
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "缺少访问令牌或订阅令牌")
    claims = await tokens.consume(raw)
    user_repo: UserRepository = request.app.state.user_repo
    user = await user_repo.get_by_id(claims.user_id)
    if user is None or not user.is_active:
        # 令牌是签给一个已不存在/已停用的用户的（签发后发生的）。**按认证失败处理**：
        # 这里没有"他还能看到什么"的余地。签发时读过用户，但那是几十秒前的事，
        # 而这条路径的代价只有"重新登录拿令牌"
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "订阅令牌对应的用户不可用")
    return StreamPrincipal(user=user, token_task_id=claims.task_id)


StreamPrincipalDep = Annotated[StreamPrincipal, Depends(get_stream_principal)]


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
