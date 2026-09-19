"""SSE 订阅令牌（详细设计 17.3.1）。

## 为什么不能直接用 Access Token

浏览器的 `EventSource` **无法自定义请求头**，所以 SSE 的连接参数只能走 URL。
而 URL 会进访问日志、浏览器历史、Referer——把长期有效的 Access Token 放进去，
等于把它泄露给每一个能看到日志的人。

于是拆成两步：用**请求头**里的 Access Token 换一个**短时效、一次性、
只对这一个任务有效**的订阅令牌，再把它放进查询参数。

- **短时效**（`SSE_TUNING__STREAM_TOKEN_TTL_SECONDS`，默认 60 秒）：
  泄露的窗口只有一分钟，而且它多半已经被用掉了；
- **一次性**：用掉的令牌立刻作废（Redis 里的标记被 `GETDEL` 取走），
  于是"日志里翻到一条旧 URL"拿不到任何东西；
- **限定 task_id**：签名里带着任务 ID，换一个任务路径使用会被拒——
  否则它就是一个"能读任意任务事件流"的通行证。

## 拒绝时**不区分原因**

过期、已用过、签名错、任务对不上——四种都返回同一个
`AUTHENTICATION_REQUIRED`（401）。这与 `decode_access_token` 是同一条纪律：
能区分的报错等于告诉探测者"这个令牌曾经有效"，而对他排查自己的问题
并没有多出什么信息（我们这边有日志）。

**唯一的例外是任务不匹配**：令牌是好的、只是用在了别的任务上，
返回 `ACCESS_DENIED`（403）。这一条不泄露任何东西——令牌本来就只对
持有者可见，而"拿自己的令牌读别人的任务"必须留下明确的痕迹。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import redis.asyncio as aioredis
from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.core.ids import IdPrefix, new_id
from app.infrastructure.redis import RedisKey

logger = logging.getLogger(__name__)


class IssuedStreamToken(BaseModel):
    """签发的订阅令牌（17.3.1 的响应体）。"""

    token: str
    expires_in: int
    #: 带令牌的完整路径。**由服务端拼**而不是让前端拼：
    #: 前端拼的话，"令牌放哪个参数名"就成了一个没有文档的约定，
    #: 改一次客户端就静默失效（连接返回 401，而看不出是参数名错了）
    stream_url: str


class StreamTokenClaims(BaseModel):
    """验过的订阅令牌。`task_id` 是**绑定关系**，调用方必须拿它跟路径里的比对。"""

    user_id: str
    task_id: str
    expires_at: datetime


class StreamTokenService:
    """签发与消费订阅令牌。依赖 Redis（一次性标记）与配置（密钥/有效期）。"""

    def __init__(self, *, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._settings = settings
        self._ttl = settings.sse_tuning.stream_token_ttl_seconds

    async def issue(self, *, user_id: str, task_id: str) -> IssuedStreamToken:
        """签发一个只对 `task_id` 有效的订阅令牌。"""
        now = datetime.now(UTC)
        jti = new_id(IdPrefix.STREAM_TOKEN)
        payload: dict[str, Any] = {
            "sub": user_id,
            "task_id": task_id,
            "jti": jti,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=self._ttl)).timestamp()),
        }
        token = jwt.encode(
            payload, self._settings.jwt_secret, algorithm=self._settings.jwt_algorithm
        )
        # 一次性标记。**TTL 与令牌一致**：令牌过期后这个键也自然消失，
        # 不需要任何清理任务，也不会让"过期的令牌"和"未知的令牌"在 Redis 里长得不同
        await self._redis.set(RedisKey.stream_token(jti), task_id, ex=self._ttl)
        return IssuedStreamToken(
            token=token,
            expires_in=self._ttl,
            stream_url=f"/api/agent/tasks/{task_id}/stream?token={token}",
        )

    async def consume(self, token: str) -> StreamTokenClaims:
        """校验并**用掉**令牌。第二次调用同一个令牌必然失败。

        Raises:
            AgentError: `AUTHENTICATION_REQUIRED`（无效/过期/已用），
                见模块 docstring 的"不区分原因"。
        """
        claims = _decode(token, self._settings)
        # **GETDEL 而不是 GET + DEL**：后者在并发下会让同一个令牌被消费两次
        # （两个请求都 GET 到值、都通过），而"一次性"正是这个设计的全部意义。
        # 这一条是原子的，第二个请求拿到 None。
        used_for = await self._redis.getdel(RedisKey.stream_token(claims.jti))
        if used_for is None:
            logger.warning("订阅令牌不可用（过期或已被使用）", extra={"task_id": claims.task_id})
            raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "订阅令牌无效或已过期，请重新获取")
        return StreamTokenClaims(
            user_id=claims.user_id,
            task_id=claims.task_id,
            expires_at=claims.expires_at,
        )


class _RawClaims(BaseModel):
    """JWT 解出来的原始载荷（`jti` 只在消费时用，不进对外的 `StreamTokenClaims`）。"""

    user_id: str
    task_id: str
    jti: str
    expires_at: datetime


def _decode(token: str, settings: Settings) -> _RawClaims:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "iat", "sub", "task_id", "jti"]},
        )
    except jwt.InvalidTokenError as exc:
        # 过期、签名错、字段缺失都落这里——**对外是同一个 401**（见模块 docstring）
        raise AgentError(
            ErrorCode.AUTHENTICATION_REQUIRED, "订阅令牌无效或已过期，请重新获取"
        ) from exc

    user_id, task_id, jti = payload.get("sub"), payload.get("task_id"), payload.get("jti")
    if not all(isinstance(value, str) and value for value in (user_id, task_id, jti)):
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "订阅令牌无效或已过期，请重新获取")
    return _RawClaims(
        user_id=user_id,
        task_id=task_id,
        jti=jti,
        expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=UTC),
    )


__all__ = ["IssuedStreamToken", "StreamTokenClaims", "StreamTokenService"]
