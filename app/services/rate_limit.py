"""限流。

两套实现，靠 `RateLimiter` 协议对齐：

- `RedisFixedWindowLimiter` —— **生产实现**，Redis + Lua 固定窗口计数
  （详细设计 19.3）。多实例共享同一份计数，重启不重置。
- `InProcessTokenBucket` —— 两个身份：Phase 1 的占位实现，
  以及 Redis 不可用时 `RedisFixedWindowLimiter` 的**降级后端**。

**降级为什么不是简单放行**：Redis 挂掉时若直接放行，限流就从「保护」变成了
「攻击者只要把 Redis 打挂就拿到无限配额」。因此降级路径把配额按实例数均分后
**收紧**（`rate_limit_degraded_factor`，0.5 即等价于按 2 个实例均分），
并置 `degraded=True` 让客户端知道当前的额度不是全量。

**降级路径里最容易被忽略的一环是熔断**：Redis 挂掉后每个请求都要先等满一次
命令超时才走降级，1 秒的超时就是每个请求 1 秒的延迟——降级保住了正确性，
却把延迟打穿了。每次失败后冷却一段时间直接走降级，才是真的「不可用时仍可用」。
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import redis.asyncio as aioredis

from app.core.config import Settings
from app.infrastructure.redis import RedisKey, register_scripts

logger = logging.getLogger(__name__)

#: 这些异常都意味着「Redis 这次没帮上忙」，而不是「用户超限了」。
#: 漏掉任何一个都会让一次 Redis 抖动变成一次 500。
_REDIS_FAILURES = (aioredis.RedisError, OSError, asyncio.TimeoutError)


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    #: 距窗口重置还有多少秒——即客户端最早可以重试的时刻
    reset_after_seconds: int
    degraded: bool = False

    def headers(self) -> dict[str, str]:
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_after_seconds),
            "X-RateLimit-Degraded": "true" if self.degraded else "false",
        }


class RateLimiter(Protocol):
    async def check(
        self, *, scope: str, subject: str, limit: int, window_seconds: int
    ) -> RateLimitResult: ...


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class InProcessTokenBucket:
    """按 `(scope, subject)` 分桶的进程内令牌桶。

    桶数量受「已认证用户数 × 限流档位数」约束，不会因请求量增长；
    进程重启即清空——这正是生产路径要换掉它的原因（重启不该重置配额）。

    作为降级后端时，它的**平滑**特性是一个副作用优点：固定窗口在窗口边界
    会放行两倍突发，而令牌桶按恒定速率补充，降级期间的流量更平缓。
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        # 时钟可注入，使限流用例不必真的 sleep
        self._clock = clock
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    async def check(
        self, *, scope: str, subject: str, limit: int, window_seconds: int
    ) -> RateLimitResult:
        if limit < 1 or window_seconds < 1:
            raise ValueError("limit 与 window_seconds 必须为正整数")

        now = self._clock()
        refill_per_second = limit / window_seconds
        key = (scope, subject)

        bucket = self._buckets.get(key)
        if bucket is None:
            # 新桶按满额起步：首次请求不该被上一分钟的余额影响
            bucket = _Bucket(tokens=float(limit), updated_at=now)
        else:
            bucket.tokens = min(
                float(limit), bucket.tokens + (now - bucket.updated_at) * refill_per_second
            )
            bucket.updated_at = now

        allowed = bucket.tokens >= 1.0
        if allowed:
            bucket.tokens -= 1.0
        self._buckets[key] = bucket

        deficit = max(0.0, 1.0 - bucket.tokens)
        return RateLimitResult(
            allowed=allowed,
            limit=limit,
            remaining=max(0, int(bucket.tokens)),
            reset_after_seconds=math.ceil(deficit / refill_per_second),
        )


class RedisFixedWindowLimiter:
    """Redis 固定窗口计数 + 本地降级。**生产实现**。

    `scope` 与 `subject` 拼成 `rl:{scope}:{subject}`（详细设计 4.4），
    与 `InProcessTokenBucket` 的分桶键语义一致，因此换实现时调用方无感。
    """

    def __init__(
        self,
        *,
        redis: aioredis.Redis,
        settings: Settings,
        fallback: RateLimiter | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._redis = redis
        self._tuning = settings.redis_tuning
        self._fallback = fallback or InProcessTokenBucket(clock=clock)
        self._clock = clock
        self._scripts = register_scripts(redis)
        #: 熔断到期时刻（单调时钟）。早于它就直接走降级，不再打扰 Redis。
        self._degraded_until = 0.0
        #: 仅供健康检查与用例观察，不是限流判定的一部分
        self.last_degraded_reason: str | None = None

    @property
    def is_degraded(self) -> bool:
        return self._clock() < self._degraded_until

    async def check(
        self, *, scope: str, subject: str, limit: int, window_seconds: int
    ) -> RateLimitResult:
        if limit < 1 or window_seconds < 1:
            raise ValueError("limit 与 window_seconds 必须为正整数")

        if self.is_degraded:
            return await self._degraded(scope, subject, limit, window_seconds, "熔断冷却中")

        key = RedisKey.rate_limit(scope, subject)
        try:
            # wait_for 而不是只靠 socket_timeout：socket_timeout 管的是单条命令的
            # 读写，连接池耗尽时的等待不在其覆盖范围内，那种情况下命令会排队到
            # 连接可用为止——限流路径上这是不可接受的挂起。
            raw = await asyncio.wait_for(
                self._scripts["rate_limit"](keys=[key], args=[window_seconds * 1000, limit]),
                timeout=self._tuning.operation_timeout_seconds,
            )
        except _REDIS_FAILURES as exc:
            return await self._degraded(scope, subject, limit, window_seconds, type(exc).__name__)

        self._degraded_until = 0.0
        self.last_degraded_reason = None

        count, ttl_ms = int(raw[0]), int(raw[1])
        # PTTL 在键无过期时间时返回 -1、键不存在时返回 -2。两者都不该出现在这里
        # （脚本一定刚 SET 过 EXPIRE），真出现就按整窗处理，宁可多限一会儿。
        ttl_seconds = math.ceil(ttl_ms / 1000) if ttl_ms > 0 else window_seconds
        return RateLimitResult(
            allowed=count <= limit,
            limit=limit,
            remaining=max(0, limit - count),
            reset_after_seconds=ttl_seconds,
        )

    async def _degraded(
        self, scope: str, subject: str, limit: int, window_seconds: int, reason: str
    ) -> RateLimitResult:
        """收紧配额后走本地降级，并把熔断窗口往后推。"""
        self._degraded_until = self._clock() + self._tuning.rate_limit_breaker_seconds
        if reason != self.last_degraded_reason:
            # 只在状态变化时打日志：Redis 挂掉期间每个请求一条 warning 会把日志刷爆，
            # 真正有用的信息反而被埋掉。
            logger.warning("限流降级为本地令牌桶：%s", reason)
            self.last_degraded_reason = reason

        degraded_limit = self._degraded_limit(limit)
        result = await self._fallback.check(
            scope=scope, subject=subject, limit=degraded_limit, window_seconds=window_seconds
        )
        # limit 也一并换成本地生效的那个：响应头必须反映**真实生效**的额度，
        # 否则客户端会按 10 次/分钟退避，而实际只有 5 次。
        return replace(result, limit=degraded_limit, degraded=True)

    def _degraded_limit(self, limit: int) -> int:
        """收紧后的配额。下限锁死在 1，不能因为系数小而变成 0（等于全拒）。"""
        return max(1, int(limit * self._tuning.rate_limit_degraded_factor))
