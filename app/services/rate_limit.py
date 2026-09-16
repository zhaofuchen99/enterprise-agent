"""限流。

Phase 1 用**进程内令牌桶**占位，Phase 1.5 换成 Redis + Lua 的固定窗口计数
（详细设计 19.3）。`RateLimiter` 协议保持不变，换的只是实现——
协议从第一天就定义成 `async`，正是为了让这次替换不需要改调用方。

**为什么占位实现也值得认真写**：令牌桶而非固定窗口，是因为 Phase 1 多实例各算各的，
固定窗口在窗口边界上会被两个实例同时放行出两倍流量；令牌桶的平滑特性
至少不会主动制造这种尖峰。真正的跨实例一致只能靠 Phase 1.5 的 Redis 计数。

**降级语义**：Phase 1 不依赖 Redis，`degraded` 恒为 False。
Phase 1.5 在 Redis 不可用时把配额按实例数均分后收紧，并置 `degraded=True`，
对应响应头 `X-RateLimit-Degraded`。降级是可用性妥协，**不得**因此放宽配额。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    #: 距恢复 1 个令牌还有多少秒——即客户端最早可以重试的时刻
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
    进程重启即清空——这正是 Phase 1.5 要换掉它的原因（重启不该重置配额）。
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
