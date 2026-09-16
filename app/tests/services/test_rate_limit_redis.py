"""Redis 限流器与降级路径（开发流程 6.3 施工项 2）。

用 fakeredis 跑**真实的 Lua 脚本**而不是替身：限流最容易错的地方就在
「计数与过期是不是原子的」，而那恰恰是替身模拟不出来的部分。

降级用例注入一个会抛异常的 Redis 替身。注意它抛的是 `RedisError` 家族，
而不是随便一个 `Exception`——限流器只该吞掉「Redis 没帮上忙」这几类，
把编程错误（AttributeError、TypeError）也吞掉会让降级路径掩盖真实 bug。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
import redis.asyncio as aioredis

from app.core.config import Settings
from app.services.rate_limit import InProcessTokenBucket, RedisFixedWindowLimiter


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BrokenRedis:
    """模拟 Redis 不可达，并记录被调用了几次。

    调用次数是关键断言：降级之后如果仍然每次都去敲一遍 Redis，
    每个请求都要白等一次超时，降级就从「保住可用性」变成「给每个请求加延迟」。
    """

    def __init__(self) -> None:
        self.calls = 0

    def register_script(self, source: str) -> object:
        del source
        return self._fail

    async def _fail(self, *, keys: list[str], args: list[object]) -> object:
        del keys, args
        self.calls += 1
        raise aioredis.ConnectionError("redis 不可达")


def build_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "model_provider": "p",
        "model_name": "m",
        "model_api_key": "k",
        "embedding_model": "e",
        "embedding_api_key": "k",
        "database_url_agent": "mysql+asyncmy://a@localhost/a",
        "database_url_business_ro": "mysql+asyncmy://a@localhost/b",
        "redis_url": "redis://localhost:6379/0",
        "milvus_uri": "http://localhost:19530",
        "jwt_secret": "x" * 32,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def build_limiter(
    redis: object, *, clock: FakeClock | None = None, **overrides: object
) -> RedisFixedWindowLimiter:
    return RedisFixedWindowLimiter(
        redis=redis,  # type: ignore[arg-type]
        settings=build_settings(**overrides),
        clock=clock or FakeClock(),
    )


# ------------------------------------------------------------------ 基本行为
async def test_allows_up_to_limit_then_denies(
    redis: fakeredis.aioredis.FakeRedis, clock: FakeClock
) -> None:
    limiter = build_limiter(redis, clock=clock)
    results = [
        await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)
        for _ in range(4)
    ]

    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    assert all(not r.degraded for r in results)


async def test_reset_after_reports_real_remaining_window(
    redis: fakeredis.aioredis.FakeRedis, clock: FakeClock
) -> None:
    """X-RateLimit-Reset 报的是真实剩余时间，不是让客户端猜的固定值。

    手工把键的 TTL 改短来验证：若实现里写死了 `window_seconds`，
    这里会返回 60 而不是 15。窗口剩余时间的来源必须是 Redis 的 PTTL。
    """
    limiter = build_limiter(redis, clock=clock)
    await limiter.check(scope="s", subject="u", limit=1, window_seconds=60)
    await redis.expire("rl:s:u", 15)

    result = await limiter.check(scope="s", subject="u", limit=1, window_seconds=60)

    assert 14 <= result.reset_after_seconds <= 15


async def test_window_expiry_restores_quota(
    redis: fakeredis.aioredis.FakeRedis, clock: FakeClock
) -> None:
    """窗口过去后配额恢复。这里删键来模拟到期——真等 60 秒会让用例又慢又不稳。"""
    limiter = build_limiter(redis, clock=clock)
    await limiter.check(scope="s", subject="u", limit=1, window_seconds=60)
    assert not (await limiter.check(scope="s", subject="u", limit=1, window_seconds=60)).allowed

    await redis.delete("rl:s:u")
    assert (await limiter.check(scope="s", subject="u", limit=1, window_seconds=60)).allowed


# --------------------------------------------------------- 多实例一致性（门禁）
async def test_two_instances_share_one_quota(redis: fakeredis.aioredis.FakeRedis) -> None:
    """门禁：两个 API 实例合计放行数不超过配置上限，不出现翻倍。

    这是 Phase 1.5 换掉进程内令牌桶的**唯一理由**。用两个 limiter 对象
    代表两个实例，共用一个 Redis——进程内实现下这里会放行 2 倍。
    """
    instance_a = build_limiter(redis)
    instance_b = build_limiter(redis)

    results = []
    for _ in range(3):
        results.append(
            await instance_a.check(scope="task:create", subject="usr_1", limit=2, window_seconds=60)
        )
        results.append(
            await instance_b.check(scope="task:create", subject="usr_1", limit=2, window_seconds=60)
        )

    assert [r.allowed for r in results] == [True, True, False, False, False, False]


async def test_restart_does_not_reset_quota(redis: fakeredis.aioredis.FakeRedis) -> None:
    """进程重启不该重置配额——Phase 1 的进程内实现正是在这里失守的。"""
    await build_limiter(redis).check(scope="s", subject="u", limit=2, window_seconds=60)
    await build_limiter(redis).check(scope="s", subject="u", limit=2, window_seconds=60)

    fresh_instance = build_limiter(redis)
    assert not (
        await fresh_instance.check(scope="s", subject="u", limit=2, window_seconds=60)
    ).allowed


# ------------------------------------------------------------------ 降级
async def test_degrades_to_local_bucket_when_redis_is_down(clock: FakeClock) -> None:
    limiter = build_limiter(BrokenRedis(), clock=clock)
    result = await limiter.check(scope="s", subject="u", limit=10, window_seconds=60)

    assert result.degraded is True
    assert result.allowed is True


async def test_degraded_quota_is_tightened_not_relaxed(clock: FakeClock) -> None:
    """降级**不得**放宽配额：否则把 Redis 打挂就等于拿到无限配额。"""
    limiter = build_limiter(BrokenRedis(), clock=clock, rate_limit_degraded_factor=None)
    result = await limiter.check(scope="s", subject="u", limit=10, window_seconds=60)

    # 配置里 redis_tuning.rate_limit_degraded_factor 默认 0.5，等价于按 2 个实例均分
    assert result.limit == 5
    assert result.headers()["X-RateLimit-Limit"] == "5"


async def test_degraded_quota_never_reaches_zero(clock: FakeClock) -> None:
    """系数再小也要留 1 个名额，取整成 0 等于把所有人拒之门外。"""
    limiter = build_limiter(
        BrokenRedis(), clock=clock, redis_tuning={"rate_limit_degraded_factor": 0.01}
    )
    result = await limiter.check(scope="s", subject="u", limit=10, window_seconds=60)

    assert result.limit == 1


async def test_degraded_responses_carry_the_header(clock: FakeClock) -> None:
    limiter = build_limiter(BrokenRedis(), clock=clock)
    headers = (await limiter.check(scope="s", subject="u", limit=4, window_seconds=60)).headers()

    assert headers["X-RateLimit-Degraded"] == "true"


async def test_breaker_stops_hammering_a_dead_redis(clock: FakeClock) -> None:
    """熔断：失败之后的一段时间内直接走降级，不再每个请求都等一次超时。

    没有这一层，「降级」的实际效果是每个请求白等一个 socket 超时，
    可用性没保住，延迟先崩了。
    """
    broken = BrokenRedis()
    limiter = build_limiter(broken, clock=clock)

    for _ in range(5):
        result = await limiter.check(scope="s", subject="u", limit=10, window_seconds=60)
        assert result.degraded is True

    assert broken.calls == 1


async def test_recovers_after_breaker_cooldown(redis: fakeredis.aioredis.FakeRedis) -> None:
    """Redis 恢复后要能自动切回来，而不是永久停在降级模式。"""
    broken = BrokenRedis()
    clock = FakeClock()
    limiter = build_limiter(broken, clock=clock)
    assert (await limiter.check(scope="s", subject="u", limit=10, window_seconds=60)).degraded

    # 冷却结束后仍然指向坏的客户端，此时应重新尝试（而不是继续跳过）
    clock.advance(60)
    await limiter.check(scope="s", subject="u", limit=10, window_seconds=60)
    assert broken.calls == 2

    # 换成可用的 Redis：新实例模拟恢复后的下一次尝试
    healthy = build_limiter(redis, clock=clock)
    assert not (await healthy.check(scope="s", subject="u", limit=10, window_seconds=60)).degraded


async def test_fallback_backend_is_injectable(clock: FakeClock) -> None:
    """降级后端可注入，便于在别处复用同一份收紧后的配额语义。"""
    fallback = InProcessTokenBucket(clock=clock)
    limiter = RedisFixedWindowLimiter(
        redis=BrokenRedis(),  # type: ignore[arg-type]
        settings=build_settings(),
        fallback=fallback,
        clock=clock,
    )

    first = await limiter.check(scope="s", subject="u", limit=2, window_seconds=60)
    second = await limiter.check(scope="s", subject="u", limit=2, window_seconds=60)
    third = await limiter.check(scope="s", subject="u", limit=2, window_seconds=60)

    # 收紧后额度是 1，因此第二次就该被拒
    assert first.allowed and first.limit == 1
    assert not second.allowed
    assert not third.allowed
