"""令牌桶限流（Phase 1 的进程内占位实现）。

时钟是注入的，用例靠推进假时钟而不是 `sleep`：
真 sleep 会让「窗口没到不放行」这类断言要么慢、要么不稳定。
"""

from __future__ import annotations

import pytest

from app.services.rate_limit import InProcessTokenBucket


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def limiter(clock: FakeClock) -> InProcessTokenBucket:
    return InProcessTokenBucket(clock=clock)


async def test_allows_up_to_limit_then_denies(
    limiter: InProcessTokenBucket, clock: FakeClock
) -> None:
    results = [
        await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)
        for _ in range(4)
    ]

    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]


async def test_denied_result_tells_when_to_retry(
    limiter: InProcessTokenBucket, clock: FakeClock
) -> None:
    for _ in range(3):
        await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)

    denied = await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)
    assert not denied.allowed
    # 3 个/60 秒 -> 每秒补 0.05 个，补满 1 个需要 20 秒
    assert denied.reset_after_seconds == 20


async def test_refills_over_time(limiter: InProcessTokenBucket, clock: FakeClock) -> None:
    for _ in range(3):
        await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)
    assert not (
        await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)
    ).allowed

    clock.advance(60)
    assert (
        await limiter.check(scope="task:create", subject="usr_1", limit=3, window_seconds=60)
    ).allowed


async def test_buckets_are_isolated_by_subject_and_scope(
    limiter: InProcessTokenBucket,
) -> None:
    """配额按用户隔离，不同限流档位之间也不互相消耗。"""
    for _ in range(2):
        await limiter.check(scope="task:create", subject="usr_1", limit=2, window_seconds=60)

    assert not (
        await limiter.check(scope="task:create", subject="usr_1", limit=2, window_seconds=60)
    ).allowed
    # 另一个用户
    assert (
        await limiter.check(scope="task:create", subject="usr_2", limit=2, window_seconds=60)
    ).allowed
    # 同一个用户、另一个档位
    assert (
        await limiter.check(scope="task:status", subject="usr_1", limit=2, window_seconds=60)
    ).allowed


async def test_partial_refill_only_restores_what_elapsed(
    limiter: InProcessTokenBucket, clock: FakeClock
) -> None:
    """半途补充的量要按比例，不能一次性恢复满额。"""
    for _ in range(3):
        await limiter.check(scope="s", subject="u", limit=3, window_seconds=60)

    clock.advance(20)  # 补回 1 个令牌
    assert (await limiter.check(scope="s", subject="u", limit=3, window_seconds=60)).allowed
    assert not (await limiter.check(scope="s", subject="u", limit=3, window_seconds=60)).allowed


async def test_new_subject_starts_with_full_bucket(limiter: InProcessTokenBucket) -> None:
    first = await limiter.check(scope="s", subject="brand-new", limit=5, window_seconds=60)
    assert first.allowed
    assert first.remaining == 4


async def test_rejects_nonsense_parameters(limiter: InProcessTokenBucket) -> None:
    """配额参数写错时要立刻炸，而不是悄悄变成一个「永不限流」的桶。"""
    with pytest.raises(ValueError):
        await limiter.check(scope="s", subject="u", limit=0, window_seconds=60)
    with pytest.raises(ValueError):
        await limiter.check(scope="s", subject="u", limit=1, window_seconds=0)


async def test_headers_expose_degraded_flag(limiter: InProcessTokenBucket) -> None:
    """Phase 1 不依赖 Redis，degraded 恒为 false；Phase 1.5 起它会变 true。"""
    result = await limiter.check(scope="s", subject="u", limit=1, window_seconds=60)
    headers = result.headers()

    assert headers["X-RateLimit-Limit"] == "1"
    assert headers["X-RateLimit-Remaining"] == "0"
    assert headers["X-RateLimit-Degraded"] == "false"
