"""版本化缓存（开发流程 6.3 施工项 3）。

缓存最容易出的两类问题在这里被钉住：

1. **版本隔离**——新版本必须读不到旧版本的内容，这是「不做主动失效」成立的前提；
2. **不可用时退化为未命中**——缓存挂掉不能让业务挂掉。

另有一条容易被忽略的：缓存里存的是**旧代码**写下的数据。版本号只覆盖数据版本，
覆盖不了代码版本，因此读出来必须再过一次 Pydantic 校验。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
from pydantic import BaseModel

from app.infrastructure.cache import VersionedCache


class SchemaContext(BaseModel):
    """借用详细设计 10.2 里会真正缓存的这类模型。"""

    tables: list[str]
    version: str


class RenamedSchemaContext(BaseModel):
    """模拟「代码换了字段名，但数据版本号没变」。"""

    table_names: list[str]
    version: str


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def cache(redis: fakeredis.aioredis.FakeRedis) -> VersionedCache:
    return VersionedCache(redis, default_ttl_seconds=3600)


async def test_get_returns_none_before_anything_is_cached(cache: VersionedCache) -> None:
    assert await cache.get("schema", "v1", model=SchemaContext) is None


async def test_set_then_get_round_trips(cache: VersionedCache) -> None:
    value = SchemaContext(tables=["fact_sales_order_item", "dim_region"], version="v1")

    await cache.set("schema", "v1", value, ttl_seconds=60)

    assert await cache.get("schema", "v1", model=SchemaContext) == value


async def test_versions_do_not_see_each_other(cache: VersionedCache) -> None:
    """4.4 纪律 1：版本号进了键名，发布新版本天然失效，不需要任何 DEL 逻辑。"""
    await cache.set("schema", "v1", SchemaContext(tables=["旧表"], version="v1"))

    assert await cache.get("schema", "v2", model=SchemaContext) is None


async def test_key_carries_the_version(redis: fakeredis.aioredis.FakeRedis) -> None:
    cache = VersionedCache(redis)
    await cache.set("schema", "2026-09-16", SchemaContext(tables=[], version="v1"))

    assert await redis.exists("cache:schema:2026-09-16") == 1


async def test_ttl_is_applied(redis: fakeredis.aioredis.FakeRedis) -> None:
    cache = VersionedCache(redis, default_ttl_seconds=120)
    await cache.set("schema", "v1", SchemaContext(tables=[], version="v1"))

    assert 0 < await redis.ttl("cache:schema:v1") <= 120


async def test_incompatible_content_is_treated_as_a_miss(
    cache: VersionedCache, redis: fakeredis.aioredis.FakeRedis
) -> None:
    """代码版本与数据版本不一致时按未命中回源，而不是把半截对象带下去。

    这里模拟的是真实会发生的场景：发布改了模型字段，而
    `schema_catalog.version` 没变，旧键仍然被新代码读到。
    """
    await cache.set("schema", "v1", SchemaContext(tables=["t"], version="v1"))

    assert await cache.get("schema", "v1", model=RenamedSchemaContext) is None


async def test_unavailable_redis_degrades_to_a_miss(cache: VersionedCache) -> None:
    """缓存不可用不是故障，是「没缓存」——不能让一次请求因此变成 500。"""
    cache._redis = BrokenRedis()

    assert await cache.get("schema", "v1", model=SchemaContext) is None


async def test_get_or_set_loads_once_and_then_hits_cache(cache: VersionedCache) -> None:
    calls = 0

    async def loader() -> SchemaContext:
        nonlocal calls
        calls += 1
        return SchemaContext(tables=["t"], version="v1")

    first = await cache.get_or_set("schema", "v1", loader, model=SchemaContext)
    second = await cache.get_or_set("schema", "v1", loader, model=SchemaContext)

    assert first == second
    assert calls == 1


async def test_loader_errors_are_not_swallowed(cache: VersionedCache) -> None:
    """回源失败是真实故障，伪装成「缓存问题」会让排查方向整个跑偏。"""

    async def broken_loader() -> SchemaContext:
        raise ConnectionError("库连不上")

    with pytest.raises(ConnectionError):
        await cache.get_or_set("schema", "v1", broken_loader, model=SchemaContext)


class BrokenRedis:
    async def get(self, name: str) -> object:
        raise ConnectionError("redis 不可达")

    async def set(self, name: str, value: str, *, ex: int | None = None) -> object:
        raise ConnectionError("redis 不可达")
