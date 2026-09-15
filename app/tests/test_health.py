"""健康探针测试。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def app(fake_redis: fakeredis.aioredis.FakeRedis, _settings_env: None) -> FastAPI:
    """绕开 lifespan：`/health/ready` 只读 app.state.redis，直接注入假客户端。"""
    application = create_app()
    application.state.redis = fake_redis
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_live_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_reports_redis(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checked"] == ["redis"]


async def test_ready_degrades_when_redis_down(app: FastAPI) -> None:
    """Redis 不可用时必须是 503 not_ready，而不是 500 或挂起。"""

    class _Broken:
        async def ping(self) -> None:
            raise ConnectionError("redis 不可达")

    app.state.redis = _Broken()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["failed"] == ["redis"]
