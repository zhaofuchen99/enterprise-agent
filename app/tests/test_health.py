"""健康探针测试。夹具在 conftest.py（fakeredis 注入 + 显式装配依赖）。"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from app.tests.conftest import build_client


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
    async with build_client(app) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["failed"] == ["redis"]
