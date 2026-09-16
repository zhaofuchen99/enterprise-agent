"""健康探针测试。夹具在 conftest.py（fakeredis 注入 + 显式装配依赖）。"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from app.tests.conftest import build_client


async def test_live_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_reports_mysql_and_redis(client: AsyncClient) -> None:
    """两个依赖都要报出来。Phase 2 起 MySQL 是权威存储，它的状态同样是就绪信息。"""
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checked"] == ["mysql", "redis"]


async def test_ready_reports_degraded_when_redis_down(app: FastAPI) -> None:
    """Redis 不可用时必须报 degraded，而不是 500、挂起、或直接判死。

    **为什么是 200 而不是 503**：Phase 1.5 起 Redis 有了降级路径
    （限流退回本地令牌桶、投递失败由补偿扫描兜底），API 仍然能对外服务。
    此时报 not_ready 会让编排层摘掉这个实例——而摘掉它恰恰是最不该做的事，
    降级路径本来就是为了「Redis 挂了也要撑住」才存在的。
    开发流程 6.3 的验收命令同样要求「期望 degraded，而非 500」。
    """

    class _Broken:
        async def ping(self) -> None:
            raise ConnectionError("redis 不可达")

    app.state.redis = _Broken()
    async with build_client(app) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["degraded"] == ["redis(ConnectionError)"]
    assert body["checked"] == ["mysql", "redis"]
