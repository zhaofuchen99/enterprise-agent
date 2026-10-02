"""健康探针测试。夹具在 conftest.py（fakeredis 注入 + 显式装配依赖）。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from app.tests.conftest import build_client


async def test_live_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_reports_every_dependency(client: AsyncClient) -> None:
    """每一条依赖都要报出来。

    Phase 2 起 MySQL 是权威存储，它的状态同样是就绪信息；Phase 13 起
    Qdrant / 业务库 / 对象存储也进来（详见 `infrastructure/health.py`）。
    **`checked` 是有意义的**：少一项就说明那条依赖从没被探过，而响应
    看起来仍然是 `ready`——一条永远绿的探针与一条不存在的探针长得一样。
    """
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["checked"] == [
        "mysql",
        "redis",
        "qdrant",
        "business_db",
        "object_storage",
    ]


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
    assert "redis" in body["checked"]


async def test_ready_reports_degraded_when_a_probe_fails(app: FastAPI) -> None:
    """**Qdrant / 业务库 / 对象存储挂了同样只标 degraded，绝不 503。**

    判 503 的唯一标准是「这个实例完全不能用」。这几条只影响**部分功能**
    ——状态查询与已缓存结果的读取照常可用——所以摘掉实例是最不该做的事
    （详设 20.3、需求 8.2）。

    这条用例同时钉住"降级状态必须出现在健康检查里"：不接进来的话，
    依赖挂了而 ready 一路 `ready`，编排层与运维都看不见。
    """
    from app.tests.fakes import FakeProbe

    app.state.dependency_probes = [
        FakeProbe("qdrant", ok=False, reason="ResponseHandlingException"),
        *[FakeProbe(name) for name in ("business_db", "object_storage")],
    ]
    async with build_client(app) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    # 报的是**哪一层、什么异常**——一句"依赖不可用"帮不上排查
    assert body["degraded"] == ["qdrant(ResponseHandlingException)"]


async def test_ready_is_503_when_mysql_is_down(app: FastAPI) -> None:
    """MySQL 不可用 → **503**，与其余依赖反过来。

    它是任务、会话、用户的**唯一权威**，没有任何降级路径：库连不上时登录、
    建任务、查状态全部失败。此时如实报 not_ready 让编排层摘掉实例才是对的。

    ⚠️ 这条分支此前**没有用例**（`FakeSessionFactory` 只模拟成功，
    文档说"靠 integration 对真库验"，而 integration 里也没有断库的用例）
    ——一次改写把 503 写成 200 不会有任何症状，直到现场才发现实例摘不掉。
    """

    class _BrokenSessions:
        def __call__(self) -> Any:
            raise ConnectionError("mysql 不可达")

    app.state.sessions = _BrokenSessions()
    async with build_client(app) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["failed"] == ["mysql(ConnectionError)"]
