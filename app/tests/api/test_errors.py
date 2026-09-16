"""错误码 -> HTTP 状态映射（开发流程 6.2 的门禁：400/401/403/404/429/500 映射正确）。

每个状态码一条用例，且都断言**统一响应外壳**的五个字段齐全——
只断言状态码会漏掉「状态对了但包体形状不对」这种情况，
而前端恰恰是按包体解析的。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.tests.api.conftest import app_client
from app.tests.conftest import build_client

ENVELOPE_KEYS = {"code", "message", "data", "trace_id", "retryable"}


def assert_envelope(body: dict[str, Any]) -> None:
    assert set(body) == ENVELOPE_KEYS, f"响应外壳字段不符：{sorted(body)}"


async def test_400_invalid_argument_on_empty_message(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    response = await client.post("/api/agent/chat", json={"message": ""}, headers=analyst_headers)
    assert response.status_code == 400
    body = response.json()
    assert_envelope(body)
    assert body["code"] == "INVALID_ARGUMENT"
    assert body["retryable"] is False
    # 文案是给用户看的，必须是中文（CLAUDE.md 代码约定）
    assert body["message"] == "message：长度不足"


async def test_400_invalid_argument_on_bad_conversation_id(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """会话 ID 格式不对也归 400：这类错误用户改一下就能过，不是权限问题。"""
    response = await client.post(
        "/api/agent/chat",
        json={"message": "销售额多少", "conversation_id": "not-a-conversation-id"},
        headers=analyst_headers,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ARGUMENT"


async def test_401_when_token_missing(client: AsyncClient) -> None:
    """缺令牌必须是 401 而非 403。

    FastAPI 的 HTTPBearer 默认在这里返回 403，直接用它就把需求规格 7.3
    的语义搞反了，因此 deps.py 里关掉了 auto_error。
    """
    response = await client.get("/api/agent/tasks/tsk_0000000000000000000000")
    assert response.status_code == 401
    body = response.json()
    assert_envelope(body)
    assert body["code"] == "AUTHENTICATION_REQUIRED"


async def test_401_when_token_malformed(client: AsyncClient) -> None:
    response = await client.get(
        "/api/agent/tasks/tsk_0000000000000000000000",
        # 头里只能用 ASCII：httpx 按 latin-1 编码请求头，塞中文会在发出去之前就炸
        headers={"Authorization": "Bearer not-a-valid-jwt-at-all"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"


async def test_404_when_task_absent(client: AsyncClient, analyst_headers: dict[str, str]) -> None:
    response = await client.get(
        "/api/agent/tasks/tsk_0000000000000000000000", headers=analyst_headers
    )
    assert response.status_code == 404
    body = response.json()
    assert_envelope(body)
    assert body["code"] == "TASK_NOT_FOUND"


async def test_404_on_unknown_route(client: AsyncClient) -> None:
    """未注册路径也要走统一外壳，不能漏出 Starlette 默认的 {"detail": ...}。"""
    response = await client.get("/api/agent/不存在的接口")
    assert response.status_code == 404
    body = response.json()
    assert_envelope(body)
    assert body["code"] == "TASK_NOT_FOUND"


async def test_405_on_wrong_method(client: AsyncClient) -> None:
    response = await client.get("/api/auth/login")
    assert response.status_code == 405
    assert_envelope(response.json())


async def test_429_when_rate_limit_exceeded(make_app: Any) -> None:
    """限流触发时返回 429，并带上 X-RateLimit-* 头。

    配额设为 1、并发上限放宽到 99，确保命中的是**限流**而不是并发上限——
    两者都是 429，混在一起测等于没测。
    """
    app: FastAPI = make_app(RATE_LIMIT_CREATE_PER_MINUTE="1", MAX_RUNNING_TASKS_PER_USER="99")
    async with app_client(app) as (client, headers):
        first = await client.post("/api/agent/chat", json={"message": "销售额"}, headers=headers)
        second = await client.post("/api/agent/chat", json={"message": "销售额"}, headers=headers)

    assert first.status_code == 202
    assert first.headers["X-RateLimit-Limit"] == "1"
    assert second.status_code == 429
    body = second.json()
    assert_envelope(body)
    assert body["code"] == "RATE_LIMITED"
    assert body["retryable"] is True
    # 异常路径也要带上限流头：客户端靠它决定退避多久
    assert second.headers["X-RateLimit-Limit"] == "1"


async def test_429_when_concurrent_quota_exceeded(make_app: Any) -> None:
    """「同时运行任务数」超限也是 429，但走的是任务服务而非限流器。"""
    app: FastAPI = make_app(MAX_RUNNING_TASKS_PER_USER="1")
    async with app_client(app) as (client, headers):
        first = await client.post("/api/agent/chat", json={"message": "第一个"}, headers=headers)
        second = await client.post("/api/agent/chat", json={"message": "第二个"}, headers=headers)

    assert first.status_code == 202
    assert second.status_code == 429
    assert "上限" in second.json()["message"]


async def test_500_hides_internals(app: FastAPI) -> None:
    """未捕获异常：外壳正确、状态 500，且**内部细节不泄露**到响应体。"""

    secret = "内部细节：连接串 mysql://user:pw@host"

    @app.get("/__boom", include_in_schema=False)
    async def boom() -> None:
        raise RuntimeError(secret)

    async with build_client(app) as client:
        response = await client.get("/__boom")

    assert response.status_code == 500
    body = response.json()
    assert_envelope(body)
    assert body["code"] == "INTERNAL_ERROR"
    assert secret not in response.text
    assert "RuntimeError" not in response.text


async def test_trace_id_present_and_consistent(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """响应体的 trace_id 必须与响应头一致，且是 trc_ 前缀。

    两者不一致等于给排查埋坑：用户报的是包体里的 ID，日志里存的是头里的那个。
    """
    response = await client.post(
        "/api/agent/chat", json={"message": "销售额多少"}, headers=analyst_headers
    )
    body = response.json()
    assert body["trace_id"].startswith("trc_")
    # 26 位含前缀，与 agent_task.trace_id 的 CHAR(26) 对齐（见 core/ids.py）
    assert len(body["trace_id"]) == 26
    assert response.headers["X-Trace-Id"] == body["trace_id"]


@pytest.mark.parametrize("path", ["/health/live", "/api/agent/tasks/tsk_x"])
async def test_trace_id_on_every_response(client: AsyncClient, path: str) -> None:
    response = await client.get(path)
    assert response.headers["X-Trace-Id"].startswith("trc_")


async def test_redis_down_does_not_block_task_creation(app: FastAPI) -> None:
    """Redis 掉线**不再**让创建任务失败——这是 Phase 2 相对 Phase 1.5 的改进。

    Phase 1.5 的任务仓储还是 Redis 实现，因此 Redis 一挂
    `POST /api/agent/chat` 就返回 503 `REDIS_UNAVAILABLE`
    （当时在 CLAUDE.md 里明确登记为**临时状态**，Phase 2 换成 MySQL 后消失）。
    Phase 2 把权威存储换成 `agent_task` 之后，Redis 不可用只剩两处影响：

    1. 限流退回本地收紧配额，并带上 `x-ratelimit-degraded` 头；
    2. 入队失败——但任务已经落库，由补偿扫描重新投递（详细设计 17.1）。

    因此这里断言的是**更严的契约**：Redis 全挂，接口照样 202，任务照样建了出来。
    """
    import json

    from app.tests.api.conftest import login_headers

    async with build_client(app) as client:
        headers = await login_headers(client)
        for key in await app.state.redis.keys("rl:*"):
            await app.state.redis.delete(key)
        # 让所有 Redis 命令都抛异常，模拟 Redis 整条挂掉
        app.state.redis.execute_command = _raise

        response = await client.post("/api/agent/chat", json={"message": "问题"}, headers=headers)

    assert response.status_code == 202
    body = json.loads(response.content)
    assert body["code"] == "ACCEPTED"
    # 降级必须**对调用方可见**：限流配额被收紧到配置值的一半（19.3）。
    # 少了这个头，调用方会把「被降级地拒绝了」当成「配额本来就这么小」。
    assert response.headers.get("x-ratelimit-degraded") == "true"


async def _raise(*args: object, **kwargs: object) -> None:
    import redis.asyncio as aioredis

    raise aioredis.ConnectionError("redis 不可达")
