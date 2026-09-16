"""取消任务接口（详细设计 17.5）。

Phase 1.5 把取消机制接起来了（`TaskRunner` 的取消键 + Worker 领取时的检查），
本文件验证接口这一侧：状态怎么转、信号有没有写下去、越权和重复取消怎么处理。

**「接受」不等于「已取消」**：接口返回 202 与 `CANCEL_REQUESTED`，
真正的 `CANCELLED` 要等 Worker 确认——Worker 可能在另一个实例上，
也可能正阻塞在一次模型调用里。断言上必须区分这两者，
否则测试会掩盖「取消请求丢在队列里没人处理」这类问题。
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from httpx import AsyncClient

from app.domain.task import TaskStatus
from app.infrastructure.redis import RedisKey
from app.repositories.task_repo import TaskPatch
from app.tests.api.conftest import issue_headers

_QUESTION = "分析华东2025年第三季度销售额同比下降原因"


async def _create_task(client: AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post("/api/agent/chat", json={"message": _QUESTION}, headers=headers)
    assert response.status_code == 202, response.text
    return str(response.json()["data"]["task_id"])


async def test_cancel_returns_202_and_records_the_request(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    task_id = await _create_task(client, analyst_headers)

    response = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)

    assert response.status_code == 202
    body = response.json()
    assert body["code"] == "ACCEPTED"
    assert body["data"]["status"] == TaskStatus.CANCEL_REQUESTED.value
    assert body["data"]["task_id"] == task_id


async def test_cancel_writes_the_cross_process_signal(
    client: AsyncClient, analyst_headers: dict[str, str], app: FastAPI
) -> None:
    """Worker 可能在另一个实例上，因此取消**不能**用进程内信号（17.5 第 2 条）。"""
    task_id = await _create_task(client, analyst_headers)

    await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)

    assert await app.state.redis.exists(RedisKey.task_cancel(task_id)) == 1


async def test_cancel_is_idempotent(client: AsyncClient, analyst_headers: dict[str, str]) -> None:
    """17.5：重复取消同一任务返回相同结果，不报错。

    客户端重试、用户连点两下都会走到这里，报错只会让前端显示一个假故障。
    """
    task_id = await _create_task(client, analyst_headers)

    first = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)
    second = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)

    assert first.status_code == second.status_code == 202
    assert first.json()["data"]["status"] == second.json()["data"]["status"]


async def test_cancel_of_a_finished_task_is_a_conflict(
    client: AsyncClient, analyst_headers: dict[str, str], app: FastAPI
) -> None:
    """已结束的任务没有「取消」可言，返回 409 而不是假装接受了。"""
    task_id = await _create_task(client, analyst_headers)

    await app.state.task_repo.update(
        task_id, TaskPatch(status=TaskStatus.SUCCEEDED), at=datetime.now(UTC)
    )

    response = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)

    assert response.status_code == 409
    assert response.json()["code"] == "TASK_CONFLICT"


async def test_cannot_cancel_someone_elses_task(
    client: AsyncClient, analyst_headers: dict[str, str], app: FastAPI
) -> None:
    """归属校验与查询接口一致：跨用户访问是 403，不是 404。"""
    task_id = await _create_task(client, analyst_headers)
    other = issue_headers(app, username="intruder")

    response = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=other)

    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"


async def test_cancel_of_unknown_task_is_404(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    response = await client.post("/api/agent/tasks/tsk_不存在/cancel", headers=analyst_headers)

    assert response.status_code == 404
    assert response.json()["code"] == "TASK_NOT_FOUND"


async def test_cancel_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/agent/tasks/tsk_x/cancel")

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"


async def test_admin_can_cancel_any_task(
    client: AsyncClient, analyst_headers: dict[str, str], admin_headers: dict[str, str]
) -> None:
    """7.2 的鉴权列规定 ADMIN 可跨用户查看，取消同样适用。"""
    task_id = await _create_task(client, analyst_headers)

    response = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=admin_headers)

    assert response.status_code == 202


async def test_cancel_response_carries_rate_limit_headers(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    task_id = await _create_task(client, analyst_headers)

    response = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)

    assert "X-RateLimit-Degraded" in response.headers
    assert response.headers["X-RateLimit-Degraded"] == "false"


async def test_cancel_is_audited_with_trace_id(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    task_id = await _create_task(client, analyst_headers)

    response = await client.post(f"/api/agent/tasks/{task_id}/cancel", headers=analyst_headers)

    assert response.json()["trace_id"].startswith("trc_")
    assert response.headers["X-Trace-Id"] == response.json()["trace_id"]
