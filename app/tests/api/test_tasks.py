"""任务查询的归属校验（FR-CHAT-002）。"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient

from app.tests.api.conftest import issue_headers

_QUESTION = "华东区的销售额是多少"


async def _create_task(client: AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post("/api/agent/chat", json={"message": _QUESTION}, headers=headers)
    assert response.status_code == 202, response.text
    return str(response.json()["data"]["task_id"])


async def test_owner_can_read_own_task(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    task_id = await _create_task(client, analyst_headers)
    response = await client.get(f"/api/agent/tasks/{task_id}", headers=analyst_headers)
    assert response.status_code == 200
    assert response.json()["code"] == "OK"


async def test_other_user_gets_403(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """FR-CHAT-002 异常情况：跨用户访问返回 403。"""
    task_id = await _create_task(client, analyst_headers)

    other = issue_headers(app, username="other-analyst")
    response = await client.get(f"/api/agent/tasks/{task_id}", headers=other)

    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"
    # 403 不回显任务内容，避免「拿 403 当存在性探针」还能顺便读到数据
    assert _QUESTION not in response.text


async def test_admin_can_read_any_task(
    client: AsyncClient, app: FastAPI, admin_headers: dict[str, str]
) -> None:
    """需求规格 7.2：任务查询的鉴权是「任务所有者 / ADMIN」。"""
    owner = issue_headers(app, username="task-owner")
    task_id = await _create_task(client, owner)

    response = await client.get(f"/api/agent/tasks/{task_id}", headers=admin_headers)
    assert response.status_code == 200
    assert response.json()["data"]["task_id"] == task_id
