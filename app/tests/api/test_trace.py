"""执行轨迹接口（详细设计 17.4 / FR-TRACE-002）。

三件事在这里被钉住：

1. **权限判据与任务详情同源**——接口先走一次 `TaskService.get_task`，
   跨用户 403、未知任务 404 都从那里来。轨迹里带着节点名与耗时，
   是未授权用户不该看到的执行细节，因此不能绕过那一步直接读表。
2. **顺序是语义**：18.3 说 `(task_id, sequence)` 的唯一约束就是顺序保证本身，
   而 `after_sequence` 的语义是「我已经有的最后一条」——严格大于。
3. **`trace_incomplete` 真的透传**：不置位时，一份缺了几段的轨迹
   看起来是完整的（缺的正好是任务没跑完的那几段）。

轨迹数据用内存仓储直接写入：这个接口读的就是 `Repositories.artifacts`
（见该字段的说明），单元测试不连 MySQL。
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from httpx import AsyncClient

from app.domain.trace import NodeTrace
from app.repositories.task_repo import TaskPatch
from app.tests.api.conftest import issue_headers

_QUESTION = "华东区的销售额是多少"

#: 一条被固定下来的执行轨迹：两个节点、四个事件。
_ROWS: list[tuple[str, str, int | None]] = [
    ("supervisor", "node.started", None),
    ("supervisor", "node.completed", 2631),
    ("sql", "node.started", None),
    ("sql", "node.completed", 1568),
]


def _traced(node: str, event_type: str, duration: int | None) -> NodeTrace:
    return NodeTrace(
        node=node,
        event_type=event_type,
        status="RUNNING" if duration is None else "SUCCEEDED",
        duration_ms=duration,
        created_at=datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC),
    )


async def _create_task(client: AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post("/api/agent/chat", json={"message": _QUESTION}, headers=headers)
    assert response.status_code == 202, response.text
    return str(response.json()["data"]["task_id"])


async def _seed(
    app: FastAPI, task_id: str, rows: list[tuple[str, str, int | None]] = _ROWS
) -> None:
    await app.state.artifacts.save(
        task_id,
        trace_id="trc_0000000000000000000001",
        trace_events=[_traced(*row) for row in rows],
    )


# ------------------------------------------------------------------ 权限


async def test_trace_requires_authentication(client: AsyncClient) -> None:
    """轨迹是执行细节，匿名一律不返回——**先鉴权再判存在性**。"""
    response = await client.get("/api/agent/tasks/tsk_0000000000000000000001/trace")

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"


async def test_unknown_task_is_404(client: AsyncClient, analyst_headers: dict[str, str]) -> None:
    response = await client.get(
        "/api/agent/tasks/tsk_0000000000000000000001/trace", headers=analyst_headers
    )

    assert response.status_code == 404
    assert response.json()["code"] == "TASK_NOT_FOUND"


async def test_other_user_gets_403(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """FR-TRACE-002 的前置条件：用户有任务访问权限。"""
    task_id = await _create_task(client, analyst_headers)
    await _seed(app, task_id)

    other = issue_headers(app, username="trace-other")
    response = await client.get(f"/api/agent/tasks/{task_id}/trace", headers=other)

    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"
    # **403 不回显轨迹内容**：否则"拿 403 当探针"也能顺便读到别人的执行细节
    assert "supervisor" not in response.text


async def test_admin_can_read_any_trace(
    client: AsyncClient, app: FastAPI, admin_headers: dict[str, str]
) -> None:
    """需求规格 7.2：任务查询的鉴权是「任务所有者 / ADMIN」。"""
    owner = issue_headers(app, username="trace-owner")
    task_id = await _create_task(client, owner)
    await _seed(app, task_id)

    response = await client.get(f"/api/agent/tasks/{task_id}/trace", headers=admin_headers)

    assert response.status_code == 200
    assert len(response.json()["data"]["events"]) == len(_ROWS)


# ------------------------------------------------------------------ 内容


async def test_events_come_back_in_execution_order(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """**顺序不是排序的副产品，是重放的依据**（18.3）。

    `sequence` 由写入侧按执行顺序分配（图是串行的），读的时候按它升序——
    客户端拿到的顺序要与真实执行顺序一致，否则"哪一步慢"根本读不出来。
    """
    task_id = await _create_task(client, analyst_headers)
    await _seed(app, task_id)

    response = await client.get(f"/api/agent/tasks/{task_id}/trace", headers=analyst_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "OK"
    data = body["data"]
    assert data["task_id"] == task_id
    assert data["trace_incomplete"] is False
    events = data["events"]
    assert [item["sequence"] for item in events] == [1, 2, 3, 4]
    assert [item["type"] for item in events] == [row[1] for row in _ROWS]
    assert [item["node"] for item in events] == [row[0] for row in _ROWS]
    assert events[0]["status"] == "RUNNING"
    # 进入事件没有耗时可言——给它 0 会被读成"瞬间完成"
    assert events[0]["duration_ms"] is None
    assert events[1]["duration_ms"] == 2631


async def test_a_task_without_a_trace_returns_an_empty_list(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """排队中的任务还没有轨迹。**空列表与 404 是两件事**：
    前者是"这个任务还没跑到"，后者是"这个任务不是你的/不存在"。
    """
    task_id = await _create_task(client, analyst_headers)

    response = await client.get(f"/api/agent/tasks/{task_id}/trace", headers=analyst_headers)

    assert response.status_code == 200
    assert response.json()["data"]["events"] == []


async def test_after_sequence_is_strictly_greater(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """语义是「我已经有的最后一条」，因此**严格大于**。

    用 `>=` 会让每次重连都重复拿到同一条——在"按序号去重"的客户端上
    表现为"最后一条永远处理两遍"。
    """
    task_id = await _create_task(client, analyst_headers)
    await _seed(app, task_id)

    response = await client.get(
        f"/api/agent/tasks/{task_id}/trace?after_sequence=2", headers=analyst_headers
    )

    assert response.status_code == 200
    assert [item["sequence"] for item in response.json()["data"]["events"]] == [3, 4]


async def test_limit_caps_the_page(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """`limit` 分页取**前面若干条**，与 `after_sequence` 组合时可增量拉取。"""
    task_id = await _create_task(client, analyst_headers)
    await _seed(app, task_id)

    response = await client.get(
        f"/api/agent/tasks/{task_id}/trace?limit=1&after_sequence=1", headers=analyst_headers
    )

    assert response.status_code == 200
    assert [item["sequence"] for item in response.json()["data"]["events"]] == [2]


async def test_out_of_range_parameters_are_rejected(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """参数越界走 400（统一异常映射），而不是悄悄忽略。

    被忽略的话，"我明明传了 limit=9999"与"它只返回了 4 条"之间
    没有任何提示，客户端会以为任务只有 4 条事件。
    """
    task_id = await _create_task(client, analyst_headers)

    for query in ("after_sequence=-1", "limit=0", "limit=501"):
        response = await client.get(
            f"/api/agent/tasks/{task_id}/trace?{query}", headers=analyst_headers
        )
        assert response.status_code == 400, query


async def test_trace_incomplete_is_carried_through(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """FR-TRACE-001 的异常情况：轨迹写入失败要能被读出来。

    置位说明"你看到的不是全部"——不置位时，一份缺了几段的轨迹
    看起来是完整的，而读它的人会拿它当完整的执行过程。
    """
    task_id = await _create_task(client, analyst_headers)
    await _seed(app, task_id)
    await app.state.task_repo.update(
        task_id,
        TaskPatch(trace_incomplete=True),
        at=datetime(2026, 9, 18, 12, 0, 30, tzinfo=UTC),
    )

    response = await client.get(f"/api/agent/tasks/{task_id}/trace", headers=analyst_headers)

    assert response.status_code == 200
    assert response.json()["data"]["trace_incomplete"] is True
