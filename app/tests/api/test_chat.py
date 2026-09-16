"""创建任务接口测试（详细设计 17.1、FR-CHAT-001）。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import AsyncClient

from app.tests.api.conftest import app_client, issue_headers

_QUESTION = "结合销售数据和渠道政策，分析华东2025年第三季度销售额同比下降原因"


async def _count_tasks(app: FastAPI) -> int:
    """数一数存储里真的建了几条任务记录。

    「幂等键在有效期内只能创建一个任务」（FR-CHAT-001 业务规则）
    是这条接口最容易写错的地方，而接口本身不提供任务列表，只能直接看存储。
    这里刻意白盒，换来的是一条真正断言了不变量、而不是「两次返回的 task_id
    一样」这种可能被巧合满足的用例。

    Phase 1.5 起任务记录存在 Redis（`task:{id}:record`），因此这里扫键。
    用 KEYS 而不是维护一个计数器：测试要看到的是**存储的真实状态**，
    再维护一份计数就变成「用一个可能同样写错的实现去验证另一个实现」。
    """
    redis = app.state.redis
    return len(await redis.keys("task:tsk_*:record"))


async def test_create_task_returns_202_with_accepted(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/agent/chat", json={"message": _QUESTION}, headers=analyst_headers
    )
    assert response.status_code == 202

    body = response.json()
    assert body["code"] == "ACCEPTED"
    data = body["data"]
    assert data["task_id"].startswith("tsk_")
    assert data["conversation_id"].startswith("cnv_")
    assert data["trace_id"].startswith("trc_")
    assert data["status"] == "QUEUED"
    assert data["stream_url"] == f"/api/agent/tasks/{data['task_id']}/stream"


async def test_task_trace_id_equals_request_trace_id(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """任务的 trace_id 必须**就是**本次请求的 trace_id。

    详细设计 19.4.1 要求「从 /api/agent/chat 到最终 final_answer 的完整链路
    属于同一个 trace」。若建档时另起一个 trace，API 侧的 HTTP span 与 Worker 侧的
    节点 span 会落在两条链上，Phase 11 的跨进程追踪验收必然失败——
    而那时再回头改，已经写进 agent_task 的历史数据对不上了。
    """
    response = await client.post(
        "/api/agent/chat", json={"message": _QUESTION}, headers=analyst_headers
    )
    body = response.json()

    assert body["data"]["trace_id"] == body["trace_id"]
    assert body["trace_id"] == response.headers["X-Trace-Id"]


async def test_created_task_is_queryable(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """创建后能用返回的 task_id 查回来，且会话归属正确。"""
    created = await client.post(
        "/api/agent/chat", json={"message": _QUESTION}, headers=analyst_headers
    )
    task_id = created.json()["data"]["task_id"]

    response = await client.get(f"/api/agent/tasks/{task_id}", headers=analyst_headers)
    assert response.status_code == 200

    data = response.json()["data"]
    assert data["task_id"] == task_id
    assert data["status"] == "QUEUED"
    assert data["query_text"] == _QUESTION
    assert data["final_answer_md"] is None
    assert data["error_code"] is None
    assert data["trace_id"] == created.json()["data"]["trace_id"]


async def test_reusing_conversation_id(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """FR-CHAT-001：创建或复用会话。传了 conversation_id 就该复用它。"""
    first = await client.post(
        "/api/agent/chat", json={"message": "第一个问题"}, headers=analyst_headers
    )
    conversation_id = first.json()["data"]["conversation_id"]

    second = await client.post(
        "/api/agent/chat",
        json={"message": "第二个问题", "conversation_id": conversation_id},
        headers=analyst_headers,
    )
    assert second.status_code == 202
    assert second.json()["data"]["conversation_id"] == conversation_id
    assert second.json()["data"]["task_id"] != first.json()["data"]["task_id"]


async def test_unknown_conversation_is_400(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    response = await client.post(
        "/api/agent/chat",
        json={"message": _QUESTION, "conversation_id": "cnv_0000000000000000000000"},
        headers=analyst_headers,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ARGUMENT"


async def test_conversation_of_another_user_is_403(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """FR-CHAT-001 异常情况：会话不属于当前用户返回 403。"""
    created = await client.post(
        "/api/agent/chat", json={"message": _QUESTION}, headers=analyst_headers
    )
    conversation_id = created.json()["data"]["conversation_id"]

    other = issue_headers(app, username="other-for-conversation")
    response = await client.post(
        "/api/agent/chat",
        json={"message": _QUESTION, "conversation_id": conversation_id},
        headers=other,
    )
    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"


async def test_idempotent_replay_returns_same_task(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """同一 Idempotency-Key + 同一请求体 -> 同一个任务，且不新建。"""
    headers = {**analyst_headers, "Idempotency-Key": "key-replay-1"}

    first = await client.post("/api/agent/chat", json={"message": _QUESTION}, headers=headers)
    second = await client.post("/api/agent/chat", json={"message": _QUESTION}, headers=headers)

    assert first.status_code == second.status_code == 202
    assert first.json()["data"]["task_id"] == second.json()["data"]["task_id"]
    assert first.json()["message"] != second.json()["message"]  # 首次创建 vs 命中幂等
    assert await _count_tasks(app) == 1


async def test_idempotency_key_reused_with_different_body_is_409(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """同一个键配不同请求体是客户端用错了键，必须报冲突而不是悄悄返回旧任务。"""
    headers = {**analyst_headers, "Idempotency-Key": "key-conflict-1"}

    await client.post("/api/agent/chat", json={"message": "第一个问题"}, headers=headers)
    response = await client.post(
        "/api/agent/chat", json={"message": "完全不一样的问题"}, headers=headers
    )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "TASK_CONFLICT"
    assert body["retryable"] is False


async def test_idempotency_key_is_scoped_to_user(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """幂等键只在同用户范围内唯一（uk_task_user_idempotency）。

    两个用户碰巧用了同一个键，不该互相干扰。
    """
    headers = {**analyst_headers, "Idempotency-Key": "shared-key"}

    mine = await client.post("/api/agent/chat", json={"message": "我的问题"}, headers=headers)
    other_headers = {
        **issue_headers(app, username="other-for-idem"),
        "Idempotency-Key": "shared-key",
    }
    theirs = await client.post(
        "/api/agent/chat", json={"message": "他的问题"}, headers=other_headers
    )

    assert mine.status_code == theirs.status_code == 202
    assert mine.json()["data"]["task_id"] != theirs.json()["data"]["task_id"]
    assert await _count_tasks(app) == 2


async def test_replay_works_even_when_quota_is_full(make_app: Any) -> None:
    """已满额时重放仍应成功：重放没有创建新任务，不该消耗配额。

    若把配额检查放在幂等检查之前，这里会错误地返回 429。
    """
    app: FastAPI = make_app(MAX_RUNNING_TASKS_PER_USER="1")
    async with app_client(app) as (client, auth):
        headers = {**auth, "Idempotency-Key": "k"}
        first = await client.post("/api/agent/chat", json={"message": "问题"}, headers=headers)
        replay = await client.post("/api/agent/chat", json={"message": "问题"}, headers=headers)
        fresh = await client.post(
            "/api/agent/chat",
            json={"message": "另一个问题"},
            headers={**headers, "Idempotency-Key": "k2"},
        )

    assert first.status_code == 202
    assert replay.status_code == 202
    assert replay.json()["data"]["task_id"] == first.json()["data"]["task_id"]
    assert fresh.status_code == 429


async def test_create_task_enqueues_the_job(
    client: AsyncClient, analyst_headers: dict[str, str], app: FastAPI
) -> None:
    """Phase 1.5 起创建任务会真的投递（开发流程 6.3 的「空任务闭环」）。

    投递载荷里必须带上 trace 上下文，否则 Worker 侧的 span 会另起一条 trace，
    Phase 11 的跨进程追踪验收过不了（详细设计 19.4.1）。
    """
    response = await client.post(
        "/api/agent/chat", json={"message": _QUESTION}, headers=analyst_headers
    )
    task_id = response.json()["data"]["task_id"]

    queued = app.state.job_queue.jobs
    assert [job.task_id for job in queued] == [task_id]
    assert queued[0].trace_id == response.json()["data"]["trace_id"]
    assert isinstance(queued[0].trace_context, dict)


async def test_task_is_still_accepted_when_enqueue_fails(
    client: AsyncClient, analyst_headers: dict[str, str], app: FastAPI
) -> None:
    """投递失败**不**改变接口结果（详细设计 17.1）。

    任务已经写库成功了，它只是还没进队列——由 Worker 侧的补偿扫描重投。
    把这种情况报成 500 会让客户端以为任务没建，于是重试；而重试带幂等键时
    会命中同一个任务、看起来「成功」了却仍然没入队，问题反而更难查。
    """
    app.state.job_queue.fail_with = True

    response = await client.post(
        "/api/agent/chat", json={"message": _QUESTION}, headers=analyst_headers
    )

    assert response.status_code == 202
    task_id = response.json()["data"]["task_id"]
    # 任务确实存在，状态停在 QUEUED 等补偿扫描
    detail = await client.get(f"/api/agent/tasks/{task_id}", headers=analyst_headers)
    assert detail.json()["data"]["status"] == "QUEUED"


async def test_message_too_long_is_400(make_app: Any) -> None:
    """长度上限由配置决定（FR-CHAT-001：默认 4,000 字，可配置）。"""
    app: FastAPI = make_app(CHAT_MESSAGE_MAX_LENGTH="10")
    async with app_client(app) as (client, headers):
        response = await client.post(
            "/api/agent/chat", json={"message": "十" * 11}, headers=headers
        )

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ARGUMENT"
    assert "10" in response.json()["message"]
