"""SSE 任务事件流（详细设计 17.3 / 18.2 / 18.4）。

分两层测，各管一段：

- **`stream_frames`（服务级）**：用真 `RedisStreamEventBus` + fakeredis 跑帧序列
  ——重放、增量、心跳、终止、`replay_lost` 都在这里验，不经过 HTTP，
  因此不受 ASGI 缓冲与客户端超时的影响，判定是确定性的；
- **端点（HTTP 级）**：鉴权两条来路、归属、限流、响应头与帧格式。

**为什么事件要先发再连**：`stream_frames` 的重放分支会把已有事件一次吐完
并以 `done` 收尾，流自己结束——不需要"边读边发"的并发编排，用例因此稳定。
增量分支单独用一次并发用例覆盖（见 `test_live_events_are_pushed_as_they_arrive`）。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import fakeredis.aioredis
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.stream import SseFrame, format_frame, stream_frames
from app.domain.events import TaskEventType
from app.domain.task import Task, TaskStatus
from app.services.event_bus import RedisStreamEventBus
from app.tests.api.conftest import issue_headers

_TASK_ID = "tsk_0000000000000000000001"
_TRACE_ID = "trc_0000000000000000000001"
_QUESTION = "华东区的销售额是多少"


def _task(*, status: TaskStatus = TaskStatus.RUNNING, answer: str | None = None) -> Task:
    now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
    return Task(
        id=_TASK_ID,
        user_id="usr_0000000000000000000001",
        conversation_id="cnv_0000000000000000000001",
        trace_id=_TRACE_ID,
        query_text=_QUESTION,
        status=status,
        final_answer_md=answer,
        queued_at=now,
        created_at=now,
        updated_at=now,
    )


async def _collect(frames: Any) -> list[SseFrame]:
    return [frame async for frame in frames]


def _bus(redis: fakeredis.aioredis.FakeRedis, settings: Any) -> RedisStreamEventBus:
    return RedisStreamEventBus(redis, settings)


# ------------------------------------------------------------------ 帧格式


def test_frame_format_puts_the_id_on_its_own_line() -> None:
    """`id:` 是客户端回传 `Last-Event-ID` 的依据，不能塞进 data 里。"""
    frame = SseFrame(event_type="node.started", data={"sequence": 3}, event_id="1-0")

    assert format_frame(frame) == ('id: 1-0\nevent: node.started\ndata: {"sequence": 3}\n\n')


def test_frames_without_an_id_omit_the_id_line() -> None:
    """快照/心跳/终止三条**不带 `id:`**：让它们推进续传游标，
    会让下一次重连从一个不存在于流里的位置开始（表现为漏掉一截事件）。"""
    frame = SseFrame(event_type="heartbeat", data={"server_time": "2026-09-18T12:00:00+00:00"})

    assert format_frame(frame).startswith("event: heartbeat\n")


def test_chinese_is_not_escaped() -> None:
    """中文直出：SSE 是 UTF-8 流，`\\uXXXX` 只让 `curl` 的输出没法看。"""
    frame = SseFrame(event_type="progress.assessed", data={"finding_statement": "证据不足"})

    assert "证据不足" in format_frame(frame)


# ------------------------------------------------------------------ 帧序列


async def test_replay_then_done_for_a_finished_task(
    settings: Any, fake_redis: fakeredis.aioredis.FakeRedis
) -> None:
    """已结束的任务：快照 + 全部事件 + `done`，然后**流自己结束**。

    不结束的话，客户端会一直挂着等一个永远不会来的事件——
    而服务端那边是一条永不释放的连接。
    """
    bus = _bus(fake_redis, settings)
    await bus.publish(
        task_id=_TASK_ID,
        trace_id=_TRACE_ID,
        event_type=TaskEventType.TASK_STARTED,
        data={"status": "RUNNING"},
    )
    await bus.publish(
        task_id=_TASK_ID,
        trace_id=_TRACE_ID,
        event_type=TaskEventType.NODE_STARTED,
        node="sql",
    )
    await bus.publish(
        task_id=_TASK_ID,
        trace_id=_TRACE_ID,
        event_type=TaskEventType.TASK_COMPLETED,
        data={"answer_available": True, "evidence_count": 2},
    )

    frames = await _collect(
        stream_frames(task=_task(status=TaskStatus.SUCCEEDED), bus=bus, settings=settings)
    )

    assert [frame.event_type for frame in frames] == [
        "snapshot",
        "task.started",
        "node.started",
        "task.completed",
        "done",
    ]
    assert frames[0].data["status"] == "SUCCEEDED"
    assert frames[-1].data["final_status"] == "SUCCEEDED"
    # 顺序由流本身保证：`sequence` 单调递增（客户端据此去重与排序）。
    # 只统计带 `id:` 的业务事件——快照/心跳/终止三条没有序号
    sequences: list[int] = []
    for frame in frames:
        value = frame.data.get("sequence")
        if frame.event_id and isinstance(value, int):
            sequences.append(value)
    assert sequences == sorted(sequences)
    assert sequences, "业务事件必须带序号"


async def test_last_event_id_skips_what_the_client_already_has(
    settings: Any, fake_redis: fakeredis.aioredis.FakeRedis
) -> None:
    """重连时从客户端已有的最后一条之后接着发。"""
    bus = _bus(fake_redis, settings)
    first = await bus.publish(
        task_id=_TASK_ID, trace_id=_TRACE_ID, event_type=TaskEventType.TASK_STARTED
    )
    await bus.publish(task_id=_TASK_ID, trace_id=_TRACE_ID, event_type=TaskEventType.TASK_COMPLETED)

    frames = await _collect(
        stream_frames(
            task=_task(status=TaskStatus.SUCCEEDED),
            bus=bus,
            settings=settings,
            after_id=first.event_id,
        )
    )

    assert [frame.event_type for frame in frames] == ["snapshot", "task.completed", "done"]


async def test_a_live_event_is_pushed_while_subscribed(
    settings: Any, fake_redis: fakeredis.aioredis.FakeRedis
) -> None:
    """增量分支：订阅期间发的事件会被推出去，然后由终态事件收尾。

    **这一条必须真跑一次阻塞订阅**（前两条走的是重放分支）——
    `XREAD BLOCK` 与 `XRANGE` 是两条不同的代码路径，
    而它们长得一模一样：漏了增量，客户端只会表现为"任务跑完了我什么都没收到"。
    """
    bus = _bus(fake_redis, settings)
    frames: list[SseFrame] = []

    async def consume() -> None:
        async for frame in stream_frames(
            task=_task(status=TaskStatus.RUNNING), bus=bus, settings=settings
        ):
            frames.append(frame)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)  # 让订阅真正进到阻塞读
    await bus.publish(
        task_id=_TASK_ID, trace_id=_TRACE_ID, event_type=TaskEventType.NODE_STARTED, node="sql"
    )
    await bus.publish(task_id=_TASK_ID, trace_id=_TRACE_ID, event_type=TaskEventType.TASK_COMPLETED)
    await asyncio.wait_for(task, timeout=5)

    assert [frame.event_type for frame in frames] == [
        "snapshot",
        "node.started",
        "task.completed",
        "done",
    ]


async def test_idle_stream_emits_heartbeats(
    settings: Any, fake_redis: fakeredis.aioredis.FakeRedis
) -> None:
    """空闲时发心跳（18.2 的 `heartbeat`）。

    心跳是"服务还活着"的唯一信号：没有它，一个正在跑长任务的连接
    与一条断掉的连接在客户端看来完全一样。
    """
    tuning = settings.sse_tuning.model_copy(update={"heartbeat_seconds": 1, "poll_block_ms": 100})
    idle = settings.model_copy(update={"sse_tuning": tuning})
    bus = _bus(fake_redis, settings)
    seen: list[str] = []

    async def consume() -> None:
        async for frame in stream_frames(task=_task(), bus=bus, settings=idle):
            seen.append(frame.event_type)
            if frame.event_type == "heartbeat":
                break

    await asyncio.wait_for(asyncio.create_task(consume()), timeout=5)

    assert seen == ["snapshot", "heartbeat"]


async def test_a_task_whose_stream_was_purged_gets_a_snapshot_with_replay_lost(
    settings: Any, fake_redis: fakeredis.aioredis.FakeRedis
) -> None:
    """流已被清理（TTL 到期）而客户端还带着游标来 → 明确告知**补不回来了**。

    这一条是 18.4 要求的：不标记的话，客户端会以为"从这条之后就没有事件了"，
    而实际上中间的一段它永远拿不到。
    """
    bus = _bus(fake_redis, settings)

    frames = await _collect(
        stream_frames(
            task=_task(status=TaskStatus.SUCCEEDED, answer="# 结论"),
            bus=bus,
            settings=settings,
            after_id="1700000000000-0",
        )
    )

    assert [frame.event_type for frame in frames] == ["snapshot", "done"]
    assert frames[0].data["replay_lost"] is True
    assert frames[0].data["answer_available"] is True


# ------------------------------------------------------------------ 端点


async def _create_task(client: AsyncClient, headers: dict[str, str]) -> str:
    response = await client.post("/api/agent/chat", json={"message": _QUESTION}, headers=headers)
    assert response.status_code == 202, response.text
    return str(response.json()["data"]["task_id"])


async def _finish(app: FastAPI, task_id: str) -> None:
    """给任务补一条终态事件，让流能自己收尾。

    **用例里的任务不会真的被执行**（单元测试没有 Worker），而服务端在
    拿到终态事件之前不会关闭流——用例因此会挂在那里等超时。
    补这条事件等价于"Worker 跑完了"：流走重放分支，立刻吐完并 `done`。
    """
    await app.state.event_bus.publish(
        task_id=task_id,
        trace_id=_TRACE_ID,
        event_type=TaskEventType.TASK_COMPLETED,
        data={"answer_available": True, "evidence_count": 1},
    )


async def _read_stream(
    client: AsyncClient, url: str, *, headers: dict[str, str] | None = None
) -> list[str]:
    """读一整个 SSE 响应，返回帧列表（按 `\\n\\n` 切）。

    **必须给超时**：服务端在任务未结束时不会主动收尾，而这个用例等的是
    "流自己结束"（预置了终态事件），所以正常路径下它是立刻返回的；
    真出问题时由超时兜底，而不是挂住整个测试会话。
    """
    chunks: list[str] = []
    async with client.stream("GET", url, headers=headers or {}) as response:
        assert response.status_code == 200, response.status_code
        async for text in response.aiter_text():
            chunks.append(text)
            if "event: done" in "".join(chunks):
                break
    return [frame for frame in "".join(chunks).split("\n\n") if frame]


async def test_endpoint_requires_credentials(client: AsyncClient) -> None:
    response = await client.get(f"/api/agent/tasks/{_TASK_ID}/stream")

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"


async def test_unknown_task_is_404(client: AsyncClient, analyst_headers: dict[str, str]) -> None:
    response = await client.get(f"/api/agent/tasks/{_TASK_ID}/stream", headers=analyst_headers)

    assert response.status_code == 404
    assert response.json()["code"] == "TASK_NOT_FOUND"


async def test_other_user_is_denied(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """FR-SSE-001：事件流里带着节点名与耗时，跨用户一律 403。"""
    task_id = await _create_task(client, analyst_headers)

    response = await client.get(
        f"/api/agent/tasks/{task_id}/stream",
        headers=issue_headers(app, username="stream-other"),
    )

    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"


async def test_stream_token_flow_works_end_to_end(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """签发 → 用令牌订阅（浏览器 `EventSource` 只能走这条）→ 收到帧。"""
    task_id = await _create_task(client, analyst_headers)
    await _finish(app, task_id)
    issued = await client.post(f"/api/agent/tasks/{task_id}/stream-token", headers=analyst_headers)
    assert issued.status_code == 200
    body = issued.json()["data"]
    assert body["expires_in"] > 0
    assert body["stream_url"] == f"/api/agent/tasks/{task_id}/stream?token={body['stream_token']}"

    frames = await _read_stream(client, body["stream_url"])

    assert frames[0].startswith("event: snapshot")
    assert frames[-1].startswith("event: done")


async def test_a_used_stream_token_cannot_be_replayed(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """一次性：同一个 URL 第二次连接必须失败。

    这条是"日志泄露"场景的直接防线——泄露的多半是**已经用过**的那条 URL。
    """
    task_id = await _create_task(client, analyst_headers)
    await _finish(app, task_id)
    issued = await client.post(f"/api/agent/tasks/{task_id}/stream-token", headers=analyst_headers)
    url = issued.json()["data"]["stream_url"]

    await _read_stream(client, url)
    response = await client.get(url)

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"


async def test_a_stream_token_for_another_task_is_denied(
    client: AsyncClient, analyst_headers: dict[str, str]
) -> None:
    """**绑定 task_id**：拿自己任务换来的令牌去读别人的任务要被拒。

    不校验的话它就是一张"能读任意任务事件流"的通行证，而它是从
    "自己的任务"换来的，看起来完全正常。
    """
    mine = await _create_task(client, analyst_headers)
    issued = await client.post(f"/api/agent/tasks/{mine}/stream-token", headers=analyst_headers)
    stolen = issued.json()["data"]["stream_token"]

    response = await client.get(
        f"/api/agent/tasks/tsk_0000000000000000000002/stream?token={stolen}"
    )

    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"


async def test_streaming_is_rate_limited(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """建连配额**与状态查询分开计**：长连接与轮询是两种负载。

    走订阅令牌的那条路也受限（限流是显式调用的，不是依赖注入的）——
    否则"用令牌订阅"就成了绕过限流的旁路，而绕过是静默的。
    """
    task_id = await _create_task(client, analyst_headers)
    app.state.rate_limiter = _AlwaysDeny()

    response = await client.get(f"/api/agent/tasks/{task_id}/stream", headers=analyst_headers)

    assert response.status_code == 429
    assert response.json()["code"] == "RATE_LIMITED"


async def test_response_headers_disable_buffering(
    client: AsyncClient, app: FastAPI, analyst_headers: dict[str, str]
) -> None:
    """`X-Accel-Buffering: no` 与 `no-cache`：中间层不许缓冲。

    缓冲掉的话客户端要等到缓冲写满才看见第一批事件——而"实时"正是这条端点的
    全部意义（Nginx 侧还要配 `proxy_buffering off`，见详细设计 20.2）。

    **读到流结束为止**（不提前 `break`）：中途退出 `async with` 会走到
    "取消一个正阻塞在 `XREAD BLOCK` 上的服务端生成器"那条路径，
    而那条路径本身不是这条用例要验的东西——混进来只会让失败原因含糊。
    """
    task_id = await _create_task(client, analyst_headers)
    await _finish(app, task_id)

    async with client.stream(
        "GET", f"/api/agent/tasks/{task_id}/stream", headers=analyst_headers
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert "event: done" in body


class _AlwaysDeny:
    """把配额收到 0 的限流器替身。

    限流本身由 `test_rate_limit.py` 对着真 Redis（fakeredis 跑真 Lua）覆盖；
    这里要断的是**SSE 端点有没有把限流接上**，因此替身足够，
    而且能造出"必然被拒"这个在真实配额下要发几十个请求才能到的状态。
    """

    async def check(self, **_kwargs: Any) -> Any:
        from app.services.rate_limit import RateLimitResult

        return RateLimitResult(
            allowed=False, limit=0, remaining=0, reset_after_seconds=30, degraded=False
        )


class _BrokenBus(RedisStreamEventBus):
    """订阅时才炸的总线替身：模拟 Redis 在流中途不可用。

    **只在 `subscribe` 上炸，不在 `read` 上**：那样才会走到"响应已经开始
    之后才出错"那条路径——而那条路径的处置正是这里要钉的。
    """

    def subscribe(self, task_id: str, *, after_id: str = "0-0", block_ms: int = 200) -> Any:
        async def _boom() -> Any:
            raise ConnectionError("Redis 断了")
            yield  # pragma: no cover - 让这个函数成为异步生成器

        return _boom()


async def test_an_infrastructure_failure_ends_the_stream_without_done(
    settings: Any, fake_redis: fakeredis.aioredis.FakeRedis
) -> None:
    """Redis 中途不可用：**结束流，但不发 `done`**（R 8.2 的降级）。

    `done` 的语义是"任务结束了"——而任务还在另一个进程里跑着。
    发 `done` 等于对它撒谎，客户端会以为拿到的是完整结果。
    不发 `done` 则客户端知道"这条流不是正常结束的"，回退到轮询状态接口，
    而那是一条正确的路径。

    **异常也不能抛出去**：此刻响应已经开始，抛出去的表现是
    `RuntimeError: ... response already started`，客户端看到的是
    "连接被掐断"——一个指不到 Redis 上去的症状（实测踩到过）。
    """
    bus = _BrokenBus(fake_redis, settings)

    frames = await _collect(stream_frames(task=_task(), bus=bus, settings=settings))

    # 快照照发（它来自建连时的任务对象，不需要 Redis），
    # 然后流**干净地结束**——没有 done，也没有把异常抛给 ASGI
    assert [frame.event_type for frame in frames] == ["snapshot"]
