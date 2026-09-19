"""任务事件总线（开发流程 6.3 施工项 8）。

事件契约来自详细设计 18.1，事件清单来自 18.2。**本阶段只落地骨架**：
Phase 1.5 需要的是「Worker 发得出、别的进程收得到」这条通路，
真正的订阅端（SSE 端点、MySQL 权威重放、Last-Event-ID 补齐）在 Phase 10。

**一个必须先说清的顺序问题**：18.3 规定「先写 MySQL 取得 sequence，
再 XADD 到 Redis Stream」，Redis 只负责快、MySQL 负责对。
Phase 1.5 还没有 MySQL（Phase 2 才有），因此这里由 Redis Stream ID
反推一个单调递增的 `sequence`——注意流 ID 的 `<毫秒>-<毫秒内序号>`
只要不做裁剪删除就是严格递增的，所以 `ms * 1000 + n` 是一个可用的临时序号。
Phase 2 接入 `agent_trace_event` 后，这个推导会被那张表的计数器取代，
**事件模型本身不变**，改的只是 `publish` 里赋值的那两行。

**`task.cancelled` 不在 18.2 的清单里**：17.5 要求取消必须可用，
但 18.2 没有对应事件，取消后的任务在事件流上无话可说。已作为文档缺口登记，
建议在 18.2 补一行 `| task.cancelled | Worker 确认取消 | final_status |`。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import redis.asyncio as aioredis

from app.core.config import Settings
from app.domain.events import TaskEvent, TaskEventType
from app.infrastructure.redis import RedisKey, load_json

#: 事件模型与类型定义在 `app/domain/events.py`——**图里的节点也要发事件**，
#: 而 `app/agent/**` 不能反向依赖 services（依赖只能向右）。这里再导出一次，
#: 让"事件总线这一侧"的调用方（`TaskRunner`、测试）不必同时记两个模块路径。
__all__ = ["EventBus", "RedisStreamEventBus", "TaskEvent", "TaskEventType", "last_event_id"]

logger = logging.getLogger(__name__)


class EventBus(Protocol):
    async def publish(
        self,
        *,
        task_id: str,
        trace_id: str,
        event_type: TaskEventType,
        data: dict[str, Any] | None = None,
        node: str | None = None,
        step_id: str | None = None,
    ) -> TaskEvent: ...

    async def read(
        self, task_id: str, *, after_id: str = "0-0", count: int = 100
    ) -> list[TaskEvent]: ...

    def subscribe(
        self, task_id: str, *, after_id: str = "0-0", block_ms: int = 1000
    ) -> AsyncIterator[TaskEvent | None]: ...

    async def finish(self, task_id: str) -> None: ...


class RedisStreamEventBus:
    """Redis Stream 实现。

    每条事件序列化成**单个 field**（`payload`）再入流，而不是把模型字段
    逐列摊成 Stream 的多个 field。这样事件模型的演进（加字段、改类型）
    完全不用动存储层，代价只是从 `redis-cli` 直接看流时看到一坨 JSON——
    而排查时真正要看的是顺序和内容，不是字段在不在列上。
    """

    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._tuning = settings.redis_tuning

    async def publish(
        self,
        *,
        task_id: str,
        trace_id: str,
        event_type: TaskEventType,
        data: dict[str, Any] | None = None,
        node: str | None = None,
        step_id: str | None = None,
    ) -> TaskEvent:
        event = TaskEvent(
            type=event_type,
            timestamp=datetime.now(UTC),
            task_id=task_id,
            trace_id=trace_id,
            node=node,
            step_id=step_id,
            data=data or {},
        )
        key = RedisKey.task_events(task_id)
        stream_id = _as_text(
            await self._redis.xadd(
                key,
                # **不把 event_id / sequence 存进流里**：它们由 Stream ID 派生，
                # 存一份就等于存了第二份真相，而两份写法一定会在某次改动后分家
                # （这里原本就写错过一版：入流的是派生之前的那份，sequence 全是 0）。
                {"payload": event.model_dump_json(exclude={"event_id", "sequence"})},
                maxlen=self._tuning.stream_maxlen,
                approximate=True,
            )
        )
        return event.model_copy(update={"event_id": stream_id, "sequence": _sequence_of(stream_id)})

    async def read(
        self, task_id: str, *, after_id: str = "0-0", count: int = 100
    ) -> list[TaskEvent]:
        """读 `after_id` 之后的事件（不含 `after_id`）。

        用 XRANGE 而不是 XREAD：XRANGE 是纯查询、无连接状态，
        Phase 10 的 SSE 端点需要的是「按区间补齐」；阻塞订阅等真正有
        长连接需求时再加，现在加进来只是一段没有调用方的代码。
        """
        key = RedisKey.task_events(task_id)
        raw = await self._redis.xrange(key, min=f"({after_id}", max="+", count=count)
        events: list[TaskEvent] = []
        for stream_id, fields in raw:
            payload = load_json(fields.get("payload"))
            if payload is None:
                # 流里有本进程写不出的条目，只能是别的东西写错了键。
                # 跳过并告警，不让一条脏数据毁掉整次重放。
                logger.warning("事件流条目无法解析，已跳过：%s", RedisKey.task_events(task_id))
                continue
            # 标识从流本身恢复，与 publish 的派生方式同一份逻辑（见 _sequence_of）
            text_id = _as_text(stream_id)
            events.append(
                TaskEvent.model_validate(
                    {**payload, "event_id": text_id, "sequence": _sequence_of(text_id)}
                )
            )
        return events

    async def subscribe(
        self, task_id: str, *, after_id: str = "0-0", block_ms: int = 1000
    ) -> AsyncIterator[TaskEvent | None]:
        """阻塞订阅：从 `after_id` 之后开始，持续产出新事件。

        **与 `read` 的分工**：`read` 是"按区间取一次"（重连补齐用），
        这里是"一直等"（实时增量用）。两条通道对应 18.3 的双通道设计，
        而它们读的是**同一条流**——这正是"Redis 只负责快"的含义：
        实时与补齐不需要两套数据。

        **超时产出 `None`，不是什么都不产出**：调用方要的正是这个
        "这段时间什么都没发生"的信号——心跳就是靠它发的。把这个信号
        在总线里吞掉（`continue`），订阅端就永远不知道自己闲了多久，
        心跳只能靠另起一个定时器，而那会与事件循环抢同一份状态。

        **不用 `XREADGROUP`**（消费者组）：消费者组适合"一条消息只被一个
        消费者处理"，而 SSE 是**广播**——多个客户端订阅同一任务时每条事件
        都要发给每一个。用消费者组会让第二个订阅者什么都收不到。
        """
        cursor = after_id
        while True:
            raw = await self._redis.xread(
                {RedisKey.task_events(task_id): cursor}, count=100, block=block_ms
            )
            if not raw:
                yield None
                continue
            for _key, entries in raw:
                for stream_id, fields in entries:
                    text_id = _as_text(stream_id)
                    cursor = text_id
                    payload = load_json(fields.get("payload"))
                    if payload is None:
                        logger.warning("事件流条目无法解析，已跳过：%s", text_id)
                        continue
                    yield TaskEvent.model_validate(
                        {**payload, "event_id": text_id, "sequence": _sequence_of(text_id)}
                    )

    async def finish(self, task_id: str) -> None:
        """任务结束：给流设置保留期（详细设计 4.4：结束后保留 1 小时）。

        只设 TTL，不删流——重连的客户端还要靠它补齐，
        这正是「MySQL 权威 + Redis 快通道」里快通道的职责边界。
        """
        await self._redis.expire(RedisKey.task_events(task_id), self._tuning.stream_ttl_seconds)


def _as_text(value: Any) -> str:
    """`decode_responses` 可能是 False（比如测试替身），两种都要能处理。"""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sequence_of(stream_id: str) -> int:
    """`<毫秒>-<毫秒内序号>` -> 单调递增整数。

    解析失败时回落成 0 而不是抛异常：事件已经写进流了，
    为了一个序号把这次发布判成失败，会让调用方以为事件没发出去。
    代价是排序退化，值得用一条日志换。
    """
    head = stream_id.split("-", 1)
    try:
        return int(head[0]) * 1000 + (int(head[1]) if len(head) > 1 else 0)
    except ValueError:
        logger.warning("无法从流 ID 推导序号，按 0 处理：%s", stream_id)
        return 0


def last_event_id(events: Sequence[TaskEvent]) -> str:
    """取最后一条事件的 id，供断线重连时作为 `after_id`。"""
    return events[-1].event_id if events else "0-0"
