"""SSE 帧的组装与任务事件流的生成（详细设计 17.3 / 18.2 / 18.3 / 18.4）。

## 双通道：先用 `read` 补齐，再用 `subscribe` 增量

```text
建连 → 发 snapshot（任务当前状态）
     → read(after_id=Last-Event-ID)  ← 补齐断开期间的事件
     → subscribe(...)                ← 实时增量（同一条流，只是换个读法）
     → 见到终态事件 → 发 done → 关闭
```

两条通道读的是**同一条 Redis Stream**（`task:{id}:events`）。18.3 的原设计是
"MySQL 权威重放 + Redis 实时增量"，本实现把重放也放在 Redis 上——
**这是一处如实记下的偏离**：`agent_trace_event` 目前只在任务收尾时批量落
节点级轨迹，与流上的事件不是同一批（登记在 CLAUDE.md 的「两条通道尚未合并
成一条」）。硬把一个批次不同的来源当重放源，比"只用流"更误导。
流在任务结束后保留 1 小时（`REDIS_TUNING__STREAM_TTL_SECONDS`），
超出这个窗口的重连走 `snapshot` + `replay_lost=true`，这也是 18.4 允许的形态。

## 帧格式

```text
id: <event_id>          ← 只有业务事件带，客户端回传它作 Last-Event-ID
event: <type>
data: <18.1 的事件模型，去掉 event_id（它已经在 id 行上）>
```

**`snapshot` / `heartbeat` / `done` 三条不带 `id:`**：它们不是任务事件，
让它们推进客户端的续传游标，会让下一次重连从一个"不存在于流里的位置"开始
（表现为重连后收到的第一条事件莫名其妙地晚了一截）。

## 心跳是"空闲触发"

15 秒内没有业务事件才发一条。**不做无条件定时**：事件密集的任务里，
无条件心跳会让流里一半的帧是心跳，而客户端要的是"服务还活着"这一个信号。

## 出基础设施故障时：结束流，但**不发 `done`**

`done` 的语义是"任务结束了"。Redis 断连、socket 超时这类事故发生时，
任务**还在跑**（它在另一个进程里）——发 `done` 等于对它撒谎。
客户端的契约因此是「收到 `done` 才算正常结束」，异常终止时它回退到
`GET /tasks/{id}` 轮询（R 8.2 的「SSE 回退为轮询状态接口」），
而那是一条**正确**的路径：事件流只是"看得见"，不影响任务本身。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.config import Settings
from app.domain.events import TaskEvent, TaskEventType
from app.domain.task import Task, TaskStatus
from app.services.event_bus import EventBus

logger = logging.getLogger(__name__)

#: 终态事件 → 任务的最终状态。**由事件本身推**，不回头查库：
#: 建连时读到的任务状态与流里的终态事件之间有一个天然的竞态窗口
#: （Worker 恰好在这一瞬间收尾），查库会让重连的客户端偶发地拿不到 done。
_FINAL_STATUS: dict[TaskEventType, TaskStatus] = {
    TaskEventType.TASK_COMPLETED: TaskStatus.SUCCEEDED,
    TaskEventType.TASK_FAILED: TaskStatus.FAILED,
    TaskEventType.TASK_CANCELLED: TaskStatus.CANCELLED,
}


@dataclass(frozen=True, slots=True)
class SseFrame:
    """一条待发送的 SSE 帧。**只描述内容，不拼字符串**——
    拼装（含换行与转义）统一走 `format_frame`，那里是唯一会被测的出口。
    """

    event_type: str
    data: dict[str, object]
    #: 空表示这条帧不带 `id:` 行（快照/心跳/终止三条，见模块 docstring）
    event_id: str = ""


def format_frame(frame: SseFrame) -> str:
    """SSE 帧的文本形式。

    `data` 用 `json.dumps(..., ensure_ascii=False)`：SSE 是 UTF-8 文本流，
    中文直出可读（排查时用 `curl` 就能看），而转义成 `\\uXXXX` 只是徒增体积。

    **`data` 里不出现换行**：SSE 的 `data:` 行遇到裸换行就断了，事件会被
    切成两条残缺的。JSON 序列化天然把换行转义掉（`\\n`），所以这里安全——
    但这也意味着**不能改用 `str()` 之类的手工拼接**（那是这条注释存在的理由）。
    """
    lines: list[str] = []
    if frame.event_id:
        lines.append(f"id: {frame.event_id}")
    lines.append(f"event: {frame.event_type}")
    lines.append(f"data: {json.dumps(frame.data, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


def _event_frame(event: TaskEvent) -> SseFrame:
    """业务事件 → 帧。`event_id` 提到帧头，其余按 18.1 进 `data`。"""
    return SseFrame(
        event_type=event.type.value,
        data=event.model_dump(mode="json", exclude={"event_id"}),
        event_id=event.event_id,
    )


def snapshot_frame(task: Task, *, replay_lost: bool) -> SseFrame:
    """建连时的状态快照。

    **每次建连都发**，不只在上次断线的地方：客户端拿它对齐"我看到的进度"
    与服务端的当前状态，而它只有几十个字节。`replay_lost=true` 表示
    **有事件已经取不回来了**（流被清理），此时快照不是冗余，而是客户端
    唯一能拿到的状态——18.4 要求的正是这个标记。
    """
    return SseFrame(
        event_type=TaskEventType.SNAPSHOT.value,
        data={
            "task_id": task.id,
            "status": task.status.value,
            "answer_available": task.final_answer_md is not None,
            "replay_lost": replay_lost,
        },
    )


def heartbeat_frame() -> SseFrame:
    """心跳。`server_time` 让客户端能算出自己的时钟偏差（18.2 的字段）。"""
    return SseFrame(
        event_type=TaskEventType.HEARTBEAT.value,
        data={"server_time": datetime.now(UTC).isoformat()},
    )


def done_frame(status: TaskStatus) -> SseFrame:
    """流终止（18.2）。**由订阅端发出，不由 Worker 发**。"""
    return SseFrame(event_type=TaskEventType.DONE.value, data={"final_status": status.value})


async def _replay_lost(bus: EventBus, task_id: str, after_id: str | None) -> bool:
    """客户端要的续传位置**早于流里最早的条目**吗（= 中间那段已经取不回来）。

    只在客户端带了 `Last-Event-ID` 时判断，且**证据必须是流本身**：
    "重放返回空"不能当判据——那既可能是"确实没有新事件"，
    也可能是"你要的位置早被裁掉了"。前者的客户端看到 `replay_lost=true`
    会以为丢了事件，然后去做一次多余的全量对齐。

    流为空时返回 True：要么被 TTL 清理了，要么这个任务从来没有过事件。
    对带着游标来的客户端，这两种情况是同一件事——它要的位置已经不在。
    """
    if not after_id:
        return False
    earliest = await bus.read(task_id, count=1)
    if not earliest:
        return True
    return _stream_id_less(after_id, earliest[0].event_id)


def _stream_id_less(left: str, right: str) -> bool:
    """比较两个 Redis Stream ID（`<毫秒>-<序号>`）。解析不了就说"不早于"。"""
    try:
        left_ms, left_seq = (int(part) for part in left.split("-", 1))
        right_ms, right_seq = (int(part) for part in right.split("-", 1))
    except ValueError:
        # ID 形状不认识（例如客户端回传了一个我们自己没发过的东西）：
        # **不断言丢事件**——误报会让客户端做一次没有必要的全量对齐
        return False
    return (left_ms, left_seq) < (right_ms, right_seq)


async def stream_frames(
    *,
    task: Task,
    bus: EventBus,
    settings: Settings,
    after_id: str | None = None,
) -> AsyncIterator[SseFrame]:
    """一个任务的 SSE 帧序列（不含 HTTP 细节，因此可以脱离 ASGI 测）。

    Args:
        task: 建连时读到的任务（已过鉴权与归属校验）。
        after_id: 客户端带回来的 `Last-Event-ID`；没有就是重头。
    """
    tuning = settings.sse_tuning
    cursor = after_id or "0-0"

    # ① 快照。**先发它**：客户端立刻知道"现在是什么状态"，
    #    而不必等第一批事件（长任务的第一条可能几秒后才来）
    #
    # **重放不设条数上限**：它天然被流的 `MAXLEN`（默认 10000）框住，
    # 而一个任务的事件是几十条量级。设一个更小的上限等于**丢事件**——
    # 丢掉的那些之后再也拿不回来（游标已经越过它们），
    # 而"丢了一部分"与"全都补上了"在客户端看来是一样的。
    try:
        replayed = await bus.read(task.id, after_id=cursor)
        yield snapshot_frame(task, replay_lost=await _replay_lost(bus, task.id, after_id))

        # ② 补齐
        last_emitted = datetime.now(UTC)
        for event in replayed:
            cursor = event.event_id
            last_emitted = datetime.now(UTC)
            yield _event_frame(event)
            if (final := _FINAL_STATUS.get(event.type)) is not None:
                yield done_frame(final)
                return

        # 任务在建连时就已结束、而流里没有终态事件（被清理过）：没有别的可等，
        # 直接收尾。**不继续阻塞订阅**——那个 wait 会一直挂到客户端超时
        if task.is_terminal and not replayed:
            yield done_frame(task.status)
            return

        # ③ 实时增量。**`None` 是"这一段空闲"的信号**，不是错误——
        # 心跳就是靠它发的（见 `EventBus.subscribe` 的说明）
        async for live in bus.subscribe(task.id, after_id=cursor, block_ms=tuning.poll_block_ms):
            if live is None:
                idle = (datetime.now(UTC) - last_emitted).total_seconds()
                if idle >= tuning.heartbeat_seconds:
                    last_emitted = datetime.now(UTC)
                    yield heartbeat_frame()
                continue
            last_emitted = datetime.now(UTC)
            yield _event_frame(live)
            if (final := _FINAL_STATUS.get(live.type)) is not None:
                yield done_frame(final)
                return
    except Exception:
        # **降级：结束这条流，但不发 `done`**（R 8.2「SSE 回退为轮询状态接口」）。
        #
        # 不把异常抛出去，是因为此刻响应已经开始了——抛出去的表现是
        # `RuntimeError: Caught handled exception, but response already started`，
        # 客户端看到的是"连接被掐断"（chunked 响应不完整），而那个症状
        # 与"浏览器/代理把连接断了"完全一样，指不到 Redis 上去（实测踩到过）。
        #
        # 也**不发 `done`**：`done` 的语义是"任务结束了"。这条流是异常终止的，
        # 客户端的契约正是"没有 `done` 就不是正常结束"——它据此回退到轮询
        # `GET /tasks/{id}`，那是一条**正确**的路径（任务是独立进程在跑的，
        # 事件流断掉不影响它）。
        bound = {"task_id": task.id, "conversation_id": task.conversation_id}
        logger.exception("事件流异常终止（客户端应退回轮询状态接口）", extra=bound)
        return


__all__ = [
    "SseFrame",
    "done_frame",
    "format_frame",
    "heartbeat_frame",
    "snapshot_frame",
    "stream_frames",
]
