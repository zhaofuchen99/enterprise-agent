"""节点埋点：把每个节点的进入与离开记成轨迹事件（16.7）。

## 为什么用包装器而不是在每个节点里写埋点

八个节点各写一遍「记开始、记结束、算耗时」，漏掉一处不会有任何症状——
那个节点在轨迹里就是"不存在"，而轨迹本来就不完整，看不出少了什么。
包一层之后，「每个节点都有轨迹」变成结构性保证。

## 为什么两个事件都要

只记 `node.completed` 的话，一个卡住的节点在轨迹上表现为"什么都没发生"。
读者需要知道的是"它进去了、还没出来"——这与"压根没跑到"是不同的故障，
而两者在只有完成事件的轨迹上长得一样。

## 异常路径在这里记不下来，得靠 `trace_incomplete`

节点抛异常时图会中止，而**中止节点的更新不会被写进 State**（LangGraph 的语义），
因此"进得去出不来"的那条事件在进程内就丢了，落库时也就没有这一任务的任何轨迹。

这是本层的能力边界，不假装它管用：异常路径的线索是任务的 `error_code`
与 `trace_incomplete`（`TaskRunner` 在收尾时置位，见 FR-TRACE-001 的异常情况）。
**节点"返回了错误"（不抛异常）那条路径才是这里管得住的**——它更常见，
且返回值会被真的写进 State。

要连异常路径也留下痕迹，得在节点入口就把事件投递到事件总线（而不是等 State 合并），
那是 Phase 10 的 SSE 侧要解决的事，登记在案。
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from app.agent.state import AgentState
from app.domain.task import TaskStatus
from app.domain.trace import NodeTrace


def traced(name: str, node: Callable[[AgentState], Any]) -> Callable[[AgentState], Awaitable[Any]]:
    """包装一个节点，产出 `node.started` + `node.completed` / `node.failed`。

    包装成 `async def` 对同步与异步节点都成立：同步节点直接调用即可，
    LangGraph 看到的是同一个异步契约。**不给两类节点写两条包装路径**——
    那会让"某个节点忘了被包"变成一件从签名上看不出来的事。
    """

    async def wrapper(state: AgentState) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        started = time.monotonic()
        enter = NodeTrace(
            node=name,
            event_type="node.started",
            status="RUNNING",
            created_at=started_at,
        )
        update = node(state)
        if inspect.isawaitable(update):
            update = await update
        update = update or {}
        duration_ms = int((time.monotonic() - started) * 1000)
        status = str(update.get("execution_status", TaskStatus.SUCCEEDED))
        leave = NodeTrace(
            node=name,
            # **失败换事件类型，不只换 status**：订阅方按 `type` 分支
            # （18.2 的事件清单就是这个粒度），折在 `status` 里等于要求
            # 每个订阅方自己再判一次——而漏判的症状是"失败被当成完成"。
            event_type="node.failed" if status == TaskStatus.FAILED else "node.completed",
            status=status,
            duration_ms=duration_ms,
            created_at=datetime.now(UTC),
        )
        return {**update, "trace_events": [enter, leave]}

    return wrapper


__all__ = ["traced"]
