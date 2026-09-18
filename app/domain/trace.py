"""任务的执行轨迹事件（详细设计 16.7 的 `agent_trace_event`）。

## 它是事件流的**权威来源**，Redis 只负责快

18.3 写得很明确：Redis Stream 会被 `MAXLEN` 裁剪，客户端断线重连后要补历史时
必须回到这张表。因此 `(task_id, sequence)` 的唯一约束**不是"防重复"的保险，
而是顺序保证本身**——重放时按 `sequence` 排序得到的就是真实执行顺序。

## 为什么不让 Redis 当权威

Stream 的保留期是配出来的（`stream_ttl_seconds`），而"这个任务当时是怎么跑的"
是排查问题时要问一年后的问题。把可裁剪的东西当权威，等于把"能查多久"交给
一个 TTL 参数决定。

## `sequence` 在这里由代码分配，不来自 Stream ID

18.2 的落地记录说 Phase 1.5 时它由 Stream ID 派生（`ms * 1000 + 毫秒内序号`），
并注明"Phase 2 接入后改由该表提供，**事件模型不变，改的只是赋值的那两行**"。
本实现按执行顺序连续编号：图的节点是串行的，列表顺序就是执行顺序。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.ids import IdPrefix, new_id

#: 节点级事件的类型（18.2 的事件清单）：`node.started` / `node.completed` /
#: `node.failed`。**故意不写成 Enum**：这一列与工具级事件（`tool.completed`
#: 等，属 Phase 10）共用同一张表，而读路径 `list_trace_events` 会把整张表
#: 的事件都读出来——写死成封闭取值之后，第一条工具事件就会让 `/trace` 报校验错。
#: 封闭性写在写入侧：只有 `agent/tracing.py` 一个地方造这些值。
NodeEventType = str


class NodeTrace(BaseModel):
    """一次节点执行的进入/离开事件（16.7 的一行）。

    **成对出现**：`node.started` 与 `node.completed` / `node.failed`。
    只记后者的话，一个卡住的节点在轨迹上表现为"什么都没发生"——
    而那正是最需要看出来的情况（18.4 的 `heartbeat` 事件是同一个问题的
    流侧答案：15 秒无事件就要发心跳，让客户端知道服务还活着）。
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id(IdPrefix.TRACE_EVENT))
    #: 该任务内的执行序号，从 1 起。**落库时统一分配**（见模块 docstring）。
    sequence: int = 0
    #: 节点名（`supervisor` / `sql` / `rag` / `reflect` / `conflict` /
    #: `analysis` / `reviewer` / `final`）
    node: str
    #: `node.started` / `node.completed` / `node.failed`
    event_type: NodeEventType
    #: 节点离开时的状态。进入时恒为 `RUNNING`。
    status: str
    #: **只有离开事件有值**：进入事件没有耗时可言，给它 0 会被读成"瞬间完成"。
    duration_ms: int | None = None
    error_code: str | None = None
    created_at: datetime


__all__ = ["NodeEventType", "NodeTrace"]
