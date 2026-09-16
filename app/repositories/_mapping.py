"""ORM 行 <-> 领域模型的转换。

集中在一处而不是散在各个仓储里：转换里有三类**静默出错**的地方，
分散实现时几乎必然漏掉一两处，而且症状出现的位置离原因很远。

**1. 时区**（最容易踩）
领域层用 aware UTC（`datetime.now(UTC)`），而 MySQL `DATETIME(3)` 不存时区，
读回来是 naive。两者直接比较会抛
`TypeError: can't compare offset-naive and offset-aware datetimes`——
症状出现在「孤儿回收扫描」这类**定时任务**里，而不是写入路径上，
本地手测根本碰不到。因此写入前一律转 UTC 后去掉 tzinfo，读出后一律补回 UTC。

**2. 枚举**
`TaskStatus` / `ErrorCode` 都是 `StrEnum`，写库时取 `.value` 存字符串；
读出来交给 Pydantic 校验还原成枚举。**不用 MySQL ENUM 类型**——
加一个状态值就要改表结构。

**3. JSON 包装**
`User.region_ids` 是 `tuple[str, ...]`，库里是 `data_scope_json = {"region_ids": [...]}`
（TBC-03 的决议）。多一层包装是为了将来能加 `{"product_lines": [...]}` 而不改列。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, overload

from app.domain.conversation import Conversation
from app.domain.task import Task
from app.domain.user import User
from app.infrastructure.models.identity import AgentConversation, AppUser


# --------------------------------------------------------------- 时间
# 用 overload 而不是「返回 datetime | None」：后者会让每个写入非空列的地方
# 都多一次不必要的 None 判断，或者被迫写 `or at` 这类掩盖问题的兜底。
@overload
def to_db_time(value: datetime) -> datetime: ...


@overload
def to_db_time(value: None) -> None: ...


def to_db_time(value: datetime | None) -> datetime | None:
    """aware -> naive UTC（写库）。naive 输入按 UTC 解释，不按本地时区。

    按本地时区解释会让结果随开发机所在时区变化，是「换台机器跑就错」的来源。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


@overload
def from_db_time(value: datetime) -> datetime: ...


@overload
def from_db_time(value: None) -> None: ...


def from_db_time(value: datetime | None) -> datetime | None:
    """naive UTC -> aware UTC（读库）。"""
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(UTC)
    return value.replace(tzinfo=UTC)


# --------------------------------------------------------------- 任务
#: 与 `app/domain/task.py` 的 `Task` 逐字段对应。
#: 单独列出来而不是 `model_dump()` 一把梭：这样多一个领域字段而没同步这里时，
#: `Task.model_validate` 会因为缺字段直接报错，而不是悄悄用默认值。
_TASK_FIELDS: tuple[str, ...] = (
    "id",
    "user_id",
    "conversation_id",
    "trace_id",
    "query_text",
    "status",
    "parent_task_id",
    "idempotency_key",
    "intent",
    "worker_id",
    "heartbeat_at",
    "final_answer_md",
    "error_code",
    "error_message",
    "trace_incomplete",
    "queued_at",
    "started_at",
    "finished_at",
    "created_at",
    "updated_at",
)

_TASK_TIME_FIELDS: tuple[str, ...] = (
    "heartbeat_at",
    "queued_at",
    "started_at",
    "finished_at",
    "created_at",
    "updated_at",
)


def task_to_row_values(task: Task) -> dict[str, Any]:
    """`Task` -> `agent_task` 的列值。"""
    values: dict[str, Any] = {name: getattr(task, name) for name in _TASK_FIELDS}
    values["status"] = task.status.value
    # ErrorCode 是 StrEnum，取 value 落库；None 保持 None，不写成空串——
    # 空串在读回来时会被 Pydantic 拒绝（不是合法错误码），且掩盖「没有错误」的语义。
    values["error_code"] = task.error_code.value if task.error_code is not None else None
    for name in _TASK_TIME_FIELDS:
        values[name] = to_db_time(values[name])
    return values


def task_from_row(row: Any) -> Task:
    """`agent_task` 行 -> `Task`。枚举与时间由 Pydantic 与本模块还原。"""
    data: dict[str, Any] = {name: getattr(row, name) for name in _TASK_FIELDS}
    for name in _TASK_TIME_FIELDS:
        data[name] = from_db_time(data[name])
    return Task.model_validate(data)


# --------------------------------------------------------------- 用户
def user_to_row_values(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "password_hash": user.password_hash,
        "display_name": user.display_name,
        "role": user.role.value,
        "status": user.status.value,
        # TBC-03：数据范围就是区域列表
        "data_scope_json": {"region_ids": list(user.region_ids)},
    }


def user_from_row(row: AppUser) -> User:
    scope = row.data_scope_json or {}
    # 只认 region_ids 一个键；将来加了别的维度时，这里要显式扩展而不是整个透传，
    # 否则领域模型会悄悄拿到一堆没校验的结构。
    region_ids = scope.get("region_ids") or []
    return User.model_validate(
        {
            "id": row.id,
            "username": row.username,
            "password_hash": row.password_hash,
            "display_name": row.display_name,
            "role": row.role,
            "status": row.status,
            "region_ids": tuple(region_ids),
        }
    )


# ------------------------------------------------------------- 会话
_CONVERSATION_TIME_FIELDS: tuple[str, ...] = ("last_message_at", "created_at", "updated_at")


def conversation_to_row_values(conversation: Conversation) -> dict[str, Any]:
    values: dict[str, Any] = {
        "id": conversation.id,
        "user_id": conversation.user_id,
        "title": conversation.title,
        "last_message_at": conversation.last_message_at,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }
    for name in _CONVERSATION_TIME_FIELDS:
        values[name] = to_db_time(values[name])
    return values


def conversation_from_row(row: AgentConversation) -> Conversation:
    data: dict[str, Any] = {
        "id": row.id,
        "user_id": row.user_id,
        "title": row.title,
        "last_message_at": row.last_message_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
    for name in _CONVERSATION_TIME_FIELDS:
        data[name] = from_db_time(data[name])
    return Conversation.model_validate(data)
