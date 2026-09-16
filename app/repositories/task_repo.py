"""任务仓储。Phase 2 换 MySQL 实现（详细设计 16.5 的 agent_task）。

内存实现里所有方法内部都**没有 await 点**，因此 asyncio 下一个
「先查幂等键、再写入」的调用序列不会被其它协程插队。
换成真实数据库后这个假设不再成立，唯一约束由
`uk_task_user_idempotency` 兜底，调用方按 `DuplicateTaskError` 处理——
`TaskService.create_task` 两种实现下写法一致，不需要为换库改逻辑。
"""

from __future__ import annotations

from typing import Protocol

from app.domain.task import ACTIVE_STATUSES, Task


class DuplicateTaskError(Exception):
    """唯一约束冲突：task_id 重复，或 (user_id, idempotency_key) 已存在。"""


class TaskRepository(Protocol):
    async def add(self, task: Task) -> None: ...

    async def get(self, task_id: str) -> Task | None: ...

    async def find_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Task | None: ...

    async def count_active(self, user_id: str) -> int:
        """统计占用并发配额的任务数（终态不计，见 `ACTIVE_STATUSES`）。"""
        ...


class InMemoryTaskRepository:
    def __init__(self) -> None:
        self._by_id: dict[str, Task] = {}
        self._idempotency: dict[tuple[str, str], str] = {}

    async def add(self, task: Task) -> None:
        if task.id in self._by_id:
            raise DuplicateTaskError(f"task_id 重复：{task.id}")
        if task.idempotency_key is not None:
            index_key = (task.user_id, task.idempotency_key)
            if index_key in self._idempotency:
                raise DuplicateTaskError(f"幂等键重复：{task.idempotency_key}")
            self._idempotency[index_key] = task.id
        self._by_id[task.id] = task

    async def get(self, task_id: str) -> Task | None:
        return self._by_id.get(task_id)

    async def find_by_idempotency_key(self, user_id: str, idempotency_key: str) -> Task | None:
        task_id = self._idempotency.get((user_id, idempotency_key))
        return self._by_id.get(task_id) if task_id else None

    async def count_active(self, user_id: str) -> int:
        return sum(
            1
            for task in self._by_id.values()
            if task.user_id == user_id and task.status in ACTIVE_STATUSES
        )
