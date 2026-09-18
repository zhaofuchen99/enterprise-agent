"""把一个任务交给 Graph 跑（`TaskRunner` 的任务体实现）。

## 权限的范围**在这里装载**，不在 `TaskRunner` 里

`agent_task` 表**没有数据范围字段**——范围属于用户（`app_user.data_scope_json`，
TBC-03 的决议）。所以跑图之前必须用 `task.user_id` 去查一次用户，
把 `permission_scope` 装进 State。

**查不到用户时拒绝执行，绝不按"不限"兜底**：`PermissionScope` 的空
`region_ids` 语义是**不限**（TBC-03），拿它当兜底等于让一个已删除用户的
任务拿到全量数据——而这条路径不会有任何报错，只会让受限用户看到不该看的数字。

## 为什么不在 `TaskGraph.run` 里查

`TaskGraph` 只依赖工具与网关，不认识用户仓储；把仓储塞进去会让它多一个
与"跑图"无关的依赖。装载权限是**调用方**的职责——它已经持有仓储了。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.graph import TaskGraph
from app.core.errors import AgentError, ErrorCode
from app.domain.task import Task, TaskOutcome
from app.repositories.user_repo import UserRepository

logger = logging.getLogger(__name__)


def build_task_body(
    graph: TaskGraph, users: UserRepository
) -> Callable[[Task], Awaitable[TaskOutcome]]:
    """构造 `TaskRunner` 要的任务体。"""

    async def run_task(task: Task) -> TaskOutcome:
        user = await users.get_by_id(task.user_id)
        if user is None:
            # **显式失败，不兜底成"不限"**，见模块 docstring
            raise AgentError(
                ErrorCode.INTERNAL_ERROR,
                "任务所属用户不存在，拒绝以未限定的数据范围执行",
                details={"task_id": task.id, "user_id": task.user_id},
            )
        if not user.is_active:
            # 用户在处理期间被停用：范围可能已经变了，而任务是按旧范围建的。
            # 按旧范围跑等于继续用一份已失效的授权。
            raise AgentError(
                ErrorCode.ACCESS_DENIED,
                "任务所属用户已停用",
                details={"task_id": task.id, "user_id": task.user_id},
            )
        logger.info("任务体开始执行", extra={"task_id": task.id})
        return await graph.run(task, permission_scope=user.permission_scope())

    return run_task


__all__ = ["build_task_body"]
