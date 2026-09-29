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

## 会话上下文为什么也在这里装载

同一条理由，而且更强：上下文是**图的输入**，不是图里的一步。
装载它要读两个仓储（消息 + 任务），而 `TaskGraph` 一个都不认识。

**归纳逻辑不在这个文件里**——`group_turns` / `build_conversation_context`
是 `app/domain/memory.py` 的纯函数，读路径与写路径（`services/conversation_service.py`）
共用同一份"怎么归纳历史"。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from app.agent.graph import TaskGraph
from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.domain.memory import (
    ConversationContext,
    build_conversation_context,
    group_turns,
    resolution_from_payload,
)
from app.domain.task import Task, TaskOutcome
from app.repositories.message_repo import MessageRepository
from app.repositories.task_repo import TaskRepository
from app.repositories.user_repo import UserRepository

logger = logging.getLogger(__name__)


async def load_conversation_context(
    task: Task,
    *,
    messages: MessageRepository,
    tasks: TaskRepository,
    settings: Settings,
) -> ConversationContext | None:
    """把最近几轮对话归纳成图的输入；没有历史时返回 `None`。

    **两次读、一次归纳**：消息给出「谁问了什么、答了什么」（时间正序），
    任务给出「那一轮把问题解析成了什么口径」。口径来自任务而不是消息，
    是因为它本来就是 supervisor 的产物、由 `agent_task.result_json` 落库
    （见 `final._resolution_of`）——在消息上再抄一份就是两份会漂移的真相。

    读失败**不抛**：一次多轮上下文读不到，正确的行为是"当成首轮"继续跑，
    而不是让整条任务失败。用户仍然会得到答案，只是这一轮的代词不会被解析——
    而 `memory.record_failed` 那种"写失败"的日志在这里没有对应物，
    因为读失败的后果看得见（下一轮问「那 Q2 呢」会去澄清）。
    """
    conversation_id = task.conversation_id or ""
    if not conversation_id:
        # 空串直接返回：拿它去查库会命中 0 行，而"0 行"与"新会话"同形，
        # 白白多一次往返，还会让"会话 ID 丢了"这件事看起来像"没有历史"。
        return None

    max_turns = settings.memory.max_turns
    try:
        recent_messages = await messages.list_recent(conversation_id, limit=2 * max_turns)
        if not recent_messages:
            return None
        recent_tasks = await tasks.list_recent_by_conversation(conversation_id, limit=max_turns)
    except Exception:
        logger.warning(
            "memory.load_failed",
            extra={"task_id": task.id, "conversation_id": conversation_id},
        )
        return None

    resolutions = {item.id: resolution_from_payload(item.result_json) for item in recent_tasks}
    turns = group_turns(recent_messages, resolutions, max_turns=max_turns)
    return build_conversation_context(turns)


def build_task_body(
    graph: TaskGraph,
    users: UserRepository,
    *,
    messages: MessageRepository,
    tasks: TaskRepository,
    settings: Settings,
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
        context = await load_conversation_context(
            task, messages=messages, tasks=tasks, settings=settings
        )
        return await graph.run(task, permission_scope=user.permission_scope(), context=context)

    return run_task


__all__ = ["build_task_body", "load_conversation_context"]
