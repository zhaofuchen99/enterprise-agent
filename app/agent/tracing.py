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
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

from app.agent.state import AgentState, pending_steps
from app.domain.events import TaskEventType
from app.domain.task import TaskStatus
from app.domain.trace import NodeTrace

logger = logging.getLogger(__name__)


class EventPublisher(Protocol):
    """发事件所需的最小形状。

    **不 import `services/event_bus.py` 的 `EventBus`**：依赖方向是
    `services → agent`，反向 import 会把两层耦成一个环。这里按结构声明
    真正用到的那一个方法，`RedisStreamEventBus` 结构上满足它——
    与 `ModelGateway` 的 `PromptSource` 是同一条理由。
    """

    async def publish(
        self,
        *,
        task_id: str,
        trace_id: str,
        event_type: TaskEventType,
        data: dict[str, Any] | None = None,
        node: str | None = None,
        step_id: str | None = None,
    ) -> Any: ...


def traced(
    name: str,
    node: Callable[[AgentState], Any],
    events: EventPublisher | None = None,
) -> Callable[[AgentState], Awaitable[Any]]:
    """包装一个节点，产出 `node.started` + `node.completed` / `node.failed`。

    包装成 `async def` 对同步与异步节点都成立：同步节点直接调用即可，
    LangGraph 看到的是同一个异步契约。**不给两类节点写两条包装路径**——
    那会让"某个节点忘了被包"变成一件从签名上看不出来的事。

    它同时是**节点级事件的唯一出口**：`node.*` 与从节点返回值里派生出来的
    领域事件（`plan.created` / `progress.assessed` / `review.completed` …）
    都在这里发。理由与埋点本身相同——散到八个节点里各写一遍，
    漏掉一处不会有任何症状。

    Args:
        events: 事件总线。**`None` 表示不发事件**（单元测试与不关心事件的
            装配路径），此时行为与加事件之前完全一致：只记状态内的轨迹。
    """

    async def wrapper(state: AgentState) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        started = time.monotonic()
        # 进入时要报的 step：**这个节点即将执行的那一步**（工具节点才有，
        # 其余节点为 None）。从 State 现推而不是让节点自己传——节点不知道
        # 自己被包了，这正是包装器的意义
        pending = pending_steps(state)
        step_id = pending[0].id if pending else None
        await _emit(
            events,
            state,
            event_type=TaskEventType.NODE_STARTED,
            node=name,
            step_id=step_id,
        )
        enter = NodeTrace(
            node=name,
            event_type=TaskEventType.NODE_STARTED.value,
            status="RUNNING",
            created_at=started_at,
        )
        update = node(state)
        if inspect.isawaitable(update):
            update = await update
        update = update or {}
        duration_ms = int((time.monotonic() - started) * 1000)
        status = str(update.get("execution_status", TaskStatus.SUCCEEDED))
        failed = status == TaskStatus.FAILED
        leave = NodeTrace(
            node=name,
            # **失败换事件类型，不只换 status**：订阅方按 `type` 分支
            # （18.2 的事件清单就是这个粒度），折在 `status` 里等于要求
            # 每个订阅方自己再判一次——而漏判的症状是"失败被当成完成"。
            event_type=(
                TaskEventType.NODE_FAILED.value if failed else TaskEventType.NODE_COMPLETED.value
            ),
            status=status,
            duration_ms=duration_ms,
            created_at=datetime.now(UTC),
        )
        await _emit(
            events,
            state,
            event_type=TaskEventType.NODE_FAILED if failed else TaskEventType.NODE_COMPLETED,
            node=name,
            step_id=step_id,
            data={"status": status, "duration_ms": duration_ms},
        )
        # **派生事件在离开事件之后**：它们是这个节点**产出的东西**
        # （计划、判定、审查结论），排在"节点完成"后面读起来才是因果顺序
        for event_type, data in _derived_events(state, update):
            await _emit(events, state, event_type=event_type, node=name, data=data)
        return {**update, "trace_events": [enter, leave]}

    return wrapper


async def _emit(
    events: EventPublisher | None,
    state: AgentState,
    *,
    event_type: TaskEventType,
    node: str | None = None,
    step_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """发一条事件。**失败只记日志，不让任务跟着倒**。

    埋点与事件流都是"关于执行的事后信息"，而 Redis 抖一下不该把一次
    正常的分析判成失败——那是拿可观测性换可用性，方向反了。
    代价如实记下：**这时事件流会缺一段，而任务照常成功**
    （18.3 的"MySQL 权威重放"落成之后，缺的那段还能从表里补回来；
    现在流就是唯一来源，缺了就是缺了）。
    """
    if events is None:
        return
    try:
        await events.publish(
            task_id=str(state.get("task_id") or ""),
            trace_id=str(state.get("trace_id") or ""),
            event_type=event_type,
            node=node,
            step_id=step_id,
            data=data,
        )
    except Exception as exc:
        logger.warning(
            "事件发布失败（不影响任务）：%s",
            type(exc).__name__,
            extra={"node": node, "status": event_type.value},
        )


def _derived_events(
    state: AgentState, update: dict[str, Any]
) -> list[tuple[TaskEventType, dict[str, Any]]]:
    """从节点的返回值里派生领域事件（18.2 的 `plan.*` / `progress.*` / `review.*`）。

    **按 State 契约判定，不按节点名**：判据是"这次更新里有没有那个字段"
    ——`task_list`、`progress_assessment`、`review_result` 都是经过 Pydantic
    校验的对象（开发流程 5.3），比"哪个节点返回了它"稳定得多。
    将来 `conflict` 或新节点产出同类字段时，事件自动跟着有，不必回来加分支。

    `plan.created` 与 `plan.updated` 靠 `plan_revision` 区分：
    `supervisor` 写的是当前值（首次成计划），`reflect` 演进时写 +1。
    """
    derived: list[tuple[TaskEventType, dict[str, Any]]] = []

    intent = update.get("intent")
    plan = update.get("task_list")
    revision = update.get("plan_revision")
    previous_revision = state.get("plan_revision", 0)

    if plan and revision is not None:
        # **空列表不算"有计划"**：`supervisor` 判成澄清或不受支持时返回的是
        # `task_list=[]`，那时报 `plan.created` 等于说"计划里有 0 步"——
        # 而真相是"这次没有计划"，两者在客户端看来一个是空跑、一个是澄清。
        # 判据写成 truthiness 就是为了让这种情形落到下面的 elif 去
        if revision > previous_revision:
            added = [
                step.id for step in plan if step.id not in {old.id for old in state["task_list"]}
            ]
            derived.append(
                (
                    TaskEventType.PLAN_UPDATED,
                    {
                        "revision_no": revision,
                        "added_steps": added,
                        # 切片内 `reflect` 只追加、不放弃步骤，所以恒为空；
                        # 放弃步骤要等模型的 EXPAND 判定（【后续扩展】）
                        "skipped_steps": [],
                        "trigger_finding_id": _trigger_finding_id(state),
                        "reason": _assessment_reason(update),
                    },
                )
            )
        else:
            derived.append(
                (
                    TaskEventType.PLAN_CREATED,
                    {
                        "step_count": len(plan),
                        "tool_types": sorted({step.tool for step in plan}),
                    },
                )
            )
    elif intent is not None and getattr(intent, "intent", None) == "CLARIFICATION":
        # 没有可执行计划 + 判成澄清：这正是"需要用户补充输入"那句话的落点
        derived.append(
            (
                TaskEventType.CLARIFICATION_REQUIRED,
                {
                    "question": getattr(intent, "clarification_question", None),
                    "missing_fields": list(getattr(intent, "missing_fields", ()) or ()),
                },
            )
        )

    assessment = update.get("progress_assessment")
    if assessment is not None:
        derived.append(
            (
                TaskEventType.PROGRESS_ASSESSED,
                {
                    "decision": assessment.decision,
                    # 18.2 的字段名是 `finding_statement`，而模型侧的字段叫 `reason`
                    # （`ProgressAssessment` 逐字对齐详设 6.3）。这里做一次映射并
                    # 在此说明：两处名字不同是文档与实现的既有差异，不是笔误。
                    # 18.2 另外要求它是"经 Pydantic 校验的短陈述"——`reason` 正是
                    # `ProgressAssessment` 的字段，天然满足
                    "finding_statement": assessment.reason,
                    "open_question_count": len(update.get("open_questions") or ()),
                },
            )
        )

    review = update.get("review_result")
    if review is not None:
        derived.append(
            (
                TaskEventType.REVIEW_COMPLETED,
                {
                    "status": review.status,
                    "score": review.score,
                    "issue_count": len(review.issues),
                },
            )
        )
    return derived


def _trigger_finding_id(state: AgentState) -> str | None:
    """触发这次演进的结论 id（18.2 的 `trigger_finding_id`）。

    **切片内恒为 None**：演进由 `reflect` 的确定性规则触发（"某一路跑了但空"），
    不来自某条 `Finding`——模型提出下钻理由的能力属【后续扩展】。
    留这个字段而不是删掉：客户端的事件 schema 不该因为一个后置能力而变形。
    """
    return None


def _assessment_reason(update: dict[str, Any]) -> str:
    assessment = update.get("progress_assessment")
    return str(getattr(assessment, "reason", "")) if assessment is not None else ""


__all__ = ["EventPublisher", "traced"]
