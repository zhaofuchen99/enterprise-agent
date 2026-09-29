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
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from app.agent.schemas.plan import StepResult
from app.agent.state import AgentState, pending_steps
from app.domain.events import TaskEventType
from app.domain.task import TaskStatus
from app.domain.trace import NodeTrace
from app.infrastructure.observability import counter, histogram

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
        # 节点耗时（19.4.3）。**记在这里而不是八个节点里各写一遍**——
        # 理由与埋点本身完全相同：漏掉一个不会有任何症状，
        # 那个节点的耗时永远不出现（见模块 docstring）
        histogram("agent.node.duration", unit="ms", description="单个节点的执行耗时").record(
            duration_ms, {"node": name, "status": status}
        )
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
        # 这次跑完的工具步 → 一条轨迹行 + 一条 SSE 事件（`tool.completed`）。
        # **只算一次**：`_tool_completed` 内部会造 `NodeTrace`（每条一个新 id），
        # 调两次会得到两组互不相干的行
        completed = _tool_completed(state, update, node=name)

        # **派生事件在离开事件之后**：它们是这个节点**产出的东西**
        # （计划、判定、审查结论），排在"节点完成"后面读起来才是因果顺序
        for event_type, data, event_step in _derived_events(state, update, completed):
            await _emit(
                events, state, event_type=event_type, node=name, step_id=event_step, data=data
            )
        return {
            **update,
            # **合并而不是覆盖**：`update` 里可能已经有节点自己放的 `trace_events`。
            # 当前没有这样的生产者（工具轨迹由上面那个函数产出），但覆盖的那天
            # **不会有任何症状**——节点照跑、答案照出，只有轨迹里少一段，
            # 而"轨迹不完整"本就是这个系统的常态。多一个分支换掉一类静默故障。
            # 顺序即真实执行顺序（进节点 → 工具执行 → 出节点），而落库的
            # `sequence` 正是按列表顺序分配的（约定 52）
            "trace_events": [
                enter,
                *[item for item, _, _ in completed],
                *(update.get("trace_events") or []),
                leave,
            ],
        }

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


def _tool_completed(
    state: AgentState, update: dict[str, Any], *, node: str
) -> list[tuple[NodeTrace, str, dict[str, Any]]]:
    """这次更新里跑完的每一步工具 → `(轨迹行, step_id, SSE 事件 data)`。

    **判据是 `step_results` 的增量，不是 `tool_calls`**：后者一次尝试一行
    （SQL 自修复会产 3 行，见 `tool_nodes._tool_calls`），而 `tool.completed`
    的语义是"一次工具结束"——一条。`step_results` 的键就是 step_id、
    由 `merge_step_results` 按键合并，因此"这次更新里有几个键"恰好等于
    "这次跑完了几步"。`duration_ms` 同理取 `StepResult` 的（整次调用的耗时），
    不是某一次尝试的。

    **一个函数产出两种表示**（轨迹行与 SSE 事件）而不是两个函数各写一遍：
    两者说的是同一件事，分开写必然漂移，而漂移的症状是"轨迹里有、流里没有"，
    两边都看不出异常（约定 87：判据只有一份）。

    工具名从 `task_list` 反查——`StepResult` 上不带它，而 `task_list` 是
    State 里工具名的权威。**反查不到也照发**（`tool=None`）：宁可留一条
    字段缺失的事件，也不静默丢掉一条，因为轨迹是权威，"哪一步慢"整段
    读不出来比读到一条不完整的严重。
    """
    results = update.get("step_results") or {}
    if not results:
        return []
    tools = {step.id: step.tool for step in state["task_list"]}
    completed: list[tuple[NodeTrace, str, dict[str, Any]]] = []
    for step_id, result in results.items():
        tool = tools.get(step_id)
        _record_tool_metrics(tool, result)
        completed.append(
            (
                NodeTrace(
                    node=node,
                    tool=tool,
                    event_type=TaskEventType.TOOL_COMPLETED.value,
                    status=result.status,
                    duration_ms=result.duration_ms,
                    error_code=result.error_code,
                    payload={"summary": result.summary},
                    created_at=datetime.now(UTC),
                ),
                step_id,
                {
                    "tool": tool,
                    "status": result.status,
                    "summary": result.summary,
                    "duration_ms": result.duration_ms,
                },
            )
        )
    _record_sql_attempts(update)
    return completed


def _record_tool_metrics(tool: str | None, result: StepResult) -> None:
    """工具的调用数与空结果数（19.4.3 的"Tool 成功率"与"RAG 空结果率"）。

    两条都挂在**这次工具执行**上（不是每次 attempt），与 `tool.completed`
    同源——于是"事件里看到的次数"与"指标里数到的次数"对得上。

    ⚠️ 记录本身不会抛：OTel 的导出失败发生在后台线程，而 `add` 只是在
    进程内追加一条 measurement。`_instrument` 那次创建是唯一会抛的点
    （它校验名字格式），而名字都是本文件里的字面量。
    """
    label = tool or ""
    counter("agent.tool.calls", unit="{call}", description="工具调用数").add(
        1, {"tool": label, "status": str(result.status)}
    )
    if result.empty:
        # **空结果单独计数**：它不是失败（详设 9.4 明写 SQL 空集不算失败），
        # 而是"关于数据的事实"。混进 `status=FAILED` 的话，
        # "没有这条数据"与"服务挂了"在报表上就是同一个数
        counter("agent.tool.empty_results", unit="{call}", description="工具返回空结果数").add(
            1, {"tool": label}
        )


def _record_sql_attempts(update: dict[str, Any]) -> None:
    """SQL 的尝试数与修复、拒绝情况（19.4.3 的"SQL 拒绝与修复率"）。

    **判据是 `ToolCallRecord.result_summary["stage"]`**——`tool_nodes._tool_calls`
    写进去的 `GENERATE` / `REPAIR` / `EXECUTE`，配上它的 `status`
    （`REJECTED` 就是被校验拦下的那些）。

    修复率与拒绝率都**不单独记账**，而是同一条计数器上的两个维度：
    分母（总尝试数）就在 `stage` 那一维里，分成两个指标会造出两份
    可能对不上的数——而"对不上"在这里没有任何症状，只会让人对指标失去信任。

    RAG 的 `tool_calls` 没有 `stage`（它不分阶段），因此被跳过——**跳过是有意的**，
    不是漏算：把它记成 `stage=""` 会让"SQL 的某一阶段"这个维度凭空多出一个
    空值桶，读的人会去猜那个空值是什么。
    """
    for call in update.get("tool_calls") or ():
        stage = str((call.result_summary or {}).get("stage") or "")
        if not stage:
            continue
        counter("agent.sql.attempts", unit="{attempt}", description="SQL 尝试数（按阶段）").add(
            1, {"stage": stage, "status": str(call.status)}
        )


def _derived_events(
    state: AgentState,
    update: dict[str, Any],
    completed: Sequence[tuple[NodeTrace, str, dict[str, Any]]],
) -> list[tuple[TaskEventType, dict[str, Any], str | None]]:
    """从节点的返回值里派生领域事件（18.2 的 `tool.*` / `plan.*` / `progress.*` / `review.*`）。

    **按 State 契约判定，不按节点名**：判据是"这次更新里有没有那个字段"
    ——`task_list`、`progress_assessment`、`review_result` 都是经过 Pydantic
    校验的对象（开发流程 5.3），比"哪个节点返回了它"稳定得多。
    将来 `conflict` 或新节点产出同类字段时，事件自动跟着有，不必回来加分支。

    `plan.created` 与 `plan.updated` 靠 `plan_revision` 区分：
    `supervisor` 写的是当前值（首次成计划），`plan_extend` 演进时写 +1。

    返回**三元组**：`step_id` 只有工具级事件有（其余领域事件没有对应的步骤），
    单独带出来而不是塞进 `data`——它是 18.2 事件模型里与 `data` **同级**的字段。
    """
    derived: list[tuple[TaskEventType, dict[str, Any], str | None]] = []

    # 工具级事件。**由调用方算好传进来**（`_tool_completed`）：那里同时产出了
    # 落库用的轨迹行，两处共用同一份数据，因此不会出现"轨迹里有、流里没有"
    for _, step_id, data in completed:
        derived.append((TaskEventType.TOOL_COMPLETED, data, step_id))

    # 合法重试（18.2 的 `task.retrying`）。判据是 `retry_route` 与
    # `retry_attempt` **同时**被这次更新写入——两者只在真的路由出去时才写，
    # 降级（预算耗尽）与不重试只写 `retry_route=None`。只看后者会把一次
    # "没重试成"报成"重试了"，而那个结论是**关于这次执行的处置**的
    attempt = update.get("retry_attempt")
    route = update.get("retry_route")
    if attempt and route is not None:
        review = state.get("review_result")
        derived.append(
            (
                TaskEventType.TASK_RETRYING,
                {
                    "target": str(route),
                    # **不给假值**：审查没给 `reason_code` 时就是空串。
                    # 回填一个 "UNKNOWN" 会让"没记"与"记了个未知码"同形，
                    # 而前者说明判据该补，后者说明审查那边出了问题
                    "reason_code": (review.reason_code if review else None) or "",
                    "attempt": attempt,
                },
                None,
            )
        )

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
            # 三样都取**这次更新的那条 delta**，不现推：`trigger_finding_id`
            # 推不出来（它到底指哪条 finding 只有 `plan_extend` 知道），
            # 而 `skipped_steps` 在它之前恒为空——那两处都是"看起来正常"的
            # 空值，读的人会把"没有"与"没记"当成一回事。
            delta = _latest_delta(update)
            derived.append(
                (
                    TaskEventType.PLAN_UPDATED,
                    {
                        "revision_no": revision,
                        "added_steps": added,
                        "skipped_steps": list(delta.skipped_step_ids) if delta else [],
                        "trigger_finding_id": delta.trigger_finding_id if delta else None,
                        # delta 的 `reason` 比 `progress_assessment.reason` 更贴切：
                        # 后者是"为什么该继续查"，前者是"为什么加了这几步"。
                        "reason": delta.reason if delta else _assessment_reason(update),
                    },
                    None,
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
                    None,
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
                None,
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
                None,
            )
        )

    review = update.get("review_result")
    if review is not None:
        # 审查判定分布（19.4.3）。**按 `status` 分维度**：PASS / RETRY / CLARIFY /
        # FAIL 各自的占比是"这道门禁到底在拦什么"唯一的量化口径——
        # 只看 `review.completed` 事件的条数是看不出来的
        counter("agent.review.verdicts", unit="{review}", description="审查判定数").add(
            1, {"status": str(review.status)}
        )
        derived.append(
            (
                TaskEventType.REVIEW_COMPLETED,
                {
                    "status": review.status,
                    "score": review.score,
                    "issue_count": len(review.issues),
                },
                None,
            )
        )

    # **只在有内容时发**：`plan_extend` 每次运行都会返回这个字段（哪怕是空
    # 列表），无条件发的话，每个正常任务都会多出一条"这次演进没有提出步骤"
    # 的事件——而它是噪声，真被拒的时候反而没人看（同 `DISABLED` 不进
    # `warnings` 的理由，约定 58）。
    rejected = update.get("plan_extend_rejected")
    if rejected:
        derived.append(
            (TaskEventType.PLAN_EXTEND_REJECTED, {"reasons": list(rejected)}, None),
        )
    return derived


def _latest_delta(update: dict[str, Any]) -> Any | None:
    """这次更新带回来的那条 `PlanDelta`（没有则 `None`）。

    **取最后一条而不是第一条**：一次节点调用最多产生一条 delta
    （`plan_extend` 每次只加一版），但 reducer 是追加语义——写成"最后一条"
    不会在将来某天有人一次追加多条时静默取到旧的那条。
    """
    deltas = update.get("plan_deltas") or []
    return deltas[-1] if deltas else None


def _assessment_reason(update: dict[str, Any]) -> str:
    assessment = update.get("progress_assessment")
    return str(getattr(assessment, "reason", "")) if assessment is not None else ""


__all__ = ["EventPublisher", "traced"]
