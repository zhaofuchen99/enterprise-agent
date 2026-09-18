"""LangGraph 的 `AgentState`（详细设计 7.1 字段表 / 7.2 reducer 策略 / 7.3 示例）。

## 形状定全，但只有六个节点读写它

详设 7.1 有 33 个字段。冲刺方案 §8.1 的最小 Graph 只有
`supervisor / sql / rag / reflect / analysis / final` 六个节点，用不到
`plan_deltas` / `conflicts` / `review_result` / `sql_candidate` 这些。
**它们仍然在这里**，理由是：往 TypedDict 里加字段要连带改所有节点的签名与
测试；而 `test_state.py` 钉住"字段表与详设 7.1 一致"这条对应关系之后，
漏一个字段会在**读代码时**就发现，而不是等到接 Reviewer 那天。

每个未使用的字段都在下面标了它归属哪个 Phase。

## 两个 reducer，一处默认行为

LangGraph 里带 `Annotated[..., reducer]` 的字段走 reducer 合并，
其余字段是**后写覆盖**。详设 7.2 定的三种合并语义落成：

| 语义 | 字段 | 实现 |
|---|---|---|
| 按 id 去重追加 | `evidence` / `findings` | `merge_by_id` |
| 按键合并 | `step_results` | `merge_step_results` |
| 整体替换 | `task_list` / `open_questions` / `progress_assessment` | 默认（后写覆盖） |

**`errors` 用 `(错误码, 文案)` 去重而不是 `id`**：`AgentError` 没有 `id`
（它是异常、不是实体）。详设 7.2 写的是"按 id 去重"，那条按当时的
`Evidence`/`Finding` 写的；异常这一类没有标识，硬造一个只会让
`merge_by_id` 对两种形状都要判一次。

## `task_list` 只允许整体替换

详设 7.2 明写「只允许 Planner、PlanExtend 或 Replan 整体替换，其余节点只读」。
在这一版里 `task_list` 由 `supervisor` 一次性产出、`reflect` 演进时替换，
其余节点只读——**这条纪律没有类型层面的强制**，靠的是
`test_state.py` 里那条"除 supervisor / reflect 外没有节点写 task_list"的断言。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, TypedDict

from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import Finding, IntentResult, ProgressAssessment, StepResult, TaskStep
from app.core.errors import AgentError
from app.domain.evidence import Evidence
from app.domain.task import TaskStatus, ToolCallRecord
from app.domain.trace import NodeTrace

#: 单值字段（后写覆盖）不需要 `Annotated`，直接写类型即可；
#: 这里给它起个名字只是为了让下面的字段表读起来一致。
Scalar = Any


class Route(StrEnum):
    """条件边返回的路由值（详细设计 6.3）。

    **条件边只返回枚举值，不直接拼接节点名**：节点名是本模块的实现细节，
    散在路由函数里的话，改一个节点名要翻遍所有返回字符串的地方，
    而漏改的那一处会在运行时抛"节点不存在"——一个只说结果不说原因的错。
    """

    SQL = "sql"
    RAG = "rag"
    ANALYSIS = "analysis"
    CLARIFY = "clarify"
    FAIL = "fail"


def merge_by_id(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """按 `item.id` 去重的追加合并（详设 7.3）。

    右侧覆盖左侧的同 id 项（新值胜），顺序按"先左侧、再右侧的新项"。
    字典插入序天然保证这一点：`{**left, **right}` 不会把已存在的键挪到末尾。
    """
    merged: dict[str, Any] = {}
    for item in [*(left or []), *(right or [])]:
        merged[item.id] = item
    return list(merged.values())


def merge_step_results(
    left: dict[str, StepResult] | None, right: dict[str, StepResult] | None
) -> dict[str, StepResult]:
    """按键合并，同一步的新状态覆盖旧的（详设 7.2）。

    **不做"只许前进不许后退"的状态机校验**：`reflect` 演进后重跑同一步是
    合法的（比如补一路召回），而后退（SUCCEEDED → PENDING）在本实现里
    不会发生。加一道校验只会让演进那条路径多一个要维护的分支。
    """
    return {**(left or {}), **(right or {})}


def merge_errors(left: list[AgentError] | None, right: list[AgentError] | None) -> list[AgentError]:
    """错误去重追加。**按 `(错误码, 文案)` 而不是 id**，见模块 docstring。"""
    merged: dict[tuple[str, str], AgentError] = {}
    for error in [*(left or []), *(right or [])]:
        merged[(error.code.value, error.message)] = error
    return list(merged.values())


class AgentState(TypedDict, total=False):
    """图的状态（详细设计 7.1）。

    `total=False`：节点只返回自己改动的字段。设成 `total=True` 的话，
    每个节点都要把 33 个字段全填一遍，而其中绝大多数与它无关——
    那些"原样回填"的地方正是漏改的高发区。
    """

    # ---------------------------------------------------------- 任务身份（API 写入）
    user_query: str
    sanitized_query: str
    user_id: str
    conversation_id: str
    task_id: str
    trace_id: str
    #: 数据权限范围。**由 `_run_body` 从用户记录装载**，不是从任务里读——
    #: `agent_task` 没有这个字段，而"读不到就按不限"会是一次静默的越权。
    permission_scope: Scalar
    #: 本步骤的截止时刻（绝对时刻，见 `ToolContext` 的说明）
    deadline_at: Scalar

    # ---------------------------------------------------------- 意图与计划
    #: 会话结构化摘要。【Phase 6 未接】memory 未实现，恒为空 dict
    context_summary: dict[str, Any]
    intent: IntentResult | None
    task_list: list[TaskStep]
    current_step_index: int
    step_results: Annotated[dict[str, StepResult], merge_step_results]

    # ---------------------------------------------------------- 循环与发现
    findings: Annotated[list[Finding], merge_by_id]
    open_questions: list[str]
    progress_assessment: ProgressAssessment | None
    #: 【Phase 6 未接】演进预算与修订记录，属 plan_extend（后置）
    plan_revision: int
    plan_deltas: list[Scalar]

    #: 每个节点的进入/离开事件（16.7 的 `agent_trace_event`）。
    #: **图的节点是串行的，所以列表顺序就是执行顺序**——`sequence` 在落库时
    #: 按这个顺序分配（见 `domain/trace.py`）。
    trace_events: Annotated[list[NodeTrace], merge_by_id]

    #: 每一次工具**尝试**（SQL 的自修复会产生多条）。落 `agent_tool_call`。
    #: 与 `evidence` 同样用按 id 去重的追加 reducer——`id` 是每条独立的行键。
    tool_calls: Annotated[list[ToolCallRecord], merge_by_id]

    # ---------------------------------------------------------- 工具中间产物
    #: 【Phase 4/5 未接】SQL 与 RAG 的分步中间态留在各自 Tool 内部，
    #: 这里只放归一化之后的 `step_results` 与 `evidence`
    schema_context: Scalar
    sql_candidate: Scalar
    sql_result: Scalar
    retrieval_result: Scalar
    search_result: Scalar

    # ---------------------------------------------------------- 证据与结论
    evidence: Annotated[list[Evidence], merge_by_id]
    #: 【Phase 9 未接】多源冲突检测
    conflicts: list[Scalar]
    analysis_result: AnalysisResult | None
    #: 【Phase 8 未接】Reviewer
    review_result: Scalar

    # ---------------------------------------------------------- 收敛与产出
    errors: Annotated[list[AgentError], merge_errors]
    #: 【Phase 7 未接】四类循环预算尚只用到 `max_expansions`，
    #: 完整实现见详设 6.6.1（四类相互独立、不可借用）
    expansions_left: int
    execution_status: TaskStatus
    final_answer: str | None
    answer_payload: dict[str, Any] | None
    next_route: Route | None


def pending_steps(state: AgentState) -> list[TaskStep]:
    """还没跑的步骤（详设 6.3 的 `next_ready_step` 的简化版）。

    **只返回 PENDING**：SUCCEEDED / FAILED / SKIPPED 都不会被重复执行——
    这条是详设 6.3 明写的，也是"演进后重跑旧步骤"这类错误的挡板。
    """
    results = state.get("step_results") or {}
    return [
        step
        for step in state.get("task_list") or []
        if step.id not in results or results[step.id].status.value == "PENDING"
    ]


def has_pending_step(state: AgentState) -> bool:
    return bool(pending_steps(state))


def executed_sources(state: AgentState) -> set[str]:
    """已经**跑过**（不论成败）的工具集合。

    `reflect` 用它判断"该调的那一路调了没有"。按工具而不是按 step_id：
    计划里同一路可能有多个步骤（演进之后），而问的问题是"这一路查过没有"。
    """
    results = state.get("step_results") or {}
    done: set[str] = {
        step.tool
        for step in state.get("task_list") or []
        if step.id in results and results[step.id].status.value != "PENDING"
    }
    return done


__all__ = [
    "AgentState",
    "Route",
    "executed_sources",
    "has_pending_step",
    "merge_by_id",
    "merge_errors",
    "merge_step_results",
    "pending_steps",
]
