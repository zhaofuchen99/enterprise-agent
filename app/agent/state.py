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
| 按 id 去重追加 | `evidence` / `findings` / `plan_deltas` | `merge_by_id` |
| 按键合并 | `step_results` | `merge_step_results` |
| 整体替换 | `task_list` / `open_questions` / `progress_assessment` | 默认（后写覆盖） |

`plan_deltas` 走 `merge_by_id` 而不是整体替换，因为 `PlanDelta.id` 就是
`"{task_id}:{revision_no}"`（详设 7.2 要的去重键）——两次演进各是一条记录，
整体覆盖会让第一次的演进记录**静默消失**，而"计划只演进过一次"看起来完全正常。

**`errors` 用 `(错误码, 文案)` 去重而不是 `id`**：`AgentError` 没有 `id`
（它是异常、不是实体）。详设 7.2 写的是"按 id 去重"，那条按当时的
`Evidence`/`Finding` 写的；异常这一类没有标识，硬造一个只会让
`merge_by_id` 对两种形状都要判一次。

## `task_list` 只允许整体替换

详设 7.2 明写「只允许 Planner、PlanExtend 或 Replan 整体替换，其余节点只读」。
在这一版里 `task_list` 的写者只有两个：`supervisor` 一次性产出，`plan_extend`
演进时追加（`replan` 那条路是 `retry_router` 把它清空，等 supervisor 重来）。

**`reflect` 不再是写者**：它只出判定，追加步骤交给 `plan_extend`。
这不是搬代码——两个写者各写一份"追加步骤 + 升 revision + 扣预算"的话，
漂移的症状是**演进记录与实际计划对不上**（`plan_deltas` 说有一步、
`task_list` 里没有），而两边都看不出来。单一写者是让 `plan_revision`、
`plan_deltas`、`expansions_left` 三者始终一致的前提。

**这条纪律没有类型层面的强制**：`task_list` 是普通字段，任何节点都能返回它。
原先这里写的是"靠 `test_state.py` 的断言钉住"，而那个文件在仓库里
**并不存在**——这条注释因此是没有保障的自我安慰，已如实改掉。
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Any, TypedDict

from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import (
    Finding,
    IntentResult,
    PlanDelta,
    ProgressAssessment,
    StepResult,
    TaskStep,
)
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
    #: **计划演进**（详设 14.4 的 `retry_target=expand`）。落到 `plan_extend`：
    #: 审查说的是"缺一整类信息"，而具体加哪几步由那个节点决定
    #: （14.3 明写 Reviewer 不直接指定 SQL）。
    EXPAND = "expand"
    #: ⚠️ 落到 **`conflict`** 而不是 `analysis`（`_TARGETS` 里那条映射）。
    #: 冲突检测要的是"证据都到齐了"这个时点，路由回这里时会重跑它再进
    #: `analysis`——重试补完证据之后，冲突也确实该重新算一遍。
    ANALYSIS = "analysis"
    #: 作废整份计划回到 `supervisor`（详设 14.4 的 `REPLAN`）。
    #: **它不是一个循环**：换的是计划，不是"再试一次同一个动作"。
    REPLAN = "replan"
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
    #: 计划版本号，初始为 0。**只有 `plan_extend` 会加一**（详设 6.6.3：
    #: 它是"循环中唯一能修改 task_list 的节点"）。
    plan_revision: int
    #: 每次计划变更的原因与增删步骤（详设 7.4）。走 `merge_by_id` 是因为
    #: `PlanDelta.id` 就是 `"{task_id}:{revision_no}"`——那正是 7.2 要的去重键。
    plan_deltas: Annotated[list[PlanDelta], merge_by_id]
    #: **本次演进中被拒的步骤与原因**（详设 6.6.3 的 `plan_extend_rejected`）。
    #:
    #: 为什么不是一个错误码：19.1 的错误码表是封闭的，"演进提出的一步没通过校验"
    #: 也不该让任务失败——6.6.3 的原话是"丢弃未通过步骤、按 SUFFICIENT 收敛"。
    #: 但**完全静默更糟**：那时"模型提了没通过的步骤"与"模型压根没提"在产物上
    #: 长得一样。所以走"节点返回字段 → `tracing._derived_events` 派生事件"这条
    #: 既有范式（约定 70），与 `open_questions` 同形。
    plan_extend_rejected: list[str]

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
    #: `ReviewResult`（标成 `Scalar` 是为了避开与 `schemas.review` 的循环引用）
    review_result: Scalar
    #: `retry_router` 的判定结果（详设 14.4）。**条件边的唯一依据**。
    #:
    #: 为什么不像 `reflect` 那样"由条件边从更新后的 State 推出来"：
    #: `retry_target=analysis`（重跑分析）与"不重试了，去 final"在 State 上
    #: **长得一模一样**（两者都没有 PENDING 步骤），条件边推不出来。
    #: 见 `nodes/retry_router.py` 的模块说明。
    retry_route: Route | None

    # ---------------------------------------------------------- 收敛与产出
    errors: Annotated[list[AgentError], merge_errors]
    #: **计划演进**剩余次数（详设 6.6.1 第一类）。执行阶段的 `reflect` 与
    #: 审查阶段的 `retry_target=expand` 共用这一份——两者之和不超过配置值，
    #: 这是 14.3 明写的（避免"执行阶段激进 + 审查阶段再来一轮"）。
    #: **只有 `plan_extend` 扣它**，且只在步骤真的被合入时扣：
    #: 一条被校验拒掉的演进没有产生新的取证动作，不该算一次演进。
    expansions_left: int
    #: **Reviewer 补证**剩余次数（第二类，`max_reviewer_evidence`）。
    #: 约束 `reviewer → retry_router → Tool/analysis` 那条回边。
    review_retries_left: int
    #: **重新规划**剩余次数（第三类，`max_replans`）。它**不是循环**：
    #: 作废整份计划回到 supervisor，只在计划本身不可执行时用。
    replans_left: int
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


def next_step_id(state: AgentState, *, taken: Iterable[str] = ()) -> str:
    """下一个可用的步骤 id（`step_01` / `step_02`…）。

    **序号接着计划往下排，不重用已经出现过的 id**：`step_results` 是按
    step_id 索引的，重用一个已存在的 id 会让新步骤的结果**覆盖**旧步骤的
    （`merge_step_results` 的设计就是"新值覆盖"）。这类覆盖不报错，
    只会让"这一步跑了两次、第一次是空的"事后无从分辨。

    **提到这里是因为它有三个调用方**（`reflect` 出判定、`plan_extend` 合入、
    `retry_router` 追加补证步骤）。三份各写一份的症状是某一天其中一份改了
    规则（比如改用更短的 id），而撞号只在**那一条路径**上发生——
    另外两条照常工作，排查时会以为是数据问题。

    `taken` 是**同一次调用里已经分配出去的 id**：一次演进可能加不止一步，
    而它们都还没进 `task_list`，只查 State 会给两步发同一个 id。
    """
    used = {step.id for step in state.get("task_list") or []} | set(taken)
    index = len(used) + 1
    while f"step_{index:02d}" in used:  # pragma: no cover - 正常路径不会进循环
        index += 1
    return f"step_{index:02d}"


#: 两路取证工具：**工具名 ↔ 路由值 ↔ 补它时给步骤写的目标说明**。
#:
#: **这张表只写一处**。读它的地方有三个：`reflect` 的"该补哪一路"、
#: `retry_router` 的补证步骤、`reviewer` 的"还有哪一路没查过"。
#: 三处各写一份的症状是**"某类补证再也不发生"**——没有任何地方会报错，
#: 只是行为静默地少了一种（同 `tools/sql/schemas.SCOPE_COLUMNS` 的理由，
#: 那里写的是"漂移时的症状是某类冲突再也检不出来"）。
COMPLEMENTS: tuple[tuple[str, str, Route, str], ...] = (
    (
        "sql_query",
        "rag_retrieve",
        Route.RAG,
        "从企业制度与报告知识库补查该问题涉及的规定与解释",
    ),
    (
        "rag_retrieve",
        "sql_query",
        Route.SQL,
        "从业务数据库补查该问题涉及的指标数值",
    ),
)


def missing_complement(sources: set[str]) -> tuple[str, str] | None:
    """已跑过的工具集合 → **该补的那一路**：`(工具名, 目标说明)`。

    没有可补的（两路都跑了、或两路都没跑）时返回 `None`。
    `reflect` 与 `reviewer` 问的是同一个问题："还差哪一路没查"。
    """
    for already, missing, _route, objective in COMPLEMENTS:
        if already in sources and missing not in sources:
            return missing, objective
    return None


def unconsulted_source(state: AgentState) -> Route | None:
    """**还没查过的那一路**（`Route.SQL` / `Route.RAG`）；两路都查过或都没查过时 None。

    给 `reviewer` 用：一条结论没有引用时，"去补一路证据"与"重跑分析"
    是两种处置，而**只有前者可能改变结果**——证据没变，重跑分析只会
    得到同样的结论。

    "都没查过"与"都查过了"都返回 None，但含义不同：前者是计划本身有问题
    （不在"补证"能解决的范围内），后者是补也无处可补。
    """
    done = executed_sources(state)
    for already, _missing, route, _objective in COMPLEMENTS:
        if already in done and _missing not in done:
            return route
    return None


def route_for_tool(tool: str) -> Route | None:
    """工具名 → 它对应的路由值（`sql_query` → `Route.SQL`）。"""
    for already, _missing, route, _objective in COMPLEMENTS:
        if already == tool:
            return route
    return None


def tool_for_route(route: Route) -> str:
    """路由值 → **要跑的那个**工具名。**不在表里直接抛**：那说明调用方传错了。

    ⚠️ 返回的是 `_missing` 而不是 `_already`：这张表一行读作
    "跑过 A 之后该补 B"，而 `route` 指的是 **B**。
    返回 `_already` 的症状是**补证步骤去重跑了同一条路**——
    任务照跑、结果照有，只是那一轮什么都没补到。
    """
    for _already, missing, known, _objective in COMPLEMENTS:
        if known == route:
            return missing
    raise ValueError(f"{route} 不是一条取证路由")


def objective_for_route(route: Route) -> str:
    """补那一路时给步骤写的目标说明。**不写具体 SQL**——14.3 明写
    "Reviewer 不得直接指定 SQL"，它只说清要补什么。"""
    for _already, _missing, known, objective in COMPLEMENTS:
        if known == route:
            return objective
    raise ValueError(f"{route} 不是一条取证路由")


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
    "next_step_id",
    "pending_steps",
]
