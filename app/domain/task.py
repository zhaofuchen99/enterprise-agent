"""任务领域模型（对应详细设计 16.5 的 agent_task 表）。

字段与表结构一一对应，字段名刻意与列名保持一致，使 Phase 2 接入 MySQL 时
不需要再做一层映射（少一层映射就少一处能写错的地方）。

未纳入的列：`plan_json` / `result_json` 是执行期中间产物，
对外只暴露**已校验**的结构化字段，不把裸 JSON 直接回给客户端（开发流程 5.2）。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.errors import ErrorCode
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Conflict, Evidence
from app.domain.trace import NodeTrace


class TaskStatus(StrEnum):
    """任务状态（详细设计 2.2）。

    `CANCEL_REQUESTED` 只在详细设计 17.5 出现，需求规格 FR-CHAT-002 未列出。
    以详细设计为准：它区分了「用户已请求取消」与「Worker 已确认取消」，
    缺少它就无法表达「取消请求已记录但尚未生效」这个中间态。
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_CLARIFICATION = "WAITING_CLARIFICATION"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)

#: 占用用户并发配额的状态。
#:
#: 取「一切非终态」而不是「仅 QUEUED/RUNNING」：详细设计 19.3 规定
#: 「任务结束（任意终态）时必须递减」，即释放配额的时机就是进入终态，
#: 在此之前都算占用。这样 WAITING_CLARIFICATION 也不会被用来绕过并发限制。
ACTIVE_STATUSES: frozenset[TaskStatus] = frozenset(TaskStatus) - TERMINAL_STATUSES


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    conversation_id: str
    trace_id: str
    query_text: str
    status: TaskStatus = TaskStatus.QUEUED
    parent_task_id: str | None = None
    #: 同用户范围内唯一（16.5 的 uk_task_user_idempotency）
    idempotency_key: str | None = None
    intent: str | None = None
    worker_id: str | None = None
    heartbeat_at: datetime | None = None
    #: 已校验的计划摘要（16.5）。**审阅一份答案时第一个要看的东西**：
    #: "它查了哪几步、每步成没成"比答案本身更能说明结论有多硬。
    plan_json: dict[str, Any] | None = None
    #: 结构化结果（16.5）。审查结论、证据、冲突、限制都在这里——
    #: 它们是**可追溯性的载体**，只留在进程内等于没记。
    result_json: dict[str, Any] | None = None
    final_answer_md: str | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    trace_incomplete: bool = False
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class TaskOutcome(BaseModel):
    """任务体的产出（16.5 的 `plan_json` / `result_json` + 答案）。

    **放在 `domain/` 而不是 `services/` 或 `agent/`**：两边都要用它——
    `agent.runner` 产出它、`services.task_runner` 收下并落库——
    而依赖方向是单向的（`services → agent`），放进任一侧都会让另一侧反向依赖。

    三个字段分开而不是塞进一个 `payload`：`answer` 要落 `final_answer_md`
    （MEDIUMTEXT，给人读），`plan` 与 `payload` 落两个 JSON 列（给程序读）。
    合成一个的话，"这条任务的答案"与"这条任务的结构化结果"就开始互相污染。
    """

    model_config = ConfigDict(frozen=True)

    answer: str | None = None
    #: 图里判定的终态是不是失败（supervisor 模型不可用、计划非法…）。
    #:
    #: **任务级状态必须跟着它走**：不跟的话，一次"模型挂了"会被记成
    #: `SUCCEEDED`，而 API、SSE 与用户都看不出区别——答案文本里那句
    #: "本次任务未能完成"是唯一线索，而它是给人读的，不是给程序读的。
    #: 实测踩到：演示脚本把一次 `MODEL_OUTPUT_INVALID` 当成了"进入澄清"，
    #: 因为两者的产物都是"没有步骤、有一句说明"。
    failed: bool = False
    #: 失败时的错误码与文案（进 `agent_task.error_code` / `error_message`）
    error_code: str | None = None
    error_message: str | None = None
    #: Supervisor 判出的意图（8.1 的 `IntentResult.intent`）。
    #: 落 `agent_task.intent`——16.5 有这个列，而它回答的是
    #: "这条任务是当查询问的、还是当制度问的"，排查误答时第一个要看的东西。
    intent: str | None = None
    plan: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    #: 落 `agent_task_step` / `agent_tool_call` / `agent_evidence` /
    #: `agent_conflict` / `agent_review` 五张表的行（16.6 / 16.7）。
    #:
    #: **它们与 `payload` 不是一个东西**：`payload` 是给 API 与前端看的
    #: 一坨 JSON，而这几张表是**能按内容查的**（按 `sql_fingerprint` 找
    #: 反复出现的烂 SQL、按 `content_hash` 找同一条证据）。
    #: 只留 JSON 的话，那些索引一个都用不上。
    steps: tuple[StepRecord, ...] = ()
    #: 计划演进的记录（`agent_plan_revision`，16.6）。
    #:
    #: **它与 `steps` 是两个问题**：`steps` 说"计划里最后有哪几步"，
    #: 这一份说"计划为什么变成这样"——哪一版加了哪几步、由哪个中间发现催生。
    #: 只有前者的话，"这一步是初始计划里的还是下钻出来的"就查不到，
    #: 而开发流程 7.5 的循环类评分正是靠这个区分。
    revisions: tuple[PlanRevisionRecord, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    review: ReviewRecord | None = None
    #: 每个节点的进入/离开事件（16.7 的 `agent_trace_event`）。
    #: **它是「可重放」的载体**：18.3 规定客户端断线重连补历史要回到这张表，
    #: 因为 Redis Stream 会被 MAXLEN 裁掉。
    trace_events: tuple[NodeTrace, ...] = ()


class PlanRevisionRecord(BaseModel):
    """`agent_plan_revision` 的一行（16.6）——**一次计划演进**。

    **不直接用 `agent/schemas/plan.py` 的 `PlanDelta`**：那条链上是
    "进 State 之前必须过 Pydantic 校验"的边界模型，而这里是**行**。
    同 `StepRecord` 的理由——`repositories/` 在依赖链的底端，
    不能 import `agent/schemas`，所以由调用方（`agent/graph.py`）合成它。

    `budget_snapshot` 是 16.6 那一列的来源，用途写在那张表的注释里：
    **复盘「预算是否被借用」**。四类预算各自独立、不可借用是 FR-REV-002
    业务规则 1 的要求，而"这次演进花的是哪一类"只有当时的余额能回答。
    """

    model_config = ConfigDict(frozen=True)

    #: 行的身份。与 `StepRecord` 同：**在这里生成而不是落库时**——
    #: 两条记录是不是同一条，这个问题在造出它的时候就该有答案。
    id: str = Field(default_factory=lambda: new_id(IdPrefix.TASK))
    revision_no: int
    #: `INITIAL` / `EXTEND` / `REPLAN`（16.6 的 `agent_plan_revision.trigger_type`）
    trigger_type: str
    trigger_finding_id: str | None = None
    added_step_ids: tuple[str, ...] = ()
    skipped_step_ids: tuple[str, ...] = ()
    reason: str | None = None
    budget_snapshot: dict[str, int] = Field(default_factory=dict)


class StepRecord(BaseModel):
    """`agent_task_step` 的一行（16.6）。

    **它不直接用 `agent/schemas/plan.py` 的 `TaskStep`**：那张表还带
    `status` / `attempt_count` / `result_summary_json`，而那些来自
    `StepResult`——两个模型合起来才是这一行。更重要的是**分层方向**：
    `repositories/` 在依赖链的底端，不能 import `agent/schemas`，
    所以"计划里的一步"与"这一步的结果"必须在**调用方**（`services/`）
    合成这个领域对象，仓储只认识它。
    """

    model_config = ConfigDict(frozen=True)

    #: 行的身份。**在这里生成而不是落库时**：`StepRecord` 会进 State 的 reducer
    #: （按 id 去重），而"两条记录是不是同一条"这个问题必须在**造出它的时候**
    #: 就有答案——等到写库时再分配，去重就只能靠内容比对了。
    id: str = Field(default_factory=lambda: new_id(IdPrefix.TASK))
    step_key: str
    objective: str
    tool: str | None = None
    depends_on: tuple[str, ...] = ()
    required: bool = True
    status: str
    attempt_count: int = 0
    result_summary: dict[str, Any] | None = None
    #: 这一步是**怎么进计划的**：`PLANNER`（初始计划）或 `EXTENDED`（下钻出来的）。
    #: 16.6 定义的第三个取值 `REPLAN` 目前没有产出者——那条路会把计划整个作废
    #: 回到 `supervisor` 重来，重来的那批仍是 `PLANNER` 的产出。
    origin: str = "PLANNER"
    #: **引入这一步的那一版计划号**，不是任务的终值。
    #:
    #: 这两个值都由 `agent/graph.py` 从 `plan_deltas` 反推——不这么做的话
    #: 每次演进都会把**所有**步骤（含初始那几条）标成新版本，
    #: 于是"哪些步骤是下钻出来的"在表里查不到，而表看起来完全正常。
    revision_no: int = 0


class ToolCallRecord(BaseModel):
    """`agent_tool_call` 的一行（16.6）——**一次工具尝试**，不是一次工具调用。

    `attempt_no` 是它有别于 `StepRecord` 的地方：SQL 的自修复会让同一个步骤
    产生多次尝试（生成 → 修复 → 修复），而"修复了几次、每次为什么失败"
    正是排查 SQL 生成质量的第一手材料（`SqlToolResult.attempts` 就是它的来源）。

    **不保存原始行**：`result_summary` 只放行数、列名与摘要（19.4 与 10.6
    的脱敏纪律）。绑定参数也不在这里——`normalized_sql` 是规范化后的语句，
    参数另行脱敏。
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id(IdPrefix.TOOL_CALL))
    step_id: str | None = None
    tool_name: str
    attempt_no: int = 1
    request_summary: dict[str, Any] | None = None
    normalized_sql: str | None = None
    sql_fingerprint: str | None = None
    result_summary: dict[str, Any] | None = None
    status: str
    error_code: str | None = None
    error_summary: str | None = None
    duration_ms: int | None = None


class ReviewRecord(BaseModel):
    """`agent_review` 的一行（16.7 / 14.2）。

    与 `StepRecord` / `ToolCallRecord` 同理：`repositories/` 不能 import
    `agent/schemas/review.py`，由 `services/` 合成这一行再传下来。
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id(IdPrefix.EVIDENCE))
    round_no: int = 1
    status: str
    score: int = 0
    coverage_score: int = 0
    evidence_score: int = 0
    consistency_score: int = 0
    issues: tuple[dict[str, Any], ...] = ()
    missing_evidence: tuple[str, ...] = ()
    retry_target: str | None = None
    reason_code: str = ""
