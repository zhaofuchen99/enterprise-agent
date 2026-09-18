"""计划与进度相关的边界模型（详细设计 7.4 / 8.1 / 6.6）。

**这些模型是"进入 State 之前必须过的那道校验"**（开发流程 5.3：禁止把裸 `dict`
写进 State）。它们直接对应详设的字段表，字段名与那里逐字一致——改名等于
让 `AgentState` 的注释与代码对不上，而那份注释是排查时唯一的对照表。

## 与详设的一处偏差：没有 `Planner` 节点

详设 6.1 的图里 `supervisor → planner → dispatch` 是三段。冲刺方案 §8.1 把
最小 Graph 定为 6 个节点（`supervisor / sql / rag / reflect / analysis / final`），
**没有 planner**。于是「从意图到计划」这一步落在 `supervisor` 里，
且是**确定性**的：`required_sources` 里有哪几路，就生成几个步骤。

这不是省事——`planner` 在详设里的职责（把意图拆成有依赖的步骤）只有到了
`plan_extend`（演进）才真正需要模型，而演进按冲刺方案是后置的。
先做确定性的版本，`task_list` 的形状与依赖字段都留着，
接模型规划时改的是**生成方式**，不是 State 形状。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.evidence import TimeRange

#: 工具名。**与 `app/tools/base.py` 的 `ToolName` 是同几个字符串**，
#: 但这里只允许两种：`web_search` 属 Search Tool（切片内后置）。
#: 直接复用 `ToolName` 会让类型允许一个本项目没有的工具。
StepTool = Literal["sql_query", "rag_retrieve"]

#: 意图分类（详设 8.1）。**取值是封闭的**：多一个值意味着多一类路由分支，
#: 而每个分支都要有对应的处理；写成 `str` 会让未知取值一路走到下游才炸。
Intent = Literal[
    "QUERY",
    "DIAGNOSIS",
    "POLICY_QA",
    "CROSS_SOURCE",
    "EXTERNAL_RESEARCH",
    "CLARIFICATION",
    "UNSUPPORTED",
]


class StepStatus(StrEnum):
    """步骤状态（详设 6.3 的 `next_ready_step` 只认 PENDING）。

    `SKIPPED` 与 `FAILED` 必须分开：前者是"计划演进后这一步不需要了"，
    后者是"跑了但没成"。两者在最终答案的"限制"一节里说法完全不同——
    前者不提，后者必须列出来。
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class TaskStep(BaseModel):
    """计划里的一步（详设 7.1 的 `task_list` 元素）。

    `depends_on` 在这一版**总是空的**：确定性计划里每一步都是独立的一路
    （SQL 拿数字、RAG 拿解释），没有先后依赖。字段留着是因为演进（`plan_extend`）
    会产生依赖步骤，而到那时再往 State 里加字段，要连带改所有节点的签名。
    """

    model_config = ConfigDict(frozen=True)

    id: str
    objective: str
    tool: StepTool
    depends_on: tuple[str, ...] = ()
    #: 必需步骤。**至少一步必需**（详设 8.3 的计划校验之一）——
    #: 全部非必需的计划允许"什么都不做就收敛"，那不是计划。
    required: bool = True


class StepResult(BaseModel):
    """一步的执行结果（详设 7.1 的 `step_results` 值）。

    **不放工具原始结果**：那些在各自的 Tool 结果模型里，且按 10.6 的纪律
    「SQL 结果不直接写应用日志、也不默认持久化原始行」，State 里这一层只留摘要。
    原始行在 `tool_results` 里，生命周期只到本任务结束。
    """

    model_config = ConfigDict(frozen=True)

    step_id: str
    status: StepStatus
    summary: str = ""
    #: 这一步产出的证据 id。**下游按它回溯**，不是按 step_id 去 re-query
    evidence_ids: tuple[str, ...] = ()
    error_code: str | None = None
    duration_ms: int = 0
    #: 这一步是否"跑了但没拿到有用东西"（SQL 空结果 / RAG 无相关知识）。
    #: 它与 `FAILED` 是两回事：前者是**关于数据的事实**，后者是执行出错。
    #: `reflect` 的演进判定读的正是它。
    empty: bool = False


class Finding(BaseModel):
    """一步提炼出的中间发现（详设 7.1 的 `findings`）。

    **`findings` 用追加 reducer 而不是覆盖**（详设 7.2 的理由）：推理链要保留
    完整历史，最终答案要能回答"你是怎么查出来的"。被覆盖掉就无法回溯。
    """

    model_config = ConfigDict(frozen=True)

    id: str
    statement: str
    step_id: str
    evidence_ids: tuple[str, ...] = ()


class ProgressAssessment(BaseModel):
    """`reflect` 的进度判定（详设 7.1 的 `progress_assessment`）。

    `decision` 的四选一来自详设 6.6.1 的完成/演进判定表。**模型的判定不直接采信**
    （详设 6.3 的 `route_after_reflect` 说明）：本实现的 `reflect` 是确定性的，
    因此这个模型由代码填、不由模型填。字段形状与详设一致，
    接模型判定时改的是**谁来填**，不是 State 形状。
    """

    model_config = ConfigDict(frozen=True)

    decision: Literal["CONTINUE", "EXPAND", "SUFFICIENT", "BLOCKED"]
    reason: str
    #: 演进时要新增的步骤（`decision=EXPAND` 时非空）
    proposed_steps: tuple[TaskStep, ...] = ()


class IntentResult(BaseModel):
    """Supervisor 的输出（详设 8.1 逐字对应）。

    **`required_sources` 是这一版的路由依据**：详设 8.2 的第 1–4 条判断规则
    （简单指标查询优先 SQL、制度问法优先 RAG、要"数据+解释"则两路都要）
    落到代码就是这一个字段。它是**模型的结论**，所以后面有一道代码校验
    （`supervisor._validate`）——模型给出空列表时不能当作"不需要任何工具"。
    """

    model_config = ConfigDict(extra="ignore")

    intent: Intent
    metrics: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    filters: dict[str, str | list[str]] = Field(default_factory=dict)
    time_range: TimeRange | None = None
    comparison: Literal["NONE", "YOY", "MOM", "TARGET"] = "NONE"
    required_sources: tuple[str, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_fields: tuple[str, ...] = ()
    clarification_question: str | None = None


__all__ = [
    "Finding",
    "Intent",
    "IntentResult",
    "ProgressAssessment",
    "StepResult",
    "StepStatus",
    "StepTool",
    "TaskStep",
]
