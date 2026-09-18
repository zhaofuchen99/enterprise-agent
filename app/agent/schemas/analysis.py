"""分析结论的边界模型（详细设计 13.5）。

**这是"答案有证据"在接口上的落点**：`AnalysisResult` 的每一条 `SupportedClaim`
都带 `evidence_ids`，而 `final` 节点渲染答案时逐条把这些 id 变成引用。
一条结论没有证据支撑，在类型上就写不出来——这比在 prompt 里写
「请为每个结论附上证据」可靠得多。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: 结论的性质（13.5）。**它约束措辞的确定性程度**：
#: FACT 必须有直接证据；INFERENCE 至少两项相互支持的证据或一条明确的计算链；
#: HYPOTHESIS 必须用不确定性语言并说明验证方法。
ClaimKind = Literal["FACT", "INFERENCE", "HYPOTHESIS"]


class SupportedClaim(BaseModel):
    """一条有证据支撑的结论（详设 13.5）。"""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    kind: ClaimKind = "FACT"


class InvestigationStep(BaseModel):
    """一轮循环的推理链节点（详设 13.5）。

    **由代码层从 `findings` 与 `plan_deltas` 组装，不由模型生成**：
    这条链要回答的是"你是怎么查出来的"，而模型回忆自己的推理过程
    正是最容易编造的地方——它总结的是"看起来合理的推理"，
    不是实际执行过的那几步。
    """

    model_config = ConfigDict(frozen=True)

    order: int
    finding_id: str
    finding_statement: str
    #: None 表示该发现没有引出新步骤（简单查询走完就结束，链是空的）
    triggered_step_id: str | None = None
    triggered_objective: str | None = None
    evidence_ids: tuple[str, ...] = ()


class AnalysisResult(BaseModel):
    """分析产物（详设 13.5 逐字对应）。

    `limitations` **不是可选的**：检索为空、某一步失败、时点未指定，
    这些都必须写进限制里。把限制留空等于声称"结论没有前提"，
    而任何结论都有前提——只是有时前提恰好都满足了。
    """

    model_config = ConfigDict(extra="ignore")

    direct_answer: str = Field(min_length=1)
    claims: tuple[SupportedClaim, ...] = ()
    #: 冲突描述。切片内的冲突检测属 Phase 9（本版恒为空，见 `nodes/analysis.py`）
    conflicts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    follow_up_questions: tuple[str, ...] = ()
    #: 任务循环产生的下钻链；没有走出循环时为空
    investigation_chain: tuple[InvestigationStep, ...] = ()


__all__ = ["AnalysisResult", "ClaimKind", "InvestigationStep", "SupportedClaim"]
