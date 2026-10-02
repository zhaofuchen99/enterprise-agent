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
#:
#: **`ABSENCE` 是本实现新增的第 4 个取值**（13.5 原文只有前三个，需回写）。
#: 它描述的是**关于证据本身**的一句话（「现有证据里没有任何 2026 年 3 月的
#: 数据」），而不是关于业务的事实。
#:
#: 为什么必须给它一个位置：这类话**按构造就不可能引用证据**——"没有证据"
#: 恰好是由"引不出证据"证明的。而 14.1 第 3 条要求每条 claim 都有
#: `evidence_ids`，于是它会被判成 `CLAIM_WITHOUT_EVIDENCE`（BLOCKING），
#: 一条**行为完全正确**的拒答因此被拦下。实测踩到（2026-09-28）。
#:
#: 与 `HYPOTHESIS` 的区别：后者是"我猜的"，前者是"这里面没有"。
#: 两者都不该按"缺少引用"判，但**理由不同**，所以是两个取值而不是一个。
ClaimKind = Literal["FACT", "INFERENCE", "HYPOTHESIS", "ABSENCE"]


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
    #: 本次回答是否**因为证据里没有问题所问的那件事**而没有作答（11.8 的拒答）。
    #:
    #: ## 为什么它是结构化的，而不是从答案文本里认
    #:
    #: 在它之前，"拒答"只活在自然语言里——判据只能拿几个词去匹配，
    #: 而模型换个说法就漏判。实测（2026-09-20）：同一条固化问题、
    #: 同一份代码，模型说「没有找到」时判对，说「证据中没有…的条文」时判错，
    #: **四次里错一次**。演示现场跑 `make demo` 就有概率看到红叉，
    #: 而那是判据在猜词，不是系统行为变了（见 CLAUDE.md 约定 77 的告诫）。
    #:
    #: ## 定义是窄的，这一点很要紧
    #:
    #: 判据是「被问的那件事在证据里根本不存在」，**不是**「答案不完整」，
    #: 也不是「这个问题我答不了」：
    #:
    #: - 问《跨境出海业务管理办法》的要求，语料里没有这份制度 →
    #:   `True`（即便检索到语义相邻的《渠道数据报送规范》并引用了它）；
    #: - 问 Q3 华南的净销售额，只覆盖 16 行里的 8 行 →
    #:   `False`——答得不全，但答的正是被问的那件事，且已声明覆盖范围；
    #: - 澄清（问题缺时间/指标）与不支持（问题类型不归这里管）→
    #:   `False`：它们不是"证据里没有"，混进来会让三种处置完全不同的事
    #:   在统计上变成一件事。
    #:
    #: ## 谁来填
    #:
    #: **工具层拒答由代码置位**（检索门禁判 `NO_RELEVANT_KNOWLEDGE`、
    #: 一路证据都没有时 `analysis` 节点直接判定），不经模型；
    #: 模型只在"检索到了语义相邻的材料、但被问的那件事并不在其中"
    #: 那条路上判——那件事只有它能判。
    refused: bool
    claims: tuple[SupportedClaim, ...] = ()
    #: 冲突描述（检测器在 `analysis` **之前**算好并交给模型披露，详设 6.1 的顺序）。
    #: 模型照抄即可，再由 `final` 落到答案里——**不给模型"要不要提"的选择权**
    conflicts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    follow_up_questions: tuple[str, ...] = ()
    #: 任务循环产生的下钻链；没有走出循环时为空
    investigation_chain: tuple[InvestigationStep, ...] = ()


__all__ = ["AnalysisResult", "ClaimKind", "InvestigationStep", "SupportedClaim"]
