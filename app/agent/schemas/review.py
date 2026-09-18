"""审查结论的边界模型（详细设计 14.2）。

字段与详设**逐字对应**，包括四个分数字段——即使切片版的算分方式比完整版简单。
理由是同样的：`answer_payload` 里要带上它，而字段名一改，
前端与评测集就都对不上了。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: 审查状态（14.3）。切片内只会产出 `PASS` 与 `FAIL`：
#: `RETRY` 要有 `retry_router` 与那四类预算的完整版，`CLARIFY` 要有
#: WAITING_CLARIFICATION 的状态位——两者都属 Phase 8，已登记。
ReviewStatus = Literal["PASS", "RETRY", "CLARIFY", "FAIL"]

#: 问题严重度。**BLOCKING 的含义是"不得交给用户"**：
#: 一条没有依据的结论看起来和别的结论一样，而它会被人拿去做决定。
Severity = Literal["INFO", "WARNING", "BLOCKING"]

#: 重试目标（14.4）。切片内恒为 None（不重试），字段留给 Phase 8。
RetryTarget = Literal["sql", "rag", "search", "analysis", "expand", "replan"]


class ReviewIssue(BaseModel):
    """一条审查发现（详细设计 14.2）。

    `code` 用**封闭取值**而不是自由字符串：它是"该不该重试、重试哪一路"
    的判据（14.4 的 RetryRouter 按 code 分流），而自由字符串会让新增一类
    问题表现为"重试路由没反应"，没有任何报错。
    """

    model_config = ConfigDict(frozen=True)

    code: str
    severity: Severity
    message: str
    #: 关联的 claim 序号（从 1 起）。`None` 表示这条不是针对某一条结论
    claim_id: str | None = None
    evidence_ids: tuple[str, ...] = ()


class ReviewResult(BaseModel):
    """审查结论（详细设计 14.2 逐字对应）。

    **分数是给人看的，`status` 才是判据**：14.3 的判定表里有几条是
    "一票否决"（BLOCKING、关键 claim 无证据），分数再高也不能 PASS。
    把 `status` 算成分数的函数会让那几条否决条件被高分盖过去。
    """

    model_config = ConfigDict(frozen=True)

    status: ReviewStatus
    score: int = Field(default=0, ge=0, le=100)
    coverage_score: int = Field(default=0, ge=0, le=100)
    evidence_score: int = Field(default=0, ge=0, le=100)
    consistency_score: int = Field(default=0, ge=0, le=100)
    issues: tuple[ReviewIssue, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    retry_target: RetryTarget | None = None
    clarification_question: str | None = None
    #: 机器可读的原因。**与 `issues` 分开**：issues 是给人看的清单，
    #: 而这一条是"为什么是这个 status"的一句话，最终答案里要引用它。
    reason_code: str = ""

    @property
    def blocking(self) -> tuple[ReviewIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "BLOCKING")

    @property
    def warnings(self) -> tuple[ReviewIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "WARNING")


__all__ = [
    "RetryTarget",
    "ReviewIssue",
    "ReviewResult",
    "ReviewStatus",
    "Severity",
]
