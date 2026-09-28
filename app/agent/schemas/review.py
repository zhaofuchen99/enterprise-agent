"""审查结论的边界模型（详细设计 14.2）。

字段与详设**逐字对应**，包括四个分数字段——即使切片版的算分方式比完整版简单。
理由是同样的：`answer_payload` 里要带上它，而字段名一改，
前端与评测集就都对不上了。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: 审查状态（14.3）。四个取值都会产出：
#: - `PASS` / `FAIL`：确定性六条（14.1 第一阶段）；
#: - `RETRY`：`retry_router` 与四类预算（14.4）；
#: - `CLARIFY`：模型审查给出澄清问题、而重试无路可走时。
#: ⚠️ **`CLARIFY` 的状态位只到这里**：任务终态仍是 `SUCCEEDED`，
#: 答案里是一句澄清问句。真正「停在澄清态等用户回复」要 API/SSE 侧的配套，
#: 与 supervisor 的澄清是同一条登记项。
ReviewStatus = Literal["PASS", "RETRY", "CLARIFY", "FAIL"]

#: 问题严重度。**BLOCKING 的含义是"不得交给用户"**：
#: 一条没有依据的结论看起来和别的结论一样，而它会被人拿去做决定。
Severity = Literal["INFO", "WARNING", "BLOCKING"]

#: 重试目标（14.4）。
#: ⚠️ **本版只有 `sql` / `rag` 会被真正产出**（`reviewer._retry_target`），
#: `analysis` 路由得到但没有产出者，`expand` / `replan` / `search` 连落点都还没有
#: ——见 `nodes/retry_router.py` 的模块说明。
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


class ModelReview(BaseModel):
    """**模型审查**的结构化输出（14.1 第二阶段）。

    14.1 把这一阶段要查的东西写死了四条：「是否回答问题、证据是否足够、
    推断是否越界、限制是否清楚，以及应补哪类证据」。这里就是那四条 + 一条。

    ## 为什么是**四个布尔**而不是让它自由写 issue

    让它写 `code` 的话，"封闭取值"就只是一句约定——模型编一个 `NEW_CODE`
    出来，`retry_router` 按 code 分流的那张表就静默地不认识它。
    改成布尔之后，**code 由代码从布尔推出来**（`reviewer._model_issues`），
    取值一定是本文件里那几个。

    另一个好处是**判据可校准**：四个布尔各自撞板的比例能单独统计，
    而一堆自由文本的 code 只能靠人读。

    ## 它**不能**产出 `BLOCKING`

    四个布尔只映射到 `WARNING`（见 `reviewer._model_issues`）。
    理由不是"模型判不准"，而是**一条能独自否决答案的模型调用是单点故障**：
    它会因为措辞、语气、偶发的过度保守而拦下一条本该发布的答案，
    而这类拦截与真拦截长得一模一样。确定性的六条负责否决，
    模型负责**要求补证据**——那是它可以被反驳、也应当被反驳的位置。
    """

    model_config = ConfigDict(frozen=True)

    #: 是否回答了用户问的那件事（问 A 答 B、只答了一半，都是 False）
    question_answered: bool = True
    #: 结论是否有足够证据支撑（不是"有没有引用"——那是第一阶段判的，
    #: 而是"引用的那些够不够支持这句话"）
    evidence_sufficient: bool = True
    #: 推断是否越过证据（把相关性说成因果、把个别说成普遍）
    inference_within_bounds: bool = True
    #: 限制是否说清楚了（语料里没有、某一步失败、时点没给）
    limitations_clear: bool = True
    #: 该补哪一类证据（14.3）。**只填"重跑某一个工具没用"的情形**——
    #: 「某个查询错了」那一类由第一阶段与 `<retry_target>` 的其他取值负责。
    retry_target: RetryTarget | None = None
    #: 要补什么，给人看的。**不写具体 SQL**（14.3 明写 Reviewer 不指定 SQL）。
    retry_reason: str | None = None
    #: 需要用户先定口径 / 时间 / 范围时填这里（14.3 的 CLARIFY）。
    #: **它必须是问用户的、用户能回答的**，不是"请你提供更多信息"这种空话。
    clarification_question: str | None = None

    @property
    def needs_retry(self) -> bool:
        return self.retry_target is not None

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """没通过的那几条（字段名）。**顺序固定**，便于与日志比对。"""
        return tuple(
            name
            for name in (
                "question_answered",
                "evidence_sufficient",
                "inference_within_bounds",
                "limitations_clear",
            )
            if not getattr(self, name)
        )


__all__ = [
    "ModelReview",
    "RetryTarget",
    "ReviewIssue",
    "ReviewResult",
    "ReviewStatus",
    "Severity",
]
