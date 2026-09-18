"""答案的落地检查（详细设计 14.1 第一阶段 / 14.3 判定表）。

**这里的用例大多在测"什么时候判 FAIL"**：审查最常见的失败形态是
"什么都通过"——规则写松了，它看起来在工作（确实在跑），
而真正该拦下的那条无依据结论照样发出去了。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.agent.nodes.reviewer import review
from app.agent.schemas.analysis import AnalysisResult, SupportedClaim
from app.agent.schemas.plan import StepResult, StepStatus, TaskStep
from app.agent.state import AgentState
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import (
    Conflict,
    ConflictResolution,
    ConflictSeverity,
    ConflictType,
    Evidence,
)


def _evidence() -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title="结果第 1 行",
        claim="net_sales=100",
        locator={},
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        content_hash="a" * 64,
    )


def _state(
    *,
    evidence: list[Evidence] | None = None,
    steps: tuple[TaskStep, ...] = (),
    results: dict[str, StepResult] | None = None,
    open_questions: tuple[str, ...] = (),
    conflicts: tuple[Conflict, ...] = (),
) -> AgentState:
    state: AgentState = {"evidence": evidence or []}
    if steps:
        state["task_list"] = list(steps)
    state["step_results"] = results or {}
    if open_questions:
        state["open_questions"] = list(open_questions)
    if conflicts:
        state["conflicts"] = list(conflicts)
    return state


def _step(step_id: str = "step_01") -> TaskStep:
    return TaskStep(id=step_id, objective="查数", tool="sql_query")


def _result(step_id: str = "step_01") -> StepResult:
    return StepResult(step_id=step_id, status=StepStatus.SUCCEEDED)


def _claim(text: str = "华东净销售额为 100", **kwargs: Any) -> SupportedClaim:
    return SupportedClaim(text=text, **kwargs)


# ---------------------------------------------------------------- PASS 一侧


def test_a_grounded_answer_passes() -> None:
    """每条结论都有引用、步骤都跑过 → PASS，且没有 issue。

    这是"审查在工作"的基线：如果这条都过不了，后面那些"该 FAIL"的用例
    测的就不是审查逻辑，而是环境问题了。
    """
    item = _evidence()
    analysis = AnalysisResult(
        direct_answer="华东净销售额为 100。",
        claims=(_claim(evidence_ids=(item.id,)),),
        limitations=("数据截止时点未明确",),
    )

    result = review(
        analysis, _state(evidence=[item], steps=(_step(),), results={"step_01": _result()})
    )

    assert result.status == "PASS"
    assert result.issues == ()
    assert result.score == 100
    assert result.evidence_score == 100
    assert result.coverage_score == 100


# ---------------------------------------------------------------- FAIL 一侧


def test_a_fact_without_evidence_is_blocking() -> None:
    """**14.3 的一票否决**：标成 `FACT` 却没有引用的结论。

    这条最值得拦：它读起来和其他结论一模一样，只是背后什么都没有，
    而用户会拿它去做决定。
    """
    analysis = AnalysisResult(
        direct_answer="华东净销售额为 100。",
        claims=(_claim(kind="FACT"),),
    )

    result = review(analysis, _state(evidence=[_evidence()]))

    assert result.status == "FAIL"
    assert result.blocking[0].code == "CLAIM_WITHOUT_EVIDENCE"
    assert result.reason_code == "CLAIM_WITHOUT_EVIDENCE"


def test_a_hypothesis_without_evidence_is_not_blocking_but_is_recorded() -> None:
    """`HYPOTHESIS` 允许没有引用（13.5），**但要记一笔**。

    - 判成 BLOCKING 的话，"可能原因"这类正常表述会让每一条答案都 FAIL，
      审查退化成对所有任务的拒绝；
    - 完全静默也不对：「本答案含未验证推测」是读者该知道的事。
    """
    analysis = AnalysisResult(
        direct_answer="可能是口径差异。",
        claims=(_claim(kind="HYPOTHESIS"),),
        limitations=("尚需验证",),
    )

    result = review(analysis, _state(evidence=[_evidence()]))

    assert result.status == "PASS"
    assert len(result.issues) == 1
    assert result.issues[0].code == "UNVERIFIED_HYPOTHESIS"
    assert result.issues[0].severity == "INFO"
    assert result.warnings == ()


def test_a_required_step_that_never_ran_is_blocking() -> None:
    """必需步骤没跑过 → 一票否决（14.1 第 1 条）。

    它意味着"计划里该查的那一路根本没查"，而这时的答案必然是片面的。
    """
    item = _evidence()
    analysis = AnalysisResult(direct_answer="答", claims=(_claim(evidence_ids=(item.id,)),))

    result = review(analysis, _state(evidence=[item], steps=(_step(),), results={}))

    assert result.status == "FAIL"
    assert result.blocking[0].code == "REQUIRED_STEP_NOT_RUN"
    assert result.coverage_score == 0


def test_an_optional_step_that_did_not_run_is_not_an_issue() -> None:
    """非必需步骤没跑不算问题——计划里本来就有可选的步骤。"""
    item = _evidence()
    optional = TaskStep(id="step_02", objective="补充", tool="rag_retrieve", required=False)
    analysis = AnalysisResult(direct_answer="答", claims=(_claim(evidence_ids=(item.id,)),))

    result = review(
        analysis,
        _state(evidence=[item], steps=(_step(), optional), results={"step_01": _result()}),
    )

    assert result.status == "PASS"


def test_a_citation_to_a_missing_evidence_is_blocking() -> None:
    """引用指向不存在的证据 → 一票否决（14.1 第 4 条）。

    **当前实现下这条不会触发**（`analysis` 已经把未知编号滤掉了），
    但那条过滤是上游的实现细节；引用指向空处是"答案里有一条点不开的引用"，
    它看起来完全正常，所以检查不该依赖上游永远正确。
    """
    analysis = AnalysisResult(
        direct_answer="答",
        claims=(_claim(evidence_ids=("evd_01M2SG0000000000000000AA",)),),
    )

    result = review(analysis, _state(evidence=[_evidence()]))

    assert result.status == "FAIL"
    assert result.blocking[0].code == "EVIDENCE_NOT_FOUND"


def test_a_leaked_credential_pattern_is_blocking() -> None:
    """答案里出现凭据形态 → 一票否决（19.2 的脱敏纪律）。

    模式**只认英文标识符与赋值形态**，不认中文关键词：语料里出现
    「密码」这个词是正常的（制度会讲口令管理），按中文判会大面积误报。
    """
    item = _evidence()
    analysis = AnalysisResult(
        direct_answer="用户记录里的 password_hash 是 abc123。",
        claims=(_claim(evidence_ids=(item.id,)),),
    )

    result = review(analysis, _state(evidence=[item]))

    assert result.status == "FAIL"
    assert result.blocking[0].code == "SENSITIVE_FIELD_LEAKED"
    # **不回显命中的内容**：那条内容本身就是不该出现的东西
    assert "abc123" not in result.blocking[0].message


def test_chinese_word_password_is_not_flagged() -> None:
    """「密码」这个词本身不触发——制度文本里完全可能正常出现。"""
    item = _evidence()
    analysis = AnalysisResult(
        direct_answer="制度要求口令（密码）每 90 天更换一次。",
        claims=(_claim(evidence_ids=(item.id,)),),
    )

    assert review(analysis, _state(evidence=[item])).status == "PASS"


def test_open_questions_must_appear_in_the_limitations() -> None:
    """未解决的问题必须列进限制（14.1 第 8 条）。

    没列出时是 WARNING 而不是 BLOCKING：结论本身仍可能有依据，
    缺的是"哪些事还没查清"的交代——那影响的是完整性，不是正确性。
    """
    item = _evidence()
    analysis = AnalysisResult(direct_answer="答", claims=(_claim(evidence_ids=(item.id,)),))

    result = review(
        analysis,
        _state(evidence=[item], open_questions=("step_02：未取得结果",)),
    )

    assert result.status == "PASS"
    assert result.warnings[0].code == "OPEN_QUESTION_NOT_LISTED"

    listed = analysis.model_copy(update={"limitations": ("step_02：未取得结果",)})
    assert (
        review(listed, _state(evidence=[item], open_questions=("step_02：未取得结果",))).warnings
        == ()
    )


def test_a_blocking_conflict_must_be_disclosed() -> None:
    """BLOCKING 冲突没披露 → 一票否决（14.1 第 5 条）。

    本版 `conflict` 节点不产出 BLOCKING（判不出谁对，取 WARNING），
    所以这条要手工构造——但判定写成通用的，接 Phase 9 时不用改。
    """
    item = _evidence()
    conflict = Conflict(
        id="cft_01M2SG0000000000000000AA",
        type=ConflictType.VALUE,
        evidence_ids=(item.id, "evd_01M2SG0000000000000000AA"),
        severity=ConflictSeverity.BLOCKING,
        description="两个来源的净销售额相差 40%",
        resolution=ConflictResolution.UNRESOLVED,
    )
    analysis = AnalysisResult(direct_answer="答", claims=(_claim(evidence_ids=(item.id,)),))

    result = review(analysis, _state(evidence=[item], conflicts=(conflict,)))

    assert result.status == "FAIL"
    assert result.blocking[0].code == "BLOCKING_CONFLICT_NOT_DISCLOSED"


# ---------------------------------------------------------------- 别的路径


def test_no_analysis_result_passes_with_an_info_issue() -> None:
    """没有分析结果（更早的节点就失败了）**不判 FAIL**。

    `final` 对这两种情形的渲染完全不同（`_failure_answer` vs `_blocked_answer`）；
    把它们混成一类会让"没跑成"看起来像"跑成了但结论没依据"。
    """
    from app.agent.nodes.reviewer import build_reviewer_node

    node = build_reviewer_node()
    outcome = node({"evidence": []})

    assert outcome["review_result"].status == "PASS"
    assert outcome["review_result"].reason_code == "NO_ANALYSIS"
    assert outcome["review_result"].issues[0].severity == "INFO"


def test_the_score_is_reported_but_does_not_override_a_blocking_verdict() -> None:
    """分数是给人看的，`status` 才是判据（14.3 的判定表有一票否决）。

    只有一条 BLOCKING 时分数是 60——它仍然不足以 PASS。
    把 status 算成分数的函数会让高分盖过否决条件。
    """
    item = _evidence()
    analysis = AnalysisResult(
        direct_answer="答",
        claims=(_claim(evidence_ids=(item.id,)), _claim(kind="FACT")),
    )

    result = review(analysis, _state(evidence=[item]))

    assert result.status == "FAIL"
    assert 0 < result.score < 100
    assert result.evidence_score == 50  # 两条可检查的结论，一条有引用
