"""`final` 按审查结论分派（详细设计 14.3 的四个 status）。

**这里钉的是"某个 status 会不会被当成正常答案渲染出去"**。
`final` 原先只认 `FAIL`，其余全部落进 `else` 分支照常渲染——
于是新加的 `RETRY` / `CLARIFY` 会被原样放行，而**放行一条本该重试的答案
与放行一条本该澄清的答案，看起来都是一条正常答案**。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.nodes.final import build_final_node
from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.review import ReviewIssue, ReviewResult


def _analysis() -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "direct_answer": "华东 Q3 净销售额为 1.12 亿元。",
            "refused": False,
            "claims": [
                {"text": "华东 Q3 净销售额为 1.12 亿元", "kind": "FACT", "evidence_ids": []}
            ],
        }
    )


def _review(**overrides: Any) -> ReviewResult:
    base: dict[str, Any] = {"status": "PASS", "reason_code": "GROUNDED"}
    return ReviewResult.model_validate({**base, **overrides})


def _run(review: ReviewResult) -> str:
    state: Any = {"analysis_result": _analysis(), "review_result": review, "evidence": []}
    answer = build_final_node()(state)["final_answer"]
    assert isinstance(answer, str)
    return answer


def test_a_passing_review_renders_the_answer() -> None:
    """基线：PASS 时答案照常出现。**没有它，下面几条"没渲染出来"
    可能只是因为 `final` 什么都不渲染。**"""
    assert "1.12 亿" in _run(_review())


def test_a_failed_review_is_not_published() -> None:
    """一票否决：结论没有依据时不把答案放出去（14.3）。"""
    answer = _run(
        _review(
            status="FAIL",
            reason_code="CLAIM_WITHOUT_EVIDENCE",
            issues=(
                ReviewIssue(
                    code="CLAIM_WITHOUT_EVIDENCE",
                    severity="BLOCKING",
                    message="第 1 条结论（FACT）没有引用任何证据",
                ),
            ),
        )
    )

    assert "未通过发布前的落地检查" in answer
    assert "1.12 亿" not in answer, "被拦下的答案不能出现在正文里"


def test_a_retry_verdict_is_not_published_either() -> None:
    """`RETRY` 走到 `final` 时**按 FAIL 处置**。

    正常路径上它到不了这里（`retry_router` 会接走），但万一到了：
    `RETRY` 的含义正是"这条答案还不该发出去"，
    渲染出去等于把一次没跑完的重试当成结论。
    """
    answer = _run(
        _review(status="RETRY", retry_target="rag", reason_code="CLAIM_WITHOUT_EVIDENCE_RETRY")
    )

    assert "1.12 亿" not in answer
    assert "未通过发布前的落地检查" in answer


def test_a_clarify_verdict_asks_the_question_and_drops_the_draft() -> None:
    """`CLARIFY` 要把**问题**交出去，而草稿不能跟着一起放出去。

    只把草稿渲染出来、附一句"请确认"是不够的：读者会直接采信那段结论，
    而它正是审查认为**还不能给**的那一段。
    """
    answer = _run(
        _review(
            status="CLARIFY",
            reason_code="NEED_SCOPE",
            clarification_question="你要看的是哪个区域、哪个季度？",
        )
    )

    assert "你要看的是哪个区域、哪个季度？" in answer
    assert "1.12 亿" not in answer, "草稿不该跟着澄清一起出去"


@pytest.mark.parametrize("status", ["RETRY", "CLARIFY", "FAIL"])
def test_every_non_pass_status_keeps_the_draft_out(status: str) -> None:
    """把四个 status 摆在一起看：**只有 PASS 会渲染草稿**。

    单条用例各自都在说"这一条没渲染"，合起来才是"分派是穷举的"——
    将来加第五个 status 时，漏掉它的症状是它落进 `else` 被当正常答案放行。
    """
    answer = _run(_review(status=status, retry_target="rag"))

    assert "1.12 亿" not in answer
