"""`reviewer` 节点：答案的落地检查（详细设计 14.1 的**第一阶段**）。

## 只做确定性那一段，而且只做判据成立的几条

14.1 把审查分成两阶段：**第一阶段是确定性检查（代码）**，第二阶段是模型审查。
本实现只做第一阶段——冲刺方案 §8.5 把 Reviewer-lite 列为时间不够时第一个砍的项，
所以它按"能确定判定的先做"来切。

**14.1 列了十条，这里做六条**，另外四条各有各的缺前提：

| 14.1 的检查 | 本版 |
|---|---|
| required 步骤是否完成 | ✅ |
| 关键 claim 是否有 `evidence_ids` | ✅ |
| `evidence_ids` 是否真实存在 | ✅ |
| BLOCKING 冲突是否披露 | ✅（本版不产出 BLOCKING，但防护写成通用的） |
| 敏感字段是否泄露 | ✅（窄口径，见 `_SENSITIVE`） |
| 未处理的 `open_questions` 是否已列为未验证事项 | ✅ |
| SQL 是否通过安全校验且未超权限 | ⛔ 校验器在 Tool 内部，被拦下的到不了这里（架构上已满足） |
| 重试预算是否非负 | ⛔ 四类预算的完整版属 Phase 7 |
| `plan_deltas` 的 EXTENDED 步骤有 `trigger_finding_id` | ⛔ 切片内恒空 |
| 是否存在"应下钻而未下钻" | ⛔ 要判"Analysis 给了可能原因但没排除竞争假设"，那是语义 |

## 为什么它能 FAIL 任务

14.3 的判定表里有一票否决：**关键 claim 没有证据**、**有 BLOCKING 冲突未披露**。
这类答案不该交给用户——它看起来和别的答案一样，只是结论没有依据，
而人拿它去做决定。所以本节点在 `status=FAIL` 时让 `final` 输出"审查未通过"
而不是把原答案放出去（见 `nodes/final.py`）。

`RETRY` / `CLARIFY` 不产出：前者要 `retry_router` 与预算，后者要
WAITING_CLARIFICATION 的状态位与续跑入口，两者都属 Phase 8。
**这不是"没检查"，是"检查了但只能整体通过或整体拦下"**，登记在案。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import StepStatus
from app.agent.schemas.review import ReviewIssue, ReviewResult
from app.agent.state import AgentState
from app.domain.evidence import ConflictSeverity

#: 答案里**不该出现**的字段名与值（19.2 的脱敏纪律）。
#: 口径很窄，只认数据库口令字段、密钥字段与 token 赋值式——
#: 语料里出现「密码」这个词是正常的（制度可能讲口令管理），
#: 所以不按中文关键词判，只认英文标识符与赋值形态。
_SENSITIVE = re.compile(
    r"(password_hash|password\s*=|api[_-]?key|secret\s*=|bearer\s+[A-Za-z0-9._-]{16,})",
    re.IGNORECASE,
)


def build_reviewer_node() -> Callable[[AgentState], dict[str, Any]]:
    """构造 `reviewer` 节点。**纯确定性，不调模型**（14.1 明写"与模型审查分离"）。"""

    def reviewer(state: AgentState) -> dict[str, Any]:
        analysis = state.get("analysis_result")
        if analysis is None:
            # supervisor 就失败了，答案本身是一条错误说明——没什么可审的。
            # **不判 FAIL**：那会把"没跑成"与"跑成了但结论没依据"混成一类。
            return {
                "review_result": ReviewResult(
                    status="PASS",
                    score=0,
                    reason_code="NO_ANALYSIS",
                    issues=(
                        ReviewIssue(
                            code="NO_ANALYSIS",
                            severity="INFO",
                            message="本次没有产出分析结果（任务在更早的节点失败）",
                        ),
                    ),
                )
            }
        return {"review_result": review(analysis, state)}

    return reviewer


def review(analysis: AnalysisResult, state: AgentState) -> ReviewResult:
    """按 14.1 第一阶段的六条检查出结论（14.3 的判定表）。"""
    issues: list[ReviewIssue] = []
    evidence_ids = {item.id for item in (state.get("evidence") or [])}
    answer_text = " ".join([analysis.direct_answer, *(claim.text for claim in analysis.claims)])

    issues.extend(_check_required_steps(state))
    issues.extend(_check_claims_have_evidence(analysis))
    issues.extend(_check_evidence_exists(analysis, evidence_ids))
    issues.extend(_check_blocking_conflicts(analysis, state))
    issues.extend(_check_open_questions(analysis, state))
    issues.extend(_check_sensitive(answer_text))

    blocking = [item for item in issues if item.severity == "BLOCKING"]
    status = "FAIL" if blocking else "PASS"
    return ReviewResult(
        status=status,
        score=_score(issues),
        coverage_score=_coverage(state),
        evidence_score=_evidence_ratio(analysis),
        consistency_score=100 if not blocking else 0,
        issues=tuple(issues),
        missing_evidence=tuple(
            f"结论「{claim.text[:20]}…」没有引用"
            for claim in analysis.claims
            if not claim.evidence_ids
        ),
        retry_target=None,
        reason_code=_reason(status, blocking),
    )


def _check_required_steps(state: AgentState) -> list[ReviewIssue]:
    """14.1 第 1 条：`required` 步骤是否完成。

    **没完成 ≠ 失败**：步骤可以是 FAILED（执行出错）或空（查不到），
    两种都该在"限制"里说清楚。这条检查的是"计划里的必需步骤有没有跑过"，
    而它没跑过的唯一原因是流程出了问题。
    """
    results = state.get("step_results") or {}
    issues: list[ReviewIssue] = []
    for step in state.get("task_list") or []:
        if not step.required:
            continue
        result = results.get(step.id)
        if result is None or result.status is StepStatus.PENDING:
            issues.append(
                ReviewIssue(
                    code="REQUIRED_STEP_NOT_RUN",
                    severity="BLOCKING",
                    message=f"必需步骤 {step.id}（{step.objective}）未执行",
                )
            )
    return issues


def _check_claims_have_evidence(analysis: AnalysisResult) -> list[ReviewIssue]:
    """14.1 第 3 条：关键 claim 是否有 `evidence_ids`。

    14.3 把它列进一票否决（"关键 claim 均有证据"是 PASS 的条件之一）。
    **`FACT` 尤其**：13.5 明写「FACT 必须由直接证据支持」——
    一条标成 FACT 却没有引用的结论，读者会当成事实。
    `INFERENCE` 也要求引用（至少两项相互支持的证据），
    只有 `HYPOTHESIS` 允许没有引用（它本来就是"证据不足时的推测"）。
    """
    issues: list[ReviewIssue] = []
    for index, claim in enumerate(analysis.claims, start=1):
        if claim.evidence_ids:
            continue
        if claim.kind == "HYPOTHESIS":
            # **不判问题，但要记一笔**：13.5 允许 HYPOTHESIS 没有引用
            # （它本来就是"证据不足时的推测"），把它也判成阻断会让每一条
            # 含"可能原因"的答案都 FAIL。但完全静默也不对——
            # "本答案含未验证推测"是读者该知道的事。
            issues.append(
                ReviewIssue(
                    code="UNVERIFIED_HYPOTHESIS",
                    severity="INFO",
                    message=f"第 {index} 条结论是未经证据验证的推测：{claim.text[:40]}",
                    claim_id=str(index),
                )
            )
            continue
        issues.append(
            ReviewIssue(
                code="CLAIM_WITHOUT_EVIDENCE",
                severity="BLOCKING" if claim.kind == "FACT" else "WARNING",
                message=f"第 {index} 条结论（{claim.kind}）没有引用任何证据：{claim.text[:40]}",
                claim_id=str(index),
            )
        )
    return issues


def _check_evidence_exists(analysis: AnalysisResult, evidence_ids: set[str]) -> list[ReviewIssue]:
    """14.1 第 4 条：`evidence_ids` 是否真实存在。

    `analysis` 节点已经把未知编号滤掉了（见 `_with_resolved_ids`），
    所以这条在当前实现下**不会触发**。留着它是因为那道过滤是"上游的一个实现细节"，
    而引用指向不存在的证据是"答案里有一条点不开的引用"——
    它看起来完全正常。这类检查不该依赖上游永远正确。
    """
    missing = sorted(
        {
            item
            for claim in analysis.claims
            for item in claim.evidence_ids
            if item not in evidence_ids
        }
    )
    if not missing:
        return []
    return [
        ReviewIssue(
            code="EVIDENCE_NOT_FOUND",
            severity="BLOCKING",
            message=f"答案引用了不存在的证据：{'、'.join(missing[:3])}",
        )
    ]


def _check_blocking_conflicts(analysis: AnalysisResult, state: AgentState) -> list[ReviewIssue]:
    """14.1 第 5 条：BLOCKING 冲突是否披露。

    **本版不产出 BLOCKING 冲突**（`conflict` 节点取 WARNING，理由是判不出谁对），
    所以这条同样不会触发。写成通用形式是为了接 Phase 9 的完整版时不用改判定——
    那时 `conflict` 会产出 BLOCKING，而"没披露"就是一票否决。
    """
    critical = [
        item
        for item in (state.get("conflicts") or [])
        if item.severity is ConflictSeverity.BLOCKING
    ]
    if not critical:
        return []
    disclosed = " ".join(analysis.conflicts) + " " + analysis.direct_answer
    missing = [item for item in critical if item.description not in disclosed]
    return (
        [
            ReviewIssue(
                code="BLOCKING_CONFLICT_NOT_DISCLOSED",
                severity="BLOCKING",
                message=f"有 {len(missing)} 条阻断性冲突未在答案中披露",
            )
        ]
        if missing
        else []
    )


def _check_open_questions(analysis: AnalysisResult, state: AgentState) -> list[ReviewIssue]:
    """14.1 第 8 条：未处理的 `open_questions` 必须作为未验证事项列出。

    "列出"的判据是**答案的限制里有对应条目**（`analysis._pipeline_limitations`
    会把它们拼进去），所以这条检查的是那条拼接有没有被绕过——
    比如以后有人改了 `analysis` 的限制组装逻辑。
    """
    questions = list(state.get("open_questions") or [])
    if not questions:
        return []
    limits = " ".join(analysis.limitations)
    missing = [item for item in questions if item.split("：", 1)[0] not in limits]
    if not missing:
        return []
    return [
        ReviewIssue(
            code="OPEN_QUESTION_NOT_LISTED",
            severity="WARNING",
            message=f"有 {len(missing)} 个未解决的问题没有出现在限制里",
        )
    ]


def _check_sensitive(text: str) -> list[ReviewIssue]:
    """14.1 第 6 条：答案是否含被禁止的敏感字段（19.2 的脱敏纪律）。"""
    found = _SENSITIVE.search(text)
    if found is None:
        return []
    return [
        ReviewIssue(
            code="SENSITIVE_FIELD_LEAKED",
            severity="BLOCKING",
            # **不回显命中的内容**：那条内容本身就是不该出现的东西，
            # 把它写进 issue 只是把泄露搬了个地方。
            message=f"答案里出现了疑似敏感字段（模式：{found.group(0)[:20]}…）",
        )
    ]


def _coverage(state: AgentState) -> int:
    """必需步骤的完成率。"""
    required = [step for step in (state.get("task_list") or []) if step.required]
    if not required:
        return 100
    results = state.get("step_results") or {}
    done = sum(1 for step in required if step.id in results)
    return int(done / len(required) * 100)


def _evidence_ratio(analysis: AnalysisResult) -> int:
    """有引用的结论占比。**HYPOTHESIS 不计入分母**——它本来就不要求引用。"""
    checkable = [claim for claim in analysis.claims if claim.kind != "HYPOTHESIS"]
    if not checkable:
        return 100
    cited = sum(1 for claim in checkable if claim.evidence_ids)
    return int(cited / len(checkable) * 100)


def _score(issues: list[ReviewIssue]) -> int:
    """综合分（14.2 的 `score`）。

    **它是给人看的**：BLOCKING 已经把 `status` 判成 FAIL 了，
    分数再高也不能翻案（见 `ReviewResult` 的说明）。
    """
    penalty = sum({"BLOCKING": 40, "WARNING": 15, "INFO": 5}[item.severity] for item in issues)
    return max(0, 100 - penalty)


def _reason(status: str, blocking: list[ReviewIssue]) -> str:
    if status == "PASS":
        return "GROUNDED" if not blocking else "PASS_WITH_ISSUES"
    return blocking[0].code if blocking else "UNKNOWN"


__all__ = ["build_reviewer_node", "review"]
