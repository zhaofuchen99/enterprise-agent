"""`reviewer` 节点：答案的落地检查（详细设计 14.1 的**两个阶段**）。

## 两阶段的分工：谁判事实、谁判尺度

| | 第一阶段（`review()`，纯代码） | 第二阶段（`_model_review()`） |
|---|---|---|
| 判什么 | 有没有引用、引用在不在、字段泄没泄 | **引用的那些够不够**支撑这句话 |
| 能不能否决 | **能**（BLOCKING → FAIL） | **不能**（一律 WARNING，见 `_MODEL_CHECKS`） |
| 能要求补证据吗 | 能（确定性的一类，见 `_retry_target`） | 能（那一类只有它判得出来） |

**"模型不能独自否决"是刻意的**：一条能独自拦下答案的模型调用是单点故障，
而它拦错的那些与拦对的在产物上长得一模一样。否决权留给判事实的那一半。

## 第一阶段的六条，以及另外四条各自缺什么前提

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

## 四个 status 现在都产得出来

- `PASS` / `FAIL`：第一阶段（一票否决）；
- `RETRY`：第一阶段那一类**或**第二阶段给出的 `retry_target`，
  且预算还有——路由由 `nodes/retry_router.py` 决定，本节点只负责**说清缺什么**，
  不决定去哪（14.3 明写"Reviewer 不直接指定 SQL"）；
- `CLARIFY`：第二阶段给出了澄清问题、且不需要重试。

⚠️ **`CLARIFY` 的状态位只到这里**：任务终态仍是 `SUCCEEDED`，
答案里是一句问句。真正"停在澄清态等用户回复"要 API/SSE 侧的配套，
与 supervisor 的澄清是同一条登记项。
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from app.agent.prompts.review import REVIEW_PROMPT
from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import StepStatus
from app.agent.schemas.review import ModelReview, RetryTarget, ReviewIssue, ReviewResult
from app.agent.state import AgentState, unconsulted_source
from app.core.errors import AgentError
from app.domain.evidence import ConflictSeverity
from app.infrastructure.model_gateway import ModelGateway

#: 答案里**不该出现**的字段名与值（19.2 的脱敏纪律）。
#: 口径很窄，只认数据库口令字段、密钥字段与 token 赋值式——
#: 语料里出现「密码」这个词是正常的（制度可能讲口令管理），
#: 所以不按中文关键词判，只认英文标识符与赋值形态。
_SENSITIVE = re.compile(
    r"(password_hash|password\s*=|api[_-]?key|secret\s*=|bearer\s+[A-Za-z0-9._-]{16,})",
    re.IGNORECASE,
)


#: 模型审查的四个布尔 → `ReviewIssue` 的 `code` 与文案。
#:
#: **code 在这里定，不在模型那边定**：让模型自由写 `code` 的话，
#: "封闭取值"就只是一句约定——它编一个 `NEW_CODE` 出来，
#: `retry_router` 按 code 分流的那张表静默地不认识它。
#: 四个布尔到四个 code 的映射写在代码里，取值就一定是这几个。
#:
#: ⚠️ **一律 `WARNING`**：模型**不能独自把答案拦下**。一条能独自否决的
#: 模型调用是单点故障——它会因为措辞、语气、偶发的过度保守而拦下一条
#: 本该发布的答案，而那种拦截与真拦截长得一模一样。否决权留给确定性六条，
#: 模型负责**要求补证据**（那是它可以被反驳、也应当被反驳的位置）。
_MODEL_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("question_answered", "QUESTION_NOT_ANSWERED", "没有回答用户问的那件事"),
    ("evidence_sufficient", "INSUFFICIENT_EVIDENCE", "引用的证据不足以支撑结论"),
    ("inference_within_bounds", "INFERENCE_OVERREACH", "推断超出了证据支持的范围"),
    ("limitations_clear", "LIMITATION_UNCLEAR", "限制没有说清楚"),
)


def build_reviewer_node(gateway: ModelGateway) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """构造 `reviewer` 节点：**确定性六条 + 模型四条**（14.1 的两个阶段）。

    第一阶段的六条走 `review()`，纯代码；第二阶段的四条走模型，
    判的是"引用的那些够不够支持这句话"——那是语义，代码判不了。
    两者合并成一份 `ReviewResult`，合并规则见 `_combine`。
    """

    async def reviewer(state: AgentState) -> dict[str, Any]:
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
        deterministic = review(analysis, state)
        model, error = await _model_review(gateway, analysis, state)
        result = _combine(deterministic, model, state, refused=analysis.refused)
        update: dict[str, Any] = {"review_result": result}
        if error is not None:
            update["errors"] = [error]
        return update

    return reviewer


async def _model_review(
    gateway: ModelGateway, analysis: AnalysisResult, state: AgentState
) -> tuple[ModelReview | None, AgentError | None]:
    """跑第二阶段。**失败不抛异常**，返回 `(None, 错误)` 让调用方记一笔。

    ## 为什么失败是"降级"而不是"失败"

    审查工具本身坏掉，不等于被审的答案有问题。把它判成 FAIL 的话，
    一次模型侧的网络抖动会让一条完全正常的答案变成"未通过发布检查"——
    而那个结论是**关于答案的**，用户会照着它行事。

    但**也不能静默**：降级意味着"这一轮只跑了确定性六条"，
    而"只跑了六条"与"两阶段都跑了"在产物上必须能分开（记一条
    `MODEL_REVIEW_UNAVAILABLE`）。这正是约定 35 说的那种"看起来完全正常"的
    静默——所以宁可记一条刺眼的 INFO，也不假装它跑过了。
    """
    try:
        result = await gateway.invoke_structured(
            REVIEW_PROMPT,
            ModelReview,
            question=state.get("user_query") or "",
            answer=_render_answer(analysis),
            evidence=_render_evidence(state),
        )
    except AgentError as exc:
        return None, exc
    return result.value, None


def _render_answer(analysis: AnalysisResult) -> str:
    lines = [f"结论：{analysis.direct_answer}"]
    for index, claim in enumerate(analysis.claims, start=1):
        citations = "、".join(claim.evidence_ids) or "（无引用）"
        lines.append(f"{index}. [{claim.kind}] {claim.text}｜引用：{citations}")
    if analysis.limitations:
        lines.append("已声明的限制：" + "；".join(analysis.limitations))
    return "\n".join(lines)


def _render_evidence(state: AgentState) -> str:
    """证据列表，**编号必须与 `analysis` 发出去的那一份逐字相同**。

    编号就是答案里 `evidence_ids` 用的那套（`E1`/`E2`…）。这里重新编一套的话，
    审查员看到的"引用：E3"会指向另一条证据——**而它不会报错**，
    只会让审查基于错误的对应关系做判断。所以直接复用 `analysis` 的那两个函数，
    而不是"照着写一份"。
    """
    from app.agent.nodes.analysis import _number, _render

    numbered = _number(list(state.get("evidence") or []))
    return _render(numbered) if numbered else "（本次没有任何证据）"


def _model_issues(model: ModelReview) -> list[ReviewIssue]:
    failed = set(model.failed_checks)
    issues = [
        ReviewIssue(
            code=code,
            severity="WARNING",
            message=f"模型审查：{message}"
            + (f"｜{model.retry_reason}" if model.retry_reason else ""),
        )
        for field, code, message in _MODEL_CHECKS
        if field in failed
    ]
    return issues


def _combine(
    deterministic: ReviewResult,
    model: ModelReview | None,
    state: AgentState,
    *,
    refused: bool = False,
) -> ReviewResult:
    """两份结论合成一份（14.3 的判定表）。

    **`RETRY` 优先于 `FAIL`**：有预算且补得到时，先去补——直接拦下等于把一条
    本来能救回来的答案扔掉。这与 `review()` 里的取舍同源。

    **`CLARIFY` 排在 `FAIL` 之后**：有 BLOCKING 就是"这条答案不发出去了"，
    而澄清是"请你补一句话，我再答"——前者更明确，也更能解释用户看到的东西。
    """
    issues = [*deterministic.issues]
    if model is not None:
        issues.extend(_model_issues(model))
    else:
        # **降级要留痕**：这一轮只跑了确定性六条，而"只跑了六条"与"两阶段都跑了"
        # 必须能分开。不记的话，一次模型侧的故障会让所有答案都少一道检查，
        # 而产物上完全看不出来（约定 35 说的那类静默）。
        issues.append(
            ReviewIssue(
                code="MODEL_REVIEW_UNAVAILABLE",
                severity="INFO",
                message="第二阶段（模型审查）本次没有跑成，本结论只经过确定性检查",
            )
        )
    blocking = [item for item in issues if item.severity == "BLOCKING"]

    target: RetryTarget | None = deterministic.retry_target
    if target is None and model is not None and model.needs_retry:
        target = model.retry_target
    retrying = target is not None and state.get("review_retries_left", 0) > 0

    if retrying:
        status = "RETRY"
    elif blocking:
        status = "FAIL"
    # ⚠️ **拒答的答案不允许被改成澄清**：`refused=True` 的含义是"证据里没有
    # 用户问的那件事"，而那时把答案换成一句问句是**答非所问**——
    # 用户已经问清楚了，是这个组织没有这条制度。实测踩到：问《直播带货专项补贴制度》
    # 时，第二阶段把"语料里没有"改判成了"你要看的是哪个季度？"。
    # **但补证据仍然允许**：那能救回真实的漏检（比如同义词导致的误拒，金标 rag-06），
    # 而它不改变"如实说不知道"这个结论的形状。
    elif not refused and model is not None and model.clarification_question:
        status = "CLARIFY"
    else:
        status = "PASS"

    return ReviewResult(
        status=status,
        score=_score(issues),
        coverage_score=deterministic.coverage_score,
        evidence_score=deterministic.evidence_score,
        consistency_score=100 if not blocking else 0,
        issues=tuple(issues),
        missing_evidence=deterministic.missing_evidence,
        retry_target=target if retrying else None,
        # **澄清问题只在 CLARIFY 时带出去**：RETRY / FAIL 时它没有落点，
        # 而 `final` 的 FAIL 分支会渲染 `blocking`——带着一句问话会让被拦下的
        # 答案看起来像在跟用户对话。
        clarification_question=(
            model.clarification_question
            if not refused and model is not None and status == "CLARIFY"
            else None
        ),
        reason_code=_reason(
            status, blocking, model_found_issues=bool(model is not None and model.failed_checks)
        ),
    )


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
    # **RETRY 优先于 FAIL**：14.3 的判定表里 RETRY 排在 FAIL 前面，而两者的
    # 前提不同——FAIL 是"这条答案不发出去了"，RETRY 是"还差一步，先去补"。
    # 有预算且补得到时当然选后者：直接拦下等于把一条**本来能救回来**的答案扔掉。
    target = _retry_target(analysis, state)
    retrying = target is not None and state.get("review_retries_left", 0) > 0
    status = "RETRY" if retrying else ("FAIL" if blocking else "PASS")
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
        retry_target=target if retrying else None,
        reason_code=_reason(status, blocking),
    )


def _retry_target(analysis: AnalysisResult, state: AgentState) -> str | None:
    """14.3 的 RETRY 判据：**存在可通过一次 Tool 修复的问题**。

    本版只认一种：**标成 `FACT` 的结论没有任何引用，而两路取证还有一路没查过**。
    这时"去把那一路查了"是一次有明确收益的动作——证据补回来之后，
    `analysis` 要么能给这条结论找到依据，要么证明它确实无据，**两种结果
    都比现在好**。

    ## 为什么不"任何 FAIL 都先重试一次"

    **重试必须能改变结果**。两路都查过之后还没有依据，再补只是把同一个
    动作重做一遍——14.3 明写"Reviewer 不得以'文风不够好'为由触发昂贵
    Tool 重试"，理由是重试要花真钱（一次 SQL 生成 + 一次执行）与真时间。

    所以这里的判据是**"还有一路没查过"**（`state.unconsulted_source`），
    而不是"有没有 BLOCKING"。两者不等价：`REQUIRED_STEP_NOT_RUN`
    同样 BLOCKING，但它是流程出了问题（计划里的必需步骤没跑），
    补一路取证解决不了——那该按 14.3 的"关键数据源不可用"判 FAIL。
    """
    uncited_fact = any(claim.kind == "FACT" and not claim.evidence_ids for claim in analysis.claims)
    if not uncited_fact:
        return None
    route = unconsulted_source(state)
    # `Route` 的取值与 14.4 的 `retry_target` 逐字相同（`"sql"` / `"rag"`）
    return None if route is None else route.value


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


def _reason(status: str, blocking: list[ReviewIssue], *, model_found_issues: bool = False) -> str:
    """`reason_code` —— 最终答案里要引用它，所以它得回答"为什么是这个 status"。

    `GROUNDED` 是 Reviewer 能给出的最强结论，因此它要求**两阶段都没话说**：
    只看确定性那一半的话，"模型记了三条 WARNING"的答案也会被标成 `GROUNDED`
    ——而那是这个码唯一不该出现的场合。
    """
    if status == "PASS":
        return "GROUNDED" if not blocking and not model_found_issues else "PASS_WITH_ISSUES"
    if status == "CLARIFY":
        return "NEED_CLARIFICATION"
    if status == "RETRY":
        # **RETRY 的原因码要指向"要补什么"，不是"哪里不合格"**：
        # 读这个码的人是在排查"为什么多跑了一轮"，而那个问题的答案是
        # "结论缺引用、还有一路没查"——`CLAIM_WITHOUT_EVIDENCE` 只说了一半。
        return "CLAIM_WITHOUT_EVIDENCE_RETRY"
    return blocking[0].code if blocking else "UNKNOWN"


__all__ = ["build_reviewer_node", "review"]
