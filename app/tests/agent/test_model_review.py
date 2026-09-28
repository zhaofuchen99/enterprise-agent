"""Reviewer 的第二阶段：**结构化模型审查**（详细设计 14.1 后半 / 14.3）。

这一阶段的职责与第一阶段完全不同，用例也围着那条界线组织：

| | 第一阶段（代码） | 第二阶段（模型） |
|---|---|---|
| 判什么 | 有没有引用、引用在不在 | **引用的那些够不够** |
| 能不能否决 | **能**（BLOCKING → FAIL） | **不能**（一律 WARNING） |
| 能要求补证据吗 | 能（确定性的一类） | 能（那一类只有它判得出来） |

**"模型不能独自否决"是这里最重要的一条**：一条能独自拦下答案的模型调用
是单点故障——它会因为措辞、语气、偶发的过度保守而拦下一条本该发布的答案，
而那种拦截与真拦截在产物上长得一模一样。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from app.agent.nodes.reviewer import build_reviewer_node
from app.agent.schemas.analysis import AnalysisResult, SupportedClaim
from app.agent.schemas.plan import StepResult, StepStatus, TaskStep
from app.agent.schemas.review import ModelReview
from app.agent.state import AgentState
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence
from app.tests.fakes import FakeFailure, FakeModelGateway


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


def _analysis(item: Evidence, *, cited: bool = True) -> AnalysisResult:
    """一份**确定性检查全过**的分析。这样剩下的就只能来自模型那一侧。

    ⚠️ 证据**由调用方传进来**，不在这里现造：`_evidence()` 每次生成一个新 id，
    而"引用的 id 不在 State 的证据列表里"是**一票否决**（`EVIDENCE_NOT_FOUND`）
    ——现造的话每条用例都会先撞上它，被测的那条判据反而看不到。
    """
    return AnalysisResult.model_validate(
        {
            "direct_answer": "华东 Q3 净销售额为 100 元。",
            "refused": False,
            "claims": (
                SupportedClaim.model_validate(
                    {
                        "text": "华东 Q3 净销售额为 100 元",
                        "kind": "FACT",
                        "evidence_ids": [item.id] if cited else [],
                    }
                ),
            ),
            "limitations": ("仅覆盖已召回的那几行",),
        }
    )


def _case(
    *, cited: bool = True, budget: int = 1, steps: tuple[str, ...] = ("sql_query",)
) -> AgentState:
    """一份状态：**证据只造一次**，分析与 State 共用同一份。

    两处各造一份的话，分析引用的 `evd_` 不在 State 的证据列表里——
    那撞的是**一票否决**（`EVIDENCE_NOT_FOUND`），每条用例都会先被它拦下，
    被测的那条判据反而看不到。
    """
    item = _evidence()
    return cast(
        AgentState,
        {
            "user_query": "华东 Q3 净销售额是多少？",
            "evidence": [item],
            "analysis_result": _analysis(item, cited=cited),
            "review_retries_left": budget,
            "expansions_left": 1,
            "replans_left": 1,
            "task_list": [
                TaskStep(id=f"step_{i:02d}", objective="查", tool=tool)
                for i, tool in enumerate(steps, start=1)
            ],
            "step_results": {
                f"step_{i:02d}": StepResult(step_id=f"step_{i:02d}", status=StepStatus.SUCCEEDED)
                for i, _ in enumerate(steps, start=1)
            },
        },
    )


def _refused(state: AgentState) -> AnalysisResult:
    """把 State 里那份分析标成"拒答"。

    **必须用 State 自带的那一份**，不能另造：`_evidence()` 每次生成一个新 id，
    另造的分析引用的 `evd_` 不在证据列表里，会先撞上一票否决
    （`EVIDENCE_NOT_FOUND`）——于是用例测到的是"引用不存在"，不是"拒答"。
    """
    return cast("AnalysisResult", state["analysis_result"]).model_copy(update={"refused": True})


async def _run(state: AgentState, reply: Any = None, *, failure: Any = None) -> Any:
    gateway = FakeModelGateway(
        responses=[] if failure is not None else [reply or ModelReview()],
        failure=failure,
    )
    node = build_reviewer_node(gateway)
    return (await node(state))["review_result"]


# ------------------------------------------------------------------ 四条判断


async def test_a_clean_answer_passes_with_no_issues() -> None:
    """基线：四个布尔全 true → PASS、零 issue、`GROUNDED`。

    **没有它，下面几条"记了 WARNING"可能只是因为审查见谁都记一笔。**
    """
    result = await _run(_case())

    assert result.status == "PASS"
    assert result.issues == ()
    assert result.reason_code == "GROUNDED"


async def test_a_failed_check_becomes_a_warning_not_a_blocking() -> None:
    """模型判否 → 记一条 **WARNING**，而 `status` 仍是 PASS。

    这是本阶段的核心边界：**模型可以要求补证据，但不能独自把答案拦下**。
    拦下（FAIL）意味着"这条答案不发给用户"，那件事只有确定性检查做得起——
    因为它们判的是事实（引用不存在、字段泄露），而模型判的是措辞与尺度。
    """
    result = await _run(_case(), ModelReview(evidence_sufficient=False))

    assert result.status == "PASS", "模型单独判否不足以拦下答案"
    assert [item.code for item in result.issues] == ["INSUFFICIENT_EVIDENCE"]
    assert result.issues[0].severity == "WARNING"
    # 但它**没有**被吞掉：`GROUNDED` 是"两阶段都没话说"，模型记了账就不能给
    assert result.reason_code == "PASS_WITH_ISSUES"


async def test_every_failed_check_maps_to_its_own_code() -> None:
    """四个布尔各对应一个 **封闭取值** 的 code，且顺序固定。

    code 由代码从布尔推出来（不是模型写的自由文本）：让模型写 code 的话，
    "封闭取值"就只是一句约定——它编一个 `NEW_CODE` 出来，
    `retry_router` 按 code 分流的那张表静默地不认识它。
    """
    result = await _run(
        _case(),
        ModelReview(
            question_answered=False,
            evidence_sufficient=False,
            inference_within_bounds=False,
            limitations_clear=False,
        ),
    )

    assert [item.code for item in result.issues] == [
        "QUESTION_NOT_ANSWERED",
        "INSUFFICIENT_EVIDENCE",
        "INFERENCE_OVERREACH",
        "LIMITATION_UNCLEAR",
    ]
    assert all(item.severity == "WARNING" for item in result.issues)


# ------------------------------------------------------------------ 重试与澄清


async def test_the_model_can_ask_for_evidence_and_the_target_is_used() -> None:
    """模型给出的 `retry_target` 会被采纳，并把结论改成 RETRY（14.3）。

    这是第二阶段存在的**主要理由**：确定性检查只认得出"有没有引用"，
    而"引用的那些够不够"只有模型判得出来——它判出来之后，
    `retry_router` 才有机会去补。
    """
    result = await _run(
        _case(),
        ModelReview(evidence_sufficient=False, retry_target="rag", retry_reason="缺渠道维度的明细"),
    )

    assert result.status == "RETRY"
    assert result.retry_target == "rag"
    assert "缺渠道维度的明细" in result.issues[0].message


async def test_the_retry_target_needs_budget_as_well() -> None:
    """有请求但没预算 → 不判 RETRY，退回 PASS（模型的话只记 WARNING）。

    **与第一阶段同一条规矩**：`RETRY` 的前提是"重试真的会发生"。
    预算为 0 时判 RETRY，`retry_router` 会立刻把它降级回去——
    而中间那份 `RETRY` 会让"到底重试了没有"在产物上变得说不清。
    """
    result = await _run(
        _case(budget=0),
        ModelReview(evidence_sufficient=False, retry_target="rag"),
    )

    assert result.status == "PASS"
    assert result.retry_target is None


async def test_a_clarification_question_becomes_clarify() -> None:
    """模型给出澄清问题、且无需重试 → **`CLARIFY`**（14.3 的第四行）。

    这一条让 `CLARIFY` 第一次有了产出者：在此之前四个 status 里只有
    `PASS` / `FAIL` 能被产出，而"需要用户定口径 / 时间 / 范围"这一类
    只能被硬塞进 FAIL——用户看到的会是"未通过发布检查"，
    而真正该看到的是"你要看哪个季度"。
    """
    result = await _run(
        _case(),
        ModelReview(question_answered=False, clarification_question="你要看的是哪个季度？"),
    )

    assert result.status == "CLARIFY"
    assert result.clarification_question == "你要看的是哪个季度？"
    assert result.reason_code == "NEED_CLARIFICATION"


async def test_the_clarification_question_is_dropped_when_we_fail_instead() -> None:
    """FAIL 时**不带**澄清问题。

    带着的话，被拦下的答案看起来像在跟用户对话（`final` 的 FAIL 分支渲染
    `blocking`，而那句问话会跟着进去），而它其实是一条不发布的结论。
    """
    # 两路都查过 → 没有可补的一路 → 确定性那条判定只能 FAIL（不是 RETRY）。
    # **只查一路时它是 RETRY**（还有可补的那一路），那是另一条用例的事。
    result = await _run(
        _case(cited=False, steps=("sql_query", "rag_retrieve")),
        ModelReview(clarification_question="你要看的是哪个季度？"),
    )

    assert result.status == "FAIL"
    assert result.clarification_question is None
    assert result.blocking, "否决来自确定性检查，不是模型"


# ------------------------------------------------------------------ 降级路径


async def test_a_model_failure_degrades_to_deterministic_only() -> None:
    """模型不可用 → **记一条 INFO 继续**，不把任务判死。

    审查工具自己坏掉，不等于被审的答案有问题——把它判成 FAIL 的话，
    一次模型侧的网络抖动会让一条完全正常的答案变成"未通过发布检查"，
    而那个结论是**关于答案的**，用户会照着它行事。

    但**也不能静默**：降级意味着"这一轮只跑了确定性六条"，
    而"只跑了六条"与"两阶段都跑了"必须能分开——所以记一条
    `MODEL_REVIEW_UNAVAILABLE`，而不是假装它跑过了（约定 35 的那类静默）。
    """
    state = _case()
    gateway = FakeModelGateway(responses=[], failure=FakeFailure.UNAVAILABLE)
    update = await build_reviewer_node(gateway)(state)

    result = update["review_result"]
    assert result.status == "PASS", "工具故障不该变成对答案的否决"
    assert "MODEL_REVIEW_UNAVAILABLE" in {item.code for item in result.issues}
    # 确定性那一半仍然生效
    assert result.evidence_score == 100


async def test_a_degraded_review_still_reports_the_deterministic_verdict() -> None:
    """降级不等于"什么都放行"：无引用的 FACT、且没得补，照旧 FAIL。"""
    state = _case(cited=False, steps=("sql_query", "rag_retrieve"))
    gateway = FakeModelGateway(responses=[], failure=FakeFailure.INVALID_OUTPUT)
    result = (await build_reviewer_node(gateway)(state))["review_result"]

    assert result.status == "FAIL"
    assert result.blocking[0].code == "CLAIM_WITHOUT_EVIDENCE"


async def test_a_refusal_is_not_turned_into_a_clarification() -> None:
    """**拒答的答案不允许被改成澄清**——实测踩到的那一条。

    `refused=True` 的含义是"证据里没有用户问的那件事"，答案就是"我们没有这条制度"。
    这时把答案换成一句问句是**答非所问**：用户已经问清楚了，
    是这个组织没有它。实测：问《直播带货专项补贴制度》时，第二阶段
    把它改判成了"你要看的是哪个季度？"——而那条用例的判据是"必须明确拒答"。

    与它成对的是下面那条：**补证据仍然允许**——那能救回真实的漏检
    （同义词导致的误拒，金标 rag-06），而它不改变"如实说不知道"这个结论的形状。
    """
    state = _case()
    state["analysis_result"] = _refused(state)

    result = await _run(
        state, ModelReview(question_answered=False, clarification_question="哪个季度？")
    )

    assert result.status == "PASS", "拒答不是失败，也不该被改写成一个问题"
    assert result.clarification_question is None


async def test_a_refusal_can_still_be_retried_for_evidence() -> None:
    """拒答**可以被要求补证据**：那能救回真实的漏检，而它不改结论的形状。

    与上一条的差别只在"模型给的是补证请求还是澄清问题"——前者动的是取证，
    后者动的是交付给用户的东西。
    """
    state = _case()
    state["analysis_result"] = _refused(state)

    result = await _run(
        state, ModelReview(evidence_sufficient=False, retry_target="rag", retry_reason="可能漏召")
    )

    assert result.status == "RETRY"
    assert result.retry_target == "rag"
