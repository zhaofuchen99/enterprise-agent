"""开发流程 6.10 的门禁：**缺陷答案阻断召回率 ≥ 90%**。

## 这条门禁在问什么

「喂给审查器一批**已知有缺陷**的答案，它拦下了几成」。
与 `test_reviewer.py` 的分工：那里一条条测"某条规则对不对"，
这里测**整体的拦截能力**——把十条缺陷一起过一遍，任何一条规则被改松
（或者被重构掉），召回率会掉下来。

它存在的理由是 14.1 那十条**每一条都是"漏了就没人知道"**：
一条 FACT 没有引用、一个点不开的引用、一处泄露的字段——
这些答案看起来与正常答案一模一样，而用户会拿它去做决定。

## 分母是「14.3 判为一票否决的那几条」，不是「14.1 的十条」

14.3 的判定表里真正否决的只有五类（`CLAIM_WITHOUT_EVIDENCE`、
`EVIDENCE_NOT_FOUND`、`BLOCKING_CONFLICT_NOT_DISCLOSED`、
`SENSITIVE_FIELD_LEAKED`、`REQUIRED_STEP_NOT_RUN`）。
其余几条按设计是 WARNING（`INFERENCE` 无引用、`OPEN_QUESTION_NOT_LISTED`），
它们**不该被算成"漏拦"**——把不该否决的算进分母，只会让这条门禁
为了凑数字去收紧规则，而那会把正常的答案一起拦下。

## ⚠️ 这条门禁**不覆盖**「答案里的数是错的」

`test_the_blind_spot_is_documented` 把这条边界写死在代码里：数字错
（把 8 行明细当整表合计，差 46%，而审查给了 100 分）**不在 14.1 的十条里**，
它由 `expect_numbers`（`make demo` / `make eval-agent` 的答案层断言）管。
不写清楚的话，"阻断率 100%"会被读成"答案都对"——而这正是 2026-09-20
那次实测打脸的地方。**那条用例断言的是"这种答案必须 PASS"**，
所以哪天有人把数字正确性塞进这六条里，它会红。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

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

#: 门禁线（开发流程 6.10）。
_BASELINE = 0.9


def _evidence() -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title="结果第 1 行",
        claim="net_sales=111967031.73",
        locator={},
        retrieved_at=datetime(2026, 9, 28, tzinfo=UTC),
        content_hash="a" * 64,
    )


#: 模块级建一次：`_clean()` 的引用必须指向**真实存在的**证据 id，
#: 否则每一条用例都会额外撞上 `EVIDENCE_NOT_FOUND`——
#: 那样测的是"引用写错了"，而不是被构造的那一处缺陷。
_EVIDENCE = _evidence()
_GOOD = _EVIDENCE.id


def _good_state(**overrides: object) -> AgentState:
    """一份**没有缺陷**的 State（步骤跑过、证据存在）。"""
    base: dict[str, object] = {
        "evidence": [_EVIDENCE],
        "steps": (_step(),),
        "results": {"step_01": _done()},
    }
    base.update(overrides)
    return _state(**base)  # type: ignore[arg-type]


def _step() -> TaskStep:
    return TaskStep(id="step_01", objective="查出净销售额", tool="sql_query", required=True)


def _state(
    *,
    evidence: list[Evidence] | None = None,
    steps: tuple[TaskStep, ...] = (),
    results: dict[str, StepResult] | None = None,
    conflicts: tuple[Conflict, ...] = (),
    open_questions: tuple[str, ...] = (),
) -> AgentState:
    state: AgentState = {"evidence": evidence if evidence is not None else []}
    if steps:
        state["task_list"] = list(steps)
    state["step_results"] = results or {}
    state["conflicts"] = list(conflicts)
    state["open_questions"] = list(open_questions)
    return state


def _done() -> StepResult:
    return StepResult(
        step_id="step_01", tool="sql_query", status=StepStatus.SUCCEEDED, summary="命中 1 条"
    )


def _blocking_conflict() -> Conflict:
    return Conflict(
        id=new_id(IdPrefix.CONFLICT),
        type=ConflictType.VALUE,
        severity=ConflictSeverity.BLOCKING,
        description="报告表格 11,039.58 万元与库 111,967,031.73 元相差 1.42%",
        resolution=ConflictResolution.UNRESOLVED,
        # 冲突天然是**两个来源**之间的，所以至少两条证据（模型里有 min_length=2）
        evidence_ids=(new_id(IdPrefix.EVIDENCE), new_id(IdPrefix.EVIDENCE)),
    )


def _clean(**overrides: object) -> AnalysisResult:
    """一份**没有缺陷**的答案，各类缺陷在它上面改一处得到。"""
    base: dict[str, object] = {
        "refused": False,
        "direct_answer": "2025年Q3华东净销售额为 111,967,031.73 元。",
        "claims": (SupportedClaim(text="华东净销售额为 111,967,031.73 元", evidence_ids=(_GOOD,)),),
        "limitations": (),
    }
    base.update(overrides)
    return AnalysisResult.model_validate(base)


#: 缺陷清单：`(名字, 答案, State, 该不该被拦下)`。
#:
#: **每一条都写成"混在正常答案里"的样子**（除了该处缺陷，其余部分都是好的）——
#: 这是它在真实产物里的形态，也是唯一能测出"逐条检查有没有漏过某一条"的形态。
#: 全部写成"整条答案都有问题"的话，一个只看第一条 claim 的实现也能全绿。
#:
#: ⚠️ 敏感字段那两条用的是**一眼假**的占位值：判据（`reviewer._SENSITIVE`）匹配的
#: 是标识符本身（`password_hash` / `api_key` 是裸词），等号后面是什么不进判定。
#: 原先填的是真实形态的假密钥（MD5 串 / `sk-live-…`），pre-commit 的 gitleaks 会
#: 据此报 `generic-api-key`——那是个**误报**，但误报会训练人绕开密钥门禁，
#: 而那道门禁是唯一挡得住真密钥入库的东西，代价比误报本身贵。
_DEFECTS: tuple[tuple[str, AnalysisResult, AgentState, bool], ...] = (
    (
        "FACT 结论没有引用",
        _clean(claims=(SupportedClaim(text="华东净销售额为 111,967,031.73 元"),)),
        _good_state(),
        True,
    ),
    (
        "同一条答案里混着一条无引用的 FACT",
        _clean(
            claims=(
                SupportedClaim(text="华东净销售额为 111,967,031.73 元", evidence_ids=(_GOOD,)),
                SupportedClaim(text="同比下降 12%"),
            )
        ),
        _good_state(),
        True,
    ),
    (
        "引用指向不存在的证据",
        _clean(
            claims=(SupportedClaim(text="华东净销售额为 111,967,031.73 元", evidence_ids=("E9",)),)
        ),
        _good_state(),
        True,
    ),
    (
        "多条引用里有一条点不开",
        _clean(
            claims=(
                SupportedClaim(
                    text="华东净销售额为 111,967,031.73 元", evidence_ids=(_GOOD, "evd_不存在")
                ),
            )
        ),
        _good_state(),
        True,
    ),
    (
        "敏感字段泄进正文",
        _clean(direct_answer="连接串是 password_hash=<占位值，非真实密钥>"),
        _good_state(),
        True,
    ),
    (
        "敏感字段泄进结论",
        _clean(
            claims=(
                SupportedClaim(
                    text="数据库口令为 api_key=<占位值，非真实密钥>",
                    evidence_ids=(_GOOD,),
                ),
            )
        ),
        _good_state(),
        True,
    ),
    (
        "BLOCKING 冲突没有披露",
        _clean(),
        _good_state(conflicts=(_blocking_conflict(),)),
        True,
    ),
    (
        "必需的步骤没有跑过",
        _clean(),
        _good_state(results={}),
        True,
    ),
    (
        "没跑的那一步只是可选的",
        _clean(),
        _good_state(
            # `required` 的默认值是 True，所以这里必须显式写 False——
            # 14.1 第 1 条管的只有必需步骤，可选步骤没跑不该拦
            steps=(
                TaskStep(id="step_01", objective="查出净销售额", tool="sql_query", required=False),
            ),
            results={},
        ),
        False,
    ),
    (
        "INFERENCE 没有引用（按设计只记 WARNING，不否决）",
        _clean(claims=(SupportedClaim(text="下滑可能与渠道结构变化有关", kind="INFERENCE"),)),
        _good_state(),
        False,
    ),
    (
        "未解决的问题没有列进限制（按设计只记 WARNING）",
        _clean(),
        _good_state(open_questions=("缺 2026 年的数据",)),
        False,
    ),
)


def test_no_defect_slips_through() -> None:
    """**该拦下的一条都不能漏**——召回率必须 ≥ 90%。

    分母是"该拦下的"那几条（见文件头：只有 14.3 判为一票否决的才算），
    不该拦的几条另算（`test_benign_shapes_are_not_blocked`）：
    把两者混在一个比率里，会让"把规则收紧到误伤正常答案"看起来像是进步。
    """
    must_block = [item for item in _DEFECTS if item[3]]
    blocked = [name for name, analysis, state, _ in must_block if review(analysis, state).blocking]

    recall = len(blocked) / len(must_block)
    assert recall >= _BASELINE, (
        f"缺陷答案阻断召回率 {recall:.0%} < {_BASELINE:.0%}——"
        f"漏掉的：{sorted({item[0] for item in must_block} - set(blocked))}"
    )


def test_benign_shapes_are_not_blocked() -> None:
    """不该被拦的**必须放行**：误伤与漏拦同样是缺陷。

    而误伤更贵——一条正常的答案被拦下，用户看到的是"审查未通过"，
    它会去改一个根本没坏的东西。
    """
    benign = [item for item in _DEFECTS if not item[3]]

    wrongly = [name for name, analysis, state, _ in benign if review(analysis, state).blocking]
    assert not wrongly, f"这些本该放行却被拦下：{wrongly}"


def test_the_clean_answer_passes() -> None:
    """基线：没有任何缺陷时必须 PASS 且零 issue。

    没有这一条的话，一个"永远 FAIL"的实现也能让上面两条全绿。
    """
    result = review(
        _clean(),
        _good_state(),
    )

    assert result.status == "PASS"
    assert result.issues == ()
    assert result.score == 100


def test_the_blind_spot_is_documented() -> None:
    """**这条门禁管不到"答案里的数是错的"**，把它写死在代码里。

    2026-09-20 实测：工具选对了、SQL 也对了、审查给了 100 分，而答案里的数
    差 46%（把 80 行明细表里的 8 行当成区域合计）。数字正确性**不在 14.1 的
    十条里**——本节的门禁对它一无所知，它由答案层的 `expect_numbers` 管
    （`make demo` / `make eval-agent`）。

    写这条用例不是为了让它通过，是为了让"阻断率 100%"**不被读成"答案都对"**：
    下面这段构造的正是那次实测的形态——**一个有引用、有证据、
    但数算错了**的答案，而它在这里**必须 PASS**。
    """
    wrong_sum = _clean(
        direct_answer="华南地区季度合计为 7,146.22 万元。",
        claims=(
            SupportedClaim(
                text="按表格明细相加，华南地区季度合计为 7,146.22 万元",
                evidence_ids=(_GOOD,),
            ),
        ),
        # 它甚至如实声明了"表格未取全"——而这仍然拦不住一个错的和
        limitations=("表格证据未取全，本次只覆盖其中 8 行",),
    )
    result = review(
        wrong_sum,
        _good_state(),
    )

    assert result.status == "PASS", (
        "这条如果变成 FAIL，说明有人把'数字正确性'塞进了 14.1 的六条里——"
        "而它不是靠引用存不存在判得出来的（见本条 docstring）"
    )


@pytest.mark.parametrize(("name", "analysis", "state", "should_block"), _DEFECTS)
def test_each_defect_is_reported_with_a_code(
    name: str, analysis: AnalysisResult, state: AgentState, should_block: bool
) -> None:
    """每一条都有**具名的代码**，不只是"没通过"。

    报告里"审查未通过"而说不出是哪一条，等于把排查推给下一个人——
    而 `retry_router` 与 `reason_code` 都按 code 分流，编不出来就是走不到。
    """
    result = review(analysis, state)

    if should_block:
        assert result.blocking, name
        assert all(item.code for item in result.blocking), name
        assert result.status == "FAIL"
    else:
        assert not result.blocking, f"{name} 不该被一票否决"
