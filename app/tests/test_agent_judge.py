"""Agent 判据本身（`scripts/agent_harness.py` 的 `evaluate`）。

**判据值得单独测**，理由与 `test_layering.py` 相同：它是演示与评测时唯一的
"这件事对不对"的依据，而它失效时的表现是**一条错误的断言一直通过**
（或者一条正确的行为一直判红）。两者都不会有人发现。

**它同时被两个入口用**（`make demo` 与 `make eval-agent`）——判据只有一份，
所以这里测好了，两边都可信；反过来，这里漏了一个分支，两边会一起错。

这里测的是判定逻辑，不是端到端——跑真链路的那部分由 `make demo` /
`make eval-agent` 覆盖。
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.agent_harness import amounts, case_turns, evaluate, missing_amounts


def _detail(
    *,
    answer: str = "华东净销售额为 111,967,031.73 元。",
    limitations: list[str] | None = None,
    refused: bool | None = False,
    conflicts: list[str] | None = None,
    tools: tuple[str, ...] = ("sql_query",),
    status: str = "SUCCEEDED",
    review: dict[str, Any] | None = None,
    progress_decision: str | None = None,
    resolved_entities: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "final_answer_md": answer,
        "limitations": limitations if limitations is not None else [],
        "refused": refused,
        "conflicts": conflicts if conflicts is not None else [],
        "steps": [{"tool": tool, "status": "SUCCEEDED"} for tool in tools],
        "review": review,
        "progress_decision": progress_decision,
        "resolved_entities": resolved_entities if resolved_entities is not None else {},
    }


# ------------------------------------------------------------ 数值还原


def test_amounts_restore_chinese_units() -> None:
    """「万元」「亿元」要还原成基准值。

    不还原的话，「11,196.70 万元」与 111,967,031.73 会被当成两个不相关的数，
    一条**正确**的答案会被判红——而误报比漏报更糟：它让人开始忽略断言。
    """
    assert amounts("净销售额 11,196.70 万元") == [pytest.approx(111_967_000)]


def test_missing_amounts_accepts_the_same_value_written_differently() -> None:
    """同一个数写成元、万元、亿元都算命中——**差的是写法不是数**。"""
    assert missing_amounts("为 111,967,031.73 元", ["111967031.73"]) == []
    assert missing_amounts("为 11,196.70 万元", ["111967031.73"]) == []
    assert missing_amounts("为 1.12 亿元", ["111967031.73"]) == []


def test_missing_amounts_catches_a_wrong_number() -> None:
    """**这是这道门禁存在的全部理由**：答案里的数错了要判红。

    那个数取的是实测踩到的那次——80 行明细只召回 8 行，模型把 8 行
    相加当成区域合计（7,146.22 万），真值 13,249.31 万，差 46%。
    """
    assert missing_amounts("华南合计 7,146.22 万元", ["132493082.21"]) == ["132493082.21"]


def test_missing_amounts_refuses_a_non_numeric_expectation() -> None:
    """配置写错时**抛**而不是跳过。

    静默跳过会让一条永远不会生效的断言看起来一直在通过——
    那正是"门禁失效"最隐蔽的形态。
    """
    with pytest.raises(ValueError, match="expect_numbers"):
        missing_amounts("答案", ["一亿"])


# ------------------------------------------------------------ 拒答与故障


def test_rejected_case_passes_on_the_structured_flag() -> None:
    """拒答判据读字段，不认答案里的词（见 `AnalysisResult.refused`）。

    这条答案**故意不含**原先那几个 marker 词（"没有检索到"/"没有找到"…），
    而它仍然该判过——判的是行为，不是措辞。
    """
    case = {"id": "demo-absent", "expect_rejected": True}
    detail = _detail(answer="证据中没有该办法的条文，现有材料只覆盖另一份规范。", refused=True)

    verdict = evaluate(case, detail)

    assert verdict.ok, verdict.note
    assert "refused=true" in verdict.note


def test_rejected_case_fails_when_the_flag_is_false() -> None:
    case = {"id": "demo-absent", "expect_rejected": True}
    detail = _detail(answer="根据该办法，海外渠道应当…", refused=False)

    verdict = evaluate(case, detail)

    assert not verdict.ok
    assert "refused=false" in verdict.note


def test_degraded_analysis_is_not_reported_as_fabrication() -> None:
    """分析生成失败时要说"未能判定"，**不能说成"没有拒答"**。

    一次故障被报成一次编造，是"失败指错了层"——排查方向会跑到
    提示词或模型上去，而真正的原因是一次调用失败。
    这里认的是 `_degraded` 自己写下的那句，是我们自己的字符串。
    """
    case = {"id": "demo-absent", "expect_rejected": True}
    detail = _detail(
        answer="已取得以下证据，但综合分析的生成失败，未能给出结论。（UPSTREAM_UNAVAILABLE）",
        limitations=["分析生成失败：上游不可用"],
        refused=False,
    )

    verdict = evaluate(case, detail)

    assert not verdict.ok
    assert "未能判定" in verdict.note
    assert verdict.unjudged, "故障要能被上层的通过率统计单独排除"


def test_task_failure_is_marked_unjudged() -> None:
    """任务没跑成时，**这条用例不该计入"行为不符"**。

    与上一条同因：`MODEL_OUTPUT_INVALID` 会让 `steps` 为空，于是
    "期望 rag 实际 []"看起来像一次工具选择错误，而它其实是一次模型抖动。
    评测报告里把两者混进同一个分母，退化趋势就看不出来了。
    """
    case = {"id": "agent-x", "expect_sources": ["sql_query"]}

    verdict = evaluate(case, _detail(status="FAILED"))

    assert not verdict.ok
    assert verdict.unjudged
    assert "任务未成功" in verdict.note


# ------------------------------------------------------------ 数值与限制


def test_expected_number_must_appear_in_the_answer() -> None:
    """**答案层的数值门禁**：工具选对了、冲突也没有，数字照样可能是错的。"""
    case = {"id": "demo-sql", "expect_sources": ["sql_query"], "expect_numbers": ["111967031.73"]}

    assert evaluate(case, _detail()).ok
    assert evaluate(case, _detail(answer="华东净销售额为 11,196.70 万元。")).ok, (
        "写成万元是同一个数，不该判红"
    )
    assert evaluate(case, _detail(answer="华东净销售额为 111,967,031.73 元。")).ok
    assert missing_amounts("没有数字的答案", ["111967031.73"]) == ["111967031.73"]
    assert not evaluate(case, _detail(answer="华东净销售额约为 1.2 亿元。")).ok, (
        "1.2 亿与 1.1196 亿差 7%，不该放过"
    )


def test_expected_limitation_must_be_present() -> None:
    """钉住代码生成的限制：断了那条链路，这里立刻变红。"""
    case = {
        "id": "demo-expand",
        "expect_sources": ["sql_query", "rag_retrieve"],
        "expect_limitations_contain": ["共 80 行，本次证据只覆盖其中"],
    }
    detail = _detail(
        tools=("sql_query", "rag_retrieve"),
        limitations=[
            "表格「分区域分渠道分产品线净销售额明细」共 80 行，本次证据只覆盖其中 3 行：…"
        ],
    )

    assert evaluate(case, detail).ok

    verdict = evaluate(case, _detail(tools=("sql_query", "rag_retrieve")))
    assert not verdict.ok
    assert "限制清单里缺少" in verdict.note


def test_tool_selection_mismatch_is_reported_first() -> None:
    """工具就没选对时，报的是它——**一条只给一个结论**。

    否则报告会同时列出"选了错的路""数字不对""限制不全"，
    读的人分不出主因，而主因往往是第一个。
    """
    case = {"id": "demo-sql", "expect_sources": ["sql_query"], "expect_numbers": ["111967031.73"]}

    verdict = evaluate(case, _detail(tools=("rag_retrieve",), answer="没有数字"))

    assert not verdict.ok
    assert "工具选择不符" in verdict.note


def test_rejected_case_still_checks_the_review() -> None:
    """拒答与澄清会**提前返回**，而审查与它们是正交的。

    不特殊处理的话，写在 `expect_rejected` 后面的 `expect_review_*`
    永远不被检查——**写了却从不生效的断言比没有断言更糟**，
    因为它看起来是有保障的。实测就是这么踩到的。
    """
    case = {"id": "agent-clarify-03", "expect_rejected": True, "expect_review_status": "PASS"}
    passing = {"status": "PASS", "reason_code": "GROUNDED", "score": 100, "issues": []}

    assert evaluate(case, _detail(refused=True, review=passing)).ok

    verdict = evaluate(case, _detail(refused=True, review={**passing, "status": "FAIL"}))
    assert not verdict.ok, "拒答的任务被审查拦下，必须报出来"
    assert "审查结论不符" in verdict.note


def test_clarification_case_still_checks_the_review() -> None:
    """澄清同理——虽然那时还没走到 reviewer（`review` 是 None），
    但一旦哪天真走到并给出了结论，断言要能生效而不是被静静跳过。"""
    case = {"id": "agent-clarify-01", "expect_clarification": True, "expect_review_status": "PASS"}

    verdict = evaluate(case, _detail(tools=(), review=None))

    assert not verdict.ok
    assert "没有审查结论" in verdict.note


# ------------------------------------------------------------ 冲突


def test_conflict_count_is_exact() -> None:
    """条数是**精确**判据，不是下限。

    它是"不该报的没报"这条断言的载体：冲突检测的漏报看不出来
    （答案照常给出），误报看得出来——所以"零条"值得单独立一条用例，
    而"至少一条"那种写法对零条与五条一视同仁，挡不住误报。
    """
    case = {"id": "agent-conflict-01", "expect_conflict_count": 0}

    assert evaluate(case, _detail(conflicts=[])).ok

    verdict = evaluate(case, _detail(conflicts=["《某报告》的net_sales为 1.00，而库查得 2.00"]))
    assert not verdict.ok
    assert "冲突条数不符" in verdict.note


def test_expected_conflict_text_must_appear() -> None:
    """钉住**报了哪一条**——只看"有没有冲突"分不出报对没报对。"""
    case = {"id": "agent-conflict-02", "expect_conflict_contain": ["相对 1.40%"]}
    conflicts = [
        "《华东区域2025年第三季度专项分析 > （二）区域分布》的net_sales为 110,395,800.00，"
        "而业务数据库查得 111,967,031.73（差 1,571,231.73，相对 1.40%），范围 region=华东"
    ]

    assert evaluate(case, _detail(conflicts=conflicts)).ok

    verdict = evaluate(case, _detail(conflicts=["另一条不相干的冲突"]))
    assert not verdict.ok
    assert "冲突描述里缺少" in verdict.note


# ------------------------------------------------------------ 循环与审查


def test_expected_progress_decision_is_checked() -> None:
    """`reflect` 的演进判定要能被断言（开发流程 7.5 的"有效性"一条）。"""
    case = {"id": "agent-loop-01", "expect_progress_decision": "EXPAND"}

    assert evaluate(case, _detail(progress_decision="EXPAND")).ok

    verdict = evaluate(case, _detail(progress_decision="SUFFICIENT"))
    assert not verdict.ok
    assert "演进判定不符" in verdict.note


def test_absent_expect_sources_means_do_not_check() -> None:
    """**缺省 ≠ 空列表**：缺省是"不关心走了哪几路"，`[]` 才是"不该调任何工具"。

    合成一个的话，一条只断审查结论的用例会被判成"工具选择不符"——
    而它压根没打算断言工具。误报比漏报更糟：一条永远红着的用例
    会让整份评测报告失去可信度，然后被忽略。
    """
    assert evaluate({"id": "agent-x"}, _detail(tools=("sql_query", "rag_retrieve"))).ok
    assert evaluate({"id": "agent-x"}, _detail(tools=())).ok

    assert not evaluate({"id": "agent-x", "expect_sources": []}, _detail(tools=("sql_query",))).ok
    assert evaluate({"id": "agent-x", "expect_sources": []}, _detail(tools=())).ok


def test_review_is_skipped_when_not_asked_for() -> None:
    """没写 `expect_review_*` 的用例**不该**因为 `review` 是 None 而判红。

    `review` 为 None 是合法状态（任务没走到审查那一步），
    把"没断言"当成"断言失败"会让一批用例无缘无故红掉。
    """
    case = {"id": "agent-x", "expect_sources": ["sql_query"]}

    assert evaluate(case, _detail(review=None)).ok


def test_expected_review_status_is_checked() -> None:
    """Reviewer 的结论形态（14.3 的一票否决是否生效）。

    它失效时的症状是"所有任务都通过"——与"所有任务都没问题"完全同形。
    """
    case = {"id": "agent-review-01", "expect_review_status": "PASS"}
    review = {"status": "PASS", "reason_code": "GROUNDED", "score": 100, "issues": []}

    assert evaluate(case, _detail(review=review)).ok

    verdict = evaluate(case, _detail(review={**review, "status": "FAIL"}))
    assert not verdict.ok
    assert "审查结论不符" in verdict.note


def test_missing_review_is_reported_as_such() -> None:
    """任务跑成了、却**没有审查结论**——这是真该红的，与"没断言"不同。"""
    case = {"id": "agent-review-01", "expect_review_status": "PASS"}

    verdict = evaluate(case, _detail(review=None))

    assert not verdict.ok
    assert "没有审查结论" in verdict.note


def test_review_issue_codes_are_compared_as_a_set() -> None:
    """issue 清单按**集合相等**比，不是包含。

    包含关系下"多记了一条 issue"永远发现不了，而那正是这一层最该被发现的
    事：Reviewer 误报会让一条好答案被扣分、被拦下，症状却是"分数低了点"。
    """
    clean = {"status": "PASS", "reason_code": "GROUNDED", "score": 100, "issues": []}
    hypothesis = {
        "status": "PASS",
        "reason_code": "GROUNDED",
        "score": 95,
        "issues": [{"code": "UNVERIFIED_HYPOTHESIS", "severity": "INFO"}],
    }

    assert evaluate({"id": "x", "expect_review_issue_codes": []}, _detail(review=clean)).ok
    assert evaluate(
        {"id": "x", "expect_review_issue_codes": ["UNVERIFIED_HYPOTHESIS"]},
        _detail(review=hypothesis),
    ).ok

    # 缺省 = 不检查：这两条都不该因为 issue 清单不同而判红
    assert evaluate({"id": "x"}, _detail(review=clean)).ok
    assert evaluate({"id": "x"}, _detail(review=hypothesis)).ok

    verdict = evaluate({"id": "x", "expect_review_issue_codes": []}, _detail(review=hypothesis))
    assert not verdict.ok
    assert "审查记录的问题不符" in verdict.note


def test_expected_review_reason_code_is_checked() -> None:
    """六条检查没有逐条的"跑过"痕迹，能钉的只有结论的归因。"""
    case = {"id": "agent-review-02", "expect_review_reason_code": "GROUNDED"}

    assert evaluate(case, _detail(review={"status": "PASS", "reason_code": "GROUNDED"})).ok

    verdict = evaluate(case, _detail(review={"status": "PASS", "reason_code": "NO_ANALYSIS"}))
    assert not verdict.ok
    assert "审查归因不符" in verdict.note


# ------------------------------------------------------- 多轮上下文（FR-CHAT-003）
def test_case_turns_expands_and_merges() -> None:
    """多轮用例展开成**结构相同的子用例**。

    展开而不是另立一套多轮判据，是为了让 `evaluate` 原样复用——
    多一份判据就多一处会与演示各说各话的地方（约定 87）。
    """
    case = {
        "id": "demo-x",
        "stability": "model-dependent",
        "turns": [
            {"question": "第一问"},
            {"question": "那Q2呢", "expect_resolved_contain": {"period": "2025-Q2"}},
        ],
    }

    turns = case_turns(case)

    assert [turn["question"] for turn in turns] == ["第一问", "那Q2呢"]
    # 顶层字段被带下去（否则每轮都要重抄一遍 stability / proves）
    assert all(turn["stability"] == "model-dependent" for turn in turns)
    # `turns` 本身不再出现——留着会让展开变成递归
    assert all("turns" not in turn for turn in turns)


def test_case_turns_of_a_single_turn_case_is_itself() -> None:
    case = {"id": "demo-y", "question": "普通问题"}
    assert case_turns(case) == [case]


def test_the_resolved_entities_must_match() -> None:
    """`expect_resolved_contain` 逐键比对。

    **这条断言必须真的会红**：多轮追问最典型的失败样式是
    "代词没被解析、于是判成缺前提去澄清"，而那时用例如果只钉工具选择，
    报告会显示"进入澄清 ✓"——一个看起来通过了的失败。
    """
    case = {"id": "x", "expect_resolved_contain": {"region": "华东", "period": "2025-Q2"}}
    good = _detail(resolved_entities={"region": "华东", "period": "2025-Q2", "metric": "净销售额"})

    assert evaluate(case, good).ok

    # 期间没换过来（上一轮是 Q3）——这正是"继承错了"的样子
    stale = _detail(resolved_entities={"region": "华东", "period": "2025-Q3"})
    verdict = evaluate(case, stale)
    assert not verdict.ok
    assert "上一轮口径没有被继承" in verdict.note


def test_the_resolved_check_runs_even_when_the_case_clarifies() -> None:
    """**它排在模式分支之前**，所以澄清 / 拒答的轮次照样被检查。

    写在后面的话，这条断言只在"没拒答也没澄清"时生效——而代词没解析出来
    最典型的结果**恰恰是去澄清**。那时用例会报"进入澄清 ✓"，
    而真正的缺陷（记忆没生效）一声不响。
    """
    case = {
        "id": "x",
        "expect_clarification": True,
        "expect_resolved_contain": {"period": "2025-Q2"},
    }

    verdict = evaluate(case, _detail(tools=(), resolved_entities={}))

    assert not verdict.ok
    assert "上一轮口径没有被继承" in verdict.note
