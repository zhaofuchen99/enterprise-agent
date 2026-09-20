"""演示判据本身（`scripts/demo.py` 的 `_evaluate`）。

**判据值得单独测**，理由与 `test_layering.py` 相同：它是演示时唯一的
"这件事对不对"的依据，而它失效时的表现是**一条错误的断言一直通过**
（或者一条正确的行为一直判红）。两者都不会有人发现。

这里测的是判定逻辑，不是端到端——跑真链路的那部分由 `make demo` 覆盖。
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.demo import _amounts, _evaluate, _missing_amounts


def _detail(
    *,
    answer: str = "华东净销售额为 111,967,031.73 元。",
    limitations: list[str] | None = None,
    refused: bool | None = False,
    conflicts: list[str] | None = None,
    tools: tuple[str, ...] = ("sql_query",),
    status: str = "SUCCEEDED",
) -> dict[str, Any]:
    return {
        "status": status,
        "final_answer_md": answer,
        "limitations": limitations if limitations is not None else [],
        "refused": refused,
        "conflicts": conflicts if conflicts is not None else [],
        "steps": [{"tool": tool, "status": "SUCCEEDED"} for tool in tools],
    }


# ------------------------------------------------------------ 数值还原


def test_amounts_restore_chinese_units() -> None:
    """「万元」「亿元」要还原成基准值。

    不还原的话，「11,196.70 万元」与 111,967,031.73 会被当成两个不相关的数，
    一条**正确**的答案会被判红——而误报比漏报更糟：它让人开始忽略断言。
    """
    assert _amounts("净销售额 11,196.70 万元") == [pytest.approx(111_967_000)]


def test_missing_amounts_accepts_the_same_value_written_differently() -> None:
    """同一个数写成元、万元、亿元都算命中——**差的是写法不是数**。"""
    assert _missing_amounts("为 111,967,031.73 元", ["111967031.73"]) == []
    assert _missing_amounts("为 11,196.70 万元", ["111967031.73"]) == []
    assert _missing_amounts("为 1.12 亿元", ["111967031.73"]) == []


def test_missing_amounts_catches_a_wrong_number() -> None:
    """**这是这道门禁存在的全部理由**：答案里的数错了要判红。

    那个数取的是实测踩到的那次——80 行明细只召回 8 行，模型把 8 行
    相加当成区域合计（7,146.22 万），真值 13,249.31 万，差 46%。
    """
    assert _missing_amounts("华南合计 7,146.22 万元", ["132493082.21"]) == ["132493082.21"]


def test_missing_amounts_refuses_a_non_numeric_expectation() -> None:
    """配置写错时**抛**而不是跳过。

    静默跳过会让一条永远不会生效的断言看起来一直在通过——
    那正是"门禁失效"最隐蔽的形态。
    """
    with pytest.raises(ValueError, match="expect_numbers"):
        _missing_amounts("答案", ["一亿"])


# ------------------------------------------------------------ 判据


def test_rejected_case_passes_on_the_structured_flag() -> None:
    """拒答判据读字段，不认答案里的词（见 `AnalysisResult.refused`）。

    这条答案**故意不含**原先那几个 marker 词（"没有检索到"/"没有找到"…），
    而它仍然该判过——判的是行为，不是措辞。
    """
    case = {"id": "demo-absent", "expect_rejected": True}
    detail = _detail(answer="证据中没有该办法的条文，现有材料只覆盖另一份规范。", refused=True)

    passed, reason = _evaluate(case, detail)

    assert passed, reason
    assert "refused=true" in reason


def test_rejected_case_fails_when_the_flag_is_false() -> None:
    case = {"id": "demo-absent", "expect_rejected": True}
    detail = _detail(answer="根据该办法，海外渠道应当…", refused=False)

    passed, reason = _evaluate(case, detail)

    assert not passed
    assert "refused=false" in reason


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

    passed, reason = _evaluate(case, detail)

    assert not passed
    assert "未能判定" in reason


def test_expected_number_must_appear_in_the_answer() -> None:
    """**答案层的数值门禁**：工具选对了、冲突也没有，数字照样可能是错的。"""
    case = {"id": "demo-sql", "expect_sources": ["sql_query"], "expect_numbers": ["111967031.73"]}

    assert _evaluate(case, _detail())[0]
    passed, reason = _evaluate(case, _detail(answer="华东净销售额为 11,196.70 万元。"))
    assert passed, "写成万元是同一个数，不该判红"

    passed, reason = _evaluate(case, _detail(answer="华东净销售额为 111,967,031.73 元。"))
    assert passed, reason
    assert _missing_amounts("没有数字的答案", ["111967031.73"]) == ["111967031.73"]
    assert not _evaluate(case, _detail(answer="华东净销售额约为 1.2 亿元。"))[0], (
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

    assert _evaluate(case, detail)[0]

    passed, reason = _evaluate(case, _detail(tools=("sql_query", "rag_retrieve")))
    assert not passed
    assert "限制清单里缺少" in reason


def test_tool_selection_mismatch_is_reported_first() -> None:
    """工具就没选对时，报的是它——**一条只给一个结论**。

    否则报告会同时列出"选了错的路""数字不对""限制不全"，
    读的人分不出主因，而主因往往是第一个。
    """
    case = {"id": "demo-sql", "expect_sources": ["sql_query"], "expect_numbers": ["111967031.73"]}

    passed, reason = _evaluate(case, _detail(tools=("rag_retrieve",), answer="没有数字"))

    assert not passed
    assert "工具选择不符" in reason
