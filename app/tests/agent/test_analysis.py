"""分析节点里**由代码判定**的那几件事（详设 6.1 / 13.5）。

模型写的那部分（结论、措辞）不在这里测——它每次都不一样，断言只能写成
"包含某个词"，那种用例在被改坏时照样通过。这里测的是模型**看不到也推不出**
的事实：哪一步空了、证据只覆盖了表的一部分、证据来自未经权限过滤的来源。
这些必须由代码补进 `limitations`，否则它们在最终答案里根本不会出现。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from app.agent.nodes.analysis import _incomplete_tables, _pipeline_limitations, _render, _table_note
from app.agent.schemas.plan import StepResult, StepStatus
from app.agent.state import AgentState
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence


def _doc_evidence(
    *,
    caption: str = "分区域分渠道分产品线净销售额明细",
    table_row: tuple[int, int] | None = (1, 4),
    document_id: str = "report/2025q3@v1.0",
    section: list[str] | None = None,
) -> Evidence:
    locator: dict[str, object] = {
        "chunk_id": new_id(IdPrefix.EVIDENCE),
        "document_id": document_id,
        "section_path": section or ["2025年第三季度经营分析", "五、风险提示"],
        "is_table": table_row is not None,
        "table_caption": caption,
        "table_row": list(table_row) if table_row else None,
    }
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="DOCUMENT",
        title=f"2025年第三季度经营分析 > {caption}",
        claim="华南 | 直营 | 智能家居 | 934.21 | 15.05",
        locator=locator,
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        content_hash="a" * 64,
    )


def _state(evidence: list[Evidence], **extra: Any) -> AgentState:
    """构造一份**只带被测字段**的 State。

    `AgentState` 是 `total=False` 的 TypedDict，部分字段正是它的正常形态——
    这几个函数只读自己关心的那几个键。用 `cast` 而不是补齐所有字段：
    补出来的字段会让人以为它们参与了判定。
    """
    return cast("AgentState", {"evidence": evidence, "step_results": {}, **extra})


# ---------------------------------------------------- 表格证据覆盖不完整


def test_table_note_names_the_position_for_table_rows() -> None:
    """位置写在**证据行上**，而不是靠 prompt 里一句"注意证据可能是残缺的"。

    通则要求模型自己想起这一条可能不完整；位置是这条证据自带的属性，
    没有推理余地。
    """
    assert _table_note(_doc_evidence(table_row=(7, 16))) == "｜该表第 7 行，共 16 行"


def test_table_note_is_empty_for_non_table_evidence() -> None:
    """非表格证据不要冒出一句"第 0 行"——那会让模型以为它也是表格的一部分。"""
    assert _table_note(_doc_evidence(table_row=None)) == ""


def test_incomplete_table_is_reported_with_both_numbers() -> None:
    """**只引用了一部分行**必须被写出来，且两个数都要有。

    这是实测踩到的那个错误答案的补丁：模型拿 8 行求和当区域合计
    （7,146.22 万，真值 13,249.31 万），而 Reviewer 六条检查一条都看不出来。
    只写"部分"不给数，读者没法判断差多少；只给总数不给引用数，他不知道缺的是哪一半。
    """
    state = _state([_doc_evidence(table_row=(1, 16)), _doc_evidence(table_row=(2, 16))])

    limits = _incomplete_tables(state)

    assert len(limits) == 1
    assert "共 16 行" in limits[0]
    assert "只覆盖其中 2 行" in limits[0]
    assert "不得用这些行求和" in limits[0]


def test_complete_table_is_not_reported() -> None:
    """整张表都在证据里时不写这条——否则它每跑一次都出现，真的缺口就没人看了。"""
    state = _state([_doc_evidence(table_row=(index, 2)) for index in (1, 2)])

    assert _incomplete_tables(state) == []


def test_duplicate_rows_are_counted_once() -> None:
    """同一行出现两次不算两行。

    按证据条数计会把"同一条被两个查询召回"读成"多覆盖了一行"，
    于是残缺的表被判成完整的——**这个错误的方向恰好是漏报**。
    """
    state = _state([_doc_evidence(table_row=(1, 4)), _doc_evidence(table_row=(1, 4))])

    assert len(_incomplete_tables(state)) == 1


def test_same_caption_in_different_documents_stays_separate() -> None:
    """两张同名的表**分别说**：它们本来就不是一张表，合并会把总数算错。"""
    state = _state(
        [
            _doc_evidence(table_row=(1, 4), document_id="report/a@v1.0"),
            _doc_evidence(table_row=(1, 9), document_id="report/b@v1.0"),
        ]
    )

    limits = _incomplete_tables(state)

    assert len(limits) == 2
    assert any("共 4 行" in item for item in limits)
    assert any("共 9 行" in item for item in limits)


def test_same_caption_different_documents_are_tellable_apart() -> None:
    """分开说还不够，**得让人看得出为什么有两条**。

    实测踩到（2026-09-20）：两张来自不同经营月报的同名表渲染成了两句话
    一模一样的限制，读者只会以为系统重复输出了一遍——而真相是
    "有两张同名的表，都只覆盖了一部分"。文件名要进这一句。
    """
    state = _state(
        [
            _doc_evidence(
                table_row=(1, 4),
                document_id="report/a@v1.0",
                section=["2025年1月经营月报", "二、经营业绩回顾"],
            ),
            _doc_evidence(
                table_row=(1, 9),
                document_id="report/b@v1.0",
                section=["2025年2月经营月报", "二、经营业绩回顾"],
            ),
        ]
    )

    limits = _incomplete_tables(state)

    assert "2025年1月经营月报" in limits[0] and "2025年2月经营月报" in limits[1]
    assert limits[0] != limits[1]


def test_long_table_list_is_truncated_but_says_so() -> None:
    """限制清单是给人读的，列太多会把要看的那条淹掉——但**截断必须说出来**。

    静默截断会让读者以为列出来的就是全部，而那正是这条限制要防的错觉。
    """
    state = _state(
        [_doc_evidence(table_row=(1, 5), document_id=f"report/{index}@v1.0") for index in range(5)]
    )

    limits = _incomplete_tables(state)

    assert len(limits) == 4
    assert "另有 2 张表" in limits[-1]


# ---------------------------------------------------- 与其它限制并列


def test_pipeline_limitations_include_the_table_gap() -> None:
    """它必须与"哪一步空了"并列出现在同一份限制里。

    散在两个出口是这类事实最常见的失效方式：一处写了、另一处没写，
    而读的人只看到其中一处。
    """
    state = _state(
        [_doc_evidence(table_row=(1, 16))],
        step_results={
            "step_02": StepResult(
                step_id="step_02", status=StepStatus.SUCCEEDED, summary="命中 1 条", empty=False
            )
        },
    )

    limits = _pipeline_limitations(state)

    assert any("共 16 行" in item for item in limits)


def test_render_puts_the_position_next_to_the_evidence() -> None:
    """模型看到的是渲染后的文本，所以位置必须出现在那里，而不是只进 locator。"""
    evidence = _doc_evidence(table_row=(7, 16))

    rendered = _render({"E1": evidence})

    assert "该表第 7 行，共 16 行" in rendered
