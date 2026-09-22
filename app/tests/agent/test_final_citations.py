"""`final` 的引用渲染（详设 13.5 的引用、CLAUDE.md 约定 86）。

**同源的引用必须合并，而合并之后行的信息不能丢。** 前者防的是
"8 句一模一样的话让读者以为格式坏了"；后者防的是合并把
"引用了整张表"与"引用了其中 8 行"渲染成同一句话——
约定 78 那个缺口（8 行当整表合计）就住在它们的差别里。
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.agent.nodes.final import _cite_all, _ranges
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence


def _document(
    *, section: list[str] | None = None, page: int | None = 4, **locator: object
) -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="DOCUMENT",
        title="2025年第三季度经营分析 > 五、风险提示",
        claim="华南 | 直营 | 智能家居 | 934.21 | 15.05",
        locator={
            "section_path": section or ["2025年第三季度经营分析", "五、风险提示"],
            "page_no": page,
            **locator,
        },
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        content_hash="a" * 64,
    )


def _sql(index: int, *, total: int = 3, fingerprint: str = "d050c1d91ad8") -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title=f"问题（结果第 {index + 1} 行）",
        claim=f"net_sales={index}",
        locator={"sql_fingerprint": fingerprint + "0" * 52, "result_slice": [index, index + 1]},
        retrieved_at=datetime(2026, 9, 20, tzinfo=UTC),
        content_hash=f"{index:064d}",
    )


# ------------------------------------------------------------ 行号压缩


def test_ranges_compress_consecutive_runs() -> None:
    assert _ranges([1, 2, 3, 5]) == "1–3、5"
    assert _ranges([1]) == "1"
    assert _ranges([3, 1, 2]) == "1–3", "乱序进来也要排好"
    assert _ranges([2, 2, 2]) == "2", "重复行只算一次"
    assert _ranges([]) == ""


# ------------------------------------------------------------ 合并


def test_table_rows_merge_into_one_citation_with_rows() -> None:
    """**这是那个缺陷的正面用例**：8 个表格行块原先渲染出 8 句一模一样的话。

    合并之后是一句，而且要说清引用的是哪几行——读者据此判断
    "这是整张表"还是"只是其中一段"。
    """
    items = [
        _document(table_row=[row, 16], table_caption="分区域分渠道分产品线净销售额明细")
        for row in (1, 2, 3, 5, 8)
    ]

    rendered = _cite_all(items)

    assert rendered.count("《") == 1, "同一个出处只该出现一次"
    assert "第 1–3、5、8 行" in rendered
    assert "分区域分渠道分产品线净销售额明细" in rendered


def test_sql_rows_merge_into_one_citation_with_slices() -> None:
    """SQL 侧同理：一次查询的多行原先也只有指纹、没有行号，且照样重复。"""
    rendered = _cite_all([_sql(0), _sql(1), _sql(2)])

    assert rendered.count("业务数据库查询") == 1
    assert "结果第 1–3 行" in rendered


def test_different_places_stay_separate() -> None:
    """不同出处**不能**被合并——合并的条件是"读起来是同一个地方"。"""
    rendered = _cite_all([_document(), _document(page=5)])

    assert rendered.count("《") == 2


def test_single_evidence_gets_no_suffix() -> None:
    """只有一条时不加任何后缀：`（1 条）` 是噪声。"""
    assert _cite_all([_document()]).endswith("第 4 页")


def test_section_is_not_repeated_when_the_title_already_has_it() -> None:
    """**书名号里外不能是同一串**。

    文档证据的 `title` 是「文档名 > 末级章节」，而 `section_path` 是完整路径——
    两者拼出来常常是「《A > B》（A > B）」，读起来像渲染坏了，
    而它只是同一个地方被说了两遍。
    """
    rendered = _cite_all([_document()])

    assert rendered == "《2025年第三季度经营分析 > 五、风险提示》，第 4 页"


def test_section_still_shows_when_the_title_does_not_contain_it() -> None:
    """标题里没有那段章节时**照旧要拼**：省掉它会让两条不同章节的证据
    看起来出自同一个地方。"""
    item = _document(section=["2025年第三季度经营分析", "六、附录"])

    assert "（2025年第三季度经营分析 > 六、附录）" in _cite_all([item])


def test_merged_blocks_without_rows_say_how_many() -> None:
    """既不是表格行也不是 SQL 行、却撞在同一句出处上时，**如实说有几条**。

    否则合并会把"2 条证据"说得像"1 条"——而"证据有几条"是读者判断
    结论有多硬的依据之一。
    """
    rendered = _cite_all([_document(), _document()])

    assert rendered.count("《") == 1
    assert rendered.endswith("（2 条）")
