"""多源冲突检测（详细设计 13.3 / 13.4 的 VALUE 一类）。

**这里最容易犯的错是"报出一个看起来被算出来的冲突"**——它格式正确、
有数字、有差值，只是配错了对。所以用例大多在测"不该报的时候不报"：
指标名不同不报、范围不同不报、容差内不报、单位读不出来不报。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.agent.nodes.conflict import detect_value_conflicts, render
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence, TimeRange
from app.tools.sql.schemas import MetricSpec, SchemaCatalog


def _catalog(*metrics: MetricSpec) -> SchemaCatalog:
    return SchemaCatalog(version="test-1", tables=(), metrics=metrics)


def _metric(code: str, name: str, aliases: tuple[str, ...] = ()) -> MetricSpec:
    return MetricSpec(
        code=code,
        name=name,
        aliases=aliases,
        expression="SUM(x)",
        unit="CNY",
        grain="订单行",
    )


def _sql_evidence(value: float, *, metric: str = "net_sales", region: str = "华东") -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title="问题（结果第 1 行）",
        claim=f"region_name={region}；{metric}={value}",
        locator={"sql_fingerprint": "f" * 64, "result_slice": [0, 1]},
        event_time=TimeRange(
            start=datetime(2025, 7, 1, tzinfo=UTC), end=datetime(2025, 10, 1, tzinfo=UTC)
        ),
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        metric_code=metric,
        scope={"region": region},
        reliability="HIGH",
        content_hash="a" * 64,
    )


def _document_evidence(text: str, *, title: str = "华东区域2025年第三季度专项分析") -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="DOCUMENT",
        title=title,
        claim=text,
        locator={"section_path": [title, "二、经营业绩回顾"], "chunk_id": "chk_x"},
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        reliability="MEDIUM",
        content_hash="b" * 64,
    )


_TABLE = """华东区域2025年第三季度专项分析 > 二、经营业绩回顾
表：分区域经营情况
区域 | 销售额（万元） | 净销售额（万元） | 同比 | 占比 | 省份数
华东 | 11,233.87 | 11,039.58 | -13.2% | 100.0% | 4"""


def test_detects_a_value_conflict_across_sources() -> None:
    """文档表格里的净销售额 vs 库里的净销售额，超出容差 → 报冲突。

    单位（万元）**必须从表头读出来**：不换算的话 11,039.58 与 1.1 亿
    差四个数量级，报出来的差值毫无意义，而它看起来是一条正常的冲突。
    """
    catalog = _catalog(_metric("net_sales", "净销售额", ("净销售", "销售额")))
    conflicts = detect_value_conflicts(
        [_document_evidence(_TABLE), _sql_evidence(111_967_031.73)], catalog=catalog
    )

    assert len(conflicts) == 1
    item = conflicts[0]
    assert item.type.value == "VALUE"
    # 11,039.58 万元 = 110,395,800 元
    assert item.detected_difference["document_value"] == pytest.approx(110_395_800.0)
    assert item.detected_difference["database_value"] == pytest.approx(111_967_031.73)
    # **用的是精确名而不是别名**：别名里那个「销售额」是含税口径，用它匹配会报假冲突
    assert item.detected_difference["matched_by"] == "exact"
    # 未判定谁对——那是 Reviewer 的事（14.3），不是检测器的
    assert item.resolution.value == "UNRESOLVED"
    assert len(item.evidence_ids) == 2


def test_the_exact_column_wins_over_the_aliased_one() -> None:
    """**同一张表里精确名列优先，别名列让位**——这是那个陷阱的正面防线。

    语料里的报表同时有这两列：`区域 | 销售额（万元） | 净销售额（万元）`，
    而 `net_sales` 的别名表里**包含「销售额」**（它是含税 − 折扣口径）。
    不处理的话两列都会映射到 `net_sales`，对同一个 SQL 数字报出**两条冲突**，
    其中拿含税口径比的那一条是假的——而它和真冲突长得一模一样。

    上一条用例（`test_detects_a_value_conflict_across_sources` 断言只有 1 条）
    就是这条规则的另一个侧面。
    """
    catalog = _catalog(_metric("net_sales", "净销售额", ("销售额",)))
    only_aliased = _TABLE.replace("净销售额（万元） | ", "")
    conflicts = detect_value_conflicts(
        [_document_evidence(only_aliased), _sql_evidence(111_967_031.73)], catalog=catalog
    )

    # 只剩别名列时仍然比——只是要标出来它是靠别名对上的
    assert len(conflicts) == 1
    assert conflicts[0].detected_difference["matched_by"] == "alias"


def test_no_conflict_when_the_difference_is_within_tolerance() -> None:
    """容差内不报（13.4 第 4 步：绝对 1 元与相对 0.1% 取较大者）。

    这条防的是"任何舍入差异都报冲突"——那会让冲突列表变成噪声，
    而真冲突被淹在里面。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    # 差 0.05%（小于 0.1% 的相对容差）
    close = 110_395_800.0 * 1.0005

    conflicts = detect_value_conflicts(
        [_document_evidence(_TABLE), _sql_evidence(close)], catalog=catalog
    )

    assert conflicts == ()


def test_no_conflict_when_the_scope_differs() -> None:
    """范围不同不报——两个不同区域的数本来就不该相等。

    不比对范围的话，"华东 1.1 亿"与"华南 1.06 亿"会被报成一条冲突，
    而它其实只是两个地方。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = detect_value_conflicts(
        [_document_evidence(_TABLE), _sql_evidence(111_967_031.73, region="华南")],
        catalog=catalog,
    )

    assert conflicts == ()


def test_a_multi_dimension_table_is_not_compared_to_a_region_total() -> None:
    """**三个维度列的表，每一行是明细，不是区域合计。**

    实测踩到的假冲突：语料里这张表

        区域 | 渠道 | 产品线 | 净销售额（万元） | 退货金额（万元）
        华东 | 电商 | 智能家居 | 764.63 | 13.57

    第一版只把"第一个维度列"（区域=华东）当成范围，看到与 SQL 的 scope 相同
    就比了，于是拿 764.63 万去对华东季度总额 1.12 亿，报出**相对差 93%** 的
    "冲突"——而两个数压根不是一回事。

    判据是**维度集合相等**：SQL 证据的 `scope` 键就是它的粒度，
    文档表格的维度列给出另一份。这条比"识别合计行"更根本，
    而且不需要语料做任何改动。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    detail = _document_evidence(
        """2025年第三季度经营分析 > 五、风险提示
表：分区域分渠道分产品线净销售额明细
区域 | 渠道 | 产品线 | 净销售额（万元） | 退货金额（万元）
华东 | 电商 | 智能家居 | 764.63 | 13.57"""
    )

    # SQL 只按区域聚合 → 粒度 {region}；文档行是 {region, channel, product_line}
    assert detect_value_conflicts([detail, _sql_evidence(111_967_031.73)], catalog=catalog) == ()


def test_a_single_dimension_table_is_still_compared() -> None:
    """对照：单维度列的表**仍然要比**——那条是真冲突（`demo-cross` 就是它）。

    只加约束不加对照的话，一次"全都不比了"的改动也能让上一条通过。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    single = _document_evidence(_TABLE)  # 表头只有「区域」一个维度列

    conflicts = detect_value_conflicts([single, _sql_evidence(111_967_031.73)], catalog=catalog)

    assert len(conflicts) == 1


def test_an_unscoped_sql_aggregate_still_compares() -> None:
    """SQL 的 `scope` 为空**不代表它是全量**，这条要照比。

    `SELECT SUM(net_amount) WHERE region='华东'` 这种带 WHERE 的标量聚合，
    结果集只有一个聚合列，`_scope_of` 从列名里提取不到 `region`——
    粒度只存在于 SQL 文本里。

    这是实测踩到的：第一版把判据写成"两边维度集合相等"，跑一遍发现
    `demo-cross` 的真冲突没了——因为它正是这条路径。**只看文档侧的维度列数
    才既拦得住明细行的假冲突、又留得住这一条。**
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    unscoped = _sql_evidence(111_967_031.73).model_copy(update={"scope": {}})

    conflicts = detect_value_conflicts([_document_evidence(_TABLE), unscoped], catalog=catalog)

    assert len(conflicts) == 1


def test_data_scope_is_not_a_dimension() -> None:
    """受限用户的证据多一个 `data_scope` 标注，而**它不是维度**。

    把它算进粒度的话，受限用户的证据与任何文档表格都对不上——
    全部跳过，而那表现为"冲突检测对某些用户突然不工作了"，不指向这里。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    scoped = _sql_evidence(111_967_031.73).model_copy(
        update={"scope": {"region": "华东", "data_scope": ["华东"]}}
    )

    conflicts = detect_value_conflicts([_document_evidence(_TABLE), scoped], catalog=catalog)

    assert len(conflicts) == 1


def test_no_conflict_when_the_metric_differs() -> None:
    """指标不同不报。`销售额` 那一列（含税口径）不该与净销售额比。"""
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = detect_value_conflicts(
        [
            _document_evidence(_TABLE),
            _sql_evidence(111_967_031.73, metric="gross_sales"),
        ],
        catalog=catalog,
    )

    assert conflicts == ()


def test_sql_without_a_matching_value_in_the_claim_is_skipped() -> None:
    """SQL 证据的 `claim` 里找不到 `metric_code` 的值时**跳过，不猜**。

    这条是"报出一个看起来被算出来的冲突"最可能的来源：
    从别的列里取一个数去比对，配出来的冲突格式完全正常。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    mismatched = _sql_evidence(111_967_031.73).model_copy(
        update={"claim": "region_name=华东；gross_sales=12345"}
    )

    assert detect_value_conflicts([_document_evidence(_TABLE), mismatched], catalog=catalog) == ()


def test_a_document_without_tables_produces_nothing() -> None:
    """正文里的数字不解析——从自由文本取数要知道"这个数说的是哪个指标"，
    那是语义，纯正则硬做出来的是一堆看起来像冲突的噪声。"""
    catalog = _catalog(_metric("net_sales", "净销售额"))
    prose = _document_evidence("报告期内，华东实现销售额 11,233.87 万元，净销售额11,039.58 万元。")

    assert detect_value_conflicts([prose, _sql_evidence(111_967_031.73)], catalog=catalog) == ()


def test_without_a_catalog_nothing_is_compared() -> None:
    """没有指标目录就没有"表头 → metric_code"的桥，整条检测跳过。

    **跳过不等于"没有冲突"**：`final` 的渲染会把这两种情形分开说
    （见 `nodes/final.py`），否则读答案的人会以为五类都比过了。
    """
    from app.agent.nodes.conflict import build_conflict_node

    node = build_conflict_node(catalog=None)
    state: Any = {"evidence": [_document_evidence(_TABLE), _sql_evidence(111_967_031.73)]}

    assert node(state) == {"conflicts": []}


def test_render_lists_the_possible_explanations() -> None:
    """渲染给模型看的文本要带上可能原因——那是它披露冲突时的措辞依据。"""
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = detect_value_conflicts(
        [_document_evidence(_TABLE), _sql_evidence(111_967_031.73)], catalog=catalog
    )

    text = render(conflicts)

    assert "[C1]" in text
    assert "可能原因" in text
    assert "UNRESOLVED" in text
    assert render(()) == "（未检出冲突）"


#: 一张**五个区域各一行**的汇总表。补块（11.7 第 ⑧ 步）会把这种表整张放进
#: 证据，于是每一行都会被拿去比对——这正是实测踩到四条假冲突的那张表。
_REGION_TABLE = """华东区域2025年第三季度专项分析 > 二、经营业绩回顾
表：分区域经营情况
区域 | 净销售额（万元）
华东 | 11,039.58
华南 | 13,062.07
华北 | 12,792.96
华中 | 12,206.40
西南 | 13,017.66"""


def test_other_regions_are_not_compared_against_a_scoped_sql_aggregate() -> None:
    """**SQL 侧有范围时，别的区域的行不能被拿去比。**

    实测（2026-09-20）：检索侧开始把同表的多行一起放进证据之后，
    这张五行的表里**四条别的区域**都被拿去对华东的库值，
    报出相对差 9–16% 的假冲突——而它们压根不是同一件事。

    修法在 SQL 侧：`SELECT SUM(...) WHERE region_name='华东'` 的粒度
    只存在于 WHERE 里，把它抽出来放进证据的 `scope`，比对就自然对上了。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = detect_value_conflicts(
        [_document_evidence(_REGION_TABLE), _sql_evidence(111_967_031.73)], catalog=catalog
    )

    assert len(conflicts) == 1, "只有华东那一行该被比"
    assert conflicts[0].detected_difference["document_value"] == pytest.approx(110_395_800.0)


def test_without_a_sql_side_scope_every_row_still_compares() -> None:
    """反过来说清楚：**SQL 侧范围未知时仍然照比**（约定 41 的有意放行）。

    这条不是"理想行为"，而是**权衡**：把判据收紧成"范围未知就不比"，
    `demo-cross` 那条真冲突会一起消失——而那条正是这个双源切片存在的理由。
    多报看得出来（读者会问"华南这行凭什么对华东的库值"），
    漏报看不出来。所以宁可多报。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    unscoped = _sql_evidence(111_967_031.73).model_copy(update={"scope": {}})

    conflicts = detect_value_conflicts(
        [_document_evidence(_REGION_TABLE), unscoped], catalog=catalog
    )

    assert len(conflicts) == 5
