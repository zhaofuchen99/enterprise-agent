"""多源冲突检测（详细设计 13.3 / 13.4 的 VALUE / DEFINITION / TIME 三类）。

**这里最容易犯的错是"报出一个看起来被算出来的冲突"**——它格式正确、
有数字、有差值，只是配错了对。所以用例大多在测"不该报的时候不报"：
指标名不同不报、范围不同不报、容差内不报、单位读不出来不报；
口径版本相同不报、"不知道版本"不报、截止日落在期末不报、期间解不出来不报。

VALUE 用例统一走 `_only_value`（本文件的主要部分）；DEFINITION 与 TIME
各自独立成段。**SCOPE 与 SOURCE 没有用例**——那两类还没实现，
理由写在 `app/agent/nodes/conflict.py` 的模块 docstring 里。
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from app.agent.nodes.conflict import detect_conflicts, render
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Conflict, ConflictType, Evidence, TimeRange
from app.tools.sql.schemas import MetricSpec, SchemaCatalog


def _only_value(evidence: list[Evidence], *, catalog: SchemaCatalog) -> tuple[Conflict, ...]:
    """只取 VALUE 类。

    本文件测的是「文档表格 × SQL 数值」这条配对路径上的**数值比对**，
    而 `detect_conflicts` 是三类冲突的总入口。DEFINITION 走的是另一条路
    （只看证据对，与表格无关），TIME 与 VALUE 共用这条配对路径但判据不同
    ——各自有各自的用例文件。这里挑一类出来，用例才只说一件事。
    """
    return tuple(
        item
        for item in detect_conflicts(evidence, catalog=catalog)
        if item.type is ConflictType.VALUE
    )


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


def _sql_evidence(
    value: float,
    *,
    metric: str = "net_sales",
    region: str = "华东",
    definition_version: str | None = None,
) -> Evidence:
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
        definition_version=definition_version,
        scope={"region": region},
        reliability="HIGH",
        content_hash="a" * 64,
    )


def _document_evidence(
    text: str,
    *,
    title: str = "华东区域2025年第三季度专项分析",
    logical_key: str = "report/special-east-china-2025Q3",
    metric_code: str | None = None,
    definition_version: str | None = None,
    stat_period: str | None = None,
    stat_cutoff: date | None = None,
) -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="DOCUMENT",
        title=title,
        claim=text,
        locator={
            "section_path": [title, "二、经营业绩回顾"],
            "chunk_id": "chk_x",
            # 冲突去重按它分组（同一篇文档可能召回多块），见 `_definition_conflicts`
            "logical_key": logical_key,
        },
        retrieved_at=datetime(2026, 9, 18, tzinfo=UTC),
        metric_code=metric_code,
        definition_version=definition_version,
        stat_period=stat_period,
        stat_cutoff=stat_cutoff,
        reliability="MEDIUM",
        content_hash="b" * 64,
    )


_TABLE = """华东区域2025年第三季度专项分析 > 二、经营业绩回顾
表：分区域经营情况
区域 | 销售额（万元） | 净销售额（万元） | 同比 | 占比 | 省份数
华东 | 11,233.87 | 11,039.58 | -13.2% | 100.0% | 4"""

#: 一篇口径说明的正文。**它没有表格**——DEFINITION 冲突看的是文档身份上的
#: `metric_code` 与 `definition_version`，与分块正文里有几张表无关。
#: 写成正文而不是表格，正是为了钉住这一点。
_METRIC_DOC = """订单行数口径说明 > 一、指标定义
指标编码：order_count
口径版本：v1.1
订单行数按销售订单明细表逐行计数，一个订单含三行商品即计 3。"""


def test_detects_a_value_conflict_across_sources() -> None:
    """文档表格里的净销售额 vs 库里的净销售额，超出容差 → 报冲突。

    单位（万元）**必须从表头读出来**：不换算的话 11,039.58 与 1.1 亿
    差四个数量级，报出来的差值毫无意义，而它看起来是一条正常的冲突。
    """
    catalog = _catalog(_metric("net_sales", "净销售额", ("净销售", "销售额")))
    conflicts = _only_value(
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
    conflicts = _only_value(
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

    conflicts = _only_value([_document_evidence(_TABLE), _sql_evidence(close)], catalog=catalog)

    assert conflicts == ()


def test_no_conflict_when_the_scope_differs() -> None:
    """范围不同不报——两个不同区域的数本来就不该相等。

    不比对范围的话，"华东 1.1 亿"与"华南 1.06 亿"会被报成一条冲突，
    而它其实只是两个地方。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = _only_value(
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

    判据是**文档侧的维度列超过一个**：SQL 那边的粒度未必全在结果列里
    （标量聚合的 `scope` 是空的），所以只能从文档侧判"这一行是不是比
    按一个维度汇总更细"。这条比"识别合计行"更根本，
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
    assert _only_value([detail, _sql_evidence(111_967_031.73)], catalog=catalog) == ()


def test_a_single_dimension_table_is_still_compared() -> None:
    """对照：单维度列的表**仍然要比**——那条是真冲突（`demo-cross` 就是它）。

    只加约束不加对照的话，一次"全都不比了"的改动也能让上一条通过。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    single = _document_evidence(_TABLE)  # 表头只有「区域」一个维度列

    conflicts = _only_value([single, _sql_evidence(111_967_031.73)], catalog=catalog)

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

    conflicts = _only_value([_document_evidence(_TABLE), unscoped], catalog=catalog)

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

    conflicts = _only_value([_document_evidence(_TABLE), scoped], catalog=catalog)

    assert len(conflicts) == 1


def test_no_conflict_when_the_metric_differs() -> None:
    """指标不同不报。`销售额` 那一列（含税口径）不该与净销售额比。"""
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = _only_value(
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

    assert _only_value([_document_evidence(_TABLE), mismatched], catalog=catalog) == ()


def test_a_document_without_tables_produces_nothing() -> None:
    """正文里的数字不解析——从自由文本取数要知道"这个数说的是哪个指标"，
    那是语义，纯正则硬做出来的是一堆看起来像冲突的噪声。"""
    catalog = _catalog(_metric("net_sales", "净销售额"))
    prose = _document_evidence("报告期内，华东实现销售额 11,233.87 万元，净销售额11,039.58 万元。")

    assert _only_value([prose, _sql_evidence(111_967_031.73)], catalog=catalog) == ()


def test_without_a_catalog_the_table_route_is_skipped() -> None:
    """没有指标目录就没有"表头 → metric_code"的桥，**表格那一路**（VALUE / TIME）跳过。

    **跳过不等于"没有冲突"**：`final` 的渲染会把这两种情形分开说
    （见 `nodes/final.py`），否则读答案的人会以为三类都比过了。
    """
    from app.agent.nodes.conflict import build_conflict_node

    node = build_conflict_node(catalog=None)
    state: Any = {"evidence": [_document_evidence(_TABLE), _sql_evidence(111_967_031.73)]}

    assert node(state) == {"conflicts": []}


def test_definition_conflicts_do_not_need_a_catalog() -> None:
    """对照：**DEFINITION 那一路不依赖目录**，`catalog=None` 时照报。

    理由：它的两侧——文档 payload 里的口径版本与 SQL 证据里的——**都已经长在
    证据上**，目录只是"表头 → metric_code"的桥，与版本比对无关。
    把它们一起关掉的话，一个"没有目录"的装配（或目录加载失败）会连
    "文档讲的是一版口径、库里算的是另一版"也一起静默——而那是**已经拿在手上的
    两条事实**，不该因为另一个组件缺席就不说。
    """
    from app.agent.nodes.conflict import build_conflict_node

    node = build_conflict_node(catalog=None)
    state: Any = {
        "evidence": [
            _document_evidence(
                _METRIC_DOC,
                title="订单行数口径说明",
                metric_code="order_count",
                definition_version="v1.1",
            ),
            _sql_evidence(12_345, metric="order_count", definition_version="v1.0"),
        ]
    }

    conflicts = node(state)["conflicts"]

    assert [item.type for item in conflicts] == [ConflictType.DEFINITION]


def test_render_lists_the_possible_explanations() -> None:
    """渲染给模型看的文本要带上可能原因——那是它披露冲突时的措辞依据。"""
    catalog = _catalog(_metric("net_sales", "净销售额"))
    conflicts = _only_value(
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
    conflicts = _only_value(
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

    conflicts = _only_value([_document_evidence(_REGION_TABLE), unscoped], catalog=catalog)

    assert len(conflicts) == 5


#: 一张**只有产品线一个维度列**的表——表头里没有区域列，所以它表达的是
#: **全公司**口径（`claim.scope` 只有 `{product_line: …}`）。
_PRODUCT_LINE_TABLE = """2025年第三季度经营分析 > 五、产品线表现
表：分产品线净销售额
产品线 | 净销售额（万元）
智能家居 | 12,334.70"""


def test_a_coarser_document_row_is_not_compared_to_a_finer_sql_point() -> None:
    """**SQL 点比文档行多一个维度 → 不比**（约定 41 的另一个方向）。

    实测踩到的（2026-09-22）：文档「产品线表现」表是**全公司**口径
    （表头只有产品线列，`claim.scope = {product_line: 智能家居}`），
    而 SQL 查的是**华东**
    （`base.scope = {region: 华东, product_line: 智能家居}`）。

    原先的判据只遍历**文档侧**的键，`base` 侧多出来的 `region` 无人过问，
    于是判为可比，报出相对差 **499.55%** 的假冲突——而两个数比的是
    两个不同的总体（全公司 vs 华东）。**SQL 更细**这一侧原先没有挡。

    只加约束不加对照的话，一次"全都不比了"的改动也能让这条通过，
    所以下面有一条对照用例断言"该比的照比"。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    finer = _sql_evidence(111_967_031.73).model_copy(
        update={"scope": {"region": "华东", "product_line": "智能家居"}}
    )

    conflicts = _only_value(
        [_document_evidence(_PRODUCT_LINE_TABLE, title="2025年第三季度经营分析"), finer],
        catalog=catalog,
    )

    assert conflicts == ()


def test_a_coarser_document_row_still_compares_when_the_sql_side_is_coarse_too() -> None:
    """对照：**SQL 侧也只有产品线时照比**——那条是真冲突。

    与上一条的差别只有 SQL 侧的 `scope`：这条没有 `region`。
    两条合起来才钉住"多出来的那个维度"是判据本身，而不是别的东西。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    same_grain = _sql_evidence(111_967_031.73).model_copy(
        update={"scope": {"product_line": "智能家居"}}
    )

    conflicts = _only_value(
        [_document_evidence(_PRODUCT_LINE_TABLE, title="2025年第三季度经营分析"), same_grain],
        catalog=catalog,
    )

    assert len(conflicts) == 1
    # **两个方向的粒度都进明细**：判"可比"要两边都对上，只记文档侧的话，
    # "凭什么拿这一行对那个库值"在产物上看不出来（499.55% 那条就是这么溜过去的）
    assert conflicts[0].detected_difference["scope"] == {"product_line": "智能家居"}
    assert conflicts[0].detected_difference["database_scope"] == {"product_line": "智能家居"}


# ------------------------------------------------------------------ 统计期间


def test_a_row_from_another_period_is_not_compared() -> None:
    """**跨期间的数不可比**——表格行是它那个期间的**合计**。

    实测（2026-09-22）：这一条缺失时，问「2025年8月直营渠道净销售额」会把
    8 月的库值拿去和《上半年经营回顾》《年度经营分析》《Q1 经营分析》
    《下半年经营展望》的同类表格比，报出 4 条 **157%–1133%** 的假冲突
    （只有 1 条是真的——8 月那份月报）。

    `_sql_evidence` 的 `event_time` 是 2025Q3，所以文档侧说 8 月的那一行
    与它**不同期**：数字不同是应该的。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    august = _document_evidence(_TABLE).model_copy(update={"stat_period": "2025-08"})
    same_quarter = _document_evidence(_TABLE).model_copy(update={"stat_period": "2025-Q3"})

    assert _only_value([august, _sql_evidence(111_967_031.73)], catalog=catalog) == ()
    # 对照：同期的那一行照比——只加约束不加对照的话，
    # 一次"期间对不上就全不比"的改动也能让上面那条通过
    assert len(_only_value([same_quarter, _sql_evidence(111_967_031.73)], catalog=catalog)) == 1


@pytest.mark.parametrize("period", ["2025-H1", "2025-H2", "2025", "2025-Q1", "2025-07"])
def test_periods_that_merely_contain_the_query_period_are_still_skipped(period: str) -> None:
    """**取「相等」而不是「相交」**：包含关系不算同期。

    H1 / 全年 / H2 都**包含** 8 月，按"相交"判的话它们会全部放行——
    而它们说的是三个不同的期间。这条正是"别把判据写成区间重叠"的钉子。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    containing = _document_evidence(_TABLE).model_copy(update={"stat_period": period})

    assert _only_value([containing, _sql_evidence(111_967_031.73)], catalog=catalog) == ()


def test_an_unknown_period_still_compares() -> None:
    """**期间未知 → 放行**。`None` 是"不知道它的期间"，不是"它不限期间"。

    与空 `scope` 同一条取舍：收紧成"不知道就不比"会丢掉真冲突，而漏报
    看不出来。代价是没有时间条件的查询会与任何报告照比——**宁可多报**。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))

    assert (
        len(
            _only_value(
                [_document_evidence(_TABLE), _sql_evidence(111_967_031.73)], catalog=catalog
            )
        )
        == 1
    )


def test_a_query_whose_range_is_not_a_period_has_no_period() -> None:
    """归不到规范跨度的区间**没有期间**（不是"硬凑一个"）。

    九个月、同比查询包出来的 15 个月、没有时间条件——都落在这里。
    硬凑一个记号会让一条跨期的比对照常发生，而它看起来完全正常。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    nine_months = _sql_evidence(111_967_031.73).model_copy(
        update={
            "event_time": TimeRange(
                start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2025, 10, 1, tzinfo=UTC)
            )
        }
    )
    monthly = _document_evidence(_TABLE).model_copy(update={"stat_period": "2025-08"})

    # SQL 侧期间未知 → 放行（多报方向），所以这条**照比**
    assert len(_only_value([monthly, nine_months], catalog=catalog)) == 1


def test_the_document_level_scope_is_merged_into_every_row() -> None:
    """**文档身份上的范围要并进每一行**——否则区域报告会被当成全公司。

    `SR-EC`（《华东区域2025年第三季度专项分析》）的「分产品线」表**只有产品线列**
    ——区域不在表头里，而在文档身份上。并进文档级范围之后，它才是"华东的各产品线"，
    才能与华东的库值比；不并的话它会因为"少了 region 这个维度"而整张跳过，
    而那是**漏报**：一条真冲突安静地不见了。

    与 `test_a_coarser_document_row_is_not_compared_to_a_finer_sql_point` 成对：
    同样是"文档行少一个维度"，**文档级范围能补上就不算少**，
    补不上（真的是全公司口径）就跳过。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    regional = _document_evidence(_PRODUCT_LINE_TABLE, title="2025年第三季度经营分析").model_copy(
        update={"scope": {"region": "华东"}}
    )
    finer = _sql_evidence(111_967_031.73).model_copy(
        update={"scope": {"region": "华东", "product_line": "智能家居"}}
    )

    conflicts = _only_value([regional, finer], catalog=catalog)

    assert len(conflicts) == 1, "文档级范围补上了 region，两个方向就对得上了"
    assert conflicts[0].detected_difference["scope"] == {
        "region": "华东",
        "product_line": "智能家居",
    }


# ==================================================================== DEFINITION
def test_a_definition_version_mismatch_is_reported() -> None:
    """同一指标、两侧口径版本不等 → DEFINITION 冲突（13.4 第 5 步）。

    载体是**口径说明文档**：它逐字声明 `metric_code` 与 `definition_version`
    （清单里写的），而目录给 SQL 侧证据填的是目录自己的 `version`。
    两者不等意味着「文档讲的是一版口径、库里算的是另一版」——
    这比数值差异更根本：数的差往往正是它引出来的。
    """
    document = _document_evidence(
        _METRIC_DOC, title="订单行数口径说明", metric_code="order_count", definition_version="v1.1"
    )
    base = _sql_evidence(12_345, metric="order_count", definition_version="v1.0")

    conflicts = detect_conflicts([document, base], catalog=None)

    assert len(conflicts) == 1
    item = conflicts[0]
    assert item.type is ConflictType.DEFINITION
    assert item.evidence_ids == (document.id, base.id)
    assert item.detected_difference["document_definition_version"] == "v1.1"
    assert item.detected_difference["database_definition_version"] == "v1.0"
    # 检测器不判谁对，同 VALUE
    assert item.resolution.value == "UNRESOLVED"
    # 版本号要出现在描述里，读的人才知道差在哪
    assert "v1.1" in item.description
    assert "v1.0" in item.description


def test_matching_definition_versions_are_not_reported() -> None:
    """对照：两边版本相同就没什么可说的。没有这条，一个"永远报一条"的实现也能过。"""
    document = _document_evidence(
        _METRIC_DOC, title="订单行数口径说明", metric_code="order_count", definition_version="v1.0"
    )
    base = _sql_evidence(12_345, metric="order_count", definition_version="v1.0")

    assert detect_conflicts([document, base], catalog=None) == ()


def test_a_definition_conflict_needs_both_versions_known() -> None:
    """任一侧口径版本未知 → 不报。

    **"不知道版本"推不出"版本不一致"。** 注意这条路与 `_comparable` 的
    "未知就放行"方向看似相反，但动作不是一回事：那里放行的是**允许数值比对**
    （多报方向），这里要产生的是**一条明确的口径指控**——凭不知道去指控就是编。
    """
    known = _document_evidence(
        _METRIC_DOC, title="订单行数口径说明", metric_code="order_count", definition_version="v1.1"
    )
    unknown_base = _sql_evidence(12_345, metric="order_count")

    assert detect_conflicts([known, unknown_base], catalog=None) == ()


def test_one_metric_document_is_reported_once() -> None:
    """**一篇口径说明被切成多块时只报一条**。

    Top-8 召回里同一篇文档常有两块以上，每块一条证据。不去重的话
    冲突列表里会出现两条一模一样的话——读的人会以为有两个问题。
    按 `locator.logical_key` 去重（跨版本稳定），不按 `id`（每块一个）。
    """
    first = _document_evidence(
        _METRIC_DOC, title="订单行数口径说明", metric_code="order_count", definition_version="v1.1"
    )
    second = first.model_copy(update={"id": new_id(IdPrefix.EVIDENCE)})
    base = _sql_evidence(12_345, metric="order_count", definition_version="v1.0")

    assert len(detect_conflicts([first, second, base], catalog=None)) == 1


def test_definition_conflicts_ignore_scope_and_period() -> None:
    """口径是**指标级**属性——范围与期间对不上也照报。

    强行要求 scope 相同会把这条判据关掉：口径说明文档没有 `report` 块，
    它的 `scope` 与 `stat_period` **恒为空**，而 SQL 侧通常带着 `region`。
    """
    document = _document_evidence(
        _METRIC_DOC, title="订单行数口径说明", metric_code="order_count", definition_version="v1.1"
    )
    base = _sql_evidence(12_345, metric="order_count", region="华南", definition_version="v1.0")

    assert len(detect_conflicts([document, base], catalog=None)) == 1


# ======================================================================== TIME
def test_a_truncated_period_is_reported_as_time() -> None:
    """文档自称 2025-Q3，但统计截止日是 9/25（该季到 9/30）→ TIME 冲突。

    判据是**文档自己前后矛盾**，不是"两个来源的期间不同"——
    后者会把"问 Q3 却召回年度报告"这种常见情况全报出来（实测过 157%–1133%
    的那一批假冲突），噪声大一个数量级。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    document = _document_evidence(_TABLE, stat_period="2025-Q3", stat_cutoff=date(2025, 9, 25))

    conflicts = detect_conflicts([document, _sql_evidence(111_967_031.73)], catalog=catalog)

    assert len(conflicts) == 1
    item = conflicts[0]
    assert item.type is ConflictType.TIME
    assert item.detected_difference["stat_period"] == "2025-Q3"
    assert item.detected_difference["declared_cutoff"] == "2025-09-25"
    assert item.detected_difference["period_end"] == "2025-09-30"
    # 描述里必须有期间末，读的人才看得出"早于哪一天"
    assert "2025-09-30" in item.description


def test_month_end_is_never_treated_as_truncated() -> None:
    """截止日落在期间末（`month_end` 解析出来的就是它）→ 不是截短，**不报**。

    这是这条判据不误报的根据：语料里绝大多数报告都是 `month_end`，
    它们必须一条 TIME 都不产生。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    document = _document_evidence(_TABLE, stat_period="2025-Q3", stat_cutoff=date(2025, 9, 30))

    conflicts = detect_conflicts([document, _sql_evidence(111_967_031.73)], catalog=catalog)

    assert [item.type for item in conflicts] == [ConflictType.VALUE], "只剩数值那条"


def test_time_replaces_value_on_the_same_pair() -> None:
    """期间被截短的同一对证据**只报 TIME，不报 VALUE**。

    两个数既然不是同一个区间的合计，数值比对的前提就不成立。
    再报一条"数值差 1.4%"是把「少统计了几天」说成「数字错了」——
    而 13.3 的 `possible_explanations` 里原本只能**猜**时点不同，现在能**判**了。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    document = _document_evidence(_TABLE, stat_period="2025-Q3", stat_cutoff=date(2025, 9, 25))

    conflicts = detect_conflicts([document, _sql_evidence(111_967_031.73)], catalog=catalog)

    assert [item.type for item in conflicts] == [ConflictType.TIME]


def test_time_is_reported_even_when_the_numbers_agree() -> None:
    """数值一致**不影响** TIME——它说的是声明，不是取值。

    这是判据的已知代价，如实钉住：语料里 9/28 截止的那两份报告，
    因为业务库的日粒度数据只到每月 27 日，数字其实与完整期间一致；
    但"自称 Q3、截止 9/28"这件事本身仍然矛盾，仍然该说。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    document = _document_evidence(_TABLE, stat_period="2025-Q3", stat_cutoff=date(2025, 9, 25))

    conflicts = detect_conflicts([document, _sql_evidence(110_395_800.0)], catalog=catalog)

    assert [item.type for item in conflicts] == [ConflictType.TIME]


def test_an_unknown_cutoff_falls_back_to_value() -> None:
    """截止日未知 → 不是 TIME，而且**不抑制** VALUE。

    "不知道"不能推出"不一致"，但也没有理由因此剥夺数值比对
    （与 `_comparable` 的"未知就放行"同向）。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    document = _document_evidence(_TABLE, stat_period="2025-Q3")

    conflicts = detect_conflicts([document, _sql_evidence(111_967_031.73)], catalog=catalog)

    assert [item.type for item in conflicts] == [ConflictType.VALUE]


def test_an_unknown_period_is_not_a_time_conflict() -> None:
    """期间未知（解不成四种记号）→ 不报 TIME，即使截止日早得离谱。

    没有"自称的期间"就没有可矛盾的对象——`2025-01` 到 `2025-09` 这种
    九个月的区间归不到任何记号，它就不是"某一期被截短"。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    document = _document_evidence(_TABLE, stat_cutoff=date(2025, 9, 25))

    conflicts = detect_conflicts([document, _sql_evidence(111_967_031.73)], catalog=catalog)

    assert [item.type for item in conflicts] == [ConflictType.VALUE]


def test_the_time_description_carries_the_scope_suffix() -> None:
    """TIME 的描述必须带上 `，范围 region=…` 后缀。

    评测判据（`scripts/agent_harness.py`）对描述做的是**子串匹配**，
    几条真冲突用例正是靠 `region=华南` / `region=华东` 钉住"报的是哪一行"。
    TIME 与 VALUE 各写一遍这个后缀的话，改了一处会让另一处的用例变成
    永远匹配不上的红，而红的原因看起来与文案毫无关系。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    row = _TABLE.replace("华东 |", "华南 |")
    document = _document_evidence(row, stat_period="2025-Q3", stat_cutoff=date(2025, 9, 25))

    conflicts = detect_conflicts(
        [document, _sql_evidence(111_967_031.73, region="华南")], catalog=catalog
    )

    assert len(conflicts) == 1
    assert "region=华南" in conflicts[0].description


def test_one_time_conflict_per_document_per_period() -> None:
    """同一篇文档的多个行块只报**一条** TIME。

    一篇报告被切成多块、一张表有好几行——不去重的话"自称 Q3 却只统计到 9/25"
    会按行重复报出来，冲突列表里堆着十几条同义句，而读者会以为有十几个问题。
    TIME 说的是**文档级**的声明，收敛成一条才对应得上。
    """
    catalog = _catalog(_metric("net_sales", "净销售额"))
    first = _document_evidence(_TABLE, stat_period="2025-Q3", stat_cutoff=date(2025, 9, 25))
    second = first.model_copy(update={"id": new_id(IdPrefix.EVIDENCE)})
    scoped = _sql_evidence(111_967_031.73)

    conflicts = detect_conflicts([first, second, scoped], catalog=catalog)

    assert [item.type for item in conflicts] == [ConflictType.TIME]
