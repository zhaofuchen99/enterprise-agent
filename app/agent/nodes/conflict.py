"""`conflict` 节点：多源冲突检测（详细设计 13.3 / 13.4 的**切片版**）。

## 这一版只做一类冲突，而且是有理由的

13.4 定义了五类。本实现只做 **VALUE**（同口径数值超容差），覆盖的载体只有两种：

文档侧认**表格行**：表头给出列名与单位（`区域 | 销售额（万元） | 净销售额（万元）`），
数据行给出范围与取值（`华东 | 11,233.87 | 11,039.58`）。
SQL 侧认证据 `claim` 里渲染出来的 `列=值`。

**另外四类为什么不在这里**：

- `DEFINITION` / `SCOPE`：要比较口径版本与维度范围，而**文档证据的
  `metric_code` 与 `scope` 现在是空的**——分块不带指标 code，按 `logical_key`
  反推是把自由文本约定当语义用（已登记）。两端缺一端就比不了。
- `TIME`：文档的统计期间不在 `Evidence` 里（`event_time` 是文档的**生效区间**，
  对报告恒为空）。没有它就无法区分"同期不同数"与"不同期的数"。
- `SOURCE`：外部与内部相悖属 16.11.2 的注入缺陷，判据是"方向相反"，
  要靠 Phase 9 的完整版；本版的 `source_kind` 已经进了证据，接口留着。

**这份清单就是面试口径**：被问"冲突检测做了多少"时，答案是
「一类（VALUE）做了，四类没做，各缺什么前提写在这里」，而不是"做了冲突检测"。

## 一个实测出来的关键约束：**两个方向的维度都要对上**

第一版只把"表里的第一个维度列"当作范围，于是这样一张表会出事：

    区域 | 渠道 | 产品线 | 净销售额（万元） | 退货金额（万元）
    华东 | 电商 | 智能家居 | 764.63 | 13.57

它**有三个维度列**，所以每一行是「华东/电商/智能家居」的**明细**，而不是
华东的合计。检测器看到"区域=华东"与 SQL 的 scope 相同就比了，
于是拿 764.63 万去对华东季度总额 1.12 亿，报出**相对差 93%** 的"冲突"——
而两个数压根不是一回事。

判据在 `_comparable` 里，两个方向各遍历一次：

| 文档表格 | SQL 证据 | 比不比 |
|---|---|---|
| 单维度（区域），华东 | `scope` 含 region_name=华东 | ✅ 共有维度取值相同 |
| 单维度（区域），华东 | `scope={}`（WHERE 里筛的华东） | ✅ 无法反驳，比 |
| 单维度（区域），华南 | `scope` 含 region_name=华东 | ⛔ 取值不同 |
| 三维度（区域/渠道/产品线） | 任意 | ⛔ 文档行更细，比了就是假冲突 |
| 单维度（产品线），全公司 | `scope` 含 region=华东 + product_line | ⛔ SQL 更细，比的是两个总体 |

**最后一行是后补的**（2026-09-22 实测）：原先只判"文档行比 SQL 更细"这**一个
方向**，反方向没有挡。于是文档里**全公司**口径的「产品线表现」表被拿去对
**华东**的库值——两边产品线相同、`any(...)` 对一个空迭代恒为假——报出相对差
**499.55%** 的假冲突。**SQL 点比文档行多一个维度，说明它更细**，
而更细的数与更粗的行比的是两个不同的总体。

**判据不是"两边维度集合相等"**：那个看着更严格，但会把上表第二行那种
**真冲突**也挡掉——而那条正是 `demo-cross` 的 headline。这是实测出来的：
先写成集合相等，跑一遍发现真冲突没了。真正的规则是**两个方向各遍历一次**，
而空 `scope` 两边都遍历零个键、恒为真——**放行**。

这条比"识别合计行"更根本，而且不需要语料做任何改动——
**表里没有合计标记时，"这一行跨了几个维度"就是可比的判据**。

⚠️ **放行的方向是有代价的，如实记下**：SQL 侧范围真的未知时（`scope={}`，
可能是 `WHERE` 里的标量聚合，也可能是真没有范围限定的全量聚合），文档的
每一行都会照比。**多报看得出来**（读者会问"华南这行凭什么对华东的库值"），
**漏报看不出来**——所以取舍是往多报那边倒的。


文档表头写的是 `净销售额（万元）`，SQL 证据带的是 `metric_code=net_sales`。
两者能对上，靠的是 `schema_catalog.yaml` 里的**指标名与别名**。

⚠️ **别名匹配有个陷阱，代码里防着**：`net_sales` 的别名里有「销售额」，
而报告里的「销售额」列是**含税 − 折扣**口径（not 净销售额）——
按别名匹配会把它错配到 `net_sales` 上，然后报一个**假冲突**。
所以匹配**先精确名、后退别名**，并在 `detected_difference` 里写明用的是哪一种，
让读冲突的人知道这个结论有多硬。

## 容差（13.4 第 4 步）

金额取「绝对 1 元」与「相对 0.1%」中**较大**者。切片内固定这两个值，
不给配置项：13.4 写的就是"默认"，而 22.6 的多源冲突评测集还没扩
（扩集时会连同容差边界一起校准）。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from app.agent.state import AgentState
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import (
    Conflict,
    ConflictResolution,
    ConflictSeverity,
    ConflictType,
    Evidence,
)
from app.domain.period import canonical_period, canonical_token

#: 绝对容差（元）。13.4 第 4 步的默认值。
_ABS_TOLERANCE = 1.0
#: 相对容差。**与绝对容差取较大者**：大额时相对容差生效，小额时绝对容差生效。
_REL_TOLERANCE = 0.001

#: 万元 → 元。表头的单位后缀只有这一种在语料里出现，但它必须是**从表头读出来的**
#: 而不是假设的：不读单位的话，11,039.58 会被当成 11,039.58 元与库里的
#: 1.1 亿去比，差 4 个数量级，报出来的冲突毫无意义。
_UNITS: dict[str, float] = {"万元": 10_000.0, "元": 1.0, "亿元": 100_000_000.0}

#: 表头里的单位后缀，如 `净销售额（万元）`。半角括号也要认——归一化会把
#: 全角折成半角，两条路径产出的表头括号形态不同。
_UNIT_SUFFIX = re.compile(r"[（(]\s*(万元|亿元|元)\s*[)）]\s*$")

#: 数字：允许千分位与小数。**不认科学计数法**——语料里没有，
#: 而放宽会让 `1e5` 这类被当成 100000，那是猜。
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

#: 表格行的列分隔符，与 `chunker._COLUMN_SEPARATOR` 一致。
_COLUMN_SEPARATOR = " | "


def build_conflict_node(
    catalog: Any | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """构造 `conflict` 节点。

    `catalog` 是 `SchemaCatalog`（指标目录）。**缺省时整条检测静默跳过**——
    这看起来危险，但它是对的：目录只是"表头 → metric_code"的桥，
    没有桥就没有可比的配对，而**报不出冲突**与**没有冲突**在最终答案里
    说法不同（见 `final` 的渲染）。为了让节点在有目录时能工作、
    无目录时能跑，这里不做断言。
    """

    def conflict(state: AgentState) -> dict[str, Any]:
        evidence = list(state.get("evidence") or [])
        if catalog is None or len(evidence) < 2:
            return {"conflicts": []}
        found = detect_value_conflicts(evidence, catalog=catalog)
        return {"conflicts": list(found)}

    return conflict


def detect_value_conflicts(evidence: Sequence[Evidence], *, catalog: Any) -> tuple[Conflict, ...]:
    """文档声称的数值 vs SQL 查出的数值（13.4 的 VALUE 一类）。

    配对规则：**指标 + 维度范围 + 统计期间**都对得上才比。
    指标由表头经目录映射得到，维度由**文档级范围**（《华东区域…专项分析》
    → `{region: 华东}`）与表格的维度列合起来给出（`区域` → `华东`）。

    期间那一项见 `_same_period`：**跨期间的数不可比**，而这一条是实测补的
    （2025 年 8 月的库值曾被拿去和《上半年经营回顾》《年度经营分析》的
    同类表格比，报出 4 条 157%–1133% 的假冲突）。
    """
    sql_points = [point for item in evidence if (point := _sql_point(item)) is not None]
    if not sql_points:
        return ()

    conflicts: list[Conflict] = []
    for item in evidence:
        if item.source_type != "DOCUMENT":
            continue
        for claim in _document_points(item, catalog=catalog):
            for base in sql_points:
                if claim.metric_code != base.metric_code:
                    continue
                # 判据见模块 docstring 的那张表。**"多维度列 = 明细行"那一条
                # 在 `_document_points` 里判**——它说的是"表格有几个维度列"，
                # 与文档级范围无关，放在这里会连文档级范围一起数进去。
                if not _comparable(claim, base):
                    continue
                if not _same_period(claim, base):
                    continue
                difference = _difference(claim.value, base.value)
                if difference is None:
                    continue
                ratio = abs(claim.value - base.value) / abs(base.value) if base.value else 0.0
                conflicts.append(
                    _build_conflict(item, base, claim, difference, difference_ratio=ratio)
                )
    return tuple(conflicts)


class _Point:
    """一个可比较的数值点（`(指标, 维度集合, 数值)`）。

    `scope` 是**维度字典**而不是单个字符串：可比性取决于两边的维度对不对得上
    （见 `_comparable`），而单个字符串表达不了"这一行同时被区域、渠道、
    产品线限定"。把三个维度里最早出现的那个当成范围，正是那个 93% 假冲突的来源。
    """

    __slots__ = ("evidence_id", "matched_by", "metric_code", "period", "scope", "value")

    def __init__(
        self,
        evidence_id: str,
        metric_code: str,
        scope: dict[str, str],
        value: float,
        matched_by: str,
        period: str | None = None,
    ) -> None:
        self.evidence_id = evidence_id
        self.metric_code = metric_code
        self.scope = scope
        self.value = value
        #: **统计期间**记号（`2025-Q3`），`None` 表示**不知道**。
        #: 未知不等于"不限期间"——两者在 `_same_period` 里处置相反。
        self.period = period
        #: `exact`（表头就是指标名）或 `alias`（命中了别名）。
        #: **它决定了这条冲突有多硬**，见模块 docstring 的陷阱说明。
        self.matched_by = matched_by


def _sql_point(item: Evidence) -> _Point | None:
    """SQL 证据 → 数值点。

    **数值从 `claim` 里解析**，而 `claim` 的格式由
    `tools/sql/evidence._render_row` 决定（`列=值；列=值`）。
    这是一个**跨模块的格式耦合**，写在这里是因为没有别的载体：
    把原始行放进 `locator` 会违反 10.6「SQL 结果不默认持久化原始行」，
    而 `locator` 是会被 `answer_payload` 带出去的。
    两侧任何一方改格式，`_render_row` 的那条断言会先炸——
    所以格式的稳定性由那条断言把着，不靠这里的注释。
    """
    if item.source_type != "SQL" or not item.metric_code:
        return None
    pairs = dict(part.split("=", 1) for part in item.claim.split("；") if "=" in part)
    raw = next((value for key, value in pairs.items() if key.strip() == item.metric_code), None)
    if raw is None:
        # 指标值不在 claim 里（例如列名与 metric_code 不同名）：**跳过而不是猜**。
        # 猜一个值去比对，得到的冲突比不报更糟——它看起来是被算出来的。
        return None
    value = _to_float(raw)
    if value is None:
        return None
    return _Point(
        item.id,
        item.metric_code,
        _sql_dimensions(item),
        value,
        "exact",
        period=_sql_period(item),
    )


def _sql_period(item: Evidence) -> str | None:
    """SQL 证据的**统计期间**：`event_time` 正好是一个规范跨度时给出记号。

    归不出来就返回 `None`（**期间未知**），而未知不等于"不限期间"。
    归不出来是常态、不是异常：

    - `order_date >= '2025-01-01' AND < '2025-10-01'`（九个月）——
      题目要的就是九个月，它不对应任何一种记号；
    - **同比查询**：`_extract_time_range` 取的是 WHERE 里全部时间谓词的
      min/max 包络，两个 `CASE WHEN` 各一个季度，包出来是 15 个月；
    - 压根没有时间条件。

    **不要为了"总能给出一个期间"去硬凑**：凑出来的记号会让一条跨期的比对
    照常发生，而它看起来完全正常（同 `_comparable` 里"空 scope 是不知道、
    不是没有"那条）。
    """
    if item.event_time is None:
        return None
    return canonical_period(item.event_time.start.date(), item.event_time.end.date())


def _same_period(claim: _Point, base: _Point) -> bool:
    """两边的数是不是**同一期**的数（13.4 第 5 步 TIME 的判据）。

    ## 取「相等」而不是「相交」

    表格里那一行是它那个期间的**合计**——跨期间不可比。按"相交"判的话
    《上半年经营回顾》《年度经营分析》《下半年经营展望》全都包含 8 月，
    照样会放行，而它们说的是三个不同的期间。实测（2026-09-22）：
    这一条缺失时，2025 年 8 月的库值被拿去对上面四张表，
    报出 4 条 **157%–1133%** 的假冲突（只有 1 条是真的）。

    ## 任一侧未知 → 放行

    与 `_comparable` 同一条取舍：`None` 是"**不知道它的期间**"，
    不是"它不限期间"。按"不知道就不比"处理会丢掉真冲突，
    而漏报看不出来。**宁可多报。**

    ⚠️ 放行的代价是：**没有时间条件的查询**（`period=None`）与任何报告
    都照比。要收紧它得先让 SQL 侧总能说清自己的期间，而那是生成侧的事
    （见本模块 docstring 的「四类为什么不做」）。
    """
    if claim.period is None or base.period is None:
        return True
    return claim.period == base.period


def _sql_dimensions(item: Evidence) -> dict[str, str]:
    """SQL 证据的粒度：`scope` 里除 `data_scope` 之外的键值。

    **`data_scope` 要排除**：它不是维度，而是"这次查询受过的限制"标注
    （见 `tools/sql/evidence._scope_of`）。把它算进去会让每个受限用户的
    证据都多出一个"维度"，于是与任何文档表格都对不上——全部跳过，
    而那表现为"冲突检测突然不工作了"，不指向这里。
    """
    return {key: str(value) for key, value in item.scope.items() if key != "data_scope"}


def _comparable(claim: _Point, base: _Point) -> bool:
    """文档的某一**行**与 SQL 的某一个**点**是不是同一个总体。

    ## 两个方向各遍历一次，而它们不是同一件事

    - **文档侧的键 → SQL 侧查得到且相等**（约定 41 立的规矩）：文档行若比
      SQL 点多一个维度，它是 SQL 结果的**明细**，拿一行去对汇总是假冲突；
    - **SQL 侧的键 → 文档侧也要有且相等**（本判据补的那一半）：反过来，
      SQL 点比文档行多一个维度时它更细，两者比的是**两个不同的总体**。

    第二半是实测补上的（2026-09-22）：文档「产品线表现」表是**全公司**口径
    （`{product_line: 智能家居}`），SQL 查的是**华东**
    （`{region: 华东, product_line: 智能家居}`）。只判第一半时，
    `any(...)` 对 SQL 侧多出来的 `region` 键一无所知，判为可比，
    报出相对差 **499.55%** 的假冲突。

    ## 任一侧为空 → 放行，这是**有意**的

    `SELECT SUM(net_amount) WHERE region_name='华东'` 这类标量聚合的结果集
    只有一个聚合列，`_scope_of` 从列名里取不到 `region`，粒度只存在于 SQL
    文本里。空 `scope` 是"**我们不知道它的粒度**"，不是"它没有粒度"——
    按"不知道就不比"处理，`demo-cross` 那条真冲突会一起消失，
    而那条正是这个双源切片存在的理由。**判据不能写成"两边维度集合相等"**：
    那是**试过并回退过**的写法
    （见 `test_an_unscoped_sql_aggregate_still_compares`）。

    代价如实记下：SQL 侧范围真未知时，文档的每一行都会被比对
    （`test_without_a_sql_side_scope_every_row_still_compares` 钉住 5 条）。
    **宁可多报**——多报看得出来（读者会问"华南这行凭什么对华东的库值"），
    漏报看不出来。

    ⚠️ 这条放行**两个方向都适用**，因此它挡不住下面这一种情形：文档行
    与 SQL 点**都不是空的、但维度互不相干**（一个只有 `product_line`、
    另一个只有 `region`）。那要靠"SQL 侧多出的键在文档侧也得有"来判——
    而那正是本函数存在的理由，它只在这条放行之后生效。
    """
    if not claim.scope or not base.scope:
        return True
    return all(claim.scope.get(k) == v for k, v in base.scope.items()) and all(
        base.scope.get(k) == v for k, v in claim.scope.items()
    )


def _document_points(item: Evidence, *, catalog: Any) -> list[_Point]:
    """文档证据 → 数值点列表（一行一个）。

    只认**表格行**：表头给出列名与单位，维度列给出范围，三者缺一不可。
    正文里的数字（「净销售额11,039.58 万元」）**不解析**——
    从自由文本里取数需要判断"这个数说的是哪个指标"，那是语义，
    纯正则做不了，硬做出来的是一堆看起来像冲突的噪声。
    """
    lines = [line for line in (item.claim or "").splitlines() if _COLUMN_SEPARATOR in line]
    if len(lines) < 2:  # 表头 + 至少一行数据
        return []
    header = [cell.strip() for cell in lines[0].split(_COLUMN_SEPARATOR)]
    dimension_indexes = _dimension_indexes(header)
    columns = [_column_of(name, catalog=catalog) for name in header]
    if all(column is None for column in columns):
        return []
    columns = _prefer_exact(columns)

    #: **文档级范围**（《华东区域2025年第三季度专项分析》 → `{region: 华东}`）。
    #: 它记在**文档身份**上而不在表头里，却决定了"这个表头说的是哪个总体"
    #: ——同一个「分产品线」表头，全公司报告与区域报告说的是两个总体。
    document_scope = {key: str(value) for key, value in item.scope.items()}

    points: list[_Point] = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split(_COLUMN_SEPARATOR)]
        # **整行的维度一起取**，不只是第一个：见模块 docstring
        row_scope = {
            key: cells[index] for index, key in dimension_indexes.items() if index < len(cells)
        }
        # **「这一行跨了几个维度」只看表格的维度列**，不含文档级范围——
        # 后者说的是"这份文档的范围"，不是"这一行比按一个维度汇总更细"。
        # 把两者混起来数，区域报告里那张单维度表会被误判成明细行而整张跳过。
        if len(row_scope) > 1:
            continue
        # 行取值优先于文档级范围（两者本该一致；不一致时行说的是更具体的那个，
        # 与 `tools/sql/evidence._scope_of` 的取舍相同）
        scope = {**document_scope, **row_scope}
        for position, column in enumerate(columns):
            if column is None or position >= len(cells):
                continue
            metric_code, unit, matched_by = column
            value = _to_float(cells[position])
            if value is None:
                continue
            # **单位只对非维度列生效**。维度列里也可能有数字（`省份数` 一列），
            # 但那一列的列名不是指标名，上面的 `_column_of` 已经把它滤掉了。
            points.append(
                _Point(
                    item.id,
                    metric_code,
                    scope,
                    value * unit,
                    matched_by,
                    period=canonical_token(item.stat_period) if item.stat_period else None,
                )
            )
    return points


def _prefer_exact(
    columns: list[tuple[str, float, str] | None],
) -> list[tuple[str, float, str] | None]:
    """同一张表里，某个指标**既有精确名列又有别名列时，别名列让位**（置 None）。

    **这是那个陷阱的正面防线**。语料里的报表同时有这两列：

        区域 | 销售额（万元） | 净销售额（万元） | ...

    而 `net_sales` 的别名表里**包含「销售额」**（它是含税口径，见目录的 note）。
    不处理的话，两列都会映射到 `net_sales`，于是同一张表对同一个 SQL 数字
    报出**两条冲突**——其中一条是假的（拿含税口径去比净额口径），
    而它和真冲突长得一模一样。

    规则是"精确名优先"而不是"别名一律不认"：只认精确名会让那些
    表头写法与目录不一致的文档（写得对、只是用了别名）永远比不了；
    两者并存时才让位——那时精确名显然是作者想表达的那个指标。
    """
    exact_metrics = {item[0] for item in columns if item is not None and item[2] == "exact"}
    return [
        None if (item is not None and item[2] == "alias" and item[0] in exact_metrics) else item
        for item in columns
    ]


def _column_of(header: str, *, catalog: Any) -> tuple[str, float, str] | None:
    """表头单元格 → `(metric_code, 单位换算, 命中方式)`；不是指标列则 None。

    匹配顺序是**精确名 → 别名**，理由见模块 docstring 的陷阱说明。
    """
    unit = 1.0
    name = header
    suffix = _UNIT_SUFFIX.search(header)
    if suffix is not None:
        unit = _UNITS.get(suffix.group(1), 1.0)
        name = header[: suffix.start()].strip()
    if not name:
        return None

    exact = [metric for metric in catalog.metrics if metric.name == name]
    if exact:
        return exact[0].code, unit, "exact"
    for metric in catalog.metrics:
        if name in (metric.aliases or ()):
            return metric.code, unit, "alias"
    return None


#: 维度列名 → 与 SQL 证据 `scope` 的键对齐。与 `tools/sql/evidence._SCOPE_COLUMNS`
#: 的值域相同——**这些字符串是两个模块的契约**，一边改了另一边要跟着改。
_DIMENSION_COLUMNS: dict[str, str] = {
    "区域": "region",
    "渠道": "channel",
    "产品线": "product_line",
    "品类": "category",
}


def _dimension_indexes(header: Sequence[str]) -> dict[int, str]:
    """表头里的**全部**维度列：`{列下标: scope 键}`。

    返回全部而不是第一个，理由见模块 docstring——第一版只取第一个，
    于是「区域 | 渠道 | 产品线 | …」的明细行被当成了区域合计。
    """
    return {
        index: _DIMENSION_COLUMNS[name.strip()]
        for index, name in enumerate(header)
        if name.strip() in _DIMENSION_COLUMNS
    }


def _to_float(text: str) -> float | None:
    match = _NUMBER.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:  # pragma: no cover - 正则保证了可解析
        return None


def _difference(left: float, right: float) -> dict[str, float] | None:
    """两个数是否**超出容差**；没超返回 None（即不构成冲突）。

    容差取「绝对 1 元」与「相对 0.1%」中较大者（13.4 第 4 步）。
    """
    delta = abs(left - right)
    tolerance = max(_ABS_TOLERANCE, abs(right) * _REL_TOLERANCE)
    if delta <= tolerance:
        return None
    ratio = delta / abs(right) if right else 0.0
    return {"delta": delta, "ratio": ratio, "tolerance": tolerance}


def _build_conflict(
    document: Evidence,
    base: _Point,
    claim: _Point,
    difference: dict[str, float],
    *,
    difference_ratio: float,
) -> Conflict:
    scope = (
        ("，范围 " + "、".join(f"{k}={v}" for k, v in claim.scope.items())) if claim.scope else ""
    )
    matched = (
        "表头与指标名精确相同"
        if claim.matched_by == "exact"
        else "表头命中的是**别名**（别名可能跨口径，可信度较低）"
    )
    return Conflict(
        id=new_id(IdPrefix.CONFLICT),
        type=ConflictType.VALUE,
        evidence_ids=(document.id, base.evidence_id),
        # **WARNING 而不是 BLOCKING**：这一版判不出谁对（见模块 docstring），
        # 而 BLOCKING 的语义是"必须处置才能继续"，那需要依据
        severity=ConflictSeverity.WARNING,
        description=(
            f"《{document.title}》的{claim.metric_code}为 {claim.value:,.2f}，"
            f"而业务数据库查得 {base.value:,.2f}"
            f"（差 {difference['delta']:,.2f}，相对 {difference_ratio:.2%}）{scope}"
        ),
        detected_difference={
            "metric_code": claim.metric_code,
            "document_value": claim.value,
            "database_value": base.value,
            "delta": difference["delta"],
            "ratio": round(difference_ratio, 6),
            "tolerance": difference["tolerance"],
            "matched_by": claim.matched_by,
            "scope": claim.scope,
            # **SQL 侧的粒度也要进明细**：判"可比"要两个方向的维度都对上，
            # 而原先只记了文档侧，于是"凭什么拿这一行对那个库值"在产物上
            # 看不出来——那条 499.55% 的假冲突正是这么溜过去的（`_comparable`）。
            "database_scope": dict(base.scope),
        },
        possible_explanations=(
            "口径不同：报告中的「销售额」按含税 − 折扣列示、未扣退货冲减，"
            "与净销售额的差额恰为退货金额（见指标目录 net_sales 的 note）",
            "统计时点不同：报告的统计截止日可能早于本次查询的区间末",
            "范围不同：报告的区域范围与查询的区域口径可能不一致",
            f"匹配可信度：{matched}",
        ),
        # **默认 UNRESOLVED**：检测器只说"发现了不一致"，
        # 谁对谁错要 Reviewer 按 13.2 的证据优先级判（14.3 的 resolution）
        resolution=ConflictResolution.UNRESOLVED,
        selected_basis=None,
    )


def render(conflicts: Sequence[Conflict]) -> str:
    """冲突 → 给模型看的文本（`analysis` 节点用它）。"""
    if not conflicts:
        return "（未检出冲突）"
    lines: list[str] = []
    for index, item in enumerate(conflicts, start=1):
        lines.append(f"[C{index}] {item.description}")
        lines.append(f"  严重度：{item.severity.value}｜状态：{item.resolution.value}")
        for explanation in item.possible_explanations:
            lines.append(f"  - 可能原因：{explanation}")
    return "\n".join(lines)


__all__ = ["build_conflict_node", "detect_value_conflicts", "render"]
