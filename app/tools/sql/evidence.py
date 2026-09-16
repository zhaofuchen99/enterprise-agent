"""从 SQL 结果生成证据（详细设计 13.1 / 10.7 的「SqlToolResult 与 Evidence 生成」）。

**这是「答案有证据」这条能力在 SQL 侧的全部产出。** 到了 Analysis 与 Reviewer
那里，看到的只有 `Evidence`——它不关心这条证据来自一次 `SELECT` 还是某篇制度的
第 3 节。这正是 Phase 5 的 RAG 能往同一张表里写、而两条链路可以被并列比较的前提。

## 三条生成规则

1. **一行一条证据**。一条 SQL 结果里的每一行都是**彼此独立的事实**
   （「华东 12.3 亿」「华南 8.7 亿」），把它们揉成一条会让下游无法引用其中某一个数。
2. **超过上限退化成一条汇总**。查了 1000 行明细时逐行生成会炸出 1000 条证据，
   而它们对答案的贡献是等价的。上限见 `SQL_TOOL__MAX_EVIDENCE_ROWS`。
3. **空结果不生成证据**。0 行的唯一可靠推论是「按当前过滤条件查不到」，
   把它包装成一条 claim 会诱导下游说出「没有销售额」——而这更可能是
   过滤条件写错了（日期少算一年、区域名拼错）。**没有证据好过一条误导性的证据。**

## `claim` 为什么由代码渲染，而不是让模型写

详设 13.5 的 `SupportedClaim` 是 Analysis 的产物，模型在那里把证据组织成叙述。
Tool 这一层要交出去的是**原始事实**：`region_name=华东, net_sales=12345678.90`。
让模型在这里润色一遍，得到的是「华东销售额约 1.2 亿」这种**已经丢失精度**的转述，
而后面所有的一致性检查都要拿它去对账。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence, TimeRange
from app.domain.user import PermissionScope
from app.tools.sql.schemas import SchemaCatalog, SqlToolResult

#: 结果列名 -> 证据 `scope` 的键（详设 13.1 的 `scope`、13.4 第 7 步的 SCOPE 冲突）。
#: 只认这几个列：scope 是给**跨来源比对**用的，把每列都塞进去会让
#: 「同口径」的判定被无关字段干扰，而 SCOPE 冲突恰恰是漏报比误报更糟的那类。
_SCOPE_COLUMNS: Final[dict[str, str]] = {
    "region_name": "region",
    "channel_name": "channel",
    "product_line_name": "product_line",
    "category": "category",
    "customer_level": "customer_level",
}

#: 敏感级别 -> 证据的 `access_level`（TBC-07 的两级 + PUBLIC）。
_ACCESS_LEVEL: Final[dict[str, str]] = {
    "PUBLIC": "PUBLIC",
    "INTERNAL": "INTERNAL",
    "SENSITIVE": "CONFIDENTIAL",
}

#: 单条 claim 最多渲染多少个「列=值」。超出部分用省略号收尾——
#: 一份 30 列的结果渲染成一行文本既读不了也占 token。
_MAX_CLAIM_FIELDS: Final[int] = 12


def build_evidence(
    result: SqlToolResult,
    *,
    question: str,
    scope: PermissionScope,
    catalog: SchemaCatalog,
    max_rows: int,
) -> list[Evidence]:
    """把一次成功的查询结果转成证据。

    Args:
        result: Tool 产出的结果。**空结果返回空列表**，见模块 docstring。
        question: 触发这次查询的业务问题，进 `title` 便于人工核对。
        scope: 调用者的数据权限范围。用于在证据里标注本次查询受过的限制——
            「这个数是全量还是只有华东」必须是证据自带的属性，
            否则 Phase 9 的冲突检测会把两个不同范围的事实当成同一个来比。
        catalog: 用于判定证据的密级（取所涉列的最高敏感级别）。
        max_rows: 逐行生成的上限。超过时退化成**一条覆盖整段切片**的汇总证据。

    Returns:
        证据列表。**结果为空时返回空列表**——包括「聚合查询返回一行 NULL」
        那种形态，见 `SqlToolResult.is_empty` 的说明。
    """
    if result.is_empty:
        return []

    columns = [column.name for column in result.columns]
    # 取**第一个**指标 code：一次查询通常只算一个指标，而 13.4 的分组是
    # 「按指标分组」——多个指标时把它们都塞进一个字段，冲突检测就分不了组了。
    # 一条 SQL 同时算两个不同口径的指标时，它们本就是两条可比性存疑的证据。
    metric_code = result.metric_codes[0] if result.metric_codes else None
    common = _Common(
        question=question,
        sql_fingerprint=result.sql_fingerprint,
        call_id=result.call_id,
        columns=columns,
        event_time=result.data_time_range,
        retrieved_at=datetime.now(UTC),
        metric_code=metric_code,
        definition_version=_definition_version(catalog, metric_code),
        access_level=_access_level(catalog, columns),
    )

    if result.row_count > max_rows:
        return [_summary_evidence(common, result, scope, max_rows)]
    return [_row_evidence(common, row, index, scope) for index, row in enumerate(result.rows)]


@dataclass(frozen=True)
class _Common:
    """逐行与汇总两条路径共用的字段。

    提出来是因为两条路径要拼的字段完全一样，只是 claim 与 locator 不同；
    各写一遍的结果一定是「改了这条忘了那条」，而其中一条会变得不可追溯。
    """

    question: str
    sql_fingerprint: str
    call_id: str
    columns: list[str]
    event_time: TimeRange | None
    retrieved_at: datetime
    metric_code: str | None
    definition_version: str | None
    access_level: str


def _row_evidence(
    common: _Common, row: Sequence[object], index: int, scope: PermissionScope
) -> Evidence:
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title=f"{common.question}（结果第 {index + 1} 行）",
        claim=_render_row(common.columns, row),
        locator={
            "call_id": common.call_id,
            "sql_fingerprint": common.sql_fingerprint,
            # 切片指向**原始结果**的行区间，而不是这条证据渲染了几个字段
            "result_slice": [index, index + 1],
            "columns": common.columns,
        },
        event_time=common.event_time,
        retrieved_at=common.retrieved_at,
        metric_code=common.metric_code,
        definition_version=common.definition_version,
        scope=_scope_of(common.columns, row, scope),
        # 详设 13.2 第 1 条：计算经营指标时，业务数据库的结果优先级最高。
        reliability="HIGH",
        content_hash=_content_hash(common.sql_fingerprint, common.columns, row),
        access_level=common.access_level,
    )


def _summary_evidence(
    common: _Common, result: SqlToolResult, scope: PermissionScope, max_rows: int
) -> Evidence:
    """行数超过上限时的汇总证据。

    `result_slice` 指向**整个结果**而不是被丢掉的那部分：这条证据的 claim
    已经明说「共 N 行，此处不逐行展开」，下游不会把它当成一行来用。
    截断本身也必须写在 claim 里——详设 10.7 的 `truncated` 是给程序看的，
    而 claim 是给人看的，两者都要能看出「这里不是全部」。
    """
    head = "；".join(_render_row(common.columns, row) for row in result.rows[:3])
    truncation = "，结果已截断" if result.truncated else ""
    remaining = max(0, result.row_count - 3)
    claim = f"共 {result.row_count} 行{truncation}；前 3 行：{head}；其余 {remaining} 行未逐行展开"
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="SQL",
        title=f"{common.question}（结果汇总，共 {result.row_count} 行）",
        claim=claim,
        locator={
            "call_id": common.call_id,
            "sql_fingerprint": common.sql_fingerprint,
            "result_slice": [0, result.row_count],
            "columns": common.columns,
        },
        event_time=common.event_time,
        retrieved_at=common.retrieved_at,
        metric_code=common.metric_code,
        definition_version=common.definition_version,
        # 汇总证据带不了单行的维度取值，只标数据范围——**不猜**某一行的 scope
        scope={"data_scope": list(scope.region_ids)} if not scope.unrestricted else {},
        reliability="HIGH",
        content_hash=_content_hash(
            common.sql_fingerprint, common.columns, [result.row_count, result.truncated]
        ),
        access_level=common.access_level,
    )


# ------------------------------------------------------------------ 内部实现
def _render_row(columns: Sequence[str], row: Sequence[object]) -> str:
    """把一行渲染成「列=值；列=值」的确定性文本。"""
    pairs = [f"{name}={value}" for name, value in zip(columns, row, strict=False)][
        :_MAX_CLAIM_FIELDS
    ]
    suffix = "；…" if len(columns) > _MAX_CLAIM_FIELDS else ""
    return "；".join(pairs) + suffix


def _scope_of(
    columns: Sequence[str], row: Sequence[object], scope: PermissionScope
) -> dict[str, str | list[str]]:
    """从结果行提取可比对的维度范围。

    受限用户带上 `data_scope` 标注：同样是「华东销售额」，
    全量用户算出的数与限华东用户算出的数**本就该不同**，
    不标注的话，两个来源的数字一旦不同就会被报成 VALUE 冲突。
    """
    extracted: dict[str, str | list[str]] = {}
    for name, value in zip(columns, row, strict=False):
        key = _SCOPE_COLUMNS.get(name)
        if key is not None and value is not None:
            extracted[key] = str(value)
    if not scope.unrestricted:
        extracted["data_scope"] = list(scope.region_ids)
    return extracted


def _access_level(catalog: SchemaCatalog, columns: Sequence[str]) -> str:
    """证据密级 = 所涉列的**最高**敏感级别（TBC-07）。

    按「最高」而不是「平均」或「主键所在表」：一条证据的密级应当由它包含的
    最敏感的那部分决定，否则把敏感列和普通列一起查就能拿到一份低密级的证据。
    """
    highest = 0
    order = ("PUBLIC", "INTERNAL", "SENSITIVE")
    for table in catalog.tables:
        for column in table.columns:
            if column.name in columns:
                highest = max(highest, order.index(column.sensitive_level))
    return _ACCESS_LEVEL[order[highest]]


def _definition_version(catalog: SchemaCatalog, metric_code: str | None) -> str | None:
    """指标口径版本（详设 13.1 的 `definition_version`）。

    口径版本不同即 DEFINITION 冲突（13.4 第 5 步），因此它必须随证据落下来——
    等到冲突检测时再去查目录，查到的是「今天」的版本，
    而证据是「当时」按哪个版本算出来的就永远说不清了。
    """
    if metric_code is None:
        return None
    metric = catalog.metric(metric_code)
    return metric.version if metric else None


def _content_hash(sql_fingerprint: str, columns: Sequence[str], row: Sequence[object]) -> str:
    """证据内容摘要。

    用 SQL 指纹而不是整条 SQL：同一条查询换个日期就是另一条 SQL，
    但它们指向的是**同一类事实**，13.4 第 1 步的「按主题分组」正是要这个粒度。
    """
    payload = "|".join([sql_fingerprint, ",".join(columns), ",".join(str(v) for v in row)])
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = ["build_evidence"]
