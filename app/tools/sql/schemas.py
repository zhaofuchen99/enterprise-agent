"""SQL Tool 的边界模型（详细设计 10.2 / 10.3 / 10.7）。

分三组：

| 组 | 模型 | 来源 |
|---|---|---|
| 目录 | `ColumnSpec` / `JoinSpec` / `TableSpec` / `MetricSpec` | 10.2，YAML 加载 |
| 生成 | `SqlCandidate` | 10.3，**模型输出，必须先经这里校验才能进 State** |
| 结果 | `ResultColumn` / `SqlToolResult` / `SqlAttempt` | 10.7 |

**`SqlCandidate` 是 `SqlCandidate` 而不是裸 dict**：开发流程 5.3 禁止把模型输出
直接写进 LangGraph State。这里的字段声明就是那道校验——`selected_tables`
之类字段模型写错了，会在这里被 Pydantic 拒绝，而不是在生成 SQL 报错时
以「模型乱写 SQL」的面目出现。

**`SqlAttempt` 的形状是照着 `agent_tool_call` 表的列定的**：详设 6.6 施工项 5
要求「每次尝试写入 agent_tool_call」，但持有 `task_id` / `step_id` 的 Tool 执行器
在 Phase 6/7 才出现。因此 Phase 4 的 Tool **把每次尝试组装成这个对象返回**，
执行器拿到后逐条落库即可，Tool 侧不依赖任务仓储——
`app/tools/**` 本来也不该 import `repositories` 去写库（详设 4.3 的依赖方向）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.evidence import TimeRange
from app.tools.base import ErrorClass

#: 列的敏感级别（详设 10.2 的 `ColumnSpec.sensitive_level`）。
#: `SENSITIVE` 的列只有 `allowed_roles` 里的角色能查——这是 10.4 第 7 步的判定依据。
SensitivityLevel = Literal["PUBLIC", "INTERNAL", "SENSITIVE"]

#: JOIN 基数。目前只用于 prompt 提示与文档，**不参与校验**：
#: 校验只问「这一对表允许不允许 JOIN」，基数写错不会造成越权，只会让模型
#: 在聚合时选错粒度，而那是 SQL 正确性问题，由金标评测覆盖。
Cardinality = Literal["many_to_one", "one_to_many", "one_to_one", "many_to_many"]

#: 一次尝试的结局。`REJECTED` 与 `FAILED` 分开：前者是校验拦下的
#: （没碰数据库），后者是执行失败的（碰了）。这两件事的排查方向完全不同。
AttemptStatus = Literal["SUCCEEDED", "FAILED", "REJECTED"]

#: 这一次尝试的**结局发生在哪个环节**：
#:
#: - `GENERATE`：首次生成的 SQL 没通过校验，被拦在生成之后；
#: - `REPAIR`：修复后重生成的 SQL 仍没通过校验；
#: - `EXECUTE`：SQL 通过了全部校验，这一次的成败由执行决定。
#:
#: 语义是「判定发生在哪」而不是「这条 SQL 从哪来」——后者可以从
#: `attempt_no > 1` 直接读出来，而前者（生成就错了，还是执行才发现错）
#: 决定了要不要继续修，是排查时真正要看的信息。
AttemptStage = Literal["GENERATE", "REPAIR", "EXECUTE"]


def has_no_values(row_count: int, rows: Sequence[Sequence[Any]]) -> bool:
    """结果里是否**没有任何有内容的值**（详设 9.4 的 `EMPTY_RESULT`）。

    判据是「零行」**或**「每一行、每一列都是 NULL」。后者不是宽容，而是必须的：
    聚合查询在没有匹配行时返回的是**一行 NULL**，不是零行——

    ```sql
    SELECT SUM(net_amount) FROM fact_sales_order_item WHERE region_name = '华南'
    -- 无匹配时返回 1 行，值为 NULL
    ```

    只看「零行」会把它当成「有一行结果」，于是工具会为 `net_sales=None`
    生成一条证据、也不给用户任何提示——而真实情况是「按当前条件查不到数据」。
    受限用户查别的区域时尤其关键：那本该是一个明确的空结果，
    而不是一个看起来像有值的 NULL。

    定义成模块级函数而不是两处各写一遍属性：执行结果与工具结果都要判这件事，
    两处判据一旦漂移，就会出现「执行器认为空、工具认为不空」这类
    只在一部分分支上显形的错。
    """
    if row_count == 0:
        return True
    return all(value is None for row in rows for value in row)


# --------------------------------------------------------------------- 目录
class ColumnSpec(BaseModel):
    """目录里的一列（详设 10.2）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    data_type: str
    description: str
    sensitive_level: SensitivityLevel = "PUBLIC"
    #: 可取该列的角色。**空表示不限制**（而不是「谁都不能取」）——
    #: 若取「空即禁止」，目录里漏写一行就会变成静默拒绝，
    #: 而漏写 allowed_roles 是常态（绝大多数列都是 PUBLIC）。
    allowed_roles: tuple[str, ...] = ()

    def visible_to(self, role: str) -> bool:
        return not self.allowed_roles or role in self.allowed_roles


class JoinSpec(BaseModel):
    """允许的 JOIN 关系（详设 10.2）。

    `condition` 保留为一条 SQL 文本而不是拆成 (左列, 右列)：它在 prompt 里要
    原样展示给模型，拆开再拼回来会出现「目录里写的」与「模型看到的」不一致。
    校验时按**表对**判定，不比对字符串（详设 10.4 第 8 步要的是
    「关系属于允许集合」，不是「写法与目录逐字相同」）。

    **字段名为什么不叫 `on`**：YAML 1.1 把 `on` / `off` / `yes` / `no` 当布尔值，
    写作映射键时会被解析成 `True`——目录能加载，但 `on` 字段永远读不到，
    报错信息还是「Field required」。改名比在每个编辑者头上悬一把刀便宜。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    to: str
    condition: str
    cardinality: Cardinality = "many_to_one"


class TableSpec(BaseModel):
    """目录里的一张表（详设 10.2）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    grain: str
    #: 时间列。同比环比一律按它切；`None` 表示该表没有业务时间轴（纯维度表）。
    time_column: str | None = None
    #: 数据权限谓词的落点列（详设 10.4 第 11 步）。`None` 表示该表不按区域划分。
    scope_column: str | None = None
    columns: tuple[ColumnSpec, ...]
    joins: tuple[JoinSpec, ...] = ()

    def column(self, name: str) -> ColumnSpec | None:
        return next((c for c in self.columns if c.name == name), None)

    def joined_tables(self) -> frozenset[str]:
        return frozenset(j.to for j in self.joins)


class MetricSpec(BaseModel):
    """指标目录的一条（详设 10.2）。

    `aliases` 是 `SchemaProvider` 做**确定性**指标匹配的依据（10.2
    「先匹配指标目录」）：命中别名就把这条指标连同它的口径表达式一起
    送进 prompt。这一步刻意不交给模型判断——口径漏带的表现是
    「模型自己发明了一套算法」，而它看起来完全正常。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    name: str
    expression: str
    unit: str
    grain: str
    aliases: tuple[str, ...] = ()
    required_filters: tuple[str, ...] = ()
    owner: str = ""
    version: str = "v1.0"
    note: str = ""

    def keywords(self) -> tuple[str, ...]:
        """参与匹配的词。名称本身也算——`aliases` 漏写名称是最常见的目录疏忽。"""
        return tuple({self.name, *self.aliases})


class ScopeResolution(BaseModel):
    """数据权限取值（区域名称）到 `scope_column` 取值的解析路径（详设 10.4 第 11 步）。

    声明在目录里而不是写死在代码中：换一套权限维度（按产品线、按渠道）时
    改的是 YAML，不是校验器。详见 `configs/schema_catalog.yaml` 的 `scope:` 段。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    match_column: str
    key_column: str


class ScopeSpec(BaseModel):
    """数据权限的范围定义。

    `resolve` 为 `None` 表示权限取值就是表上 `scope_column` 的字面值，
    不需要解析——目前没有这种表，但保留这个取值是因为「权限维度直接落在
    表上」是完全合理的一种配置，那时不该被迫写一个指向自己的解析表。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolve: ScopeResolution | None = None


class SchemaCatalog(BaseModel):
    """整份 Schema 目录（详设 16.9 的 `schema_catalog` 内容，表结构后置）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    tables: tuple[TableSpec, ...]
    metrics: tuple[MetricSpec, ...] = ()
    allowed_functions: frozenset[str] = frozenset()
    scope: ScopeSpec | None = None

    def table(self, name: str) -> TableSpec | None:
        return next((t for t in self.tables if t.name == name), None)

    def metric(self, code: str) -> MetricSpec | None:
        return next((m for m in self.metrics if m.code == code), None)

    def match_metrics(self, text: str) -> tuple[MetricSpec, ...]:
        """按别名在问题文本里做包含匹配。

        按命中词的**长度倒序**返回，长词优先是必要的：「目标销售额」
        同时命中「销售额」与「目标销售额」两条指标，先匹配短的会把
        `net_sales` 也带进来，prompt 里就出现两个口径解释同一个词。
        """
        hits = [m for m in self.metrics if any(k in text for k in m.keywords())]
        return tuple(sorted(hits, key=lambda m: -max(len(k) for k in m.keywords() if k in text)))

    def join_allowed(self, left: str, right: str) -> bool:
        """两张表之间是否存在目录声明的 JOIN 关系（方向无关）。

        方向无关是刻意的：目录按「事实表 → 维度表」声明，而模型写
        `FROM dim_region JOIN fact_sales_order_item` 在 SQL 里完全等价，
        按方向判定会把一个正确的查询拒之门外。
        """
        pair = {left, right}
        for table in self.tables:
            if table.name not in pair:
                continue
            other = next(iter(pair - {table.name}), None)
            if other and other in table.joined_tables():
                return True
        return False


class SchemaContext(BaseModel):
    """一次生成实际注入的表与指标（详设 10.2 的 `SchemaContext`）。

    `omitted_tables` **不是截断日志，而是给调用方的信号**：10.2 要求
    「超过 8 张表时应重新拆分问题，而不是把全库 Schema 发给模型」。
    Phase 4 的 Tool 没有拆问题的能力（那是 Planner 的事），因此它把
    「还有哪些表没进来」如实带出来，由上层决定重试还是拆解。
    """

    model_config = ConfigDict(frozen=True)

    version: str
    tables: tuple[TableSpec, ...]
    metrics: tuple[MetricSpec, ...] = ()
    omitted_tables: tuple[str, ...] = ()
    matched_metric_codes: tuple[str, ...] = ()

    def render(self) -> str:
        """渲染成 prompt 里的 Schema 段。

        用紧凑的缩进文本而不是 JSON：模型读 Schema 只需要「表 → 列 → 类型」，
        而 JSON 的引号与括号在长 Schema 下会显著增加 token，且不带来信息。
        """
        lines: list[str] = [f"Schema 版本：{self.version}", "", "可用表："]
        for table in self.tables:
            time_hint = f"｜时间列 {table.time_column}" if table.time_column else ""
            lines.append(f"- {table.name}（{table.description}；粒度：{table.grain}{time_hint}）")
            for column in table.columns:
                lines.append(f"    · {column.name} {column.data_type} — {column.description}")
            for join in table.joins:
                lines.append(f"    ↳ 允许 JOIN {join.to}：{join.condition}")
        if self.metrics:
            lines.extend(["", "指标口径（必须按此计算，不得自行发明算法）："])
            for metric in self.metrics:
                filters = "、".join(metric.required_filters)
                lines.append(
                    f"- {metric.name}({metric.code}) = {metric.expression}"
                    f"｜单位 {metric.unit}｜粒度 {metric.grain}"
                    + (f"｜必须过滤：{filters}" if filters else "")
                    + f"｜口径版本 {metric.version}"
                )
                if metric.note:
                    lines.append(f"    注：{metric.note}")
        if self.omitted_tables:
            lines.extend(
                [
                    "",
                    "以下表因超出上下文上限未提供，**不要引用它们**："
                    + "、".join(self.omitted_tables),
                ]
            )
        return "\n".join(lines)


# --------------------------------------------------------------------- 生成
class SqlCandidate(BaseModel):
    """模型生成的候选 SQL（详设 10.3）。

    `explanation` **只说明业务口径，不保存思维链**（详设 10.3 原文）。
    它进 Trace 的是摘要而非全文，理由同 19.4 的脱敏纪律。
    """

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(min_length=1)
    #: 命名绑定参数。日期以 ISO 字符串给出（Pydantic 的 smart 模式会保留 str，
    #: 不会把 `2025-07-01` 悄悄转成 date 对象再变一次形状）。
    parameters: dict[str, str | int | float | date] = Field(default_factory=dict)
    selected_tables: tuple[str, ...] = ()
    selected_columns: tuple[str, ...] = ()
    metric_codes: tuple[str, ...] = ()
    expected_columns: tuple[str, ...] = ()
    explanation: str = ""


# --------------------------------------------------------------------- 校验
class ValidatedSql(BaseModel):
    """通过 12 步校验、可以被执行的 SQL（详设 10.4 的输出）。

    `normalized_sql` **才是被执行的那一份**——不是模型吐出的原文。
    重写（补 LIMIT、注入权限谓词）只能通过重写后的 SQL 生效，
    执行原文等于把第 10、11 两步的成果丢掉。
    """

    model_config = ConfigDict(frozen=True)

    validated_sql: str
    normalized_sql: str
    sql_fingerprint: str
    #: 执行时要用到的**全部**绑定参数：模型给的 + 校验器注入权限谓词时加的。
    #: 由校验器一并产出，而不是让调用方自己把两部分拼起来——
    #: 拼漏了会以「MissingBindParameter」的形式在数据库层报出来，
    #: 而那个错误看起来像 SQL 写错了，排查方向会跑偏。
    bind_parameters: dict[str, object] = Field(default_factory=dict)
    #: 涉及的表与列（校验过程中收集，供 Evidence 定位与密级判定）
    used_tables: tuple[str, ...]
    used_columns: tuple[str, ...]
    #: 本次重写做了什么。空列表表示模型写的 SQL 原样通过。
    rewrites: tuple[str, ...] = ()
    #: 是否注入了数据权限谓词。**必须可观测**：静默注入会让
    #: 「为什么我只看到华东的数据」变成一个查不出来的谜。
    scope_injected: bool = False
    data_time_range: TimeRange | None = None


# --------------------------------------------------------------------- 结果
class ResultColumn(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    data_type: str


class SqlAttempt(BaseModel):
    """一次生成/修复/执行尝试，形状对齐 `agent_tool_call` 表（详设 16.6）。

    `error_summary` **必须已脱敏**：入库前会经 19.4 的纪律检查，
    因此这里存的是「哪一类错、错在第几个字符」而非模型原文或数据库回显。
    """

    model_config = ConfigDict(frozen=True)

    attempt_no: int = Field(ge=1)
    stage: AttemptStage
    sql: str | None = None
    normalized_sql: str | None = None
    sql_fingerprint: str | None = None
    status: AttemptStatus
    error_code: str | None = None
    error_class: ErrorClass | None = None
    error_summary: str | None = None
    duration_ms: int | None = None


class SqlToolResult(BaseModel):
    """SQL Tool 的结果（详设 10.7）。

    `rows` 是**唯一**保存原始取值的字段，且只活在内存里——
    详设 10.6 明写「SQL 结果不直接写应用日志，也不默认持久化原始行」。
    落库的是 `summary` 与 `attempts`，不是这个对象本身。
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    normalized_sql: str
    sql_fingerprint: str
    columns: tuple[ResultColumn, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    truncated: bool
    duration_ms: int
    data_time_range: TimeRange | None = None
    #: **本次查询实际用到的指标 code**，供 Evidence 与冲突检测按指标分组（13.4 第 1 步）。
    #: 与下面的 `metric_definitions` 分开是必要的：那个是给人读的口径说明，
    #: 拿它去 `catalog.metric()` 查不到任何东西，而查不到的表现是
    #: 「证据的 `definition_version` 静默为 None」——不报错，只是 DEFINITION
    #: 冲突从此再也检不出来。
    metric_codes: tuple[str, ...] = ()
    metric_definitions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    #: 本次查询**实际**被服务端谓词限定到的区域（详设 10.4 第 11 步）。
    #:
    #: `None` 表示没有施加任何限制，两种情况都归到它：调用者本就不受限，
    #: 或这条查询没落到带 `scope_column` 的表上。两者对下游是同一件事——
    #: **空结果不能归因于数据权限**。空元组不会出现：限定成"零个区域"
    #: 不是一种权限。
    #:
    #: 有值时下游必须把它写进限制。缺了它，`has_no_values` 判出来的
    #: 「查询未命中任何数据」会被读成"公司没有这个数据"，而真相是
    #: "你看不到这个数据"——两者处置相反：前者要换数据源，后者要找
    #: 数据负责人。⚠️ `warnings` 里那句「已按当前账号的数据权限限定
    #: 查询范围」不能替代它：那是给人读的提示串，而这是可判断的字段。
    data_scope: tuple[str, ...] | None = None
    #: 本次查询的全部尝试，供 Tool 执行器逐条写 `agent_tool_call`
    attempts: tuple[SqlAttempt, ...] = ()
    #: 生成 SQL 时使用的 Schema 目录版本，进 Trace 便于「昨天还能查今天不行」的排查
    schema_version: str = ""

    @property
    def is_empty(self) -> bool:
        """没有任何有内容的行（详设 9.4 的 `EMPTY_RESULT`）。

        **不算失败**：SQL 正确执行了、只是没有匹配的行。是否要补证、
        澄清还是受限回答，由 Reviewer 判断（9.4 原文），Tool 不替它决定。
        判据见模块级的 `has_no_values`。
        """
        return has_no_values(self.row_count, self.rows)


class SqlQueryArgs(BaseModel):
    """`sql_query` 的入参（详设 9.1 的 `BaseTool.execute(args, ctx)`）。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    #: 该步骤的分析目标，来自 Planner。与 question 分开是因为追问场景下
    #: 二者不同：「那 Q2 呢」是 question，「补齐 Q2 的同口径数值」才是目标。
    objective: str = ""
    #: 已确定的指标口径，非空时优先于别名匹配
    metric_codes: tuple[str, ...] = ()
    #: 结果截断上限的行数（None 取配置值）
    limit: int | None = None


class SqlExecutionResult(BaseModel):
    """执行器的直接产出（`executor.py` → `tool.py`）。"""

    model_config = ConfigDict(frozen=True)

    columns: tuple[ResultColumn, ...]
    rows: tuple[tuple[Any, ...], ...]
    row_count: int
    truncated: bool
    duration_ms: int
    #: 数据库实际返回的行数（截断前）。与 `row_count` 之差即被丢掉的部分，
    #: 「返回值看起来正常但其实被砍过」是这类工具最危险的状态，必须可观测。
    fetched_count: int = 0
    started_at: datetime | None = None

    @property
    def is_empty(self) -> bool:
        """与 `SqlToolResult.is_empty` 同一判据，见 `has_no_values`。"""
        return has_no_values(self.row_count, self.rows)


__all__ = [
    "AttemptStage",
    "AttemptStatus",
    "Cardinality",
    "ColumnSpec",
    "JoinSpec",
    "MetricSpec",
    "ResultColumn",
    "SchemaCatalog",
    "SchemaContext",
    "SensitivityLevel",
    "SqlAttempt",
    "SqlCandidate",
    "SqlExecutionResult",
    "SqlQueryArgs",
    "SqlToolResult",
    "TableSpec",
    "ValidatedSql",
    "has_no_values",
]
