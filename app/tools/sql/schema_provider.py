"""Schema 目录加载与上下文选择（详细设计 10.2）。

**这个模块决定「模型能看见哪些表」**，是 10.5 分层安全责任里代码层的第一个环节：
目录里没有的表，模型连知道它存在的机会都没有。

## 选择过程是确定性的，不含模型调用

10.2 的原文是「先匹配指标目录，再扩展维度表与合法 JOIN」。两步都是确定性规则：

1. **匹配指标**：按 `MetricSpec.aliases` 在问题文本里做包含匹配（见
   `SchemaCatalog.match_metrics`）。命中即带上该指标的口径表达式。
   *这一步交给模型判断的话，「该带的口径没带上」会表现为模型自己发明了
   一个算法——结果看起来完全正常，只是数是错的。*
2. **扩展表**：从命中指标的表达式里找出被引用的表，再按目录声明的 JOIN
   向外扩一层维度表。事实表 → 维度表是星型模型的标准形状，扩一层就够；
   不做传递闭包是因为那会把整库的表都拉进来，正好撞上 10.2 的 8 张上限。

没有任何指标命中时（「有多少客户」「库存够不够」这类不算指标的查询），
退化为「把事实表与它的全部维度表都带上」——这是能生成有效 SQL 的最小集合。

## 上限不是截断

超过 `SQL_TOOL__MAX_SCHEMA_TABLES` 时**不做静默截断**，而是把没进去的表
记进 `SchemaContext.omitted_tables`，在 prompt 里显式告诉模型「这些表
你没有、不要引用」。静默截断的表现是「模型生成的 SQL 引用了不存在的表」，
排查方向会完全跑偏到模型能力上。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.tools.sql.schemas import MetricSpec, SchemaCatalog, SchemaContext, TableSpec

#: JOIN 展开的最大跳数，见 `_expand_tables` 的说明。
_MAX_JOIN_HOPS = 2


class CatalogError(ValueError):
    """目录文件本身有问题（读不到、YAML 语法错、Schema 不合规）。

    继承内建 `ValueError` 而不是映射成 `AgentError`：这是**部署期错误**，
    不是运行期故障。目录配错时每个任务都会失败，把它降级成一条用户可见的
    「查询失败」只会让真正的原因（文件路径写错了）更难被发现。
    与 `model_gateway.EmbeddingDimensionError` 同一处理方式。
    """


def load_catalog(path: str | Path) -> SchemaCatalog:
    """从 YAML 加载 Schema 目录。

    每次调用都重新读文件，不做进程级缓存：目录是**安全白名单**，
    改完必须立刻生效——缓存住的话，一次「加了白名单却还是被拦」的排查
    会先从代码查起，而原因只是进程没重启。加载成本是几百微秒，
    相对一次模型调用可以忽略。
    """
    file = Path(path)
    if not file.is_file():
        raise CatalogError(
            f"Schema 目录不存在：{file}。请检查 SQL_TOOL__CATALOG_PATH，"
            "或从仓库的 configs/schema_catalog.yaml 恢复。"
        )
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CatalogError(f"Schema 目录 {file} 不是合法 YAML：{exc}") from exc
    if not isinstance(raw, dict):
        raise CatalogError(f"Schema 目录 {file} 的顶层必须是映射（表/指标/函数白名单）")
    try:
        return SchemaCatalog.model_validate(raw)
    except ValidationError as exc:
        # 只回显字段路径与原因，不回显整份目录内容（可能有几十行）
        problems = "; ".join(
            f"{'.'.join(str(p) for p in item['loc'])}: {item['msg']}"
            for item in exc.errors(include_url=False, include_input=False)[:5]
        )
        raise CatalogError(f"Schema 目录 {file} 不合规：{problems}") from exc


class SchemaProvider:
    """按问题选择注入模型的 Schema 子集（详设 10.2 的 `SchemaProvider`）。

    Attributes:
        catalog: 整份目录。校验器也要用它——白名单只有一份，
            分开加载就会出现「生成时看得到、校验时不认识」的自相矛盾。
    """

    def __init__(self, catalog: SchemaCatalog, *, max_tables: int = 8) -> None:
        self.catalog = catalog
        self._max_tables = max_tables

    @classmethod
    def from_settings(cls, settings: Settings) -> SchemaProvider:
        return cls(
            load_catalog(settings.sql_tool.catalog_path),
            max_tables=settings.sql_tool.max_schema_tables,
        )

    # ------------------------------------------------------------------ 选择
    def select(self, question: str, *, metric_codes: tuple[str, ...] = ()) -> SchemaContext:
        """按问题选出表与指标（详设 10.2）。

        `metric_codes` 非空时**以它为准**，不再做别名匹配：那是 Planner 或
        上一轮 reflect 已经定下来的口径（例如追问「那 Q2 呢」时，
        问题文本里根本没有指标词，靠别名匹配会一条都命中不了）。
        """
        metrics = self._resolve_metrics(question, metric_codes)
        wanted = self._expand_tables(metrics)

        selected: list[TableSpec] = []
        omitted: list[str] = []
        for table in self.catalog.tables:
            if table.name not in wanted:
                continue
            if len(selected) < self._max_tables:
                selected.append(table)
            else:
                omitted.append(table.name)

        return SchemaContext(
            version=self.catalog.version,
            tables=tuple(selected),
            metrics=metrics,
            omitted_tables=tuple(omitted),
            matched_metric_codes=tuple(m.code for m in metrics),
        )

    # ---------------------------------------------------------------- 内部实现
    def _resolve_metrics(
        self, question: str, metric_codes: tuple[str, ...]
    ) -> tuple[MetricSpec, ...]:
        if metric_codes:
            resolved = [m for code in metric_codes if (m := self.catalog.metric(code)) is not None]
            unknown = [code for code in metric_codes if self.catalog.metric(code) is None]
            if unknown:
                # 未知指标码意味着上游给错了口径。**不能静默忽略**：
                # 忽略之后模型会自己猜一个算法，结果看起来正常但是错的。
                raise AgentError(
                    ErrorCode.PLAN_INVALID,
                    "指定的指标口径不存在",
                    details={"unknown_metric_codes": unknown},
                )
            return tuple(resolved)
        return self.catalog.match_metrics(question)

    def _expand_tables(self, metrics: tuple[MetricSpec, ...]) -> set[str]:
        """指标表达式 → 表，再按目录声明的 JOIN 向外做**限深**闭包。

        深度取 2 而不是 1，是因为星型模型里总有一条「隔一层」的路径：
        `fact_sales_order_item → dim_product → dim_product_line`。
        只扩一跳的话，「各产品线销售额」这类问题会因为拿不到产品线表而
        生成不出正确 SQL——而这恰恰是演示下钻案例必需的那一问。
        """
        by_expression = [m.expression for m in metrics]
        reached = {
            table.name
            for table in self.catalog.tables
            if any(table.name in expression for expression in by_expression)
        }
        if not reached:
            # 没有指标命中（「库存够不够」这类不算指标的查询）：
            # 从有 JOIN 声明的表出发，也就是事实表——它能扩出整张星型图。
            reached = {table.name for table in self.catalog.tables if table.joins}

        for _ in range(_MAX_JOIN_HOPS):
            frontier: set[str] = set()
            for table in self.catalog.tables:
                if table.name in reached:
                    frontier |= table.joined_tables()
            if frontier <= reached:
                break
            reached |= frontier
        return reached
