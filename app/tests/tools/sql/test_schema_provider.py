"""SchemaProvider 与 Schema 目录（详设 10.2 / 16.9）。

这个模块决定「模型能看见哪些表」，因此它的失败模式有两类，都要测：

- **给少了**：该带的口径没带上 → 模型自己发明算法，结果看着正常但数是错的；
- **给多了**：把不该进上下文的表带上 → 模型有机会引用它，虽然校验器会拦，
  但那时已经烧掉一次生成 + 一次修复预算。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.tools.sql.schema_provider import CatalogError, SchemaProvider, load_catalog
from app.tools.sql.schemas import MetricSpec, SchemaCatalog


class TestCatalogLoading:
    def test_real_catalog_loads(self, catalog: SchemaCatalog) -> None:
        assert catalog.tables, "目录里一张表都没有，说明加载路径或 YAML 结构有问题"
        assert catalog.allowed_functions, "函数白名单为空即默认拒绝一切函数"

    def test_missing_catalog_reports_path(self, tmp_path: Path) -> None:
        # 报错里必须带**路径**：这类失败发生在启动或第一次查询时，
        # 而没有路径的「找不到文件」会让人先去翻代码而不是翻配置。
        with pytest.raises(CatalogError, match="不存在"):
            load_catalog(tmp_path / "nope.yaml")

    def test_invalid_catalog_reports_field(self, tmp_path: Path, catalog: SchemaCatalog) -> None:
        broken = tmp_path / "broken.yaml"
        # `grain` 是必填项，漏掉它必须被 Pydantic 挡在加载期而不是使用期
        broken.write_text(
            "version: v0\ntables:\n  - name: t\n    description: d\n    columns: []\n",
            encoding="utf-8",
        )
        with pytest.raises(CatalogError, match="grain"):
            load_catalog(broken)

    def test_reloads_file_every_time(self, tmp_path: Path) -> None:
        """目录是安全白名单，**不做进程级缓存**。

        缓存住的话，一次「加了白名单却还是被拦」的排查会先从代码查起，
        而原因只是进程没重启——改完立刻生效比省几百微秒重要得多。
        """
        path = tmp_path / "c.yaml"
        path.write_text(self._minimal(version="v1"), encoding="utf-8")
        assert load_catalog(path).version == "v1"
        path.write_text(self._minimal(version="v2"), encoding="utf-8")
        assert load_catalog(path).version == "v2"

    @staticmethod
    def _minimal(*, version: str) -> str:
        return (
            f"version: {version}\n"
            "tables:\n"
            "  - name: t\n"
            "    description: d\n"
            "    grain: g\n"
            "    columns:\n"
            "      - { name: a, data_type: INT, description: d }\n"
        )


class TestMetricMatching:
    """10.2 的「先匹配指标目录」——这一步是确定性的，不交给模型。"""

    def test_alias_hit_brings_metric(self, catalog: SchemaCatalog) -> None:
        hits = catalog.match_metrics("2025 年华东 Q3 净销售额是多少")
        assert "net_sales" in {m.code for m in hits}

    def test_longer_alias_wins(self, catalog: SchemaCatalog) -> None:
        """「目标销售额」同时命中 `sales_target_amount` 与 `net_sales`。

        短词排前面的话，prompt 里会同时出现两条口径解释同一个词，
        模型选哪条就成了掷骰子。
        """
        hits = catalog.match_metrics("2025 年目标销售额是多少")
        assert hits[0].code == "sales_target_amount"

    def test_no_metric_match_returns_empty(self, catalog: SchemaCatalog) -> None:
        assert catalog.match_metrics("今天天气怎么样") == ()

    def test_explicit_metric_codes_win(self, catalog: SchemaCatalog) -> None:
        """追问「那 Q2 呢」的问题文本里根本没有指标词，只能靠上游给定。"""
        provider = SchemaProvider(catalog)
        context = provider.select("那 Q2 呢", metric_codes=("net_sales",))
        assert context.matched_metric_codes == ("net_sales",)

    def test_unknown_metric_code_raises(self, catalog: SchemaCatalog) -> None:
        """静默忽略之后模型会自己猜一个算法，结果看着正常但是错的。"""
        provider = SchemaProvider(catalog)
        with pytest.raises(AgentError) as excinfo:
            provider.select("销售额", metric_codes=("no_such_metric",))
        assert excinfo.value.code is ErrorCode.PLAN_INVALID
        assert "no_such_metric" in excinfo.value.details["unknown_metric_codes"]


class TestTableSelection:
    def test_metric_tables_selected(self, catalog: SchemaCatalog) -> None:
        context = SchemaProvider(catalog).select("2025 年华东 Q3 净销售额")
        names = {t.name for t in context.tables}
        assert "fact_sales_order_item" in names
        assert "dim_region" in names

    def test_product_line_table_reachable(self, catalog: SchemaCatalog) -> None:
        """`dim_product_line` 要经 `dim_product` 再跳一层才够得到。

        只扩一跳的话，「各产品线销售额」会因为拿不到产品线表而生成不出正确 SQL——
        而这恰恰是演示下钻案例必需的那一问。最大跳数见 `_MAX_JOIN_HOPS`。
        """
        context = SchemaProvider(catalog).select("各产品线的净销售额")
        assert "dim_product_line" in {t.name for t in context.tables}

    def test_falls_back_to_star_schema(self, catalog: SchemaCatalog) -> None:
        context = SchemaProvider(catalog).select("华东有大熊猫保护区吗")
        names = {t.name for t in context.tables}
        assert "fact_sales_order_item" in names, "没有指标时应当从事实表出发"
        assert not context.metrics

    def test_reports_omitted_tables(self, catalog: SchemaCatalog) -> None:
        """10.2 要求「超过 8 张应重新拆分问题」。

        Phase 4 的 Tool 没有拆问题的能力（那是 Planner 的事），因此它必须把
        「还有哪些表没进来」交出去。静默截断的表现是「模型引用了不存在的表」，
        排查方向会完全跑偏到模型能力上。
        """
        provider = SchemaProvider(catalog, max_tables=2)
        context = provider.select("2025 年华东 Q3 净销售额")
        assert len(context.tables) == 2
        assert context.omitted_tables, "截断必须可观测"
        # 被略过的表要出现在渲染结果里，明确告诉模型「不要引用」
        assert "不要引用" in context.render()
        for name in context.omitted_tables:
            assert name in context.render()

    def test_render_includes_metric_definition(self, catalog: SchemaCatalog) -> None:
        context = SchemaProvider(catalog).select("净销售额")
        rendered = context.render()
        assert "SUM(fact_sales_order_item.net_amount)" in rendered
        assert "口径版本" in rendered


class TestJoinWhitelist:
    def test_join_allowed_both_directions(self, catalog: SchemaCatalog) -> None:
        """模型写 `FROM dim_region JOIN fact_sales_order_item` 与反向等价。

        按方向判定会把一个正确的查询拒之门外。
        """
        assert catalog.join_allowed("fact_sales_order_item", "dim_region")
        assert catalog.join_allowed("dim_region", "fact_sales_order_item")

    def test_undeclared_join_not_allowed(self, catalog: SchemaCatalog) -> None:
        assert not catalog.join_allowed("dim_channel", "dim_product_line")

    def test_self_join_not_allowed(self, catalog: SchemaCatalog) -> None:
        assert not catalog.join_allowed("dim_region", "dim_region")


class TestCatalogAsWhitelist:
    def test_sensitive_column_role_visible(self, catalog: SchemaCatalog) -> None:
        customer = catalog.table("dim_customer")
        assert customer is not None
        name_column = customer.column("customer_name")
        assert name_column is not None
        assert not name_column.visible_to("ANALYST")
        assert name_column.visible_to("ADMIN")

    def test_column_without_roles_visible_to_all(self, catalog: SchemaCatalog) -> None:
        """空 `allowed_roles` 是「不限制」而不是「谁都不能取」。

        取「空即禁止」的话，目录里漏写一行就会变成静默拒绝，
        而漏写 `allowed_roles` 是常态——绝大多数列都是 PUBLIC。
        """
        region = catalog.table("dim_region")
        assert region is not None
        code = region.column("region_code")
        assert code is not None and code.visible_to("ANALYST")

    def test_allowed_functions_exclude_dangerous(self, catalog: SchemaCatalog) -> None:
        for name in ("SLEEP", "BENCHMARK", "LOAD_FILE"):
            assert name not in catalog.allowed_functions

    def test_scope_resolution_declared(self, catalog: SchemaCatalog) -> None:
        """10.4 第 11 步的谓词形态由目录决定，见 `ScopeSpec`。"""
        assert catalog.scope is not None and catalog.scope.resolve is not None
        assert catalog.scope.resolve.table == "dim_region"


def test_build_from_settings(settings: Settings) -> None:
    provider = SchemaProvider.from_settings(settings)
    assert provider.catalog.version == load_catalog(settings.sql_tool.catalog_path).version


def test_every_table_declares_grain(catalog: SchemaCatalog) -> None:
    """粒度不是文档字段：10.6 的聚合正确性直接依赖模型知道「一行代表什么」。"""
    for table in catalog.tables:
        assert table.grain, f"{table.name} 没声明粒度"


def test_metric_expressions_reference_known_tables(catalog: SchemaCatalog) -> None:
    """指标表达式里写错表名会让模型生成引用不存在表的 SQL。

    这条断言把「口径写错」变成加载期就能发现的问题——否则它要等到
    某个具体问题被问到、模型照着错表达式生成 SQL 才会显形。
    """
    known = {table.name for table in catalog.tables}
    for metric in catalog.metrics:
        referenced = [name for name in known if name in metric.expression]
        assert referenced, f"指标 {metric.code} 的表达式没有引用任何已声明的表：{metric.expression}"


@pytest.mark.parametrize("code", ["net_sales", "sales_target_amount", "available_qty"])
def test_core_metrics_present(catalog: SchemaCatalog, code: str) -> None:
    assert isinstance(catalog.metric(code), MetricSpec)
