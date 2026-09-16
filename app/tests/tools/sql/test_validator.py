"""SQL 安全校验（详细设计 10.4 的 12 步 / 22.2 的 SQL 安全测试）。

**这个文件是「安全由代码保证」这句话的实证。** 冲刺方案 §6 明确：评测集可以
从 115 条缩到 40 条，但「SQL 安全与越权 15 条，阻断率必须 100%」是**门禁性质**的，
不是覆盖率指标——缩题数可以，缩判定标准不行。

因此这里的组织方式是：`BLOCKED_CASES` 是一张表，每一条都必须在**校验阶段**
被拒（不是执行时被数据库拒）。文件末尾有一条断言守住「至少 15 条」这个下限，
删用例会让它失败——这是刻意的，防止为了好看而削减安全用例。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.user import PermissionScope
from app.tools.sql.schemas import SchemaCatalog
from app.tools.sql.validator import SqlValidationError, SqlValidator

#: 必须被**校验器**拦下的用例（详设 22.2 的清单）。
#:
#: 每项是 (用例名, SQL, 期望命中的步号)。步号不是装饰——它把「被拦了」细化为
#: 「被哪一条规则拦了」，否则改坏第 3 步而第 4 步恰好也拦得住时，测试照样绿。
BLOCKED_CASES: tuple[tuple[str, str, int], ...] = (
    # --- 写操作与 DDL（10.4 第 3、4 步）---
    ("DROP 表", "DROP TABLE fact_sales_order_item", 3),
    ("TRUNCATE 表", "TRUNCATE TABLE fact_sales_order_item", 3),
    ("DELETE", "DELETE FROM fact_sales_order_item WHERE 1 = 1", 3),
    ("UPDATE", "UPDATE dim_region SET province_count = 0 WHERE 1 = 1", 3),
    (
        "INSERT",
        "INSERT INTO dim_region (region_id, region_code, region_name, province_count, "
        "is_anomaly, created_at) VALUES ('x', 'x', 'x', 0, 0, NOW())",
        3,
    ),
    ("CTE 里藏写操作", "WITH x AS (SELECT 1) DELETE FROM fact_sales_order_item", 3),
    ("多语句堆叠", "SELECT 1; DROP TABLE fact_sales_order_item", 2),
    ("事务控制", "ROLLBACK", 3),
    # --- 注释与旁路（10.4 第 5 步）---
    ("行注释拆分关键字", "SELECT net_amount FROM fact_sales_order_item -- 绕过\nWHERE 1 = 1", 5),
    ("块注释拆分关键字", "SELECT net_amount FROM fact_sales_order_item /* x */", 5),
    # --- 系统对象（10.4 第 5、6 步）---
    ("系统库 information_schema", "SELECT table_name FROM information_schema.tables", 6),
    ("系统库 mysql", "SELECT user FROM mysql.user", 6),
    ("会话变量", "SELECT @@version AS v FROM dim_region", 5),
    # --- 表与列权限（10.4 第 6、7 步）---
    ("非白名单表", "SELECT id FROM fact_sales_order_item_secret", 6),
    ("SELECT 星号", "SELECT * FROM dim_region", 7),
    ("带表名的星号", "SELECT r.* FROM dim_region AS r", 7),
    ("不存在的列", "SELECT r.no_such_column FROM dim_region AS r", 7),
    ("敏感字段（analyst）", "SELECT c.customer_name FROM dim_customer AS c", 7),
    (
        "敏感字段经别名绕过",
        "SELECT c.customer_name AS n FROM dim_customer AS c",
        7,
    ),
    (
        "敏感字段经派生表绕过",
        "SELECT t.n FROM (SELECT customer_name AS n FROM dim_customer) AS t",
        7,
    ),
    (
        "敏感字段经 CTE 绕过",
        "WITH x AS (SELECT customer_name FROM dim_customer) SELECT customer_name FROM x",
        7,
    ),
    (
        "敏感字段经 WHERE 别名绕过",
        "SELECT c.customer_level FROM dim_customer AS c WHERE c.customer_name = 'x'",
        7,
    ),
    # --- JOIN 关系（10.4 第 8 步）---
    (
        "逗号连接",
        "SELECT a.order_id FROM fact_sales_order_item AS a, dim_region AS r",
        8,
    ),
    (
        "JOIN 无 ON",
        "SELECT a.order_id FROM fact_sales_order_item AS a JOIN dim_region AS r",
        8,
    ),
    (
        "未声明的 JOIN 关系",
        "SELECT c.customer_level FROM dim_customer AS c "
        "JOIN dim_product_line AS p ON p.category = c.customer_level",
        8,
    ),
    # --- 函数（10.4 第 9 步）---
    ("危险函数 SLEEP", "SELECT SLEEP(5) FROM dim_region", 9),
    ("危险函数 BENCHMARK", "SELECT BENCHMARK(1000000, MD5('x')) FROM dim_region", 9),
    ("危险函数 LOAD_FILE", "SELECT LOAD_FILE('/etc/passwd') FROM dim_region", 9),
    ("未登记的函数", "SELECT GREATEST(province_count, 1) FROM dim_region", 9),
)


def _ids(cases: Sequence[Sequence[object]]) -> list[str]:
    """用例名做 parametrize 的 id，让失败信息里直接看到是哪一条被放过了。

    默认的 id 是 `sql0` / `sql1`——失败时还要回去数第几条，
    而安全用例失败恰恰是最需要一眼定位的场景。
    """
    return [str(case[0]) for case in cases]


#: 应当**通过**校验、但会被重写的用例。
REWRITE_CASES: tuple[tuple[str, str], ...] = (
    ("明细查询无 LIMIT", "SELECT order_id FROM fact_sales_order_item"),
    ("LIMIT 超过上限", "SELECT order_id FROM fact_sales_order_item LIMIT 999999"),
)

#: 应当原样通过的合法查询（正例）。没有正例的校验器测试只会逼着实现越拦越宽。
ACCEPTED_CASES: tuple[str, ...] = (
    "SELECT SUM(s.net_amount) AS net FROM fact_sales_order_item AS s "
    "WHERE s.order_date >= '2025-07-01' AND s.order_date < '2025-10-01'",
    "SELECT DATE_FORMAT(s.order_date, '%Y-%m') AS m, SUM(s.net_amount) AS net "
    "FROM fact_sales_order_item AS s GROUP BY m ORDER BY net DESC",
    "WITH x AS (SELECT region_id, SUM(net_amount) AS amt FROM fact_sales_order_item "
    "GROUP BY region_id) SELECT r.region_name, x.amt FROM x "
    "JOIN dim_region AS r ON r.region_id = x.region_id",
)


class TestBlocked:
    """详设 22.2：危险 SQL 与越权用例 **100% 阻断**。"""

    @pytest.mark.parametrize(("label", "sql", "step"), BLOCKED_CASES, ids=_ids(BLOCKED_CASES))
    def test_must_be_blocked_at_validation(
        self,
        validator: SqlValidator,
        restricted: PermissionScope,
        label: str,
        sql: str,
        step: int,
    ) -> None:
        del label
        with pytest.raises(SqlValidationError) as excinfo:
            validator.validate(sql, scope=restricted)
        assert excinfo.value.step == step, (
            f"被拦下了，但拦它的是第 {excinfo.value.step} 步而不是第 {step} 步："
            f"{excinfo.value.repair_hint()}"
        )

    def test_security_cases_at_least_fifteen(self) -> None:
        """冲刺方案 §6：题数可缩，判定标准不缩。

        这条断言存在的意义是**防止削减用例**——它不检查覆盖了什么，
        只保证「15 条」这个数量下限不被悄悄突破。
        """
        assert len(BLOCKED_CASES) >= 15, "安全与越权用例不得少于 15 条（冲刺方案 §6）"

    def test_block_rate_is_one_hundred_percent(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """逐条重跑一遍并统计，把「100%」这个数字本身变成可打印的证据。"""
        blocked = 0
        for _, sql, _ in BLOCKED_CASES:
            try:
                validator.validate(sql, scope=restricted)
            except SqlValidationError:
                blocked += 1
        assert blocked == len(BLOCKED_CASES), f"阻断率 {blocked}/{len(BLOCKED_CASES)}，必须 100%"

    def test_privileged_query_allowed_for_unrestricted(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        """「敏感字段被拦」必须真的是**角色**判定，而不是这条 SQL 本身有问题。

        少了这条反例，把第 7 步写成「凡 dim_customer.customer_name 一律拒绝」
        也能让上面的用例全绿。
        """
        result = validator.validate(
            "SELECT c.customer_name FROM dim_customer AS c", scope=unrestricted
        )
        assert "customer_name" in result.validated_sql

    def test_blocked_sql_never_reaches_executor(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """校验失败的产物只有异常，没有任何「半成品」可执行对象漏出去。

        `validate` 要么返回一个通过了 12 步的 `ValidatedSql`，要么抛异常——
        不存在「返回了但标记为不安全」的中间态，那种设计总有人会忽略标记。
        """
        with pytest.raises(SqlValidationError):
            validator.validate("DROP TABLE dim_region", scope=restricted)


class TestErrorClass:
    """详设 10.8：哪些错误回给模型修复、哪些直接终止。

    分类错了的代价不对称——把不可修复的判成可修复，会烧掉一次修复预算并
    把一个危险请求再喂模型一次；反过来则让一个拼错的列名白白失败。
    """

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            ("DROP TABLE dim_region", "SECURITY"),
            ("SELECT r.no_such_column FROM dim_region AS r", "REPAIRABLE"),
            ("SELECT * FROM dim_region", "REPAIRABLE"),
            ("SELECT c.customer_name FROM dim_customer AS c", "SECURITY"),
            ("SELECT GREATEST(province_count, 1) FROM dim_region", "REPAIRABLE"),
            ("SELECT SLEEP(1) FROM dim_region", "SECURITY"),
            (
                "SELECT r.region_code FROM dim_region AS r "
                "JOIN dim_channel AS c ON c.channel_id = r.region_id",
                "SECURITY",
            ),
        ],
    )
    def test_error_class_mapping(
        self, validator: SqlValidator, restricted: PermissionScope, sql: str, expected: str
    ) -> None:
        with pytest.raises(SqlValidationError) as excinfo:
            validator.validate(sql, scope=restricted)
        assert excinfo.value.error_class == expected

    def test_repair_hint_carries_no_raw_sql(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """回灌给模型的错因只带步号与标识符。

        数据库回显与模型原文都可能带着业务取值，而这条提示会进 prompt、
        进 Trace、进日志（19.4 的脱敏纪律）。
        """
        sql = "SELECT r.secret_column FROM dim_region AS r"
        with pytest.raises(SqlValidationError) as excinfo:
            validator.validate(sql, scope=restricted)
        hint = excinfo.value.repair_hint()
        assert "SELECT" not in hint.upper()
        assert "secret_column" in hint


class TestRewrites:
    def test_adds_limit_when_missing(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT order_id FROM fact_sales_order_item", scope=unrestricted
        )
        assert "LIMIT 1000" in result.validated_sql
        assert result.rewrites, "重写必须可观测"

    def test_clamps_limit_above_max(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT order_id FROM fact_sales_order_item LIMIT 999999", scope=unrestricted
        )
        assert "LIMIT 1000" in result.validated_sql
        assert "999999" not in result.validated_sql

    def test_keeps_limit_within_max(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        """合法的小 LIMIT 被改写成 1000 会让「只看前 5 行」的意图失效。"""
        result = validator.validate(
            "SELECT order_id FROM fact_sales_order_item LIMIT 5", scope=unrestricted
        )
        assert "LIMIT 5" in result.validated_sql
        assert not result.rewrites

    @pytest.mark.parametrize(("label", "sql"), REWRITE_CASES, ids=[c[0] for c in REWRITE_CASES])
    def test_rewritten_sql_still_passes(
        self, validator: SqlValidator, unrestricted: PermissionScope, label: str, sql: str
    ) -> None:
        del label
        assert "LIMIT 1000" in validator.validate(sql, scope=unrestricted).validated_sql


class TestDataScopePredicate:
    """详设 10.4 第 11 步：注入不可被模型覆盖的服务端谓词（TBC-03）。"""

    def test_unrestricted_user_gets_no_predicate(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item", scope=unrestricted
        )
        assert not result.scope_injected
        assert "scope_region" not in result.validated_sql

    def test_restricted_user_gets_subquery_on_fact_table(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """事实表上只有 `region_id`，而权限范围里存的是区域**名称**。

        少了名称→ID 的解析这一层，受限用户会查不到任何行——
        表现为「权限生效了但结果是空的」，比报错更难排查。
        """
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item", scope=restricted
        )
        assert result.scope_injected
        assert "SELECT dim_region.region_id FROM dim_region" in result.validated_sql
        assert result.bind_parameters["scope_region_0_0"] == "华东"

    def test_restricted_user_gets_direct_filter_on_dimension(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """`dim_region` 自己就是名称的出处，对它不需要子查询。"""
        result = validator.validate("SELECT province_count FROM dim_region", scope=restricted)
        assert "dim_region.region_name IN (:scope_region_0_0)" in result.validated_sql

    def test_scope_predicate_anded_with_existing_where(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """模型写的 `OR` 覆盖不掉权限条件。

        若改成把权限条件塞进第一个 WHERE 的位置，`WHERE a=1 OR b=2` 会让
        权限条件被绑定到 `b=2` 上——除了 b=2 的行，其余照旧全露。
        """
        result = validator.validate(
            "SELECT province_count FROM dim_region WHERE province_count = 3 OR province_count = 4",
            scope=restricted,
        )
        assert "dim_region.region_name IN (:scope_region_0_0)" in result.validated_sql
        # 权限条件必须在 OR 之外：渲染里应当看到把原条件整体括起来的括号
        assert ") AND dim_region.region_name IN (" in result.validated_sql

    def test_model_cannot_widen_scope_with_tautology(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """模型试图自己放开范围（`WHERE 1=1`）也拿不到别的区域的数据。"""
        result = validator.validate(
            "SELECT province_count FROM dim_region WHERE 1 = 1", scope=restricted
        )
        assert result.scope_injected
        assert "1 = 1" in result.validated_sql  # 原条件保留
        assert "AND dim_region.region_name IN (" in result.validated_sql

    def test_scope_injected_into_nearest_select(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """注入到「最近的 SELECT」，不是一律塞进最外层。

        CTE 里的表如果被注入到外层 WHERE，那是一个外层作用域里不存在的限定名，
        SQL 会直接报错——这是这一步最容易写错的地方。
        """
        result = validator.validate(
            "WITH x AS (SELECT region_id, SUM(net_amount) AS amt FROM fact_sales_order_item "
            "GROUP BY region_id) SELECT r.region_name, x.amt FROM x "
            "JOIN dim_region AS r ON r.region_id = x.region_id",
            scope=restricted,
        )
        sql = result.validated_sql
        # CTE 内部注入在 `fact_sales_order_item` 上（该表没有别名，限定名即表名）
        assert "fact_sales_order_item.region_id IN (" in sql
        # 外层注入在别名 `r` 上。**不是** `fact_sales_order_item.region_id` 出现在外层——
        # 那是一个外层作用域里不存在的限定名，SQL 会直接报错。
        assert "r.region_name IN (" in sql
        # 两处注入的参数名必须不同：共用名字靠「值恰好相同」维持正确是个定时炸弹
        assert len({k for k in result.bind_parameters if k.startswith("scope_region")}) == 2

    def test_tables_without_scope_column_not_injected(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """查产品目录这类问题本来就不该被区域权限影响，**不报错也不加谓词**。"""
        result = validator.validate(
            "SELECT product_name FROM dim_product WHERE status = 'ACTIVE'", scope=restricted
        )
        assert not result.scope_injected

    def test_scope_params_do_not_collide_with_model_params(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item WHERE order_date >= :start_date",
            parameters={"start_date": "2025-07-01"},
            scope=restricted,
        )
        assert result.bind_parameters["start_date"] == "2025-07-01"
        assert result.bind_parameters["scope_region_0_0"] == "华东"


class TestFingerprint:
    """详设 10.4 第 12 步 / 16.6：指纹用于「同一批烂 SQL 是否反复出现」。"""

    def test_literals_do_not_affect_fingerprint(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        first = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date >= '2025-07-01' AND order_date < '2025-10-01'",
            scope=unrestricted,
        )
        second = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date >= '2024-07-01' AND order_date < '2024-10-01'",
            scope=unrestricted,
        )
        assert first.sql_fingerprint == second.sql_fingerprint, (
            "保留字面量的话，同一个句型换个日期就是一个新指纹，按指纹聚合查不出任何模式"
        )

    def test_different_structure_yields_different_fingerprint(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        first = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item", scope=unrestricted
        )
        second = validator.validate(
            "SELECT SUM(gross_amount) FROM fact_sales_order_item", scope=unrestricted
        )
        assert first.sql_fingerprint != second.sql_fingerprint

    def test_scope_predicate_changes_fingerprint(
        self, validator: SqlValidator, unrestricted: PermissionScope, restricted: PermissionScope
    ) -> None:
        """受限用户执行的是**另一条** SQL，指纹必须反映这一点。

        否则「同一批烂 SQL 反复出现」的归因会把两种权限下的查询混在一起。
        """
        sql = "SELECT SUM(net_amount) FROM fact_sales_order_item"
        assert (
            validator.validate(sql, scope=unrestricted).sql_fingerprint
            != validator.validate(sql, scope=restricted).sql_fingerprint
        )


class TestTimeRange:
    """结果 Schema 的 `data_time_range`，最终变成 Evidence 的 `event_time`。"""

    def test_extracts_range_from_named_parameters(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date >= :start_date AND order_date < :end_date",
            parameters={"start_date": "2025-07-01", "end_date": "2025-10-01"},
            scope=unrestricted,
        )
        assert result.data_time_range is not None
        assert result.data_time_range.start.date().isoformat() == "2025-07-01"
        assert result.data_time_range.end.date().isoformat() == "2025-10-01"

    def test_extracts_range_from_literals(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date >= '2025-07-01' AND order_date < '2025-10-01'",
            scope=unrestricted,
        )
        assert result.data_time_range is not None

    def test_extracts_range_from_between(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date BETWEEN '2025-07-01' AND '2025-10-01'",
            scope=unrestricted,
        )
        assert result.data_time_range is not None

    def test_returns_none_without_time_condition(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        """宁可没有，不能猜错：一个错的时间区间会让 Phase 9 报出根本不存在的冲突。"""
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item", scope=unrestricted
        )
        assert result.data_time_range is None

    def test_non_time_comparison_is_not_a_range(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT province_count FROM dim_region WHERE province_count >= 3",
            scope=unrestricted,
        )
        assert result.data_time_range is None


class TestDerivedColumns:
    """从 CTE / 派生表取它算出来的列——这是模型写多步计算时最自然的写法。

    **一次真实缺陷**：金标评测里模型把同比算在 CTE 里、外层引用 `yoy_change_pct`，
    被第 7 步误判成「引用了不存在的列」。它不是安全问题，是校验器把一种
    合法写法当成了拼写错误。
    """

    def test_cte_output_alias_is_accepted(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "WITH yoy AS (SELECT SUM(CASE WHEN order_date >= '2025-07-01' "
            "AND order_date < '2025-10-01' THEN net_amount ELSE 0 END) AS cur_sales, "
            "SUM(CASE WHEN order_date >= '2024-07-01' AND order_date < '2024-10-01' "
            "THEN net_amount ELSE 0 END) AS prev_sales FROM fact_sales_order_item) "
            "SELECT cur_sales, prev_sales, (cur_sales - prev_sales) / prev_sales AS yoy "
            "FROM yoy",
            scope=unrestricted,
        )
        assert "yoy" in result.validated_sql

    def test_subquery_output_alias_is_accepted(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT t.total FROM (SELECT SUM(net_amount) AS total FROM fact_sales_order_item) AS t",
            scope=unrestricted,
        )
        assert "total" in result.validated_sql

    def test_catalog_column_does_not_fall_back_to_derived(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """**这条是上一类放行的安全边界。**

        `customer_name` 既是目录里的敏感列、又恰好可能是某个 CTE 的输出别名。
        如果「名字在派生列集合里」就跳过，那么把敏感列包进 CTE 再取出
        就能绕开角色检查。判据因此必须是「**目录里没有**、派生结果里有」。
        """
        with pytest.raises(SqlValidationError) as excinfo:
            validator.validate(
                "WITH x AS (SELECT customer_name AS customer_name FROM dim_customer) "
                "SELECT customer_name FROM x",
                scope=restricted,
            )
        assert excinfo.value.step == 7
        assert excinfo.value.error_class == "SECURITY"

    def test_unknown_name_in_derived_still_rejected(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        """放行的只是「真出现在派生输出里」的名字，拼错的名字照样拦。"""
        with pytest.raises(SqlValidationError) as excinfo:
            validator.validate(
                "WITH yoy AS (SELECT SUM(net_amount) AS cur_sales FROM "
                "fact_sales_order_item) SELECT typo_name FROM yoy",
                scope=unrestricted,
            )
        assert excinfo.value.step == 7
        assert excinfo.value.error_class == "REPAIRABLE"


class TestBindParameters:
    """命名参数的取值必须齐备——**详设 10.4 之外的补充**，见 `_check_bind_parameters`。

    一次真实缺陷的产物：模型写 `:start` 却给出 `start_date`，报错发生在
    SQLAlchemy 的绑定阶段（拿不到 MySQL errno），于是被归成 `TRANSIENT`
    「数据库不可用」——而它其实是模型写错了参数名，改一改就能过。
    """

    def test_missing_bind_parameter_reports_name(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        with pytest.raises(SqlValidationError) as excinfo:
            validator.validate(
                "SELECT SUM(net_amount) FROM fact_sales_order_item "
                "WHERE order_date >= :start_date AND order_date < :end_date",
                parameters={"start_date": "2025-07-01"},
                scope=unrestricted,
            )
        assert excinfo.value.step == 2
        # 归类必须是可修复：归类错了的代价是修复预算根本没被用上
        assert excinfo.value.error_class == "REPAIRABLE"
        # 参数名要带出来——那是模型唯一需要的修复线索
        assert "end_date" in excinfo.value.repair_hint()
        assert "start_date" not in (excinfo.value.safe_detail or "")

    def test_complete_bind_parameters_pass(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date >= :start_date AND order_date < :end_date",
            parameters={"start_date": "2025-07-01", "end_date": "2025-10-01"},
            scope=unrestricted,
        )
        assert result.bind_parameters["end_date"] == "2025-10-01"

    def test_extra_bind_parameters_are_ignored(
        self, validator: SqlValidator, unrestricted: PermissionScope
    ) -> None:
        """多给一个没被引用的参数不影响执行，拒绝它只会平白烧掉一次修复预算。"""
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item WHERE order_date >= :start_date",
            parameters={"start_date": "2025-07-01", "unused": "x"},
            scope=unrestricted,
        )
        assert result.bind_parameters["unused"] == "x"


class TestAcceptedAndRewritten:
    """正例：合法的 SQL 必须通过，且**执行的是重写后的那一份**。"""

    @pytest.mark.parametrize("sql", ACCEPTED_CASES, ids=range(len(ACCEPTED_CASES)))
    def test_valid_query_passes(
        self, validator: SqlValidator, unrestricted: PermissionScope, sql: str
    ) -> None:
        result = validator.validate(sql, scope=unrestricted)
        assert result.validated_sql
        assert result.sql_fingerprint

    def test_bind_parameters_travel_with_result(
        self, validator: SqlValidator, restricted: PermissionScope
    ) -> None:
        """执行时一个绑定参数都不能少。

        让调用方自己把「模型的参数」与「注入权限谓词时加的参数」拼起来，
        拼漏了会以 `MissingBindParameter` 的形式在数据库层报出来——
        那个错误看起来像 SQL 写错了，排查方向会跑偏。
        """
        result = validator.validate(
            "SELECT SUM(net_amount) FROM fact_sales_order_item "
            "WHERE order_date >= :start_date AND region_id = :region",
            parameters={"start_date": "2025-07-01", "region": "anything"},
            scope=restricted,
        )
        assert set(result.bind_parameters) >= {"start_date", "region", "scope_region_0_0"}

    def test_rejects_overlong_sql(
        self, catalog: SchemaCatalog, unrestricted: PermissionScope
    ) -> None:
        small = SqlValidator(catalog, max_sql_chars=50, max_rows=1000)
        with pytest.raises(SqlValidationError) as excinfo:
            small.validate(
                "SELECT region_name FROM dim_region WHERE region_name = 'x'", scope=unrestricted
            )
        assert excinfo.value.step == 1
        # 超长不是攻击特征，是模型跑飞了——但它也不可修复（重新猜不是修正）
        assert excinfo.value.error_class == "VALIDATION"
        assert not excinfo.value.repairable
