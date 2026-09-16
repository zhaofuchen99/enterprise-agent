"""SQL Tool 的真连集成用例（详设 22.1：集成测试覆盖 MySQL）。

`make test` 下的单元用例用 `FakeSqlRunner` 覆盖了编排与自修复；
这里补的是**只有真连业务库才能验的那部分**：

1. **数据库层的只读护栏真的生效**（详设 10.5 的第 1 条防线）——
   即使校验器被绕过，写操作也该被数据库拒绝。这条断言的对象是**数据库配置**，
   不是我们的代码，因此只能真连才测得出来。
2. **权限谓词在真实数据上确实过滤掉了别的区域**——单元测试只能断言 SQL 文本
   里有 `IN (...)`，断言不了它真的少返回了行。
3. **只读账号能读到全部演示数据**——SQL Tool 的取数路径就是它。
4. **截断与字节上限**在真实结果集上的行为。

前置：`make up` + `make seed-business`（业务库 8 表已灌数）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import Settings, get_settings
from app.domain.user import PermissionScope, UserRole
from app.infrastructure.db import create_engine
from app.tools.sql.executor import SqlExecutor
from app.tools.sql.schema_provider import SchemaProvider
from app.tools.sql.schemas import ValidatedSql
from app.tools.sql.validator import SqlValidator

pytestmark = pytest.mark.integration

#: 需要从测试环境变量里摘掉、好让 `.env` 的真实值生效的键。
_TEST_ENV_DB_KEYS = ("DATABASE_URL_BUSINESS_RO", "DATABASE_URL_BUSINESS_RW")

#: 演示库里华东 2025 Q3 的净销售额（Phase 2 反向构造出的固定结论）。
#: 写死在这里是**刻意的**：它是「数据变了没有」的哨兵。改成从库里现查，
#: 这些断言就退化成「查询等于它自己」。
HUADONG_Q3_NET_SALES = "111967031.73"


@pytest.fixture
def real_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """读**真实** `.env` 的配置，绕开 conftest 注入的测试环境变量。

    conftest 把业务库连接串钉成了与 `.env` 相同的值，但测试库是 `agent_test`；
    这里显式回落一次，保证「连的是 `.env` 里那个业务库」这件事不依赖两者的巧合。
    """
    for key in _TEST_ENV_DB_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield Settings()
    get_settings.cache_clear()


@pytest.fixture
async def executor(real_settings: Settings) -> AsyncIterator[SqlExecutor]:
    instance = SqlExecutor(
        real_settings, catalog=SchemaProvider.from_settings(real_settings).catalog
    )
    yield instance
    await instance.aclose()


@pytest.fixture
def validator(real_settings: Settings) -> SqlValidator:
    catalog = SchemaProvider.from_settings(real_settings).catalog
    tuning = real_settings.sql_tool
    return SqlValidator(catalog, max_sql_chars=tuning.max_sql_chars, max_rows=tuning.max_rows)


def _validate(sql: str, validator: SqlValidator, scope: PermissionScope) -> ValidatedSql:
    return validator.validate(sql, scope=scope)


class TestReadOnlyGuard:
    """详设 10.5 第一条防线：数据库层的只读账号 + 只读会话。

    这条用例**绕过我们的校验器**直接打数据库。理由正是它要证明的事：
    「安全不能只由我们的代码保证」。校验器有漏洞时，这一层要能兜住。
    """

    async def test_readonly_account_cannot_write(self, real_settings: Settings) -> None:
        engine = create_engine(real_settings, url=real_settings.database_url_business_ro)
        try:
            async with engine.connect() as connection:
                with pytest.raises(SQLAlchemyError):
                    await connection.execute(
                        text("UPDATE dim_region SET province_count = 0 WHERE region_name = '华东'")
                    )
        finally:
            await engine.dispose()

    async def test_readonly_account_cannot_create_table(self, real_settings: Settings) -> None:
        engine = create_engine(real_settings, url=real_settings.database_url_business_ro)
        try:
            async with engine.connect() as connection:
                with pytest.raises(SQLAlchemyError):
                    await connection.execute(text("CREATE TABLE t_probe (id INT)"))
        finally:
            await engine.dispose()

    async def test_readonly_account_can_read_demo_data(self, real_settings: Settings) -> None:
        """与上面两条互为对照：权限是「只读」而不是「读不到」。"""
        engine = create_engine(real_settings, url=real_settings.database_url_business_ro)
        try:
            async with engine.connect() as connection:
                result = await connection.execute(
                    text("SELECT COUNT(*) FROM fact_sales_order_item")
                )
                assert result.scalar_one() > 0
        finally:
            await engine.dispose()


class TestRealExecution:
    async def test_real_query_returns_real_number(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        validated = _validate(
            "SELECT SUM(s.net_amount) AS net_sales FROM fact_sales_order_item AS s "
            "JOIN dim_region AS r ON r.region_id = s.region_id "
            "WHERE r.region_name = '华东' "
            "AND s.order_date >= '2025-07-01' AND s.order_date < '2025-10-01'",
            validator,
            PermissionScope(role=UserRole.ADMIN),
        )

        result = await executor.execute(validated)

        assert result.row_count == 1
        assert result.rows[0][0] == HUADONG_Q3_NET_SALES
        assert not result.truncated
        # 金额被规范化为**字符串**而不是 float：二进制浮点表示不了 0.1，
        # 一次「对不上」的排查如果从浮点误差开始，方向就永远回不到口径上。
        assert isinstance(result.rows[0][0], str)

    async def test_column_types_from_catalog(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        validated = _validate(
            "SELECT r.region_name, r.province_count FROM dim_region AS r",
            validator,
            PermissionScope(role=UserRole.ADMIN),
        )
        result = await executor.execute(validated)

        types = {column.name: column.data_type for column in result.columns}
        assert types["region_name"] == "VARCHAR(32)"
        assert types["province_count"] == "INT"

    async def test_rewritten_limit_takes_effect(
        self, real_settings: Settings, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        """事实表有 49.9 万行，不补 LIMIT 的结果集大小与补了的不在一个量级。"""
        validated = _validate(
            "SELECT s.order_id FROM fact_sales_order_item AS s",
            validator,
            PermissionScope(role=UserRole.ADMIN),
        )
        assert f"LIMIT {real_settings.sql_tool.max_rows}" in validated.validated_sql

        result = await executor.execute(validated)

        assert result.row_count == real_settings.sql_tool.max_rows
        assert result.truncated, "恰好取满上限时必须标出「可能还有更多」"


class TestDataScopeOnRealData:
    """详设 10.4 第 11 步的谓词在真实数据上的效果。

    单元测试只能断言 SQL 文本里有 `IN (...)`；**它过滤掉了别的区域**这件事
    只有真连才验得出来。
    """

    QUERY = (
        "SELECT r.region_name, SUM(s.net_amount) AS net_sales "
        "FROM fact_sales_order_item AS s "
        "JOIN dim_region AS r ON r.region_id = s.region_id "
        "WHERE s.order_date >= '2025-07-01' AND s.order_date < '2025-10-01' "
        "GROUP BY r.region_name"
    )

    async def test_unrestricted_user_sees_all_regions(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        validated = _validate(self.QUERY, validator, PermissionScope(role=UserRole.ADMIN))
        result = await executor.execute(validated)

        assert result.row_count == 5, "演示库有五个区域"

    async def test_restricted_user_sees_own_region_only(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        validated = _validate(
            self.QUERY,
            validator,
            PermissionScope(role=UserRole.ANALYST, region_ids=("华东",)),
        )
        result = await executor.execute(validated)

        assert result.row_count == 1
        assert result.rows[0][0] == "华东"

    async def test_restricted_user_querying_other_region_gets_nothing(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        """**关键的一条**：模型被诱导去查华南时，拿不到华南的任何数据。

        返回别人的数据就是越权；报错则是可用性问题。**没有数据**是唯一正确的结局。

        注意这里的形态：`SELECT SUM(...)` 没有 GROUP BY，无匹配行时
        返回的是「一行 NULL」而不是零行——所以判据是 `is_empty`（它覆盖了
        这两种形态）而不是 `row_count == 0`。
        """
        validated = _validate(
            "SELECT SUM(s.net_amount) AS net_sales FROM fact_sales_order_item AS s "
            "JOIN dim_region AS r ON r.region_id = s.region_id "
            "WHERE r.region_name = '华南'",
            validator,
            PermissionScope(role=UserRole.ANALYST, region_ids=("华东",)),
        )
        result = await executor.execute(validated)

        # 原条件（华南）与注入条件（华东）互斥 → 聚合结果为 NULL
        assert result.rows[0][0] is None
        assert result.is_empty

    async def test_restricted_user_aggregate_over_own_region_has_value(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        """与上一条互为对照：查**自己的**区域必须拿到真实数字。

        少了这条，把权限谓词写成「恒假」（查什么都返回空）也能让上一条通过——
        那是把越权换成了不可用，同样不能交。
        """
        validated = _validate(
            "SELECT SUM(s.net_amount) AS net_sales FROM fact_sales_order_item AS s "
            "JOIN dim_region AS r ON r.region_id = s.region_id "
            "WHERE r.region_name = '华东' "
            "AND s.order_date >= '2025-07-01' AND s.order_date < '2025-10-01'",
            validator,
            PermissionScope(role=UserRole.ANALYST, region_ids=("华东",)),
        )
        result = await executor.execute(validated)

        assert result.rows[0][0] == HUADONG_Q3_NET_SALES

    async def test_dimension_table_filtered_too(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        """`dim_region` 是权限谓词的**来源表**，对它自己也要过滤。

        不过滤的话，受限用户能读到其他区域的名称与省份数——
        那本身就可能是一份不该看到的经营信息。
        """
        validated = _validate(
            "SELECT r.region_name FROM dim_region AS r",
            validator,
            PermissionScope(role=UserRole.ANALYST, region_ids=("华东",)),
        )
        result = await executor.execute(validated)

        assert [row[0] for row in result.rows] == ["华东"]


class TestErrorMapping:
    async def test_semantic_error_is_repairable(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        """通过 AST 校验、但数据库不认的写法归 REPAIRABLE。

        这条路径在单元测试里只能用替身构造，**errno 的真实取值**只有真连才知道——
        而 `_REPAIRABLE_ERRNOS` 那张表就是按真实 errno 写的，写错了只能靠这里发现。

        用非法日期而不是除零来构造：MySQL 对**除零只发 warning 并返回 NULL**，
        不是错误（`ERROR_FOR_DIVISION_BY_ZERO` 在默认 sql_mode 下只警告）。
        「以为它会报错」正是这类断言最容易写错的地方。
        """
        from app.tools.sql.executor import SqlExecutionError

        validated = _validate(
            "SELECT SUM(s.net_amount) FROM fact_sales_order_item AS s "
            "WHERE s.order_date >= '2025-13-45'",
            validator,
            PermissionScope(role=UserRole.ADMIN),
        )
        with pytest.raises(SqlExecutionError) as excinfo:
            await executor.execute(validated)

        # MySQL 8 对非法**日期**字面量报的是 1525（ER_WRONG_VALUE），
        # 不是常见的 1292（ER_TRUNCATED_WRONG_VALUE）。这个取值只有真连才知道，
        # 而搞错的代价是「可修复的错被当成不可修复」——修复预算根本没被用上。
        assert excinfo.value.errno == 1525
        assert excinfo.value.repairable, f"errno={excinfo.value.errno} 应当归入可修复类"

    async def test_db_error_is_sanitized(
        self, executor: SqlExecutor, validator: SqlValidator
    ) -> None:
        """详设 19.4：回灌给模型的错因必须已脱敏。

        这条 SQL 里的日期是**业务取值**；数据库的报错会把它原样回显，
        若不脱敏就会随修复 prompt 进 Trace 与日志。
        """
        from app.tools.sql.executor import SqlExecutionError

        validated = _validate(
            "SELECT SUM(s.net_amount) FROM fact_sales_order_item AS s "
            "WHERE s.order_date >= '2025-13-45'",
            validator,
            PermissionScope(role=UserRole.ADMIN),
        )
        with pytest.raises(SqlExecutionError) as excinfo:
            await executor.execute(validated)

        assert "2025-13-45" not in (excinfo.value.safe_detail or "")
        assert "2025-13-45" not in excinfo.value.repair_hint()
