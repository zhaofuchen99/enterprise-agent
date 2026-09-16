"""`SqlQueryTool` 编排与自修复（详设 10.8 / 9.4）。

这里测的是**调用方的行为**，因此模型与执行器都必须能被精确控制：
「固定返回同一条烂 SQL」「第 2 次执行时返回除零错误」这类场景真模型做不到，
而它们正是自修复预算、终止条件、尝试记录这三件事唯一能被验证的方式。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import InvalidRequestError, OperationalError, StatementError

from app.core.config import Settings
from app.domain.evidence import Evidence
from app.tests.fakes import FakeModelGateway, FakeSqlRunner
from app.tools.base import ToolContext
from app.tools.sql.executor import SqlExecutionError, _sanitize_db_error
from app.tools.sql.generator import SqlGenerator
from app.tools.sql.schema_provider import SchemaProvider
from app.tools.sql.schemas import (
    ResultColumn,
    SqlCandidate,
    SqlExecutionResult,
    SqlQueryArgs,
    SqlToolResult,
)
from app.tools.sql.tool import SqlQueryTool
from app.tools.sql.validator import SqlValidator

#: 一条能过全部 12 步的 SQL。
GOOD_SQL = (
    "SELECT SUM(s.net_amount) AS net_sales FROM fact_sales_order_item AS s "
    "WHERE s.order_date >= :start_date AND s.order_date < :end_date"
)
GOOD_PARAMS = {"start_date": "2025-07-01", "end_date": "2025-10-01"}

#: 一次成功的执行结果。
_EXECUTED_AT = datetime(2026, 9, 16, tzinfo=UTC)


def _candidate(sql: str = GOOD_SQL, **overrides: object) -> SqlCandidate:
    payload: dict[str, object] = {
        "sql": sql,
        "parameters": GOOD_PARAMS,
        "selected_tables": ["fact_sales_order_item"],
        "selected_columns": ["net_amount", "order_date"],
        "metric_codes": ["net_sales"],
        "expected_columns": ["net_sales"],
        "explanation": "净销售额按含税减折扣减退货计",
    }
    payload.update(overrides)
    return SqlCandidate.model_validate(payload)


def _result(rows: list[tuple[object, ...]], *, truncated: bool = False) -> SqlExecutionResult:
    return SqlExecutionResult(
        columns=(ResultColumn(name="net_sales", data_type="DECIMAL(14,2)"),),
        rows=tuple(tuple(row) for row in rows),
        row_count=len(rows),
        truncated=truncated,
        duration_ms=12,
        fetched_count=len(rows),
        started_at=_EXECUTED_AT,
    )


def build_tool(
    settings: Settings,
    gateway: FakeModelGateway,
    runner: FakeSqlRunner,
) -> SqlQueryTool:
    """按生产装配路径构造，只把网关与执行器换成替身。

    刻意用真实的 `SchemaProvider` / `SqlValidator` / `SqlGenerator`：
    它们是纯逻辑，替身只会让「校验规则写错了」这类缺陷逃过测试。
    """
    provider = SchemaProvider.from_settings(settings)
    catalog = provider.catalog
    tuning = settings.sql_tool
    return SqlQueryTool(
        settings=settings,
        gateway=gateway,
        catalog=catalog,
        provider=provider,
        generator=SqlGenerator(gateway, max_rows=tuning.max_rows),
        validator=SqlValidator(
            catalog, max_sql_chars=tuning.max_sql_chars, max_rows=tuning.max_rows
        ),
        executor=runner,
    )


@pytest.fixture
def runner() -> FakeSqlRunner:
    return FakeSqlRunner()


@pytest.fixture
def tool(settings: Settings, gateway: FakeModelGateway, runner: FakeSqlRunner) -> SqlQueryTool:
    return build_tool(settings, gateway, runner)


ARGS = SqlQueryArgs(question="2025 年 Q3 的净销售额是多少")


class TestHappyPath:
    async def test_succeeds_on_first_attempt(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        gateway.responses = [_candidate()]
        runner.results = [_result([("111967031.73",)])]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "SUCCEEDED"
        assert result.tool == "sql_query"
        assert result.error is None
        assert result.evidence, "有结果就必须有证据"
        assert "111967031.73" in result.evidence[0].claim

    async def test_executes_rewritten_sql(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """模型原文里没有 LIMIT，执行器必须收到补过 LIMIT 的那一份。

        少写下限会是「查了 50 万行才被行数截断」——而那是执行期的事，
        与「校验器补了 LIMIT」是两回事，两者都要成立。
        """
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",)])]

        await tool.execute(ARGS, ctx)

        assert "LIMIT 1000" in runner.calls[0].validated_sql

    async def test_restricted_user_executes_scoped_sql(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",)])]

        await tool.execute(ARGS, ctx)

        assert "scope_region" in runner.calls[0].validated_sql
        assert runner.calls[0].bind_parameters["scope_region_0_0"] == "华东"

    async def test_attempts_recorded(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """详设 6.6 施工项 5：每次尝试都写 `agent_tool_call`。

        Phase 4 的 Tool 不落库（没有 `task_id`/`step_id` 的持有者），
        因此这次尝试必须原样出现在结果里，由 Phase 6/7 的执行器逐条写下去。
        """
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",)])]

        result = await tool.execute(ARGS, ctx)

        attempts = (result.payload or {})["attempts"]
        assert len(attempts) == 1
        assert attempts[0]["stage"] == "EXECUTE"
        assert attempts[0]["status"] == "SUCCEEDED"
        assert attempts[0]["normalized_sql"]
        assert attempts[0]["sql_fingerprint"]


class TestEmptyAndTruncated:
    async def test_empty_result_is_not_failure(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """详设 9.4 的 `EMPTY_RESULT`：SQL 正确执行了，只是没有匹配的行。

        是否补证、澄清还是受限回答由 Reviewer 判断——Tool 不替它决定，
        因此状态仍是 SUCCEEDED，只把信号放进 warnings。
        """
        gateway.responses = [_candidate()]
        runner.results = [_result([])]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "SUCCEEDED"
        assert not result.evidence, "空结果不生成证据：没有证据好过一条误导性的证据"
        assert any("未返回任何行" in w for w in (result.payload or {})["warnings"])

    async def test_all_null_aggregate_counts_as_empty(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """聚合查询无匹配行时返回的是**一行 NULL**，不是零行。

        只看 `row_count == 0` 会把它当成「有一行结果」，于是工具会为
        `net_sales=None` 生成一条证据、也不给用户任何提示——
        而真实情况是「按当前条件查不到数据」。受限用户查别的区域时，
        那本该是一个明确的空结果，而不是一个看起来像有值的 NULL。
        """
        gateway.responses = [_candidate()]
        runner.results = [_result([(None,)])]

        result = await tool.execute(ARGS, ctx)

        assert not result.evidence
        assert any("NULL" in w for w in (result.payload or {})["warnings"])

    async def test_partial_null_is_not_empty(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """有空列的普通结果仍然是结果——按「全空才算空」判，不按「有 NULL 就算空」。"""
        gateway.responses = [_candidate()]
        runner.results = [_result([(None,), ("1",)])]

        result = await tool.execute(ARGS, ctx)

        assert len(result.evidence) == 2

    async def test_truncated_result_is_partial(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """详设 9.2 定义 `PARTIAL` 状态的全部意义：有结果但不完整。

        把它标成 SUCCEEDED 会让下游以为拿到的是全量，据此得出错误的结论。
        """
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",), ("2",)], truncated=True)]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "PARTIAL"
        assert any("截断" in w for w in (result.payload or {})["warnings"])

    async def test_scope_injection_reported_in_warnings(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """静默注入会让「为什么只查到华东的数据」变成一个查不出来的谜。"""
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",)])]

        result = await tool.execute(ARGS, ctx)

        assert any("数据权限" in w for w in (result.payload or {})["warnings"])


class TestSelfRepair:
    async def test_repair_after_repairable_error(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """第一次 `SELECT *` 被拦（可修复），修复后通过。

        修复后的 SQL 必须**重新走完整校验**，因此第二次的产物里同样有 LIMIT。
        """
        gateway.responses = [
            _candidate("SELECT * FROM fact_sales_order_item"),
            _candidate(),
        ]
        runner.results = [_result([("1",)])]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "SUCCEEDED"
        attempts = (result.payload or {})["attempts"]
        assert [a["status"] for a in attempts] == ["REJECTED", "SUCCEEDED"]
        # stage 记的是「判定发生在哪」：第一次被校验拦在生成之后，
        # 第二次过了校验，成败由执行决定——因此是 EXECUTE 而不是 REPAIR。
        assert attempts[0]["stage"] == "GENERATE"
        assert attempts[1]["stage"] == "EXECUTE"
        assert attempts[0]["error_code"] == "SQL_VALIDATION_FAILED"
        assert attempts[0]["normalized_sql"] is None, "被拦下的 SQL 没有规范化产物"
        assert attempts[1]["normalized_sql"], "执行成功的那条要留下指纹与规范化文本"

    async def test_repair_prompt_carries_sql_and_reason(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """详设 10.8：修复 prompt 只包含原 SQL、脱敏错误、同一 Schema 与规则摘要。"""
        gateway.responses = [_candidate("SELECT * FROM fact_sales_order_item"), _candidate()]
        runner.results = [_result([("1",)])]

        await tool.execute(ARGS, ctx)

        repair_call = gateway.calls[1]
        assert repair_call.prompt_name == "sql_repair"
        assert "SELECT * FROM fact_sales_order_item" in repair_call.rendered
        assert "第 7 步校验未通过" in repair_call.rendered

    async def test_stops_when_repair_budget_exhausted(
        self, settings: Settings, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """`LOOP__MAX_SQL_REPAIRS` 是**独立预算**，用尽即止。

        脚本里只放一条烂 SQL，替身会重复返回最后一项——「模型反复犯同一个错」
        正是这条预算存在的理由。
        """
        settings.loop.max_sql_repairs = 2
        tool = build_tool(settings, gateway, runner)
        gateway.responses = [_candidate("SELECT * FROM fact_sales_order_item")]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "FAILED"
        assert result.error is not None and result.error.code == "SQL_VALIDATION_FAILED"
        # 1 次生成 + 2 次修复 = 3 次尝试，一次都不多
        assert len((result.payload or {})["attempts"]) == 3
        assert not runner.calls, "被拦下的 SQL 一次都不该进入执行器"

    async def test_unrepairable_error_consumes_no_budget(
        self, settings: Settings, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """详设 10.8：危险语句、非白名单表、敏感字段直接终止，**不给第二次机会**。

        再生成一次等于把一个已经表现出注入倾向的输入再喂模型一次。
        """
        settings.loop.max_sql_repairs = 2
        tool = build_tool(settings, gateway, runner)
        gateway.responses = [_candidate("DROP TABLE fact_sales_order_item")]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "FAILED"
        assert result.error is not None and result.error.error_class == "SECURITY"
        assert len((result.payload or {})["attempts"]) == 1
        assert len(gateway.calls) == 1, "不可修复的错误不得再问一次模型"

    async def test_execution_error_enters_repair(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """AST 校验看不到语义错误，数据库看得到——两条路径共用同一份修复预算。"""
        gateway.responses = [_candidate(), _candidate()]
        runner.results = [
            SqlExecutionError(
                "除零", error_class="REPAIRABLE", safe_detail="division by zero", errno=1690
            ),
            _result([("1",)]),
        ]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "SUCCEEDED"
        attempts = (result.payload or {})["attempts"]
        assert [a["status"] for a in attempts] == ["FAILED", "SUCCEEDED"]
        assert attempts[0]["error_class"] == "REPAIRABLE"
        assert attempts[0]["error_summary"] == "division by zero"

    async def test_connection_failure_skips_repair(
        self, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext, settings: Settings
    ) -> None:
        """详设 10.8 明确把连接失败排除在修复之外：重写 SQL 不会让数据库变得可达。"""
        tool = build_tool(settings, gateway, runner)
        gateway.responses = [_candidate()]
        runner.results = [
            SqlExecutionError("连不上", error_class="TRANSIENT", safe_detail="OperationalError")
        ]

        result = await tool.execute(ARGS, ctx)

        assert result.status == "FAILED"
        assert result.error is not None and result.error.error_class == "TRANSIENT"
        assert len(gateway.calls) == 1, "连接失败不得再问一次模型"

    async def test_model_failure_skips_repair(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """修复的前提是「手上有一条 SQL」，而模型根本没给出 SQL。"""
        from app.tests.fakes import FakeFailure

        gateway.failure = FakeFailure.RATE_LIMITED

        result = await tool.execute(ARGS, ctx)

        assert result.status == "FAILED"
        assert result.error is not None and result.error.code == "MODEL_RATE_LIMITED"
        assert result.error.retryable, "限流是客户端可以稍后重试的"
        assert not (result.payload or {}).get("attempts"), "根本没生成过 SQL，没有尝试可记"

    async def test_unknown_metric_code_blocks_before_generation(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """上游给错了口径时不能静默忽略——忽略之后模型会自己猜一个算法。"""
        result = await tool.execute(SqlQueryArgs(question="销售额", metric_codes=("no_such",)), ctx)

        assert result.status == "FAILED"
        assert result.error is not None and result.error.code == "PLAN_INVALID"
        assert not gateway.calls, "口径没定下来就不该问模型"


class _FakeDBAPIError(Exception):
    """DBAPI 层异常的形状：第一个参数是 errno，第二个是消息。

    `_sanitize_db_error` 只从 `exc.orig.args` 里取东西，因此替身必须复现
    这个形状。用 `SQLAlchemyError("...")` 是不行的——它的 `args` 是空的，
    脱敏函数会直接返回 `None`，测试看着在跑其实一次都没进逻辑。
    """

    def __init__(self, errno: int | None, message: str) -> None:
        super().__init__(errno, message) if errno is not None else super().__init__(message)


def _db_error(errno: int | None, message: str) -> OperationalError:
    """造一个形状与真实 DBAPI 异常一致的 `OperationalError`。"""
    return OperationalError("SELECT 1", {}, _FakeDBAPIError(errno, message))


def _bind_error(message: str) -> StatementError:
    """SQLAlchemy 在**绑定阶段**失败时抛的就是它（拿不到 MySQL errno）。"""
    return StatementError(message, "SELECT 1", {}, InvalidRequestError(message))


class TestErrorSanitizing:
    """详设 19.4：数据库报错进修复 prompt / Trace / 日志之前必须脱敏。

    这一组用例的对象是**脱敏的边界**：值必须抹掉，而修复需要的标识符
    （列名、参数名）必须留下。抹掉了值只是提示变差，留住了值就是数据泄露，
    两者的严重性不对称——所以规则画在「这段引号里是不是取值」上。
    """

    def test_literal_values_are_stripped(self) -> None:
        cleaned = _sanitize_db_error(
            _db_error(1525, "Incorrect DATE value: '2025-13-45' for column `s`.`order_date`")
        )
        assert cleaned is not None
        assert "2025-13-45" not in cleaned
        assert "order_date" in cleaned, "列名是修复需要的，不能一起抹掉"

    def test_bind_parameter_name_is_kept(self) -> None:
        """一次真实缺陷：一刀切抹引号把参数名也抹成了 `'…'`。

        `A value is required for bind parameter 'start_date'` 里的引号括的是
        **参数名**，而那是修复唯一需要的东西——抹掉之后模型只能靠猜。
        """
        cleaned = _sanitize_db_error(
            _bind_error("A value is required for bind parameter 'start_date'")
        )
        assert cleaned is not None
        assert "start_date" in cleaned
        assert "'…'" not in cleaned

    def test_literal_and_parameter_name_together(self) -> None:
        cleaned = _sanitize_db_error(
            _db_error(1265, "Incorrect DATE value: '2025-13-45' for bind parameter 'start_date'")
        )
        assert cleaned is not None
        assert "2025-13-45" not in cleaned
        assert "start_date" in cleaned

    def test_no_message_returns_none(self) -> None:
        """拿不到消息时返回 None 而不是空串：调用方据此回落到通用文案。"""
        assert _sanitize_db_error(_db_error(None, "")) is None

    def test_long_message_is_truncated(self) -> None:
        """数据库偶尔吐一大段，而修复 prompt 的预算按 token 算。"""
        cleaned = _sanitize_db_error(_db_error(1064, "x" * 5000))
        assert cleaned is not None and len(cleaned) <= 300


class TestEvidence:
    async def test_one_evidence_per_row(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """每一行是彼此独立的事实，揉成一条会让下游无法引用其中某一个数。"""
        gateway.responses = [_candidate()]
        runner.results = [_result([("华南", 1), ("华东", 2)])]

        result = await tool.execute(ARGS, ctx)

        assert len(result.evidence) == 2
        assert [e.locator["result_slice"] for e in result.evidence] == [[0, 1], [1, 2]]

    async def test_evidence_has_locator_and_definition(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """13.1：`locator` 用 `call_id + sql_fingerprint + result_slice`，
        口径版本必须随证据落下来——事后再查目录，查到的是「今天」的版本。"""
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",)])]

        evidence: Evidence = (await tool.execute(ARGS, ctx)).evidence[0]

        assert evidence.source_type == "SQL"
        assert evidence.reliability == "HIGH"
        assert evidence.metric_code == "net_sales"
        assert evidence.definition_version == "v1.2"
        assert set(evidence.locator) >= {"call_id", "sql_fingerprint", "result_slice"}
        assert evidence.event_time is not None, "有日期条件时必须带上业务时间区间"

    async def test_restricted_evidence_records_data_scope(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """「这个数是全量还是只有华东」必须是证据自带的属性。

        不标注的话，两个来源的数字一旦不同就会被 Phase 9 报成 VALUE 冲突。
        """
        gateway.responses = [_candidate()]
        runner.results = [_result([("1",)])]

        evidence = (await tool.execute(ARGS, ctx)).evidence[0]

        assert evidence.scope["data_scope"] == ["华东"]

    async def test_degrades_to_summary_evidence(
        self, settings: Settings, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        """查了 1000 行明细时逐行生成会炸出 1000 条证据，而它们贡献等价。"""
        settings.sql_tool.max_evidence_rows = 3
        tool = build_tool(settings, gateway, runner)
        gateway.responses = [_candidate()]
        runner.results = [_result([(str(i),) for i in range(10)])]

        result = await tool.execute(ARGS, ctx)

        assert len(result.evidence) == 1
        assert "共 10 行" in result.evidence[0].claim
        assert result.evidence[0].locator["result_slice"] == [0, 10]


class TestRunSql:
    """`run_sql`：绕过生成、**不绕过校验**。"""

    async def test_run_sql_still_validates(
        self, tool: SqlQueryTool, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        result = await tool.run_sql("DROP TABLE fact_sales_order_item", ctx)

        assert result.status == "FAILED"
        assert result.error is not None
        assert result.error.code == "SQL_VALIDATION_FAILED"
        assert not runner.calls, "被拦下的 SQL 不该碰到数据库"

    async def test_run_sql_does_not_call_model(
        self, tool: SqlQueryTool, gateway: FakeModelGateway, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        runner.results = [_result([("1",)])]

        result = await tool.run_sql(GOOD_SQL, ctx, parameters=GOOD_PARAMS)

        assert result.status == "SUCCEEDED"
        assert not gateway.calls, "这条路径的要点就是不经模型"

    async def test_run_sql_injects_scope(
        self, tool: SqlQueryTool, runner: FakeSqlRunner, ctx: ToolContext
    ) -> None:
        runner.results = [_result([("1",)])]

        await tool.run_sql("SELECT SUM(net_amount) FROM fact_sales_order_item", ctx)

        assert "scope_region" in runner.calls[0].validated_sql


def test_build_produces_usable_tool(settings: Settings, gateway: FakeModelGateway) -> None:
    """`build_sql_query_tool` 是生产路径的唯一装配处，它必须能真的建出东西。"""
    from app.tools.sql.tool import build_sql_query_tool

    tool = build_sql_query_tool(settings, gateway)
    assert tool.name == "sql_query"


def test_result_model_is_json_serializable() -> None:
    """`payload` 要穿过 State 的 JSON 边界，`model_dump(mode="json")` 不能炸。

    金额是 `str` 而不是 `Decimal` 正是为了这一步（见 `executor._normalize`），
    这条断言把这个约定钉住——改回 Decimal 会在运行期才暴露。
    """
    payload = SqlToolResult(
        call_id="tcl_x",
        normalized_sql="SELECT 1",
        sql_fingerprint="f" * 64,
        columns=(ResultColumn(name="a", data_type="INT"),),
        rows=(("111967031.73",),),
        row_count=1,
        truncated=False,
        duration_ms=1,
    )
    dumped = payload.model_dump(mode="json")
    assert dumped["rows"] == [["111967031.73"]]
