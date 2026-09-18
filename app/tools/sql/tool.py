"""`sql_query` 工具：生成 → 校验 → 执行 → 自修复（详设 10.1 的子流程、10.8 的自修复）。

## 与「Tool 不自行决定重试」的关系

详设 9.3 明写「Tool 不自行决定是否重试，只返回 `retryable` 和错误类别，
由 Graph 路由统一控制」。本模块的**自修复**看似违反它，实则不是一回事：

| | 自修复（本模块） | 重试（由 Graph 控制） |
|---|---|---|
| 对象 | 模型生成的那一条 SQL | 整个工具调用 |
| 手段 | 带着错因重新生成 SQL | 再调一次工具 |
| 预算 | `LOOP__MAX_SQL_REPAIRS`（独立、不可借用） | 图的路由预算 |
| 对调用方 | **完全透明** | 调用方决定 |

调用方看到的只有「成功/失败、尝试了几次」，看不到修复过程。把生成与修复
放在 Tool 内部而不是拆成两个工具，是因为「模型没生成对」与「模型生成对了但
语义不成立」要用同一条修复路径处理——拆开会让调用方需要理解两者的区别，
而它并不需要。

## 三道门的顺序是固定的

```
生成 → 校验（12 步） → 执行
         ↑ 失败且可修复 ↓
         └── 修复（≤ LOOP__MAX_SQL_REPAIRS 次）
```

**校验永远在执行之前**，且修复后的 SQL 必须重新走完整校验（详设 10.8）。
不存在「已经校验过、只改了一点点所以可以直接执行」这条捷径——
那正是把校验器的价值作废的方式。

## 任何业务失败都返回 `ToolResult`，不抛异常

包括危险 SQL 被拦。只有「工具本身坏了」（目录文件读不出来、prompt 变量写错）
才让它抛出去。理由有两条：详设 9.2 就是这么定义这个结构的；
调用方要拿 `error.code` 去写任务状态与错误码，做成异常的话每个调用点
都得写一次 `except` 再转回来。
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final

from opentelemetry.trace import Span

from app.core.config import Settings
from app.core.errors import AgentError
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence
from app.domain.user import PermissionScope, UserRole
from app.infrastructure.model_gateway import ModelGateway, StructuredResult, TokenUsage
from app.infrastructure.observability import span
from app.tools.base import ErrorClass, ToolContext, ToolError, ToolName, ToolResult
from app.tools.sql.evidence import build_evidence
from app.tools.sql.executor import SqlExecutionError, SqlExecutor, SqlRunner
from app.tools.sql.generator import SqlGenerator
from app.tools.sql.schema_provider import SchemaProvider
from app.tools.sql.schemas import (
    SchemaCatalog,
    SchemaContext,
    SqlAttempt,
    SqlCandidate,
    SqlExecutionResult,
    SqlQueryArgs,
    SqlToolResult,
    ValidatedSql,
)
from app.tools.sql.validator import SqlValidationError, SqlValidator

#: 本工具的名字，写进 `agent_tool_call.tool_name` 与 Trace。
TOOL_NAME: Final[ToolName] = "sql_query"

#: 错误码 -> 详设 9.4 的错误类别。只覆盖「工具自己能产生的」那几个：
#: 模型限流与上游不可用在网关那侧已经分好类，这里只是把码翻回类别。
_CODE_CLASS: Final[dict[str, ErrorClass]] = {
    "SQL_VALIDATION_FAILED": "SECURITY",
    "SQL_EXECUTION_REPAIRABLE": "REPAIRABLE",
    "TASK_TIMEOUT": "TIMEOUT",
    "UPSTREAM_UNAVAILABLE": "TRANSIENT",
    "MODEL_RATE_LIMITED": "TRANSIENT",
    "MODEL_OUTPUT_INVALID": "VALIDATION",
    "PLAN_INVALID": "VALIDATION",
    "REDIS_UNAVAILABLE": "TRANSIENT",
}


@dataclass(frozen=True)
class _Outcome:
    """生成-校验-执行这条链路的产出：**要么有结果，要么有失败**。

    不用 `tuple[SqlToolResult | None, ToolResult | None]` 是因为那两个类型
    在脑内可以互换，调用方迟早会把它们写反，而写反的表现是「成功时返回失败」，
    只在特定分支才出现。`attempts` 无论成败都带出来——详设 6.6 施工项 5 要求
    每次尝试都落到 `agent_tool_call`，失败的那几次恰恰是最需要留痕的。
    """

    attempts: tuple[SqlAttempt, ...]
    result: SqlToolResult | None = None
    failure: ToolResult | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.failure is None):
            raise ValueError("_Outcome 必须恰好有 result 或 failure 之一")


class SqlQueryTool:
    """自然语言 → 可追溯的 SQL 结论。

    依赖全部由构造函数注入（网关、目录、执行器），因此单元测试可以用
    `FakeModelGateway` + 假执行器把整条链路跑完，不必连数据库也不必打网络。
    装配见 `build_sql_query_tool`。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        gateway: ModelGateway,
        catalog: SchemaCatalog,
        provider: SchemaProvider,
        generator: SqlGenerator,
        validator: SqlValidator,
        executor: SqlRunner,
    ) -> None:
        self.name: ToolName = TOOL_NAME
        self._settings = settings
        #: 公开只读：Phase 6 的冲突检测要用它把文档表头映射到 `metric_code`
        #: （`agent/nodes/conflict.py`）。给一个 property 而不是让调用方拿
        #: `_catalog`，是为了这个用途在签名上可见。
        self.catalog = catalog
        self._provider = provider
        self._generator = generator
        self._validator = validator
        self._executor = executor
        self._gateway = gateway

    # ------------------------------------------------------------------ 入口
    async def execute(self, args: SqlQueryArgs, ctx: ToolContext) -> ToolResult:
        """详设 9.1 的 `BaseTool.execute`。

        `ctx.permission_scope` 是**唯一**的数据权限来源。工具不读用户仓储、
        也不接受调用方另传一份范围——详设 10.5 把数据权限划在代码层，
        而代码层必须只有一个入口，否则「哪儿算错了」会变成一个查不清的问题。
        """
        started_at = datetime.now(UTC)
        call_id = new_id(IdPrefix.TOOL_CALL)
        scope = ctx.permission_scope

        with span(
            "tool.sql_query",
            **{
                "tool.name": TOOL_NAME,
                "tool.call_id": call_id,
                "task.id": ctx.task_id,
                "sql.scope_restricted": not scope.unrestricted,
                "sql.schema_version": self.catalog.version,
            },
        ) as current:
            try:
                context = self._provider.select(args.question, metric_codes=args.metric_codes)
            except AgentError as exc:
                return _failure_result(call_id, started_at, exc)

            outcome = await self._generate_and_run(args, ctx, call_id, context, current)
            current.set_attribute("sql.attempts", len(outcome.attempts))

            if outcome.failure is not None:
                return _with_attempts(outcome.failure, outcome.attempts)
            assert outcome.result is not None  # _Outcome 保证二者恰有其一
            current.set_attribute("sql.rows", outcome.result.row_count)
            current.set_attribute("sql.truncated", outcome.result.truncated)
            evidence = build_evidence(
                outcome.result,
                question=args.question,
                scope=scope,
                catalog=self.catalog,
                max_rows=self._settings.sql_tool.max_evidence_rows,
            )
            return _success_result(
                outcome.result, outcome.attempts, evidence, started_at=started_at
            )

    async def run_sql(
        self,
        sql: str,
        ctx: ToolContext,
        *,
        parameters: Mapping[str, str | int | float | date] | None = None,
    ) -> ToolResult:
        """直接跑一条**给定的** SQL：绕过生成，但**不绕过校验**。

        存在的理由有三个，都不是「为了方便」：

        1. **安全演示**：模型在正常对话下不会写出 `DROP TABLE`——这本身就是
           第一层防线在工作。因此「代码层拦得住」必须能脱离模型被单独证明，
           否则演示出来的只是「模型很乖」。给一条危险 SQL 直接喂进这条路径，
           拦下它的就只可能是 `validator.py`。
        2. **金标评测**：开发流程 6.6 的门禁是「金标 SQL 正确率达到阶段基线」，
           而结果集等价判定需要把金标 SQL 也真跑一遍。
        3. **Phase 6/7 的重放**：reflect 要复查上一步查到了什么时，
           重放一条已验证过的 SQL 比重新问一次模型便宜得多，也更可复现。

        **不做自修复**：修复的动作是「带着错因让模型重新生成」，
        而这条路径没有生成环节。失败就是失败——调用方要么换一条 SQL，
        要么走 `execute()` 那条有模型参与的路径。
        """
        started_at = datetime.now(UTC)
        call_id = new_id(IdPrefix.TOOL_CALL)
        try:
            validated = self._validator.validate(
                sql, parameters=parameters or {}, scope=ctx.permission_scope
            )
        except SqlValidationError as exc:
            return _with_attempts(
                _failure_result(call_id, started_at, exc),
                (
                    _rejected_attempt(
                        1, "GENERATE", _as_candidate(sql, parameters), exc, time.monotonic()
                    ),
                ),
            )

        try:
            execution = await self._executor.execute(validated)
        except (SqlExecutionError, AgentError) as exc:
            return _with_attempts(
                _failure_result(call_id, started_at, exc),
                (_failed_attempt(1, _as_candidate(sql, parameters), exc, time.monotonic()),),
            )

        result = SqlToolResult(
            call_id=call_id,
            normalized_sql=validated.normalized_sql,
            sql_fingerprint=validated.sql_fingerprint,
            columns=execution.columns,
            rows=execution.rows,
            row_count=execution.row_count,
            truncated=execution.truncated,
            duration_ms=execution.duration_ms,
            data_time_range=validated.data_time_range,
            warnings=tuple(validated.rewrites)
            + (("已按当前账号的数据权限限定查询范围",) if validated.scope_injected else ()),
            schema_version=self.catalog.version,
        )
        evidence = build_evidence(
            result,
            question=sql,
            scope=ctx.permission_scope,
            catalog=self.catalog,
            max_rows=self._settings.sql_tool.max_evidence_rows,
        )
        return _success_result(result, (), evidence, started_at=started_at)

    async def aclose(self) -> None:
        """释放执行器与模型的连接。生命周期归创建者，由装配点调用。"""
        await self._executor.aclose()
        await self._gateway.aclose()

    # ------------------------------------------------------- 生成 / 修复 / 执行
    async def _generate_and_run(
        self,
        args: SqlQueryArgs,
        ctx: ToolContext,
        call_id: str,
        context: SchemaContext,
        current: Span,
    ) -> _Outcome:
        attempts: list[SqlAttempt] = []
        repairs_left = self._settings.loop.max_sql_repairs
        started = time.monotonic()

        try:
            candidate = await self._generator.generate(
                question=args.question, objective=args.objective, context=context
            )
        except AgentError as exc:
            # 模型侧失败（限流、服务不可用、输出不合结构）已经被网关自己的
            # 两条预算处理过，到这里就是最终结论。**不进入自修复**：
            # 修复的前提是「手上有一条 SQL」，而此时一条都没有。
            return _Outcome(attempts=(), failure=_failure_result(call_id, _now(), exc))

        stage: str = "GENERATE"
        while True:
            attempt_no = len(attempts) + 1
            try:
                validated = self._validator.validate(
                    candidate.value.sql,
                    parameters=candidate.value.parameters,
                    scope=ctx.permission_scope,
                )
            except SqlValidationError as exc:
                attempts.append(_rejected_attempt(attempt_no, stage, candidate, exc, started))
                if not exc.repairable or repairs_left <= 0:
                    return _Outcome(
                        attempts=tuple(attempts), failure=_failure_result(call_id, _now(), exc)
                    )
                repairs_left -= 1
                candidate = await self._repair(
                    args, context, candidate.value.sql, exc.repair_hint()
                )
                stage = "REPAIR"
                continue

            try:
                execution = await self._executor.execute(validated)
            except SqlExecutionError as exc:
                attempts.append(_failed_attempt(attempt_no, candidate, exc, started))
                if not exc.repairable or repairs_left <= 0:
                    return _Outcome(
                        attempts=tuple(attempts), failure=_failure_result(call_id, _now(), exc)
                    )
                repairs_left -= 1
                candidate = await self._repair(
                    args, context, candidate.value.sql, exc.repair_hint()
                )
                stage = "REPAIR"
                continue
            except AgentError as exc:
                # 非 `SqlExecutionError` 的 AgentError：连接彻底不可用、被取消之类。
                # 详设 10.8 明确把它们排除在修复之外——重写 SQL 不会让数据库变得可达，
                # 也不会让它变快。这两个分支刻意分开而不是合成一个 `except`：
                # 合成之后 `repairable` 只能在运行期用 getattr 去猜，类型系统帮不上忙。
                attempts.append(_failed_attempt(attempt_no, candidate, exc, started))
                return _Outcome(
                    attempts=tuple(attempts), failure=_failure_result(call_id, _now(), exc)
                )

            attempts.append(
                SqlAttempt(
                    attempt_no=attempt_no,
                    stage="EXECUTE",
                    sql=candidate.value.sql,
                    normalized_sql=validated.normalized_sql,
                    sql_fingerprint=validated.sql_fingerprint,
                    status="SUCCEEDED",
                    duration_ms=execution.duration_ms,
                )
            )
            current.set_attribute(
                "sql.repairs_used", self._settings.loop.max_sql_repairs - repairs_left
            )
            return _Outcome(
                attempts=tuple(attempts),
                result=self._assemble(call_id, validated, execution, candidate, context, attempts),
            )

    async def _repair(
        self, args: SqlQueryArgs, context: SchemaContext, original_sql: str, hint: str
    ) -> StructuredResult[SqlCandidate]:
        return await self._generator.repair(
            question=args.question,
            objective=args.objective,
            context=context,
            original_sql=original_sql,
            error=hint,
        )

    # ------------------------------------------------------------------ 组装
    def _assemble(
        self,
        call_id: str,
        validated: ValidatedSql,
        execution: SqlExecutionResult,
        candidate: StructuredResult[SqlCandidate],
        context: SchemaContext,
        attempts: list[SqlAttempt],
    ) -> SqlToolResult:
        """把三样东西拼成 `SqlToolResult`。

        **每一条 warning 都对应一个「不写出来就查不出原因」的现象**：
        结果被截断了、LIMIT 是系统补的、权限谓词是系统注入的、一行都没查到。
        这些都不算失败，但都会让用户看到的数字与预期不同，
        因此必须随结果一起交出去。
        """
        warnings: list[str] = []
        if execution.truncated:
            warnings.append(
                f"结果已截断（返回 {execution.row_count} 行，数据库共 "
                f"{execution.fetched_count} 行），结论可能不完整"
            )
        warnings.extend(validated.rewrites)
        if validated.scope_injected:
            warnings.append("已按当前账号的数据权限限定查询范围")
        if execution.row_count == 0:
            warnings.append("查询未返回任何行，请确认时间区间、区域等过滤条件是否正确")
        elif all(value is None for row in execution.rows for value in row):
            # 聚合查询无匹配行时返回的是「一行 NULL」而不是零行——不加这条提示，
            # 用户看到的是一个看起来像有值的 NULL，而真实情况是查不到数据。
            warnings.append("查询未匹配到任何数据（聚合结果为 NULL）")

        return SqlToolResult(
            call_id=call_id,
            normalized_sql=validated.normalized_sql,
            sql_fingerprint=validated.sql_fingerprint,
            columns=execution.columns,
            rows=execution.rows,
            row_count=execution.row_count,
            truncated=execution.truncated,
            duration_ms=execution.duration_ms,
            data_time_range=validated.data_time_range,
            metric_codes=self._metric_codes(candidate, context),
            metric_definitions=self._metric_definitions(candidate, context),
            warnings=tuple(warnings),
            attempts=tuple(attempts),
            schema_version=self.catalog.version,
        )

    def _metric_codes(
        self, candidate: StructuredResult[SqlCandidate], context: SchemaContext
    ) -> tuple[str, ...]:
        """本次查询用到的指标 code。

        优先取**模型自报**的 `metric_codes`：那是它实际用的口径，与
        「按别名匹配到的」不一定相同。模型没报时才退回到上下文里的指标——
        这样「模型用了什么口径」与「我们给了什么口径」的差异是可观测的，
        而不是被我们单方面覆盖掉。
        """
        declared = [code for code in candidate.value.metric_codes if self.catalog.metric(code)]
        if declared:
            return tuple(declared)
        return tuple(metric.code for metric in context.metrics)

    def _metric_definitions(
        self, candidate: StructuredResult[SqlCandidate], context: SchemaContext
    ) -> tuple[str, ...]:
        """口径的可读说明（详设 10.7 的 `metric_definitions`）。与 code 一一对应。"""
        return tuple(
            f"{metric.name}({metric.code}) = {metric.expression}｜口径版本 {metric.version}"
            for code in self._metric_codes(candidate, context)
            if (metric := self.catalog.metric(code)) is not None
        )


# ------------------------------------------------------------------ 结果构造
def _now() -> datetime:
    return datetime.now(UTC)


def _success_result(
    result: SqlToolResult,
    attempts: tuple[SqlAttempt, ...],
    evidence: Sequence[Evidence],
    *,
    started_at: datetime,
) -> ToolResult:
    """成功（或部分成功）的 `ToolResult`。

    三种状态的判定依据：

    - `SUCCEEDED`：正常拿到完整结果；
    - `SUCCEEDED` + 空结果：详设 9.4 的 `EMPTY_RESULT` **不算失败**——
      SQL 正确执行了，只是没有匹配的行。是否补证、澄清还是受限回答
      由 Reviewer 判断（9.4 原文），Tool 不替它决定，只在 warnings 里留信号；
    - `PARTIAL`：拿到了结果但被截断。这是详设 9.2 定义第三种状态的全部意义。

    ## `payload` 里为什么带着原始行

    详设 10.6 说「SQL 结果不直接写应用日志，也不默认持久化原始行」——
    这里的 payload **不是持久化**，它是 State 内的载体：Phase 6/7 的 Analysis
    节点必须看到结果行才能写出结论，把行从这里摘掉等于让那条链路无米下锅。

    「不持久化」发生在**落库边界**：16.6 规定 `agent_tool_call.result_summary_json`
    只存行数、列名与摘要，Trace 事件同样只记摘要。因此这条纪律的实现点在
    Phase 6/7 的 Tool 执行器，不在 Tool 自己——由写库的人决定写什么，
    比由产出数据的人猜「调用方会怎么用」可靠。
    """
    if result.is_empty:
        status = "SUCCEEDED"
        summary = "查询未命中任何数据"
    elif result.truncated:
        status = "PARTIAL"
        summary = f"查询返回 {result.row_count} 行（已截断）"
    else:
        status = "SUCCEEDED"
        summary = f"查询返回 {result.row_count} 行"

    return ToolResult(
        call_id=result.call_id,
        tool=TOOL_NAME,
        status=status,
        started_at=started_at,
        finished_at=_now(),
        summary=summary,
        # `is_empty` **必须显式放进 payload**：它是 `SqlToolResult` 的
        # `@property`，而 `model_dump` 只序列化字段，property 不会进去。
        # 调用方（Phase 6 的 `reflect`）要靠它区分"查了但没数据"与"查成了"——
        # 两者在 `status` 上都是 SUCCEEDED，只看状态分不出来。
        payload={**result.model_dump(mode="json"), "is_empty": result.is_empty},
        evidence=list(evidence),
    )


def _failure_result(call_id: str, started_at: datetime, exc: AgentError) -> ToolResult:
    """失败（含被安全规则拦下）的 `ToolResult`。

    `safe_detail` 进的是 `ToolError.safe_detail` 而不是 `message`：
    `message` 面向用户，「查询涉及当前账号无权访问的字段」这句话要能看懂，
    但**不能带上具体是哪个字段**——那本身就是一条越权信息。
    """
    error_class: ErrorClass = getattr(exc, "error_class", None) or _class_of_code(exc.code.value)
    return ToolResult(
        call_id=call_id,
        tool=TOOL_NAME,
        status="FAILED",
        started_at=started_at,
        finished_at=_now(),
        summary=exc.message,
        error=ToolError(
            code=exc.code.value,
            message=exc.message,
            error_class=error_class,
            retryable=exc.retryable,
            safe_detail=getattr(exc, "safe_detail", None),
        ),
    )


def _with_attempts(result: ToolResult, attempts: tuple[SqlAttempt, ...]) -> ToolResult:
    """把尝试记录挂到失败的 `ToolResult` 上。

    走 `payload` 而不是新加一个字段：`payload` 本来就是「工具自己的结果数据」
    的落点，而尝试记录的形状就是 `agent_tool_call` 的一张行，
    Phase 6/7 的 Tool 执行器拿到后逐条落库即可。
    """
    return result.model_copy(
        update={"payload": {"attempts": [a.model_dump(mode="json") for a in attempts]}}
    )


# ------------------------------------------------------------------ 尝试记录
def _as_candidate(
    sql: str, parameters: Mapping[str, str | int | float | date] | None
) -> StructuredResult[SqlCandidate]:
    """把一条给定的 SQL 包装成 `SqlCandidate` 结果，只为复用尝试记录的构造。

    尝试记录要的是「这条 SQL 长什么样」，而它恰好与 `SqlCandidate` 的字段
    对得上。另造一个类型只会让 `_rejected_attempt` / `_failed_attempt`
    多一份分支，而这两个函数的存在意义正是「所有尝试都记成同一种形状」。
    """
    return StructuredResult[SqlCandidate](
        value=SqlCandidate(sql=sql, parameters=parameters or {}),
        usage=TokenUsage(),
        model="",
        prompt_version="",
        duration_ms=0,
        attempts=0,
    )


def _rejected_attempt(
    attempt_no: int,
    stage: str,
    candidate: StructuredResult[SqlCandidate],
    exc: SqlValidationError,
    started: float,
) -> SqlAttempt:
    """被校验拦下的一次尝试。

    `status=REJECTED` 而不是 `FAILED`：它**没有碰过数据库**。
    这两件事的排查方向完全不同，合成一个会让「是谁拦的」变得不可见。
    """
    return SqlAttempt(
        attempt_no=attempt_no,
        stage="REPAIR" if stage == "REPAIR" else "GENERATE",
        sql=candidate.value.sql,
        status="REJECTED",
        error_code=exc.code.value,
        error_class=exc.error_class,
        error_summary=exc.repair_hint(),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _failed_attempt(
    attempt_no: int,
    candidate: StructuredResult[SqlCandidate],
    exc: AgentError,
    started: float,
) -> SqlAttempt:
    error_class: ErrorClass = getattr(exc, "error_class", None) or _class_of_code(exc.code.value)
    return SqlAttempt(
        attempt_no=attempt_no,
        stage="EXECUTE",
        sql=candidate.value.sql,
        status="FAILED",
        error_code=exc.code.value,
        error_class=error_class,
        error_summary=getattr(exc, "safe_detail", None) or exc.message,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _class_of_code(code: str) -> ErrorClass:
    return _CODE_CLASS.get(code, "INTERNAL")


# ------------------------------------------------------------------ 装配点
def build_sql_query_tool(settings: Settings, gateway: ModelGateway) -> SqlQueryTool:
    """按配置装配一个可用的 SQL 工具。

    命名沿用 `build_model_gateway` / `build_job_queue` 的惯例。
    这里是**唯一**读 `SQL_TOOL__*` 来拼装各组件的地方：散在多处的结果
    一定是「某条路径用了不同的上限」，而那种差异只在特定输入下才显形。
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
        executor=SqlExecutor(settings, catalog=catalog),
    )


def build_cli_context(*, region_ids: tuple[str, ...], timeout_seconds: int) -> ToolContext:
    """给 CLI 造一个最小上下文。

    CLI 没有登录用户、也没有任务，但 `ToolContext` 的字段一个都不能少——
    与其让工具内部到处判断「是不是 CLI 调用」，不如在入口处造一个诚实的上下文。
    这里的 `task_id` / `trace_id` 是**一次性**的随机 ID，不落库，
    因此不会污染任务列表；`deadline_at` 按 `TASK_TIMEOUT_SECONDS` 给，
    与真实任务同一口径。`region_ids` 为空即全量，与 `DEMO_ACCOUNTS`
    里 admin 的取值语义一致。
    """
    return ToolContext(
        user_id="cli",
        task_id=new_id(IdPrefix.TASK),
        step_id=new_id(IdPrefix.TASK),
        trace_id=new_id(IdPrefix.TRACE),
        permission_scope=PermissionScope(role=UserRole.ANALYST, region_ids=region_ids),
        deadline_at=datetime.now(UTC) + timedelta(seconds=timeout_seconds),
    )


__all__ = ["TOOL_NAME", "SqlQueryTool", "build_cli_context", "build_sql_query_tool"]


__all__ = [
    "TOOL_NAME",
    "SqlQueryTool",
    "build_cli_context",
    "build_sql_query_tool",
]
