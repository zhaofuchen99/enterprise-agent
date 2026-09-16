"""只读 SQL 执行器（详细设计 10.6）。

**这一层是「安全」的最后一道，也是唯一一道不依赖我们自己的代码是否写对的道。**
详设 10.5 的分层里，数据库层要保证「只读账号、允许库表、连接超时、资源限制」——
即使 `validator.py` 有漏洞放过了一条写语句，它也会在这里被数据库拒绝。
两条独立的防线各自成立，才谈得上「安全由代码保证」而不是「安全由某一处保证」。

## 四件事必须在这一层做

1. **独立 engine**：不复用 Agent 运行库的连接池。业务库是只读账号、
   Agent 库是读写账号，混用池子会让「这条连接到底是哪个账号」不可推理。
2. **会话级只读事务 + 最大执行时间**：在连接上设置，而不是每条 SQL 前面拼一句
   `SET`——拼文本等于让模型生成的语句有机会影响会话设置。
3. **双重截断**：行数与序列化后的字节数。只限行数不够——一行里塞一个
   MEDIUMTEXT 就足以把内存打满。
4. **结果不进日志**：详设 10.6 明文要求。本模块因此不打任何 `logger.info`，
   只在失败时抛异常，且异常里不带行数据。

## 值为什么被规范化成 JSON 原生类型

`Decimal` / `date` / `bytes` 不能直接穿过 LangGraph State 的序列化边界
（Trace 与 `payload` 都要 dump 成 JSON）。因此这里统一转成 JSON 原生类型，
**金额转 `str` 而不是 `float`**：二进制浮点表示不了 0.1，
一次「净销售额对不上」的排查如果从浮点误差开始，方向就永远回不到口径上。
`str(Decimal)` 是精确的，下游要比数值时自己 `Decimal(...)` 解析即可。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Final, Protocol

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.infrastructure.db import create_engine
from app.tools.base import ErrorClass
from app.tools.sql.schemas import ResultColumn, SchemaCatalog, SqlExecutionResult, ValidatedSql

#: 会话级最大执行时间（毫秒，MySQL 的 `max_execution_time` 语义，只作用于 SELECT）。
_SET_MAX_EXECUTION_TIME: Final[str] = "SET SESSION max_execution_time = :milliseconds"

#: 把会话切到只读事务。此后本连接上的**任何**写操作都会被数据库拒绝，
#: 报 `ER_CANT_EXECUTE_IN_READ_ONLY_TRANSACTION`。
_SET_READ_ONLY: Final[str] = "SET SESSION TRANSACTION READ ONLY"

#: 通过 AST 校验、却在数据库层仍会失败的**可修复**错误码（详设 10.8 的
#: 「数据类型不匹配、合法聚合规则错误」）。这些错误的共同点是：
#: SQL 的结构没问题，是某处写法在语义上不成立，**改一改就能过**。
#:
#: 为什么不复用「校验失败」那条路：AST 校验发生在执行之前，看不到语义错误；
#: 而数据库看到了，它给出的 errno 就是最准的错因分类，没有理由重新推断一遍。
_REPAIRABLE_ERRNOS: Final[frozenset[int]] = frozenset(
    {
        1052,  # ER_NON_UNIQ_ERROR            列名在多表间有歧义
        1054,  # ER_BAD_FIELD_ERROR           不存在的列
        1064,  # ER_PARSE_ERROR               语法错误
        1140,  # ER_MIX_OF_GROUP_FUNC_AND_FIELDS  聚合与非聚合混用
        1146,  # ER_NO_SUCH_TABLE             不存在的表
        1241,  # ER_OPERAND_COLUMNS           操作数个数不匹配
        1247,  # ER_ILLEGAL_REFERENCE         引用了不允许引用的东西
        1292,  # ER_TRUNCATED_WRONG_VALUE     数值字面量非法
        # 1525 ER_WRONG_VALUE：MySQL 8 对**非法日期字面量**报的是它而不是 1292
        # （`WHERE order_date >= '2025-13-45'` 实测）。这张表是按真实 errno 写的，
        # 因此集成用例里专门有一条断言 errno 取值——猜错的代价是「可修复的错
        # 被当成不可修复」，表现为修复预算根本没被用上就失败了。
        1525,
        1582,  # ER_WRONG_PARAMCOUNT_TO_NATIVE_FCT  函数参数个数不对
        1690,  # ER_DATA_OUT_OF_RANGE         除零等越界
    }
)

#: MySQL 报错里的字符串字面量。数据库的报错形如
#: ``Incorrect DATE value: '2025-13-01' for column `s`.`order_date` ``——
#: 反引号里是**标识符**（对修复有用），单引号里是**取值**（就是业务数据，不能带出去）。
_DB_STRING_LITERAL: Final[re.Pattern[str]] = re.compile(r"'(?:[^'\\]|\\.)*'")

#: SQLAlchemy 报「绑定参数缺值」时的固定句式，见 `_sanitize_db_error`。
#: 只认这一种句式：它是**参数名**的标记，而参数名对修复有用、又不含业务取值。
_BIND_PARAMETER_NAME: Final[re.Pattern[str]] = re.compile(r"(bind parameter\s+)'((?:[^'\\]|\\.)*)'")

#: 回灌给模型的错因长度上限。数据库偶尔会吐出一大段，
#: 而修复 prompt 的预算是按 token 算的。
_MAX_HINT_CHARS: Final[int] = 300


class SqlExecutionError(AgentError):
    """执行期失败，带「可否修复」的分类。

    与 `SqlValidationError` 分开而不是共用一个类型：两者的**来源**不同
    （我们的规则 vs 数据库），排查时要看的东西也不同，
    合成一个会让日志里分不出「被我们拦了」和「数据库不认」。
    """

    def __init__(
        self,
        message: str,
        *,
        error_class: ErrorClass,
        safe_detail: str | None = None,
        errno: int | None = None,
    ) -> None:
        super().__init__(
            ErrorCode.SQL_EXECUTION_REPAIRABLE
            if error_class == "REPAIRABLE"
            else ErrorCode.UPSTREAM_UNAVAILABLE,
            message,
            details={"error_class": error_class, "safe_detail": safe_detail, "errno": errno},
        )
        self.error_class = error_class
        self.safe_detail = safe_detail
        self.errno = errno

    @property
    def repairable(self) -> bool:
        """是否属于详设 10.8 第一张清单里的可修复类错误。"""
        return self.error_class == "REPAIRABLE"

    def repair_hint(self) -> str:
        """回灌给模型的错因（已脱敏，见 `_sanitize_db_error`）。"""
        return f"执行失败：{self.safe_detail or self.message}"


class SqlRunner(Protocol):
    """执行器契约。

    存在的理由与 `JobQueue` / `ModelGateway` 相同：**让单元测试的替身与
    生产实现受同一份签名约束**。`SqlQueryTool` 依赖这个协议而不是
    `SqlExecutor` 具体类，因此不连数据库也能把整条编排链路跑完——
    而替身一旦与真实实现漂移，类型检查会先发现。
    """

    async def execute(self, validated: ValidatedSql) -> SqlExecutionResult: ...

    async def aclose(self) -> None: ...


class SqlExecutor:
    """在只读连接上执行校验过的 SQL。

    `validated` 是**唯一**可接受的入参类型（不是裸字符串）：
    这个类型只能由 `SqlValidator.validate()` 产出，因此「没有经过校验的 SQL
    被拿去执行」这件事在类型层面就写不出来。比在每个调用点记得先校验可靠。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        catalog: SchemaCatalog | None = None,
        engine: AsyncEngine | None = None,
    ) -> None:
        self._settings = settings
        self._tuning = settings.sql_tool
        #: 只用于把结果列名映射回目录里声明的类型，不参与安全判定。
        #: 允许为 None：执行器的安全边界不依赖目录（那是校验器的职责），
        #: 缺目录时结果列的类型标 `UNKNOWN`，功能不受影响。
        self._catalog = catalog
        # 允许注入 engine：集成测试要指向 `agent_test` 之类的库。
        # 注入时生命周期归调用方，本类不 close 别人传进来的东西。
        self._engine = engine
        self._owns_engine = engine is None

    async def aclose(self) -> None:
        if self._engine is not None and self._owns_engine:
            await self._engine.dispose()
            self._engine = None

    # ------------------------------------------------------------------ 执行
    async def execute(self, validated: ValidatedSql) -> SqlExecutionResult:
        """执行并截断。

        **绑定参数取自 `validated` 本身**，不接受调用方另外传一份：
        校验器注入权限谓词时会加入自己的绑定参数，调用方再拼一次必然漏。
        见 `ValidatedSql.bind_parameters` 的说明。

        Raises:
            SqlExecutionError: 执行失败。`error_class` 标出可否修复——
                连接不通与查询超时是 `TRANSIENT` / `TIMEOUT`，**不进入自修复**
                （详设 10.8：重写 SQL 不会让数据库变得可达，也不会让它变快）。
        """
        engine = self._ensure_engine()
        timeout = float(self._tuning.timeout_seconds)
        started = time.monotonic()

        try:
            async with engine.connect() as connection:
                await self._prepare_session(connection)
                rows, columns, fetched, truncated = await asyncio.wait_for(
                    self._fetch(connection, validated),
                    timeout=timeout,
                )
        except TimeoutError as exc:
            raise SqlExecutionError(
                "查询超时，请缩小查询范围后重试",
                error_class="TIMEOUT",
                safe_detail=f"超过 {self._tuning.timeout_seconds} 秒",
            ) from exc
        except SQLAlchemyError as exc:
            raise _execution_error(exc) from exc

        return SqlExecutionResult(
            columns=tuple(columns),
            rows=tuple(rows),
            row_count=len(rows),
            truncated=truncated,
            duration_ms=int((time.monotonic() - started) * 1000),
            fetched_count=fetched,
            started_at=datetime.now(UTC),
        )

    # ---------------------------------------------------------------- 内部实现
    def _ensure_engine(self) -> AsyncEngine:
        if self._engine is None:
            # 用业务库的**只读**连接串。这条配置不来自调用方，
            # 也不允许调用方覆盖——10.5 的第 1 条防线就是它。
            self._engine = create_engine(
                self._settings, url=self._settings.database_url_business_ro
            )
        return self._engine

    async def _prepare_session(self, connection: AsyncConnection) -> None:
        """设置会话级护栏。

        `max_execution_time` 的值来自配置，仍然走**绑定参数**——
        没有理由为一个数字开一个拼接的口子，而拼接一旦成为惯例，
        下一个人就会往这里拼真正危险的东西。
        """
        await connection.execute(text(_SET_READ_ONLY))
        await connection.execute(
            text(_SET_MAX_EXECUTION_TIME),
            {"milliseconds": self._tuning.timeout_seconds * 1000},
        )

    async def _fetch(
        self,
        connection: AsyncConnection,
        validated: ValidatedSql,
    ) -> tuple[list[tuple[Any, ...]], list[ResultColumn], int, bool]:
        """执行并物化结果。

        ## 为什么「取满即算截断」

        校验器已经把 `LIMIT max_rows` 写进了 SQL，因此**数据库不会告诉我们
        是否还有更多行**——查了 5 万行的前 1000 行与恰好只有 1000 行，
        返回的都是 1000 行。想区分只能把 SQL 写成 `LIMIT max_rows + 1`，
        但那会让「上限 1000」这句话变成「上限 1001，我们扔掉一行」，
        校验器与执行器的口径就对不上了。

        详设 10.6 的判定标准是「最大 1,000 行、2MB，**任一达到即标记 truncated**」，
        正是这个取舍：宁可对「恰好 1000 行」误报一次，也不能对「被砍掉的行」
        漏报——漏报会让下游以为拿到的是全量，据此得出错误的结论。
        """
        limit = self._tuning.max_rows
        byte_budget = self._tuning.max_result_bytes

        result = await connection.execute(
            text(validated.validated_sql), dict(validated.bind_parameters)
        )
        columns = _result_columns(result.keys(), self._catalog)
        fetched_rows = result.fetchmany(limit)

        rows: list[tuple[Any, ...]] = []
        consumed = 0
        truncated = len(fetched_rows) >= limit
        for raw in fetched_rows:
            normalized = tuple(_normalize(value) for value in raw)
            consumed += len(repr(normalized).encode())
            if consumed > byte_budget:
                truncated = True
                break
            rows.append(normalized)
        return rows, columns, len(fetched_rows), truncated


# ------------------------------------------------------------------ 错误分类
def _execution_error(exc: SQLAlchemyError) -> SqlExecutionError:
    """把 SQLAlchemy 异常转成带分类的 `SqlExecutionError`。

    **区分「SQL 语义错」与「数据库不可用」是这一段唯一的职责**，而且它必须在这里做：
    到了上层，两种失败都只剩一句「查询失败了」，再想分类就只能靠猜。
    """
    errno = _errno_of(exc)
    detail = _sanitize_db_error(exc)
    if errno in _REPAIRABLE_ERRNOS:
        return SqlExecutionError(
            "SQL 语义有误，执行被数据库拒绝",
            error_class="REPAIRABLE",
            safe_detail=detail,
            errno=errno,
        )
    # 认不出的错误按不可修复处理：详设 10.8 的可修复清单是**白名单**，
    # 一个没见过的 errno 有可能是权限拒绝或数据损坏，重试只会浪费预算。
    return SqlExecutionError(
        "数据查询服务暂时不可用，请稍后重试",
        error_class="TRANSIENT",
        safe_detail=detail,
        errno=errno,
    )


def _orig_args(exc: SQLAlchemyError) -> tuple[object, ...]:
    """取底层 DBAPI 异常的参数元组。

    `orig` 挂在 `DBAPIError` / `StatementError` 上，不在 `SQLAlchemyError`
    基类上——因此这里必须 `getattr` 而不是直接点出来。取不到时返回空元组，
    让调用方走「认不出这个错误」的分支，而不是在这里抛一个 AttributeError
    把一次可分类的失败变成一次不可分类的崩溃。
    """
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", ()) or ()
    return tuple(args)


def _errno_of(exc: SQLAlchemyError) -> int | None:
    """取 MySQL 的错误号。

    只取**整数错误号**，不带消息——错误号是有限的、可枚举的，
    而消息里可能带着取值。DBAPI 异常的第一个参数就是 errno。
    """
    args = _orig_args(exc)
    first = args[0] if args else None
    return first if isinstance(first, int) else None


def _sanitize_db_error(exc: SQLAlchemyError) -> str | None:
    """把数据库报错脱敏成可以回灌给模型的错因。

    **单引号里的内容全部抹掉**：数据库的报错形如
    ``Incorrect DATE value: '2025-13-01' for column `s`.`order_date` ``——
    反引号里是列名（正是修复需要的），单引号里是模型自己填的取值
    （也就是业务数据的一部分）。留着它，等于把一条业务取值写进修复 prompt、
    写进 Trace、写进日志，正好撞上 19.4 的脱敏纪律。

    ## 但参数名要留下

    SQLAlchemy 的 ``A value is required for bind parameter 'start_date'``
    里的单引号括的是**参数名**，而那是修复唯一需要的东西——
    一刀切抹掉之后，回灌给模型的是「A value is required for bind parameter '…'」，
    它只能靠猜。所以先按固定句式把参数名摘出来（去掉引号，后面的
    字面量规则就不再匹配它），摘不到时仍然走一刀切——
    摘不掉的代价是提示变差，抹不掉的代价是数据泄露，两者的严重性不对称。
    """
    message = next((item for item in _orig_args(exc) if isinstance(item, str)), None)
    if not message:
        return None
    named = _BIND_PARAMETER_NAME.sub(lambda match: f"{match.group(1)}{match.group(2)}", message)
    cleaned = _DB_STRING_LITERAL.sub("'…'", named).replace("\n", " ").strip()
    return cleaned[:_MAX_HINT_CHARS]


# ------------------------------------------------------------------ 值规范化
def _normalize(value: object) -> object:
    """把驱动返回的值转成 JSON 原生类型，见模块 docstring。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        # 金额一律转字符串：float(Decimal("0.1")) 已经不是 0.1 了
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        # 不把二进制内容带出去，只留长度——它既打不进日志也没有展示价值
        return f"<binary {len(value)} bytes>"
    return str(value)


def _result_columns(keys: Sequence[str] | Any, catalog: SchemaCatalog | None) -> list[ResultColumn]:
    """结果列名与类型。

    类型按列名去目录里查，查不到退回 `"UNKNOWN"`——**不猜**。
    聚合列（`SUM(x) AS net_sales`）与计算列在目录里当然没有，
    它们的口径由 `SqlToolResult.metric_definitions` 解释，
    在这里编一个类型只会让下游以为它有确定类型。

    按裸列名查而不按「表.列」查：结果集里没有表的信息，
    而同一列名在不同表上类型不一致的情况在演示库里不存在；
    真要出现，`UNKNOWN` 也比猜错好。
    """
    if catalog is None:
        return [ResultColumn(name=str(key), data_type="UNKNOWN") for key in keys]
    index = {column.name: column.data_type for table in catalog.tables for column in table.columns}
    return [ResultColumn(name=str(key), data_type=index.get(str(key), "UNKNOWN")) for key in keys]


__all__ = ["SqlExecutionError", "SqlExecutor", "SqlRunner"]
