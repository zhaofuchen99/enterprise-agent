"""确定性 SQL 安全校验（详细设计 10.4 的 12 步，顺序固定、任一步失败即不执行）。

**这是「安全由代码保证」这句话的落点。** 详设 10.5 把安全责任分了三层，
本模块是中间那层：不依赖数据库权限（那是最后一道），更不依赖模型的自觉。

## 为什么用 sqlglot 而不是正则

正则拦 SQL 的经典失败模式是「拦得住写出来的、拦不住能执行的」——
注释能拆关键字（`SEL/**/ECT`）、大小写能绕、MySQL 的版本注释
`/*!50000 DROP*/` 也能绕。AST 校验问的是**解析后它到底是什么**，
模型把语句写成什么样都不影响结论。代价是要处理 sqlglot 的方言差异，
下面每处都注明了。

## 12 步的落点

| 步 | 内容 | 函数 |
|---|---|---|
| 1 | 请求文本长度 | `_step1_length` |
| 2 | 只能解析出一条语句 | `_step2_parse` |
| 3 | 根节点必须是 SELECT | `_step3_root_select` |
| 4 | 禁止写操作与 DDL | `_step4_statement_kind` |
| 5 | 禁止注释、OUTFILE、系统变量 | `_step5_exotic` |
| 6 | 表必须在目录白名单 | `_step6_tables` |
| 7 | 列必须存在且角色可访问，禁止 `SELECT *` | `_step7_columns` |
| 8 | JOIN 必须属于允许集合 | `_step8_joins` |
| 9 | 函数属于允许集合 | `_step9_functions` |
| 10 | LIMIT 重写 | `_step10_limit` |
| 11 | 注入数据权限谓词 | `_step11_scope` |
| 12 | 规范化 SQL 与指纹 | `_step12_render` |

## 可修复与不可修复（详设 10.8）

失败按 10.8 的两张清单分类：**语法 / 别名 / 字段名 / 类型 / 聚合**类错误回给模型
修复，**权限拒绝、危险语句、非白名单表、敏感字段**类错误直接终止。结论由
`SqlValidationError.error_class` 带出，**要不要重试由 `tool.py` 决定**——
详设 9.3 明写 Tool 的执行器统一控制重试，校验器不越这个界。

两处需要说明的归类：

- **`SELECT *` 归可修复**：它是模型偷懒，不是攻击。逐列写出来就能过。
- **不在白名单的函数分两种**：`SLEEP` / `BENCHMARK` / `LOAD_FILE` 这类
  资源与侧信道函数归**不可修复**（10.4 第 9 步禁它们有明确理由）；
  其余只是没登记（`GREATEST`、`ISNULL` 之类）归可修复，给模型一次改正机会。
  这条分界线画在「函数本身危不危险」上，而不是画在「在不在名单里」。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.errors import AgentError, ErrorCode
from app.domain.evidence import TimeRange
from app.domain.user import PermissionScope
from app.tools.base import ErrorClass
from app.tools.sql.schemas import (
    ColumnSpec,
    SchemaCatalog,
    ScopeSpec,
    TableSpec,
    ValidatedSql,
)

#: 目标方言。目录、函数白名单与渲染都按它对齐。
_DIALECT: Final[str] = "mysql"

#: 禁止访问的系统库。这些库里放着库自身的元数据（账号哈希、库表结构），
#: 是信息收集的第一步。
_SYSTEM_SCHEMAS: Final[frozenset[str]] = frozenset(
    {"information_schema", "mysql", "performance_schema", "sys"}
)

#: 不可修复的函数（10.4 第 9 步点名的资源与侧信道函数）。
#: 与 `allowed_functions` 是两回事：这份是「无论怎么改都不给执行」，
#: 白名单是「登记过的才给执行」。前者是攻击特征，后者是登记制度。
_FORBIDDEN_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "SLEEP",  # 延时，用于确认盲注是否成立
        "BENCHMARK",  # 纯 CPU 消耗，拒绝服务
        "LOAD_FILE",  # 读服务器文件
        "GET_LOCK",  # 加锁，可造成拒绝服务
        "RELEASE_LOCK",
        "SYS_EXEC",  # UDF 提权链
        "SYS_EVAL",
    }
)

#: 权限谓词的绑定参数名前缀。加前缀是为了不与模型自己生成的
#: `:start_date` 之流撞名——撞名的表现是「模型给的日期被权限值覆盖了」，
#: 而这种错会静默产生一个看似合理的错误答案。
_SCOPE_PARAM_PREFIX: Final[str] = "scope_region"

_REWRITE_LIMIT: Final[str] = "补写 LIMIT {limit}（详设 10.4 第 10 步）"
_REWRITE_SCOPE: Final[str] = "注入数据权限谓词（详设 10.4 第 11 步，共 {count} 张表）"

#: 禁止出现在 AST 任何位置的写操作与管理语句（10.4 第 4 步）。
_FORBIDDEN_NODES: Final[tuple[type[exp.Expr], ...]] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
    exp.Set,
    # sqlglot 认不出的语句（CALL / PREPARE / HANDLER…）落在这里。
    # 宁可把认不出的都拒掉：白名单的意义就是「只放行认得出的」。
    exp.Command,
)


class SqlValidationError(AgentError):
    """校验未通过。

    继承 `AgentError` 而不是自造异常：它最终要变成一个错误码进 State、
    进 Trace、进日志（详设 19.1 的 `SQL_VALIDATION_FAILED`）。自造异常会让
    每个调用点都要额外写一次转换。

    Attributes:
        step: 详设 10.4 里失败的步号，进 Trace 后能直接定位到规则。
        error_class: 详设 9.4 的错误类别，决定调用方是否进入修复流程。
        safe_detail: 已脱敏的补充说明（只含标识符与计数，不含原始 SQL）。
    """

    def __init__(
        self,
        message: str,
        *,
        step: int,
        error_class: ErrorClass,
        safe_detail: str | None = None,
    ) -> None:
        super().__init__(
            ErrorCode.SQL_VALIDATION_FAILED,
            message,
            details={"step": step, "error_class": error_class, "safe_detail": safe_detail},
        )
        self.step = step
        self.error_class = error_class
        self.safe_detail = safe_detail

    @property
    def repairable(self) -> bool:
        """是否属于详设 10.8 第一张清单里的可修复类错误。"""
        return self.error_class == "REPAIRABLE"

    def repair_hint(self) -> str:
        """回灌给模型的错因。

        只带步号、原因与出问题的标识符。**不带整条 SQL**——原 SQL 由
        `tool.py` 单独放进修复 prompt（详设 10.8），不重复塞进错误里，
        也不回灌数据库回显。
        """
        suffix = f"（{self.safe_detail}）" if self.safe_detail else ""
        return f"第 {self.step} 步校验未通过：{self.message}{suffix}"


class SqlValidator:
    """按详设 10.4 的顺序校验并重写一条候选 SQL。

    Attributes:
        catalog: 整份目录。与 `SchemaProvider` **共用同一份**——
            分开加载会出现「生成时看得到、校验时不认识」的自相矛盾。
    """

    def __init__(self, catalog: SchemaCatalog, *, max_sql_chars: int, max_rows: int) -> None:
        self._catalog = catalog
        self._max_sql_chars = max_sql_chars
        self._max_rows = max_rows

    # ------------------------------------------------------------------ 入口
    def validate(
        self,
        sql: str,
        *,
        parameters: Mapping[str, str | int | float | date] | None = None,
        scope: PermissionScope | None = None,
    ) -> ValidatedSql:
        """校验、重写并返回可执行的 SQL。

        Args:
            sql: 模型生成的候选 SQL 原文。
            parameters: 候选 SQL 里的命名绑定参数取值。
            scope: 调用者的数据权限范围。**由 `User.permission_scope()` 产出**，
                工具不得自行组装。

        Returns:
            通过校验的对象；**`validated_sql` 才是应该被执行的那一份**，
            它带着补写的 LIMIT 与注入的权限谓词。

        Raises:
            SqlValidationError: 任一步失败，`error_class` 标出可否修复。
        """
        params: dict[str, object] = dict(parameters or {})
        permission = scope or PermissionScope()

        self._step1_length(sql)
        tree = self._step2_parse(sql)
        self._check_bind_parameters(tree, params)
        self._step3_root_select(tree)
        self._step4_statement_kind(tree)
        self._step5_exotic(sql, tree)

        cte_names = frozenset(cte.alias for cte in tree.find_all(exp.CTE) if cte.alias)
        aliases = self._alias_map(tree, cte_names)
        derived = self._derived_aliases(tree, cte_names)
        derived_columns = self._derived_columns(tree, cte_names)

        used_tables = self._step6_tables(tree, cte_names)
        used_columns = self._step7_columns(tree, aliases, derived, derived_columns, permission)
        self._step8_joins(tree, aliases, cte_names, derived)
        self._step9_functions(tree)

        rewrites: list[str] = []
        self._step10_limit(tree, rewrites)
        injected = self._step11_scope(tree, permission, params)

        normalized = self._step12_render(tree)
        if injected:
            rewrites.append(_REWRITE_SCOPE.format(count=injected))

        return ValidatedSql(
            validated_sql=normalized,
            normalized_sql=normalized,
            sql_fingerprint=_fingerprint(tree),
            # `params` 里既有模型给的参数，也有第 11 步注入的权限参数——
            # 执行时一个都不能少，因此整份带出去。
            bind_parameters=dict(params),
            used_tables=tuple(sorted(used_tables)),
            used_columns=tuple(sorted(used_columns)),
            rewrites=tuple(rewrites),
            scope_injected=bool(injected),
            data_time_range=self._extract_time_range(tree, aliases, params),
        )

    # ------------------------------------------------- 1 请求文本长度
    def _step1_length(self, sql: str) -> None:
        if len(sql) <= self._max_sql_chars:
            return
        # 归类 VALIDATION 而非 SECURITY：超长不是攻击特征，是模型跑飞了。
        # 但它也不可修复——再看一遍同样的上下文生成一条短十倍的正确 SQL，
        # 那不是「修正」而是重新猜，修复预算花在这里不划算。
        raise SqlValidationError(
            "生成的 SQL 过长",
            step=1,
            error_class="VALIDATION",
            safe_detail=f"{len(sql)} 字符 > 上限 {self._max_sql_chars}",
        )

    # -------------------------------------------------------- 2 单语句解析
    def _step2_parse(self, sql: str) -> exp.Expr:
        try:
            statements = [s for s in sqlglot.parse(sql, dialect=_DIALECT) if s is not None]
        except ParseError as exc:
            # sqlglot 的 ParseError 会把出错处的 SQL 片段拼进 message，
            # 只取第一行（`Invalid expression / Unexpected token. Line 1, Col: 20.`），
            # 避免把模型原文带进日志与响应体。
            raise SqlValidationError(
                "SQL 无法解析，存在语法错误",
                step=2,
                error_class="REPAIRABLE",
                safe_detail=str(exc).splitlines()[0],
            ) from exc

        if not statements:
            raise SqlValidationError("SQL 为空", step=2, error_class="REPAIRABLE")
        if len(statements) > 1:
            # **多语句归不可修复**，与语法错误分开：堆叠查询（`SELECT 1; DROP …`）
            # 是 SQL 注入的典型形态，不是手滑。给一次「再生成一遍」的机会
            # 等于把一个已经表现出注入倾向的输入再喂一次。
            raise SqlValidationError(
                "不允许一次提交多条语句",
                step=2,
                error_class="SECURITY",
                safe_detail=f"解析出 {len(statements)} 条",
            )
        return statements[0]

    # ------------------------------------ 2.5 绑定参数完整性（详设 10.4 之外的补充）
    def _check_bind_parameters(self, tree: exp.Expr, params: Mapping[str, object]) -> None:
        """SQL 里用到的每个命名参数都必须有取值。

        **这是详设 10.4 十二步之外的补充，理由是一次真实缺陷**：模型写
        `WHERE d >= :start` 却在 `parameters` 里给出 `start_date` 时，
        报错发生在 SQLAlchemy 的绑定阶段（`A value is required for bind
        parameter`）。那时异常类型是 `StatementError`、拿不到 MySQL 的 errno，
        于是被归成了 `TRANSIENT`——「数据库不可用」——而它其实是模型写错了参数名，
        正是 10.8 里「参数不匹配、改一改就能过」那一类。归类错了的代价是
        修复预算根本没被用上，用户看到的是「服务暂时不可用，请稍后重试」。

        在 AST 上校验是**确定性**的：不依赖 SQLAlchemy 的报错文案，
        也能在抛错时把缺的参数名直接交给模型（`safe_detail`）。

        **多余参数不报错**：模型多给一个没被引用的参数不影响执行，
        而拒绝它只会平白烧掉一次修复预算。
        """
        used = {
            str(node.this) for node in tree.find_all(exp.Placeholder) if isinstance(node.this, str)
        }
        if missing := sorted(used - set(params)):
            raise SqlValidationError(
                "SQL 引用了未提供取值的命名参数",
                step=2,
                error_class="REPAIRABLE",
                safe_detail="、".join(missing),
            )

    # ---------------------------------------------------- 3 根节点是 SELECT
    def _step3_root_select(self, tree: exp.Expr) -> None:
        root = tree.this if isinstance(tree, exp.Subquery) else tree
        # `WITH x AS (…) SELECT …` 的根节点已经是 Select（CTE 挂在它的 `with` 参数上），
        # 不需要额外判断；而 `WITH x AS (…) INSERT …` 的根节点是 Insert，正好被拦下。
        if not isinstance(root, exp.Select):
            raise SqlValidationError(
                "只允许 SELECT 查询",
                step=3,
                error_class="SECURITY",
                safe_detail=f"根节点是 {type(root).__name__}",
            )

    # ---------------------------------------------------- 4 禁止写与 DDL
    def _step4_statement_kind(self, tree: exp.Expr) -> None:
        """扫描 AST 的**任何位置**，而不只是根节点。

        MySQL 不允许把 INSERT 放进 CTE，但 DDL 与事务控制可以出现在
        多语句里（第 2 步已拦），存储过程调用可以出现在表达式里。
        全树扫描比逐个推理「哪些位置可能藏写操作」可靠。
        """
        for node in tree.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise SqlValidationError(
                    "检测到写操作或管理语句，已拒绝执行",
                    step=4,
                    error_class="SECURITY",
                    safe_detail=type(node).__name__,
                )

    # -------------------------------------------- 5 注释、OUTFILE、系统变量
    def _step5_exotic(self, sql: str, tree: exp.Expr) -> None:
        """拦下三类「AST 上看不出来但会被带进执行路径」的东西。

        **注释为什么必须拦**：sqlglot 解析后会把注释挂在节点上，并在
        `sql()` 重新渲染时**原样打印回来**。也就是说一条带注释的语句会带着
        注释进入执行路径。在 AST 上做判定本身没错，但既然规范化输出会保留它，
        不如在入口一次拒掉——这条规则的成本只有「模型别写注释」。
        """
        if (found := _find_comment(sql)) is not None:
            raise SqlValidationError(
                "SQL 中不允许出现注释",
                step=5,
                error_class="SECURITY",
                safe_detail=f"发现{found}",
            )
        if any(isinstance(node, exp.Into) for node in tree.walk()):
            raise SqlValidationError(
                "不允许 INTO OUTFILE / DUMPFILE 写文件",
                step=5,
                error_class="SECURITY",
            )
        if any(isinstance(node, exp.SessionParameter) for node in tree.walk()):
            raise SqlValidationError(
                "不允许读取会话或全局变量",
                step=5,
                error_class="SECURITY",
            )

    # ---------------------------------------------------------- 6 表白名单
    def _step6_tables(self, tree: exp.Expr, cte_names: frozenset[str]) -> set[str]:
        used: set[str] = set()
        for node in tree.find_all(exp.Table):
            name = node.name
            if not name or name in cte_names:
                # CTE 是查询内部定义的临时结果集，不是库里的表。
                # 拿它去查白名单会误报「引用了未授权的表」。
                continue
            if (db := node.db) and str(db).lower() in _SYSTEM_SCHEMAS:
                raise SqlValidationError(
                    "不允许访问系统库",
                    step=6,
                    error_class="SECURITY",
                    safe_detail=str(db),
                )
            # 带库名前缀（`business.fact_sales_order_item`）按裸表名比对：
            # 白名单本身是闭集，放行前缀不会带来新的可达表。
            spec = self._catalog.table(name)
            if spec is None:
                # 「非白名单表」在 10.8 里明确属不可修复类：
                # 它多半意味着模型在够别的数据，再生成一次没有意义。
                raise SqlValidationError(
                    "查询引用了未授权的表",
                    step=6,
                    error_class="SECURITY",
                    safe_detail=name,
                )
            used.add(spec.name)
        return used

    # ------------------------------------- 7 列白名单、角色可访问与 SELECT *
    def _step7_columns(
        self,
        tree: exp.Expr,
        aliases: Mapping[str, str],
        derived: frozenset[str],
        derived_columns: frozenset[str],
        scope: PermissionScope,
    ) -> set[str]:
        used: set[str] = set()
        self._reject_star(tree)

        for column in tree.find_all(exp.Column):
            if isinstance(column.this, exp.Star):
                # `COUNT(*)` 的星号：聚合函数的实参，不是列引用。
                # 顶层投影里的星号已在 `_reject_star` 拦掉。
                continue
            name = column.name
            qualifier = column.table
            if not qualifier:
                if _refers_to_output_alias(column):
                    # `GROUP BY m` / `ORDER BY net_sales` 里的 `m` / `net_sales`
                    # 是 SELECT 自己起的输出别名，不是表里的列。
                    # **只在 ORDER / GROUP / HAVING 里放行**：这三处引用输出别名
                    # 是 SQL 允许的写法，而 WHERE 里出现同名标识符只可能是拼错了列名。
                    # 别名背后的真实列引用仍然会在本函数里被单独检查一次。
                    continue
                if name in derived_columns and not self._columns_named(name):
                    # 从 CTE 或派生表里取它算出来的列：
                    # `WITH yoy AS (SELECT … AS yoy_pct …) SELECT yoy_pct FROM yoy`
                    # 这里的 `yoy_pct` 只存在于那张 CTE 的输出里，目录里当然没有。
                    #
                    # **多一个 `and not self._columns_named(name)` 是必要的**：
                    # 只有「目录里没有、派生结果里有」的名字才跳过。
                    # 否则 `WITH x AS (SELECT customer_name FROM dim_customer)
                    # SELECT customer_name FROM x` 这种绕法会因为
                    # `customer_name` 恰好也是个派生列名而直接放行——
                    # 那条路径上真实的列引用就没机会被检查了。
                    continue
                used |= self._check_unqualified_column(name, scope)
                continue
            if qualifier in derived:
                # 派生表 / CTE 的列集合由它自己的 SELECT 决定，目录里没有，
                # 无法逐列核对。**跳过是安全的**：那张派生表内部的列引用
                # 会在本函数里被单独检查一次（它一定出现在某个地方）。
                # 必须先于别名映射判断——见 `_alias_map` 的说明。
                continue
            resolved = aliases.get(qualifier)
            if resolved is None:
                # 别名对不上任何表：详设 10.8 的「别名错误」，可修复。
                raise SqlValidationError(
                    "查询使用了不存在的表别名",
                    step=7,
                    error_class="REPAIRABLE",
                    safe_detail=qualifier,
                )
            used.add(f"{resolved}.{self._require_column(resolved, name, scope).name}")
        return used

    def _reject_star(self, tree: exp.Expr) -> None:
        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                    isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
                ):
                    raise SqlValidationError(
                        "不允许 SELECT *，必须逐列写出",
                        step=7,
                        error_class="REPAIRABLE",
                        safe_detail="SELECT *",
                    )

    def _check_unqualified_column(self, name: str, scope: PermissionScope) -> set[str]:
        """未限定表名的列。

        放宽到「上下文里任意一张表有它即可通过」，是因为合格的模型在单表查询里
        通常不写表名，而单表查询本来就无从限定。命中多张表时**每张都要过角色检查**——
        否则同名但敏感的列（如 `dim_customer.customer_name`）会被另一张表的
        同名 PUBLIC 列放行。
        """
        matches = self._columns_named(name)
        if not matches:
            raise SqlValidationError(
                "查询引用了不存在的列",
                step=7,
                error_class="REPAIRABLE",
                safe_detail=name,
            )
        used: set[str] = set()
        for table, spec in matches:
            self._require_role(table, spec, scope)
            used.add(f"{table.name}.{spec.name}")
        return used

    def _require_column(self, table_name: str, column: str, scope: PermissionScope) -> ColumnSpec:
        table = self._catalog.table(table_name)
        spec = table.column(column) if table else None
        if table is None or spec is None:
            # 「白名单内字段名错误」在 10.8 里属可修复类：模型多半是拼错了列名。
            raise SqlValidationError(
                "查询引用了不存在的列",
                step=7,
                error_class="REPAIRABLE",
                safe_detail=f"{table_name}.{column}",
            )
        self._require_role(table, spec, scope)
        return spec

    def _require_role(self, table: TableSpec, column: ColumnSpec, scope: PermissionScope) -> None:
        if column.visible_to(scope.role.value):
            return
        # 「敏感字段」在 10.8 里属不可修复类：换个写法也还是同一个字段。
        raise SqlValidationError(
            "查询涉及当前账号无权访问的字段",
            step=7,
            error_class="SECURITY",
            safe_detail=f"{table.name}.{column.name}（{column.sensitive_level}）",
        )

    def _columns_named(self, name: str) -> list[tuple[TableSpec, ColumnSpec]]:
        return [
            (table, spec)
            for table in self._catalog.tables
            if (spec := table.column(name)) is not None
        ]

    # -------------------------------------------------------- 8 JOIN 白名单
    def _step8_joins(
        self,
        tree: exp.Expr,
        aliases: Mapping[str, str],
        cte_names: frozenset[str],
        derived: frozenset[str],
    ) -> None:
        for join in tree.find_all(exp.Join):
            condition = join.args.get("on")
            if condition is None and join.args.get("using") is None:
                # 无 ON 的 JOIN（含逗号连接与 CROSS JOIN）语义上是笛卡尔积：
                # 结果毫无意义，而且会让权限谓词之外的行被组合出来。
                raise SqlValidationError(
                    "JOIN 必须带 ON 条件",
                    step=8,
                    error_class="REPAIRABLE",
                    safe_detail=join.sql(dialect=_DIALECT)[:60],
                )
            if condition is None:
                # USING(col) 形式：列名相等即等价，无法用别名还原表对。
                # 直接拒绝而不是放行——白名单是闭集，认不出的不给过。
                raise SqlValidationError(
                    "JOIN 请使用 ON 条件而不是 USING",
                    step=8,
                    error_class="REPAIRABLE",
                    safe_detail=join.sql(dialect=_DIALECT)[:60],
                )

            joined = join.this.name if isinstance(join.this, exp.Table) else ""
            if not joined or joined in cte_names:
                continue

            # ON 条件里出现的限定名 -> 真实表名。ON 里写的是别名
            # （`r.region_id = s.region_id`），必须还原后才知道是哪两张表在关联。
            qualifiers = {c.table for c in condition.find_all(exp.Column) if c.table}
            unverifiable = qualifiers & derived
            partners: set[str] = set()
            for qualifier in qualifiers - unverifiable:
                if (resolved := aliases.get(qualifier)) and resolved != joined:
                    partners.add(resolved)

            if not partners:
                if unverifiable:
                    # ON 的对面是派生表或 CTE（`ON r.region_id = x.region_id`）。
                    # 这类关联无法用目录判定，**放行是安全的**：
                    # 派生表自己的数据来源已在第 6 步逐表核对过，
                    # 这里没有被访问到的新表。
                    continue
                # 形如 `ON a = b`（两侧都无限定名）无法判定表对。
                # 按可修复处理：多表查询里这么写几乎必然是错的。
                raise SqlValidationError(
                    "JOIN 的 ON 条件必须用表名或表别名限定",
                    step=8,
                    error_class="REPAIRABLE",
                    safe_detail=condition.sql(dialect=_DIALECT)[:60],
                )
            if not any(self._catalog.join_allowed(joined, partner) for partner in partners):
                # 「JOIN 关系越权」在 10.8 里属不可修复类：换一种写法还是不允许的组合。
                raise SqlValidationError(
                    "JOIN 关系不在允许范围内",
                    step=8,
                    error_class="SECURITY",
                    safe_detail=f"{joined} ⋈ {sorted(partners)}",
                )

    # -------------------------------------------------------- 9 函数白名单
    def _step9_functions(self, tree: exp.Expr) -> None:
        for node in tree.find_all(exp.Func):
            name = _function_name(node)
            if name is None or name in self._catalog.allowed_functions:
                # `None` 表示这个节点是运算符而不是函数调用（见 `_function_name`）
                continue
            if name in _FORBIDDEN_FUNCTIONS:
                raise SqlValidationError(
                    "SQL 中不允许使用该函数",
                    step=9,
                    error_class="SECURITY",
                    safe_detail=name,
                )
            raise SqlValidationError(
                "SQL 使用了不在允许清单内的函数",
                step=9,
                error_class="REPAIRABLE",
                safe_detail=name,
            )

    # -------------------------------------------------------- 10 LIMIT 重写
    def _step10_limit(self, tree: exp.Expr, rewrites: list[str]) -> None:
        """补写或收紧 LIMIT（详设 10.4 第 10 步）。

        「聚合查询也设置最大结果行数」——**不区分明细与聚合**：识别聚合的成本
        与收益不成比例，而漏判一次就是一个没有上限的查询。
        """
        select = _outer_select(tree)
        if select is None:
            return
        current = select.args.get("limit")
        value = _literal_limit(current)
        if value is not None and value <= self._max_rows:
            return
        select.set("limit", exp.Limit(expression=exp.Literal.number(self._max_rows)))
        rewrites.append(_REWRITE_LIMIT.format(limit=self._max_rows))

    # ------------------------------------------------ 11 数据权限谓词注入
    def _step11_scope(
        self,
        tree: exp.Expr,
        scope: PermissionScope,
        params: dict[str, object],
    ) -> int:
        """注入**不可被模型覆盖**的服务端谓词（详设 10.4 第 11 步）。

        ## 为什么在 SQL 层注入，而不是在提示词里要求

        `app_user.data_scope_json` 是服务端事实，模型看不到也不该看到。
        写成一句 prompt 要求（「请只查华东」）等于把权限交给模型执行，
        而这正是详设 10.5 明令不能依赖的那一层。

        ## 为什么用 AND 追加，而不是替换 WHERE

        追加并整体加括号后，语义是「原条件 AND 权限条件」，
        模型写的 `WHERE 1=1` 或 `OR` 都覆盖不掉它。若改成把权限条件
        塞进第一个 WHERE 的位置，`WHERE a=1 OR b=2` 会让权限条件
        被错误地绑定到 `b=2` 上——除了 b=2 的行，其余行照旧全露。

        ## 权限取值为什么可以进 SQL

        取值来自已认证用户的数据库记录，不是用户可编辑的自由文本；
        而且它们是**绑定参数**（`:scope_region_0_0`），不是拼进文本的字面量。
        两条加起来，注入面与「模型生成的 SQL」是同一量级，而后者本来就要过 AST 校验。

        ## 注入点按「最近的 SELECT」定位

        表节点往上找第一个 `exp.Select` 作为注入点，而不是一律塞进最外层。
        CTE 或派生表里的表如果被注入到外层 WHERE，那是一个外层作用域里
        不存在的限定名，SQL 会直接报错——这是本函数最容易写错的地方。
        """
        if scope.unrestricted:
            return 0

        injected = 0
        for node in tree.find_all(exp.Table):
            name = node.name
            if not name:
                continue
            spec = self._catalog.table(name)
            if spec is None or spec.scope_column is None:
                continue
            owner = node.find_ancestor(exp.Select)
            if owner is None:
                continue
            # 参数名按**全局**序号生成，不按「每个 SELECT 从 0 开始」：
            # 后者在 CTE 场景下会让两处注入共用 `scope_region_0_0`，
            # 靠「两处的绑定值恰好相同」维持正确——这是个定时炸弹，
            # 一旦哪天权限范围分表计算，就会变成「一处覆盖另一处」。
            parameter_names = [
                f"{_SCOPE_PARAM_PREFIX}_{injected}_{position}"
                for position in range(len(scope.region_ids))
            ]
            for param_name, value in zip(parameter_names, scope.region_ids, strict=True):
                params[param_name] = value
            predicate = self._scope_predicate(
                spec, node.alias or name, parameter_names, self._catalog.scope
            )
            existing = owner.args.get("where")
            owner.set(
                "where",
                exp.Where(this=exp.and_(existing.this, predicate) if existing else predicate),
            )
            injected += 1
        return injected

    def _scope_predicate(
        self,
        table: TableSpec,
        qualifier: str,
        parameter_names: Sequence[str],
        scope_spec: ScopeSpec | None,
    ) -> exp.Expr:
        """构造单张表的权限谓词。

        两种形态由目录决定：
        - `scope_column` 恰好是解析表的匹配列（`dim_region.region_name`）→ 直接 IN；
        - 否则（`fact_sales_order_item.region_id`）→ 子查询把范围取值解析成目标列。
          没有这一层，受限用户查事实表时会因为「名称对不上 ID」而查不到任何行——
          表现为「权限生效了但结果是空的」，比报错更难排查。
        """
        # `scope_column` 非空由调用方保证（拿不到落点列的表根本不会走到这里）。
        # 显式断言而不是 `# type: ignore`：真被改坏了要在开发期炸，不是在运行期静默跳过。
        assert table.scope_column is not None
        target = exp.column(table.scope_column, table=qualifier)
        placeholders = [_placeholder(name) for name in parameter_names]
        resolution = scope_spec.resolve if scope_spec else None
        if resolution is None or table.scope_column == resolution.match_column:
            return target.isin(*placeholders) if placeholders else exp.false()

        lookup = (
            exp.select(exp.column(resolution.key_column, table=resolution.table))
            .from_(resolution.table)
            .where(
                exp.column(resolution.match_column, table=resolution.table).isin(*placeholders)
                if placeholders
                else exp.false()
            )
        )
        return target.isin(query=lookup)

    # ---------------------------------------------------------- 12 规范化
    def _step12_render(self, tree: exp.Expr) -> str:
        """规范化 SQL（详设 10.4 第 12 步）。

        **执行的是这一份，不是模型原文**：第 10、11 两步的重写只有经由
        重新渲染才会体现出来。反过来，这也要求所有重写都用 sqlglot 的
        表达式节点完成，而不是往字符串里插文本——插文本会绕过渲染，
        重写结果与 AST 不一致。
        """
        return tree.sql(dialect=_DIALECT, pretty=False)

    # -------------------------------------------------------------- 辅助
    def _alias_map(self, tree: exp.Expr, cte_names: frozenset[str]) -> dict[str, str]:
        """限定名（别名优先，否则表名）-> 目录里的表名。

        没有它就无法把 `ON r.region_id = s.region_id` 还原成
        `dim_region ⋈ fact_sales_order_item`——而 JOIN 白名单判的正是它。

        **必须排除 CTE 引用**：`WITH x AS (…) … FROM x` 里的 `x` 在 AST 上
        也是一个 `exp.Table` 节点，不排除的话会往映射里塞一条 `x -> x`，
        随后 `x.amt` 这样的列会被当成「表 x 的列」拿去查目录，
        报出一句和真实原因（这是 CTE 的输出列）毫无关系的错误。
        """
        mapping: dict[str, str] = {}
        for node in tree.find_all(exp.Table):
            if not node.name or node.name in cte_names:
                continue
            mapping[node.alias or node.name] = node.name
        return mapping

    def _derived_aliases(self, tree: exp.Expr, cte_names: frozenset[str]) -> frozenset[str]:
        """派生表与 CTE 的**名字**集合。

         它们的列集合由各自的 SELECT 决定，目录里查不到，因此用它们限定的列
         （`yoy.amt`）没法逐列核对。**跳过是安全的**：那些 SELECT 内部的
         列引用会在 `_step7_columns` 里各自过一遍，敏感字段不会因为被包进
        子查询就漏网。
        """
        names = set(cte_names)
        for subquery in tree.find_all(exp.Subquery):
            if subquery.alias:
                names.add(subquery.alias)
        return frozenset(names)

    def _derived_columns(self, tree: exp.Expr, cte_names: frozenset[str]) -> frozenset[str]:
        """CTE 与派生表**对外暴露的列名**（它们的 SELECT 里那些别名）。

        用途是让 `WITH yoy AS (SELECT … AS yoy_pct …) SELECT yoy_pct FROM yoy`
        这类写法能通过第七步：`yoy_pct` 只存在于那张 CTE 的输出里，
        目录里当然没有它。不收集这个集合的话，所有把计算放在 CTE 里、
        外层再引用其结果的 SQL 都会被误判成「引用了不存在的列」——
        而这是模型写同比、环比这类多步计算时**最自然**的写法。

        只收集真正暴露的名字（`… AS 名`），不收集 CTE 内部的裸列名：
        后者本来就该按目录里的列去校验。
        """
        exposed: set[str] = set()
        for cte in tree.find_all(exp.CTE):
            exposed |= _projection_aliases(cte.this)
        for subquery in tree.find_all(exp.Subquery):
            if subquery.alias:
                exposed |= _projection_aliases(subquery.this)
        return frozenset(exposed)

    # ------------------------------------------------------------ 时间区间
    def _extract_time_range(
        self,
        tree: exp.Expr,
        aliases: Mapping[str, str],
        params: Mapping[str, object],
    ) -> TimeRange | None:
        """从 WHERE 提取业务时间区间，供结果 Schema 的 `data_time_range`。

        只认「时间列与值比较」这一种形态，认不出就返回 `None`——
        这个字段最终会变成 Evidence 的 `event_time`，**宁可没有，不能猜错**：
        一个错的时间区间会让 Phase 9 的 TIME 冲突检测报出根本不存在的冲突。
        """
        lower: date | None = None
        upper: date | None = None
        for node in tree.find_all(exp.GTE, exp.GT, exp.LTE, exp.LT, exp.Between):
            if not self._is_time_comparison(node, aliases):
                continue
            if isinstance(node, exp.Between):
                low = _as_date(_resolve_value(node.args.get("low"), params))
                high = _as_date(_resolve_value(node.args.get("high"), params))
                if low is not None:
                    lower = low if lower is None else min(lower, low)
                if high is not None:
                    upper = high if upper is None else max(upper, high)
                continue
            moment = _as_date(_resolve_value(node.args.get("expression"), params))
            if moment is None:
                continue
            if isinstance(node, (exp.GTE, exp.GT)):
                lower = moment if lower is None else min(lower, moment)
            else:
                upper = moment if upper is None else max(upper, moment)

        if lower is None or upper is None:
            return None
        return TimeRange(start=_midnight(lower), end=_midnight(upper))

    def _is_time_comparison(self, node: exp.Expr, aliases: Mapping[str, str]) -> bool:
        left = node.args.get("this")
        if not isinstance(left, exp.Column):
            # 也接受「值 比较 列」的写法，生成的 SQL 里偶有出现
            left = node.args.get("expression")
        if not isinstance(left, exp.Column):
            return False
        resolved = aliases.get(left.table) if left.table else None
        candidates: Iterable[TableSpec] = (
            [spec]
            if resolved and (spec := self._catalog.table(resolved)) is not None
            else self._catalog.tables
        )
        return any(
            spec.time_column is not None and spec.time_column == left.name for spec in candidates
        )


# ------------------------------------------------------------------ 模块级辅助
def _outer_select(tree: exp.Expr) -> exp.Select | None:
    """最外层的 SELECT。

    `find(exp.Select)` 是**深度优先**的，对有 CTE 的语句会先命中 CTE 里的那个，
    于是 LIMIT 被加到了子查询上、外层仍然没有上限。
    """
    if isinstance(tree, exp.Select):
        return tree
    return tree.find(exp.Select)


def _literal_limit(limit: exp.Expr | None) -> int | None:
    if limit is None:
        return None
    try:
        return int(limit.expression.this)
    except (AttributeError, TypeError, ValueError):
        # `LIMIT :n` 这类非字面量：编译期判不了大小，当作「未知」由调用方收紧。
        return None


def _find_comment(sql: str) -> str | None:
    """检测 SQL 文本里的注释，且**不误伤字符串字面量**。

    为什么不用 sqlglot 的 tokenizer：它**静默丢弃**注释 token——实测
    `Tokenizer(dialect="mysql").tokenize("SELECT a FROM t -- x")` 不产出任何
    注释 token，因此无法据此判断「有没有注释」。手写状态机只跟踪引号，
    二十行之内，比依赖一个会丢信息的 API 可靠。
    """
    index = 0
    length = len(sql)
    quote: str | None = None
    while index < length:
        char = sql[index]
        if quote is not None:
            if char == "\\" and quote in {"'", '"'}:
                index += 2  # 转义符后的字符不参与判定
                continue
            if char == quote:
                # 双写引号（`''`）在 SQL 里表示一个字面引号，不是字符串结束
                if index + 1 < length and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "#":
            return "行注释（#）"
        elif sql.startswith("--", index):
            # MySQL 要求 `--` 后跟空白才算注释（`1--2` 是表达式）；
            # 但那种写法在只读查询里没有意义，一律当注释拦掉更省心。
            return "行注释（--）"
        elif sql.startswith("/*", index):
            return "块注释（/* */）"
        index += 1
    return None


#: 函数调用的形态：标识符紧跟左括号。用来把**运算符**从函数里摘出去。
_FUNCTION_CALL = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _function_name(node: exp.Expr) -> str | None:
    """取函数在 **MySQL 方言下**的名字；**不是函数调用时返回 `None`**。

    两个坑决定了这个函数长这样：

    1. **不能用 `Func.sql_names()`**：它给的是通用名，而 `DATE_FORMAT` 在
       sqlglot 里是 `TimeToStr`，`sql_names()` 返回 `TIME_TO_STR`——
       拿它比对白名单会把一个完全正常的日期函数判成「未登记」。
       改为按目标方言渲染一次、取括号前的标识符，代价是一次字符串渲染，
       换来的是「模型看到的函数名」与「我们比对的名字」同源。
    2. **`exp.Func` 的子类远不止函数**：sqlglot 30 里 `And` / `Or` / `GTE` /
       `Add` 这些运算符都继承自 `Func`，直接遍历会把 `a AND b` 当成一个
       名叫 `r.n = 'x' AND s.d >= :a` 的函数。因此按**渲染结果**判定——
       真正的函数调用一定渲染成 `名字(` 开头，运算符不会。
    """
    if isinstance(node, exp.Anonymous):
        return node.name.upper()
    match = _FUNCTION_CALL.match(node.sql(dialect=_DIALECT))
    return match.group(1).upper() if match else None


def _fingerprint(tree: exp.Expr) -> str:
    """规范化 SQL 的指纹，**字面量全部替换为占位符**。

    这样指纹才能回答详设 16.6 提出的那个问题：「同一批烂 SQL 是否反复出现」。
    保留字面量的话，同一个句型换个日期就是一个新指纹，
    按 `sql_fingerprint` 聚合查不出任何模式。
    """
    anonymized = tree.copy()
    for literal in anonymized.find_all(exp.Literal):
        literal.replace(exp.Placeholder())
    return hashlib.sha256(anonymized.sql(dialect=_DIALECT).encode()).hexdigest()


def _placeholder(name: str) -> exp.Expr:
    return exp.Placeholder(this=name)


def _projection_aliases(node: exp.Expr) -> set[str]:
    """一层查询的投影里显式起了别名的列名。

    `WITH x AS (SELECT SUM(a) AS total …)` 对外暴露的是 `total`，
    而不是 `a`——收集错方向的话，`SELECT total FROM x` 仍会被误判。
    """
    select = node if isinstance(node, exp.Select) else node.find(exp.Select)
    if select is None:
        return set()
    return {
        projection.alias
        for projection in select.expressions
        if isinstance(projection, exp.Alias) and projection.alias
    }


def _refers_to_output_alias(column: exp.Column) -> bool:
    """判断一个未限定列名的列，是不是在引用 SELECT 的输出别名。

    只看 ORDER / GROUP / HAVING 三处：这三处引用输出别名是 SQL 允许的写法，
    而 WHERE 里出现同名标识符只可能是拼错了列名（SQL 标准本就不允许
    WHERE 引用输出别名）。**这个区分是必要的**——一律放行会让
    `WHERE customer_name = 'x'` 这种绕过角色检查的写法蒙混过关。
    """
    if not column.name:
        return False
    current = column.parent
    while current is not None:
        if isinstance(current, (exp.Order, exp.Group, exp.Having)):
            scope = current.find_ancestor(exp.Select)
            if scope is None:
                return False
            return any(
                isinstance(projection, exp.Alias) and projection.alias == column.name
                for projection in scope.expressions
            )
        if isinstance(current, exp.Select):
            return False
        current = current.parent
    return False


def _resolve_value(node: exp.Expr | None, params: Mapping[str, object]) -> object:
    if isinstance(node, exp.Placeholder):
        return params.get(str(node.this))
    if isinstance(node, exp.Literal):
        return node.this
    return None


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _midnight(day: date) -> datetime:
    """日期 → UTC 零点。

    `TimeRange` 语义是 `[start, end)`（见 `domain/evidence.py`），
    这里只做「日期到时刻」的补齐，不做时区换算——演示库里的
    `order_date` 是 DATE 型，没有时区语义，硬套一个时区只会引入偏差。
    """
    return datetime.combine(day, time.min, tzinfo=UTC)


__all__ = ["SqlValidationError", "SqlValidator"]
