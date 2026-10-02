"""`SqlExecutor` 的**失败分类**（详设 9.4 / 10.8）。

执行器把数据库侧的失败分成三类，而**分类决定下一步动作**：

| 类 | 例子 | 下一步 |
|---|---|---|
| `REPAIRABLE` | 语法、别名、字段类型错 | 带错因重写 SQL（≤2 次） |
| `TRANSIENT` | 连不上、库侧拒绝 | **不修**——重写 SQL 不会让数据库变得可达 |
| `TIMEOUT` | 超过 `SQL_TOOL__TIMEOUT_SECONDS` | **不修**——重写不会让它变快 |

"连不上也去修 SQL"是个很容易写出来的错：修复循环会白跑两次、每次都拿到
同一个连接错误，而报告上只看到"修复了 2 次仍失败"——读起来像是 SQL 写得不好。

**不需要容器**：连一个没人监听的端口就能造出"业务库不可达"。
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.domain.user import PermissionScope, UserRole
from app.infrastructure.db import create_engine
from app.tools.sql.executor import SqlExecutionError, SqlExecutor
from app.tools.sql.schemas import SchemaCatalog
from app.tools.sql.validator import SqlValidator

#: 一个没人监听的本地端口（连接被拒是即时的，用例不会挂住）。
_DEAD_URL = "mysql+asyncmy://readonly:readonly_pw@127.0.0.1:1/business"

_SQL = (
    "SELECT SUM(net_amount) AS net_sales FROM fact_sales_order_item "
    "WHERE order_date >= :start AND order_date < :end"
)
_PARAMS = {"start": "2025-07-01", "end": "2025-10-01"}


async def test_a_dead_business_db_is_transient_and_not_repairable(settings: Settings) -> None:
    """业务库连不上 → `TRANSIENT` + `UPSTREAM_UNAVAILABLE`，且**不可修复**。

    为什么码是 `UPSTREAM_UNAVAILABLE`（503）而不是 `SQL_EXECUTION_REPAIRABLE`
    （422）：前者是对客户端说的"过一会儿再试"，后者是对我们说"重写这条 SQL"。
    库连不上时后者是错的建议。

    与 A 类（Qdrant 挂）同一条纪律：**同一个"外部依赖挂了"要在三处给出
    同一种待遇**（Redis → degraded、业务库 → 可重试、Qdrant → 可重试）。
    """
    catalog_validator = SqlValidator(_load_catalog(settings), max_sql_chars=8000, max_rows=1000)
    validated = catalog_validator.validate(
        _SQL, parameters=_PARAMS, scope=PermissionScope(role=UserRole.ADMIN)
    )
    executor = SqlExecutor(settings, engine=create_engine(settings, url=_DEAD_URL))
    try:
        with pytest.raises(SqlExecutionError) as caught:
            await executor.execute(validated)
    finally:
        await executor.aclose()

    assert caught.value.error_class == "TRANSIENT"
    assert caught.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert caught.value.repairable is False


def _load_catalog(settings: Settings) -> SchemaCatalog:
    """真实目录（与 SQL Tool 用的是同一份）。"""
    from app.tools.sql.schema_provider import load_catalog

    return load_catalog(settings.sql_tool.catalog_path)


def test_the_error_code_follows_the_class_not_the_other_way_round() -> None:
    """对照：可修复类才给 `SQL_EXECUTION_REPAIRABLE`。

    没有这条对照的话，一个"所有失败都标 UPSTREAM_UNAVAILABLE"的实现
    也能让上面那条通过——而它会**关掉自修复**（`tool.py` 按 `repairable`
    决定要不要重写 SQL），症状是"模型写错一列名就不再重试了"。
    """
    repairable = SqlExecutionError("字段不存在", error_class="REPAIRABLE")

    assert repairable.repairable is True
    assert repairable.code is ErrorCode.SQL_EXECUTION_REPAIRABLE


def test_a_timeout_keeps_its_own_class_under_an_upstream_code() -> None:
    """超时：**码**是 `UPSTREAM_UNAVAILABLE`（503 / 可重试），
    **类别**是 `TIMEOUT`（下次别原样重试）。

    两条轴各司其职，别把它们压成一个：
    - `code` 是给客户端的契约（19.1 的封闭码表，**不许新增同义码**）；
      而 `TASK_TIMEOUT` 是**任务总预算**语义（504 / 不重试 /「返回已有证据和限制」），
      与"这一条查询跑了 10 秒"差一个量级——用它会让客户端误判整个任务终结了。
    - `error_class` 是内部的 9.4 分类，它才表达"是超时还是连接被拒"。
    """
    timeout = SqlExecutionError("查询超时", error_class="TIMEOUT")

    assert timeout.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert timeout.error_class == "TIMEOUT"
    assert timeout.repairable is False


@pytest.mark.parametrize("cls", ["REPAIRABLE", "TRANSIENT", "TIMEOUT"])
def test_the_source_class_survives_on_the_exception(cls: str) -> None:
    """三类的 `error_class` 都原样留在异常上——工具层据此报 `ToolError`。"""
    exc = SqlExecutionError("x", error_class=cls)  # type: ignore[arg-type]

    assert exc.error_class == cls
    assert isinstance(exc, AgentError)
