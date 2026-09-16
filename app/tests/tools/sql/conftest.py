"""SQL Tool 测试夹具。

**目录用真实的 `configs/schema_catalog.yaml`，不另造一份小的**：那份目录就是
生产里的安全白名单，测试若用另一份，「目录写错了」这件事永远不会被测出来。
代价是目录一改就可能挂测试——那正是我们要的信号。
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest

from app.core.config import Settings
from app.domain.user import PermissionScope, UserRole
from app.tests.fakes import FakeModelGateway
from app.tools.base import ToolContext
from app.tools.sql.schema_provider import SchemaProvider
from app.tools.sql.schemas import SchemaCatalog
from app.tools.sql.validator import SqlValidator

#: 演示库里两个账号的数据范围，取自 `app/repositories/user_repo.py` 的 `DEMO_ACCOUNTS`。
#: **刻意从那里抄成常量而不是 import**：这两个值是「账号的配置」，
#: 如果哪天演示账号改了范围，这里应当因为对不上而失败，而不是自动跟着变——
#: 自动跟着变会让「权限过滤生效了吗」这个问题失去一个独立的参照。
ANALYST_REGIONS: tuple[str, ...] = ("华东",)


@pytest.fixture
def catalog(settings: Settings) -> SchemaCatalog:
    """函数级而不是会话级：全局的 `settings` 夹具是函数级的（它要能按用例改环境变量），
    会话级夹具依赖函数级夹具会被 pytest 判为 ScopeMismatch。加载成本是几百微秒，
    相对一次模型调用可以忽略，不值得为它去复制一份配置解析逻辑。
    """
    return SchemaProvider.from_settings(settings).catalog


@pytest.fixture
def validator(catalog: SchemaCatalog, settings: Settings) -> SqlValidator:
    tuning = settings.sql_tool
    return SqlValidator(catalog, max_sql_chars=tuning.max_sql_chars, max_rows=tuning.max_rows)


@pytest.fixture
def unrestricted() -> PermissionScope:
    """全量用户，对应演示账号 admin（`region_ids` 为空）。"""
    return PermissionScope(role=UserRole.ADMIN)


@pytest.fixture
def restricted() -> PermissionScope:
    """限华东的分析师，对应演示账号 analyst。"""
    return PermissionScope(role=UserRole.ANALYST, region_ids=ANALYST_REGIONS)


@pytest.fixture
def ctx(restricted: PermissionScope) -> ToolContext:
    return ToolContext(
        user_id="usr_test",
        task_id="tsk_test",
        step_id="stp_test",
        trace_id="trc_test",
        permission_scope=restricted,
        deadline_at=datetime.now(UTC) + timedelta(seconds=60),
    )


@pytest.fixture
def gateway() -> FakeModelGateway:
    """模型替身。脚本由用例显式设置——**刻意没有默认响应**。

    默认给一条能过的 SQL 会让「忘了配脚本」表现为一个看似通过的用例，
    与 `FakeModelGateway` 自身的约定一致。
    """
    return FakeModelGateway()


@pytest.fixture(autouse=True)
def _reset_tracer() -> Iterator[None]:
    """每个用例前后清掉模块级 tracer。

    没有这个夹具时，第一个设置了 provider 的用例会把 trace 状态留在一个
    **已经被关闭的** InMemorySpanExporter 上，后续用例的 span 全部丢进黑洞——
    表现为「测试通过但什么都没验证」。与 `test_observability.py` 同一处理方式。
    """
    from app.infrastructure.observability import reset_for_testing

    reset_for_testing()
    yield
    reset_for_testing()
