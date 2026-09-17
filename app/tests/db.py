"""测试用的真实数据库辅助。

**只在 `@pytest.mark.integration` 的用例里使用**——它连真实的 MySQL。
放在独立的模块而不是 conftest 里：conftest 的同名夹具会被所有用例解析，
而这个模块的入口是一个显式的上下文管理器，调用点一眼能看出「这里连库了」。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.infrastructure.models  # noqa: F401  必须导入：让全部表进 Base.metadata
from app.core.config import get_settings
from app.infrastructure.db import Base, create_engine, create_session_factory

_TABLES: tuple[str, ...] = tuple(Base.metadata.tables)

#: 本进程内表结构是否已经重建过。见 `sql_sessions` 的说明。
_schema_ready = False


@asynccontextmanager
async def sql_sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """连真实 MySQL，给出会话工厂，并在进入时清空全部业务表。

    **用 `create_all` 而不是跑 alembic**：契约测试验证的是**仓储行为**，
    不是迁移脚本，而跑一轮 alembic 比 create_all 慢一个数量级。
    迁移脚本由 `make migrate` 的 downgrade/upgrade 往返单独覆盖，
    两者各管一段、不重叠也不遗漏。

    **每个进程的第一次会先 `drop_all` 再 `create_all`**，之后只清数据。
    只 `create_all` 是不够的——它的 `checkfirst` 只看"表在不在"，
    不看列全不全，于是模型加了列之后，`agent_test` 里的旧表**照样通过 checkfirst**，
    失败要到 INSERT 时才发生，报的是 `Unknown column 'source_kind' in 'field list'`。
    那个报错读起来像代码写错了列名，而不像"测试库的表结构旧了"——
    本项目已经在这上面花掉一次排查（Phase 5 加 `source_kind` 时）。

    只做一次而不是每个用例一次：16 张表的 drop+create 不便宜，
    而同一进程内表结构不会变。清数据仍然每个用例都做。
    """
    global _schema_ready

    settings = get_settings()
    engine = create_engine(settings)
    try:
        async with engine.begin() as conn:
            if not _schema_ready:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
                _schema_ready = True
            for table in reversed(_TABLES):
                await conn.execute(text(f"DELETE FROM `{table}`"))  # noqa: S608 - 表名来自 metadata，非用户输入
        yield create_session_factory(engine)
    finally:
        await engine.dispose()
