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

#: 建表只做一次（`checkfirst`），每个用例前清数据。
_TABLES: tuple[str, ...] = tuple(Base.metadata.tables)


@asynccontextmanager
async def sql_sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """连真实 MySQL，给出会话工厂，并在进入时清空全部业务表。

    **用 `create_all` 而不是跑 alembic**：契约测试验证的是**仓储行为**，
    不是迁移脚本，而跑一轮 alembic 比 create_all 慢一个数量级。
    迁移脚本由 `make migrate` 的 downgrade/upgrade 往返单独覆盖，
    两者各管一段、不重叠也不遗漏。

    清表顺序取 `reversed` 只是为了可读性：本阶段**没有物理外键**
    （见 `app/infrastructure/models/common.py` 的说明），顺序实际不影响结果。
    """
    settings = get_settings()
    engine = create_engine(settings)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            for table in reversed(_TABLES):
                await conn.execute(text(f"DELETE FROM `{table}`"))  # noqa: S608 - 表名来自 metadata，非用户输入
        yield create_session_factory(engine)
    finally:
        await engine.dispose()
