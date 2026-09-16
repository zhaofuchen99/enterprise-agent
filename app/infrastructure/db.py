"""数据库接入层（详细设计 4.2 的 `infrastructure/db.py`）。

只负责三件事：声明 `Base`、按配置建 async engine、给出会话工厂。
表的定义在 `app/infrastructure/models/`，查询在 `app/repositories/`。

**为什么用 `expire_on_commit=False`**：默认行为下 commit 会过期 ORM 对象的全部属性，
之后再读任意字段都会触发一次懒加载 IO。在 async 会话里这种隐式 IO 会直接抛
`MissingGreenlet`，而报错位置通常离真正的原因很远（常见于 commit 后访问 `obj.id`）。
关掉过期语义，commit 后对象仍是普通快照，少一类难查的错。

**为什么在这里统一 `naming_convention`**：MySQL 会自行给未命名约束生成名字，
`alembic downgrade` 时按名字删除就找不到目标。统一约定后，索引与约束的名字由
SQLAlchemy 生成且可预测，迁移脚本的 upgrade/downgrade 才能稳定往返。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings

#: 约束命名约定。`ix` 对应索引、`uq` 唯一约束、`fk` 外键、`ck` 检查约束。
#: `%(column_0_N_name)s` 会拼上涉及的列名，避免同一张表上多个同类约束重名。
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s",
    "pk": "pk_%(table_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
}


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def create_engine(settings: Settings, *, url: str | None = None) -> AsyncEngine:
    """建 async engine。

    `url` 可覆盖，供 `alembic` 与业务演示库复用同一个建法（两边池参数一致，
    否则会出现「迁移能连、运行连不上」这类只在参数上不同的故障）。
    """
    return create_async_engine(
        url or settings.database_url_agent,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        # 连接池取连接前先探活。MySQL 会主动断开空闲超时的连接，
        # 不探活的话第一个拿到死连接的请求必然报错，且只在低峰后复现。
        pool_pre_ping=True,
        # 回显 SQL 只在本地调试开；生产开着会把每条语句连同参数写进日志，
        # 与「日志默认脱敏」的约束冲突。
        echo=False,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """事务边界：正常退出提交，异常回滚。

    仓储方法不自行 commit——否则一个业务动作会拆成多次提交，
    中途失败时留下半成品状态。
    """
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
