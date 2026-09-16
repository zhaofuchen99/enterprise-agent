"""Alembic 运行环境（开发流程 6.4 第 1 项）。

连接串从 `Settings.database_url_agent` 取，与 API / Worker 用的是**同一份配置**。
不从 `alembic.ini` 读，是为了避免出现「迁移连 A 库、服务连 B 库」这种
表面通过、实际改错库的情况。

`sqlalchemy.url` 是 asyncmy 驱动，因此整个迁移过程跑在 async engine 上，
DDL 通过 `run_sync` 下推到同步上下文执行。
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# 必须导入 models 包：autogenerate 靠 Base.metadata 做差，
# 未被导入的模型不在 metadata 里，表现为「迁移脚本少了一张表」且不报错。
import app.infrastructure.models  # noqa: F401
from app.core.config import get_settings
from app.infrastructure.db import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return get_settings().database_url_agent


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连库。"""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # 打开类型比对：字段类型被改动而没被察觉，是「本地跑得好好的、
        # 上线才发现列类型不对」的典型来源。
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section, {}) or {}
    configuration["sqlalchemy.url"] = _url()

    # NullPool：迁移是一次性动作，用连接池只会在结束时留下未回收的连接，
    # 使 `alembic upgrade` 偶发挂住等池超时。
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        async with connectable.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
