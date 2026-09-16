"""独立验证与运维入口（开发流程 4.2 的 `app/cli.py`）。

`make seed` / `make cleanup` 都走这里。做成子命令而不是一堆散落脚本，
是因为这些动作**必须复用应用自己的配置与仓储**：另写一份连接串与
SQL，就会出现「CLI 灌进了 A 库、服务读的是 B 库」这类只在演示时才发现的问题。

**这个模块不得 import `app/main.py`**：main 会加载 FastAPI，
而本模块要能在没有 Web 框架的进程里跑（也避免把 API 的启动成本
转嫁到一条运维命令上）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from app.core.config import Settings, get_settings
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import SERVICE_CLI, setup_logging
from app.repositories.user_repo import SqlUserRepository, seed_demo_users


async def _seed(settings: Settings) -> int:
    """灌入演示数据，幂等。

    Phase 2 分两步：演示账号（本文件）+ 业务演示数据（`fact_sales_order_item`
    等 8 张表，施工项 5/6）。**业务数据尚未实现**，见 CLAUDE.md 的进度表。
    """
    engine = create_engine(settings)
    try:
        sessions = create_session_factory(engine)
        created = await SqlUserRepository(sessions).upsert_demo_users(seed_demo_users(settings))
        print(f"演示账号：新增 {created} 个（已存在的跳过）")
        print("【后续扩展】业务演示数据（8 张表）尚未实现，见 Phase 2 施工项 5/6")
    finally:
        await engine.dispose()
    return 0


async def _cleanup(settings: Settings) -> int:
    """按保留期清理过期数据（详细设计 16.12）。

    **幂等**，供 cron 每日调用；不引入分布式调度器，多实例重复执行也不出错。
    """
    print("【后续扩展】`cleanup` 尚未实现：需先确定各表的保留期策略（详细设计 16.12）")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="项目运维与验证命令")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="灌入演示账号与业务演示数据（幂等）")
    sub.add_parser("cleanup", help="按保留期清理过期数据（幂等，供 cron 调用）")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(SERVICE_CLI, settings.log_level)

    handler = _seed if args.command == "seed" else _cleanup
    return asyncio.run(handler(settings))


if __name__ == "__main__":
    sys.exit(main())
