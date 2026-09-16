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

from app.agent.prompts import SMOKE_PROMPT
from app.agent.schemas import SmokeAnswer
from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import SERVICE_CLI, setup_logging
from app.infrastructure.model_gateway import EmbeddingDimensionError, build_model_gateway
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


async def _model_smoke(settings: Settings) -> int:
    """打一次真实模型 + 一次真实向量化，把观测到的结果打印出来。

    **为什么需要它**：Phase 3 之后的所有节点都要过 `ModelGateway`，
    而它连的是外部服务。把「密钥对不对、端点通不通、返回的 JSON 能不能被
    Pydantic 吃下」这三件事留到第一个业务节点去发现，
    排查时会同时面对「prompt 写得对不对」和「配置对不对」两个未知数。
    自检把后者单独摘出来先验掉。

    **只打一次、不做重试放大**：网关自身的重试仍然生效，但这里不额外循环——
    自检要如实反映一次调用的代价（延迟、token），而不是压测吞吐。
    """
    gateway = build_model_gateway(settings)
    failures: list[str] = []
    try:
        print(f"主模型   ：{settings.model_name} @ {settings.model_base_url}")
        try:
            result = await gateway.invoke_structured(
                SMOKE_PROMPT, SmokeAnswer, question="请判断：1 + 1 是否等于 2？"
            )
        except AgentError as exc:
            failures.append(f"结构化调用：{exc.code.value} - {exc.message}")
            print(f"  ✗ 结构化调用失败：{exc.code.value} - {exc.message}")
            print(f"    详情（仅诊断）：{exc.details}")
        else:
            print(f"  ✓ 结构化调用成功：{result.value.model_dump()}")
            print(
                f"    耗时 {result.duration_ms}ms ｜ 请求次数 {result.attempts} ｜ "
                f"token 入/出 {result.usage.prompt_tokens}/{result.usage.completion_tokens} ｜ "
                f"prompt {SMOKE_PROMPT.name}({result.prompt_version})"
            )

        print(
            f"向量模型：{settings.embedding_model} @ "
            f"{settings.embedding_base_url or settings.model_base_url}"
        )
        try:
            vectors = await gateway.embed(["华东地区 Q3 净销售额"])
        except (AgentError, EmbeddingDimensionError) as exc:
            failures.append(f"向量化：{exc}")
            print(f"  ✗ 向量化失败：{exc}")
        else:
            print(f"  ✓ 向量化成功：{len(vectors)} 条 × {len(vectors[0])} 维")
    finally:
        await gateway.aclose()

    if failures:
        print("\n自检未通过：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\n自检通过。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="项目运维与验证命令")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="灌入演示账号与业务演示数据（幂等）")
    sub.add_parser("cleanup", help="按保留期清理过期数据（幂等，供 cron 调用）")
    sub.add_parser("model-smoke", help="打一次真实模型与向量化，验证连通性与结构化输出")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(SERVICE_CLI, settings.log_level)

    handlers = {"seed": _seed, "cleanup": _cleanup, "model-smoke": _model_smoke}
    return asyncio.run(handlers[args.command](settings))


if __name__ == "__main__":
    sys.exit(main())
