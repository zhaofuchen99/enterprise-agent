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
from typing import Any

from app.agent.prompts import SMOKE_PROMPT
from app.agent.schemas import SmokeAnswer
from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import SERVICE_CLI, setup_logging
from app.infrastructure.model_gateway import EmbeddingDimensionError, build_model_gateway
from app.repositories.user_repo import SqlUserRepository, seed_demo_users
from app.tools.sql.schemas import SqlQueryArgs
from app.tools.sql.tool import build_cli_context, build_sql_query_tool


async def _seed(settings: Settings, args: argparse.Namespace) -> int:
    """灌入演示账号，幂等。

    业务演示库（`fact_sales_order_item` 等 8 张表）**不在这里**：那套数据是
    「模拟企业既有系统」，建表与灌数属于 DBA 侧的供给，走
    `scripts/business_seed.py`（`make seed` 会把两者依次跑一遍）。
    放进 CLI 会让读者以为应用运行时也会写那个库——而它连的是只读账号。
    """
    engine = create_engine(settings)
    try:
        sessions = create_session_factory(engine)
        created = await SqlUserRepository(sessions).upsert_demo_users(seed_demo_users(settings))
        print(f"演示账号：新增 {created} 个（已存在的跳过）")
        print("业务演示库请另行执行：make seed-business（或 make seed 一并跑）")
    finally:
        await engine.dispose()
    return 0


async def _cleanup(settings: Settings, args: argparse.Namespace) -> int:
    """按保留期清理过期数据（详细设计 16.12）。

    **幂等**，供 cron 每日调用；不引入分布式调度器，多实例重复执行也不出错。
    """
    print("【后续扩展】`cleanup` 尚未实现：需先确定各表的保留期策略（详细设计 16.12）")
    return 0


async def _model_smoke(settings: Settings, args: argparse.Namespace) -> int:
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

    def report_failure(label: str, exc: AgentError | EmbeddingDimensionError) -> None:
        """报告一次失败。**必须带 details**。

        自检命令的全部价值在于给出可行动的线索。只打印 `AgentError.message`
        （面向用户的通用文案，如「向量化服务返回错误」）等于什么都没说——
        真正有用的 `status_code` / `attempts` 都在 `details` 里。
        本项目自己的诊断工具尤其不能犯这个错。
        """
        if isinstance(exc, AgentError):
            failures.append(f"{label}：{exc.code.value} - {exc.message}")
            print(f"  ✗ {label}失败：{exc.code.value} - {exc.message}")
            print(f"    详情（仅诊断）：{exc.details}")
        else:
            failures.append(f"{label}：{exc}")
            print(f"  ✗ {label}失败：{exc}")

    try:
        print(f"主模型   ：{settings.model_name} @ {settings.model_base_url}")
        try:
            result = await gateway.invoke_structured(
                SMOKE_PROMPT, SmokeAnswer, question="请判断：1 + 1 是否等于 2？"
            )
        except AgentError as exc:
            report_failure("结构化调用", exc)
        else:
            print(f"  ✓ 结构化调用成功：{result.value.model_dump()}")
            print(
                f"    耗时 {result.duration_ms}ms ｜ 请求次数 {result.attempts} ｜ "
                f"token 入/出 {result.usage.prompt_tokens}/{result.usage.completion_tokens} ｜ "
                f"prompt {SMOKE_PROMPT.name}({result.prompt_version})"
            )

        # 打印**生效的**端点而不是配置里的那一项：`EMBEDDING_BASE_URL` 缺省时
        # 会回落到 `MODEL_BASE_URL`，于是向量化请求会被悄悄发到聊天模型的服务商那里。
        # 这一行是发现那类「配漏了但没报错」问题的第一现场。
        effective_embedding_url = settings.embedding_base_url or settings.model_base_url
        print(f"向量模型：{settings.embedding_model} @ {effective_embedding_url}")
        try:
            vectors = await gateway.embed(["华东地区 Q3 净销售额"])
        except (AgentError, EmbeddingDimensionError) as exc:
            report_failure("向量化", exc)
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


async def _sql(settings: Settings, args: argparse.Namespace) -> int:
    """跑一次完整的自然语言 → SQL → 真数据 → 证据（开发流程 6.6 的验证命令）。

    **这条命令是 Phase 4 的门禁本身**：详设 21.7 要求「SQL Tool 可独立通过
    API/CLI 调用，不依赖 Graph」。因此它走的是与将来 Worker 完全相同的
    `SqlQueryTool`，而不是一份为演示写的简化逻辑——那种写法只能证明
    「演示脚本能跑」，证明不了「工具能跑」。

    `--region` 模拟 `app_user.data_scope_json`：CLI 没有登录用户，
    但数据权限的注入路径必须被真实执行过一次才算数（详设 10.4 第 11 步）。
    不传即全量，与演示库里 admin 账号的语义一致。
    """
    gateway = build_model_gateway(settings)
    tool = build_sql_query_tool(settings, gateway)
    ctx = build_cli_context(
        region_ids=tuple(args.region), timeout_seconds=settings.task_timeout_seconds
    )

    try:
        if args.sql:
            result = await tool.run_sql(args.sql, ctx)
        else:
            result = await tool.execute(SqlQueryArgs(question=args.question), ctx)
    finally:
        await tool.aclose()

    if args.sql:
        print(f"给定 SQL（跳过生成，仍走完整校验）：{args.sql}")
    else:
        print(f"问题：{args.question}")
    if args.region:
        print(f"数据范围：{'、'.join(args.region)}（模拟 data_scope）")
    print(f"状态：{result.status}｜{result.summary}｜耗时 {result.duration_ms}ms")
    print()

    if result.status == "FAILED" and result.error is not None:
        error = result.error
        print(f"✗ 已拒绝：{error.message}")
        print(f"  错误码 {error.code}｜类别 {error.error_class}｜可重试 {error.retryable}")
        if error.safe_detail:
            print(f"  详情（仅诊断）：{error.safe_detail}")
        _print_attempts(result.payload)
        return 1

    payload = result.payload or {}
    print("规范化 SQL：")
    print(f"  {payload.get('normalized_sql', '')}")
    print(f"  fingerprint {payload.get('sql_fingerprint', '')}")
    print()

    columns = [column["name"] for column in payload.get("columns", [])]
    if columns:
        print(f"结果（{payload.get('row_count', 0)} 行）：")
        print("  " + " | ".join(columns))
        for row in payload.get("rows") or []:
            print("  " + " | ".join("" if v is None else str(v) for v in row))
    else:
        print("结果：未返回任何列")
    print()

    for warning in payload.get("warnings", []):
        print(f"⚠ {warning}")
    if payload.get("warnings"):
        print()

    if result.evidence:
        print(f"证据（{len(result.evidence)} 条）：")
        for item in result.evidence:
            print(f"  · {item.claim}")
            print(f"    定位 {item.locator}｜口径 {item.metric_code}@{item.definition_version}")
        print()

    _print_attempts(result.payload)
    return 0


def _print_attempts(payload: dict[str, Any] | None) -> None:
    """打印每次尝试。**失败的尝试也要打**——详设 6.6 施工项 5 要求每次尝试
    都落 `agent_tool_call`，而在落库之前，这里是唯一能看到「修复了几次、
    每次为什么失败」的地方。"""
    attempts: list[dict[str, Any]] = (payload or {}).get("attempts") or []
    if not attempts:
        return
    print(f"尝试记录（{len(attempts)} 次，落 agent_tool_call）：")
    for item in attempts:
        detail = item.get("error_summary") or ""
        print(
            f"  #{item.get('attempt_no')} {item.get('stage')} {item.get('status')}"
            f"｜{item.get('error_code') or '—'}｜{detail}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="项目运维与验证命令")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="灌入演示账号与业务演示数据（幂等）")
    sub.add_parser("cleanup", help="按保留期清理过期数据（幂等，供 cron 调用）")
    sub.add_parser("model-smoke", help="打一次真实模型与向量化，验证连通性与结构化输出")
    sql = sub.add_parser("sql", help="跑一次自然语言 → SQL → 真数据 → 证据（Phase 4 门禁）")
    sql.add_argument(
        "question",
        nargs="?",
        default="",
        help="业务问题，如「2025 年华东地区 Q3 净销售额是多少」",
    )
    sql.add_argument(
        "--sql",
        default="",
        metavar="SQL",
        help="直接给一条 SQL：跳过生成，但仍走全部 12 步校验与只读执行。"
        "用于单独验证「安全由代码保证」与跑金标 SQL",
    )
    sql.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="名称",
        help="模拟数据权限范围（可重复）。不传即全量，对应 admin 账号",
    )
    sql.set_defaults(region=[])
    args = parser.parse_args(argv)

    if args.command == "sql" and not args.question and not args.sql:
        # 两个入口都没有时**必须报错**，不能让 `make sql` 空跑一次模型。
        # argparse 的 `nargs="?"` 做不到「二者恰有其一」，因此在这里补一条校验。
        parser.error(
            'sql 需要一个问题，或用 --sql 直接给一条 SQL，例如：sql "2025 年华东 Q3 净销售额"'
        )

    settings = get_settings()
    setup_logging(SERVICE_CLI, settings.log_level)

    handlers = {
        "seed": _seed,
        "cleanup": _cleanup,
        "model-smoke": _model_smoke,
        "sql": _sql,
    }
    # 所有子命令都收 (settings, args)：让签名统一，代价只是三个用不到 args 的
    # 函数多一个参数；不统一的话分发处就得按命令名分支，加一个命令改一次。
    return asyncio.run(handlers[args.command](settings, args))


if __name__ == "__main__":
    sys.exit(main())
