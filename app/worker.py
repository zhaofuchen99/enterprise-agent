"""Worker 进程入口（arq）。

**分层硬约束（开发流程 5.2）**：本模块及 `app/agent/`、`app/tools/` 下任何模块，
不得 import `fastapi`。破坏后 Worker 无法独立扩缩容。
该约束由 `scripts/check_layering.py` 在 CI 中强制。
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from arq.connections import RedisSettings

from app.core.config import get_settings
from app.infrastructure.logging import bind_context, clear_context, setup_logging

logger = logging.getLogger(__name__)


async def ping(ctx: dict[str, Any], payload: str = "") -> str:
    """连通性自检任务：验证「入队 -> 领取 -> 执行 -> 写回」链路可用。

    Phase 1.5 的「空任务闭环」验收即基于此任务。
    """
    bind_context(task_id=str(ctx.get("job_id", "")), tool="ping", status="running")
    try:
        logger.info("worker ping 收到载荷", extra={"payload_len": len(payload)})
        return f"pong:{payload}"
    finally:
        clear_context()


async def on_startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    setup_logging(settings.otel_service_name, settings.log_level)
    logger.info("worker 启动完成")


async def on_shutdown(ctx: dict[str, Any]) -> None:
    logger.info("worker 正在退出")


def _redis_settings() -> RedisSettings:
    """从 REDIS_URL 解析 arq 连接参数，避免维护第二份配置。"""
    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """`arq app.worker.WorkerSettings` 的入口。"""

    functions: ClassVar[list[Any]] = [ping]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings: ClassVar[RedisSettings] = _redis_settings()
    #: 任务超时与 loop.max_expansions 联动（开发流程 5.4）
    job_timeout: ClassVar[int] = get_settings().task_timeout_seconds
    max_jobs: ClassVar[int] = 4
    keep_result: ClassVar[int] = 3600
