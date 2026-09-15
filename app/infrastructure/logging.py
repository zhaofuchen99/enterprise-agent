"""JSON 结构化日志。

字段固定为开发流程 5.5 定义的那一组：
`timestamp、level、service、trace_id、conversation_id、task_id、step_id、
node、tool、status、duration_ms、error_code、message`

**默认脱敏**：调用方不得把 Token、密码、密钥、模型思维链、SQL 原始行数据写进 message。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

#: 允许出现在日志记录上的上下文字段（其余 extra 一律丢弃，防止误写敏感数据）
CONTEXT_FIELDS: tuple[str, ...] = (
    "trace_id",
    "conversation_id",
    "task_id",
    "step_id",
    "node",
    "tool",
    "status",
    "duration_ms",
    "error_code",
)

#: 跨 await 传播的上下文。API 进程按请求绑定，Worker 进程按任务绑定。
#: 默认值必须是 None 而非 `{}`：可变默认值会在所有上下文之间共享同一个 dict。
_log_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "log_context", default=None
)


def _current_context() -> dict[str, Any]:
    return dict(_log_context.get() or {})


def bind_context(**kwargs: Any) -> None:
    """把上下文字段绑定到当前执行上下文，后续日志自动携带。"""
    unknown = set(kwargs) - set(CONTEXT_FIELDS)
    if unknown:
        raise ValueError(f"不支持的日志上下文字段：{sorted(unknown)}")
    merged = {**_current_context(), **{k: v for k, v in kwargs.items() if v is not None}}
    _log_context.set(merged)


def clear_context() -> None:
    _log_context.set(None)


class JsonFormatter(logging.Formatter):
    """输出单行 JSON，便于被日志后端按字段检索。"""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 上下文先写入，显式 extra 再覆盖
        payload.update(_current_context())
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            # 堆栈只进受控日志，不回显给用户（详细设计 19.1）
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


#: uvicorn / arq 会给自己挂 handler 并置 propagate=False。不接管这几个 logger，
#: 它们的输出就会绕过 JsonFormatter，启动日志与访问日志的字段契约（5.5）直接断掉。
MANAGED_LOGGERS: tuple[str, ...] = ("uvicorn", "uvicorn.error", "uvicorn.access", "arq")

#: 只降噪，不接管的第三方 logger。注意不含 uvicorn.access——访问日志要保留。
NOISY_LOGGERS: tuple[str, ...] = ("httpx", "httpcore", "urllib3", "asyncio", "multipart")


def setup_logging(service: str, level: str = "INFO") -> None:
    """进程启动时调用一次。`service` 为 `api` 或 `worker`。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in MANAGED_LOGGERS:
        managed = logging.getLogger(name)
        managed.handlers.clear()
        managed.propagate = True

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
