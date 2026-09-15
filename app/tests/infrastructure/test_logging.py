"""结构化日志测试（开发流程 5.5）。

日志字段是跨进程排查的依据（Phase 11 的 trace 验收依赖它），
字段名写错不会有任何报错，只会让线上查不到——所以在这里钉死。
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from app.infrastructure.logging import (
    JsonFormatter,
    bind_context,
    clear_context,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    clear_context()
    yield
    clear_context()


def _emit(
    record_logger: str = "test", level: int = logging.INFO, **extra: object
) -> dict[str, Any]:
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter("api"))

    logger = logging.getLogger(record_logger)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.log(level, "测试消息", extra=extra)

    parsed: dict[str, Any] = json.loads(buffer.getvalue().strip())
    return parsed


def test_base_fields_present() -> None:
    payload = _emit()
    assert payload["service"] == "api"
    assert payload["level"] == "INFO"
    assert payload["message"] == "测试消息"
    assert "timestamp" in payload


def test_bound_context_is_attached() -> None:
    bind_context(trace_id="t-1", task_id="tsk-1", node="planner")
    payload = _emit()
    assert payload["trace_id"] == "t-1"
    assert payload["task_id"] == "tsk-1"
    assert payload["node"] == "planner"


def test_explicit_extra_overrides_bound_context() -> None:
    bind_context(trace_id="from-context")
    payload = _emit(trace_id="from-extra")
    assert payload["trace_id"] == "from-extra"


def test_none_values_are_not_bound() -> None:
    """绑定 None 不应写入字段，否则日志里会出现一堆 null 噪声。"""
    bind_context(trace_id="t-1", task_id=None)
    assert "task_id" not in _emit()


def test_unknown_context_field_rejected() -> None:
    """字段名写错必须立刻报错，而不是静默丢弃。"""
    with pytest.raises(ValueError, match="不支持的日志上下文字段"):
        bind_context(user_password="oops")


def test_context_does_not_leak_across_clears() -> None:
    bind_context(trace_id="t-1")
    clear_context()
    bind_context(task_id="tsk-1")
    payload = _emit()
    assert payload["task_id"] == "tsk-1"
    assert "trace_id" not in payload


def test_context_dict_is_not_shared_between_scopes() -> None:
    """ContextVar 默认值若是可变对象，会跨上下文共享同一个 dict。"""
    from app.infrastructure.logging import _current_context

    bind_context(trace_id="a")
    snapshot = _current_context()
    snapshot["trace_id"] = "tampered"
    assert _current_context()["trace_id"] == "a"


def test_uvicorn_loggers_are_taken_over() -> None:
    """不接管 uvicorn 的 logger，启动日志就绕过 JSON 格式，字段契约断掉。"""
    setup_logging("api", "INFO")
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        assert logger.propagate is True
        assert logger.handlers == []
