"""错误码与 HTTP 映射测试（详细设计 19.1）。"""

from __future__ import annotations

import pytest

from app.core.errors import HTTP_STATUS, AgentError, ErrorCode


def test_every_code_has_http_mapping() -> None:
    """错误码表新增了码但忘记映射，会导致线上返回 500——这里直接兜住。"""
    assert set(HTTP_STATUS) == set(ErrorCode)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ErrorCode.INVALID_ARGUMENT, 400),
        (ErrorCode.ACCESS_DENIED, 403),
        (ErrorCode.PLAN_INVALID, 422),
        (ErrorCode.SQL_VALIDATION_FAILED, 422),
        (ErrorCode.MODEL_RATE_LIMITED, 503),
        (ErrorCode.TASK_TIMEOUT, 504),
        (ErrorCode.REDIS_UNAVAILABLE, 503),
        (ErrorCode.INTERNAL_ERROR, 500),
    ],
)
def test_http_status_mapping(code: ErrorCode, expected: int) -> None:
    assert AgentError(code, "x").http_status == expected


def test_retryable_defaults_to_false() -> None:
    assert AgentError(ErrorCode.INVALID_ARGUMENT, "x").retryable is False


def test_details_are_not_part_of_message() -> None:
    """details 只进日志与 Trace，不能混进用户可见的 message。"""
    error = AgentError(ErrorCode.SQL_VALIDATION_FAILED, "查询不合法", details={"table": "secret"})
    assert "secret" not in error.message
    assert error.details == {"table": "secret"}
