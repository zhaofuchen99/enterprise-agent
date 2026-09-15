"""错误码与 AgentError 基类。

错误码表来自详细设计 19.1，**不得自行新增同义码**（开发流程 5.5）。
Graph 内异常统一包装为 AgentError 写入 State；HTTP 映射只在全局处理器里做。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """详细设计 19.1 的错误码表。"""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    ACCESS_DENIED = "ACCESS_DENIED"
    PLAN_INVALID = "PLAN_INVALID"
    SQL_VALIDATION_FAILED = "SQL_VALIDATION_FAILED"
    SQL_EXECUTION_REPAIRABLE = "SQL_EXECUTION_REPAIRABLE"
    NO_RELEVANT_KNOWLEDGE = "NO_RELEVANT_KNOWLEDGE"
    MODEL_RATE_LIMITED = "MODEL_RATE_LIMITED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    ENQUEUE_FAILED = "ENQUEUE_FAILED"
    QUEUE_BACKLOG = "QUEUE_BACKLOG"
    WORKER_INTERRUPTED = "WORKER_INTERRUPTED"
    REDIS_UNAVAILABLE = "REDIS_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


#: 错误码 -> HTTP 状态码。仅用于全局异常处理器的映射，业务代码不得直接读取。
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_ARGUMENT: 400,
    ErrorCode.ACCESS_DENIED: 403,
    ErrorCode.PLAN_INVALID: 422,
    ErrorCode.SQL_VALIDATION_FAILED: 422,
    ErrorCode.SQL_EXECUTION_REPAIRABLE: 422,
    ErrorCode.NO_RELEVANT_KNOWLEDGE: 422,
    ErrorCode.MODEL_RATE_LIMITED: 503,
    ErrorCode.UPSTREAM_UNAVAILABLE: 503,
    ErrorCode.TASK_TIMEOUT: 504,
    ErrorCode.ENQUEUE_FAILED: 503,
    ErrorCode.QUEUE_BACKLOG: 503,
    ErrorCode.WORKER_INTERRUPTED: 500,
    ErrorCode.REDIS_UNAVAILABLE: 503,
    ErrorCode.INTERNAL_ERROR: 500,
}


class AgentError(Exception):
    """Agent 内部统一异常。

    Attributes:
        code: 详细设计 19.1 的错误码。
        message: 面向用户的提示，**不得包含堆栈或敏感数据**（开发流程 5.5）。
        details: 结构化上下文，只进日志与 Trace，不回显给用户。
        retryable: 客户端是否可直接重试。
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    @property
    def http_status(self) -> int:
        return HTTP_STATUS.get(self.code, 500)

    def __repr__(self) -> str:
        return f"AgentError(code={self.code.value!r}, message={self.message!r})"
