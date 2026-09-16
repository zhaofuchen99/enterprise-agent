"""错误码与 AgentError 基类。

错误码表来自详细设计 19.1，**不得自行新增同义码**（开发流程 5.5）。
Graph 内异常统一包装为 AgentError 写入 State；HTTP 映射只在全局处理器里做。

**19.1 的补齐说明**：19.1 原表只有 400/403/422/500/503/504 六档，缺 401/404/409/429，
而开发流程 6.2 的 Phase 1 门禁要求「400/401/403/404/429/500 映射正确」。
经确认，从需求规格 7.3 取 `AUTHENTICATION_REQUIRED`、`TASK_NOT_FOUND`、
`TASK_CONFLICT`、`RATE_LIMITED` 四个码补入，命名与需求规格保持一致，
不另起同义名。已回写详细设计 19.1。

**Phase 3 的补齐说明（2026-09-16）**：开发流程 6.5 施工项 4 要求「结构化输出解析失败 →
`MODEL_OUTPUT_INVALID`」，但详细设计 19.1 的正式表里没有这个码——两份文档自相矛盾。
它也不能由现有码兼任：`PLAN_INVALID` 是计划语义，`UPSTREAM_UNAVAILABLE` 是「服务不可用」，
而这里是服务一切正常、只是输出不合规。需求 12.2.1 把「结构化输出不稳」列为本项目的
头号返工风险，需要它作为可观测落点。**这是补一个被遗漏的码，不是新增同义码**，
处理方式与上面的四项一致。已回写详细设计 19.1。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """详细设计 19.1 的错误码表（按 HTTP 状态码升序排列）。"""

    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    ACCESS_DENIED = "ACCESS_DENIED"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_CONFLICT = "TASK_CONFLICT"
    PLAN_INVALID = "PLAN_INVALID"
    SQL_VALIDATION_FAILED = "SQL_VALIDATION_FAILED"
    SQL_EXECUTION_REPAIRABLE = "SQL_EXECUTION_REPAIRABLE"
    NO_RELEVANT_KNOWLEDGE = "NO_RELEVANT_KNOWLEDGE"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    RATE_LIMITED = "RATE_LIMITED"
    WORKER_INTERRUPTED = "WORKER_INTERRUPTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ENQUEUE_FAILED = "ENQUEUE_FAILED"
    QUEUE_BACKLOG = "QUEUE_BACKLOG"
    MODEL_RATE_LIMITED = "MODEL_RATE_LIMITED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    REDIS_UNAVAILABLE = "REDIS_UNAVAILABLE"
    TASK_TIMEOUT = "TASK_TIMEOUT"


#: 错误码 -> HTTP 状态码。仅用于全局异常处理器的映射，业务代码不得直接读取。
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_ARGUMENT: 400,
    ErrorCode.AUTHENTICATION_REQUIRED: 401,
    ErrorCode.ACCESS_DENIED: 403,
    ErrorCode.TASK_NOT_FOUND: 404,
    ErrorCode.TASK_CONFLICT: 409,
    ErrorCode.PLAN_INVALID: 422,
    ErrorCode.SQL_VALIDATION_FAILED: 422,
    ErrorCode.SQL_EXECUTION_REPAIRABLE: 422,
    ErrorCode.NO_RELEVANT_KNOWLEDGE: 422,
    ErrorCode.MODEL_OUTPUT_INVALID: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.WORKER_INTERRUPTED: 500,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.ENQUEUE_FAILED: 503,
    ErrorCode.QUEUE_BACKLOG: 503,
    ErrorCode.MODEL_RATE_LIMITED: 503,
    ErrorCode.UPSTREAM_UNAVAILABLE: 503,
    ErrorCode.REDIS_UNAVAILABLE: 503,
    ErrorCode.TASK_TIMEOUT: 504,
}

#: 错误码 -> 客户端是否可直接重试。
#:
#: 需求规格 7.3 要求错误结构带 `retryable`，而「能否重试」本质上是**错误码自带的语义**，
#: 不是每个调用点的自由选择——同一码在不同地方给出不同答案才是真的出错。
#: 因此把默认值集中在这里，调用点只在确有例外时显式覆盖 `AgentError(..., retryable=)`。
DEFAULT_RETRYABLE: dict[ErrorCode, bool] = {
    ErrorCode.INVALID_ARGUMENT: False,
    ErrorCode.AUTHENTICATION_REQUIRED: False,
    ErrorCode.ACCESS_DENIED: False,
    ErrorCode.TASK_NOT_FOUND: False,
    ErrorCode.TASK_CONFLICT: False,
    ErrorCode.PLAN_INVALID: False,
    ErrorCode.SQL_VALIDATION_FAILED: False,
    ErrorCode.SQL_EXECUTION_REPAIRABLE: False,
    ErrorCode.NO_RELEVANT_KNOWLEDGE: False,
    #: 网关内部已按详设 9.4 的 VALIDATION 类别重试过一次；模型对同一 prompt
    #: 再答一遍大概率还是同样的结构，让客户端原样重试没有意义
    ErrorCode.MODEL_OUTPUT_INVALID: False,
    #: 限流与容量类：等一会儿再来是对的
    ErrorCode.RATE_LIMITED: True,
    ErrorCode.WORKER_INTERRUPTED: True,
    ErrorCode.INTERNAL_ERROR: True,
    ErrorCode.ENQUEUE_FAILED: True,
    ErrorCode.QUEUE_BACKLOG: True,
    ErrorCode.MODEL_RATE_LIMITED: True,
    ErrorCode.UPSTREAM_UNAVAILABLE: True,
    ErrorCode.REDIS_UNAVAILABLE: True,
    #: 超时代表本次预算已耗尽，原样重试大概率还是超时
    ErrorCode.TASK_TIMEOUT: False,
}


def http_status_of(code: ErrorCode) -> int:
    """错误码 -> HTTP 状态码。

    表里没有的码返回 500 而不是抛异常：这是给全局异常处理器兜底用的，
    兜底路径自己再抛异常只会把一次可诊断的失败变成一次不可诊断的失败。
    """
    return HTTP_STATUS.get(code, 500)


class AgentError(Exception):
    """Agent 内部统一异常。

    Attributes:
        code: 详细设计 19.1 的错误码。
        message: 面向用户的提示，**不得包含堆栈或敏感数据**（开发流程 5.5）。
        details: 结构化上下文，只进日志与 Trace，不回显给用户。
        retryable: 客户端是否可直接重试；缺省取 `DEFAULT_RETRYABLE`。
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = DEFAULT_RETRYABLE.get(code, False) if retryable is None else retryable

    @property
    def http_status(self) -> int:
        return http_status_of(self.code)

    def __repr__(self) -> str:
        return f"AgentError(code={self.code.value!r}, message={self.message!r})"
