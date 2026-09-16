"""全局异常处理器：把各类异常统一成 `{code, message, data, trace_id, retryable}`。

原则（详细设计 19.1）：业务异常在产生处就包装成 `AgentError`，
全局处理器**只负责 HTTP 映射**；未知异常的堆栈只进受控日志，绝不进响应体。

**框架级 404 的处理**：错误码表是封闭的（不得新增同义码），
而 Starlette 对「路径不存在」抛的 404 在表里只有 `TASK_NOT_FOUND` 一个对应项。
因此映射到它，但 `message` 明确写「接口不存在」以区分——
客户端应当按 `message` 判断，或在后续阶段补一个路径级 404 处理器消除这处含混。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import DEFAULT_RETRYABLE, AgentError, ErrorCode

logger = logging.getLogger(__name__)

#: 框架抛出的 HTTPException 没有错误码，按状态码回推一个语义最接近的码。
#: 业务错误一律走 AgentError，不经过这张表。表中不存在的状态码，
#: 按 4xx -> INVALID_ARGUMENT、5xx -> INTERNAL_ERROR 兜底，且**保留原始 HTTP 状态**。
_FRAMEWORK_STATUS_TO_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.INVALID_ARGUMENT,
    401: ErrorCode.AUTHENTICATION_REQUIRED,
    403: ErrorCode.ACCESS_DENIED,
    404: ErrorCode.TASK_NOT_FOUND,
    405: ErrorCode.INVALID_ARGUMENT,
    409: ErrorCode.TASK_CONFLICT,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.UPSTREAM_UNAVAILABLE,
}

_FRAMEWORK_MESSAGES: dict[int, str] = {
    404: "接口不存在",
    405: "请求方法不被支持",
}

#: 校验错误消息里最多回显几个字段，避免长表单刷屏
_MAX_REPORTED_FIELD_ERRORS = 3

#: pydantic 的 error type -> 中文说明。
#:
#: 不直接用 pydantic 自带的英文 msg：这些文案会原样显示在客户端上，
#: 而本项目「用户可见文案一律中文」（CLAUDE.md 代码约定）。
#: 表里没有的 type 走通用兜底，**不回显原始 msg**——它可能带出内部结构。
_REASON_ZH: dict[str, str] = {
    "missing": "缺少必填字段",
    "string_too_short": "长度不足",
    "string_too_long": "长度超限",
    "string_type": "必须是字符串",
    "string_pattern_mismatch": "格式不正确",
    "enum": "取值不在允许范围内",
    "literal_error": "取值不在允许范围内",
    "int_parsing": "必须是整数",
    "int_type": "必须是整数",
    "float_parsing": "必须是数字",
    "bool_parsing": "必须是布尔值",
    "list_type": "必须是数组",
    "dict_type": "必须是对象",
    "extra_forbidden": "不支持的字段",
    "json_invalid": "请求体不是合法的 JSON",
}


def trace_id_of(request: Request) -> str:
    """取本次请求的 trace_id（由 TraceContextMiddleware 写入 scope）。"""
    trace_id = getattr(request.state, "trace_id", None)
    return trace_id if isinstance(trace_id, str) else ""


def _envelope(
    request: Request,
    *,
    status_code: int,
    code: ErrorCode,
    message: str,
    retryable: bool,
) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content={
            "code": code.value,
            "message": message,
            "data": None,
            "trace_id": trace_id_of(request),
            "retryable": retryable,
        },
    )
    # 限流依赖会把 X-RateLimit-* 暂存在 request.state：抛异常时 FastAPI
    # 不会合并注入式 Response 的头部，只能在错误响应上手工补回来。
    rate_limit_headers = getattr(request.state, "rate_limit_headers", None)
    if isinstance(rate_limit_headers, dict):
        response.headers.update(rate_limit_headers)
    return response


def _validation_message(exc: RequestValidationError) -> str:
    """只回显字段位置与中文原因。

    **不能**把 pydantic 的 `input` 一起带出来：登录接口的校验错误里
    那个 input 就是明文口令，它会顺着错误响应、日志和前端 toast 一路扩散。
    同理也不回显原始的英文 `msg`，只按 error type 查表给中文说明。
    """
    parts: list[str] = []
    for error in exc.errors()[:_MAX_REPORTED_FIELD_ERRORS]:
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        reason = _REASON_ZH.get(str(error.get("type", "")), "取值不合法")
        parts.append(f"{location or '请求体'}：{reason}")
    return "；".join(parts) if parts else "请求参数不合法"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentError)
    async def _handle_agent_error(request: Request, exc: AgentError) -> JSONResponse:
        logger.warning(
            "业务异常：%s",
            exc.message,
            extra={"error_code": exc.code.value, "status": exc.code.value},
        )
        return _envelope(
            request,
            status_code=exc.http_status,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI 默认返回 422，但需求规格 7.3 把「参数缺失、格式错误或越界」
        # 定义为 INVALID_ARGUMENT 400。422 在本项目里留给 SQL 安全与无证据两类业务错误，
        # 不能跟「参数写错了」混成一个状态码。
        message = _validation_message(exc)
        logger.warning("请求参数校验失败：%s", message, extra={"status": "INVALID"})
        return _envelope(
            request,
            status_code=400,
            code=ErrorCode.INVALID_ARGUMENT,
            message=message,
            retryable=False,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _FRAMEWORK_STATUS_TO_CODE.get(
            exc.status_code,
            ErrorCode.INVALID_ARGUMENT if exc.status_code < 500 else ErrorCode.INTERNAL_ERROR,
        )
        return _envelope(
            request,
            status_code=exc.status_code,
            code=code,
            message=_FRAMEWORK_MESSAGES.get(exc.status_code, "请求无法处理"),
            retryable=DEFAULT_RETRYABLE.get(code, False),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # 堆栈只进日志（详细设计 19.1）；响应体里连异常类型都不给，
        # 异常类型本身就可能暴露用了哪个库、哪个连接串。
        logger.exception("未捕获异常", extra={"error_code": ErrorCode.INTERNAL_ERROR.value})
        return _envelope(
            request,
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="服务内部错误，请稍后重试",
            retryable=True,
        )
