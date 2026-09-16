"""trace_id 中间件。

为每个请求生成 `trc_` 前缀 ID（开发流程 6.2 施工项 3），并做三件事：

1. 写入 `scope["state"]`，供全局异常处理器与路由读取；
2. 绑定到日志上下文，此后本次请求的所有日志自动带 trace_id；
3. 回写 `X-Trace-Id` 响应头——用户报障时能直接给出可检索的 ID。

**不采用客户端传入的 trace_id**：请求头是可伪造的，若照单全收，
攻击者可以用同一个 trace_id 把多个请求串成一条链，污染排查。
跨服务串联由 Phase 11 的 OTel context 传播负责，那是受控通道。
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.ids import new_trace_id
from app.infrastructure.logging import bind_context, clear_context


def _with_trace_header(send: Send, trace_id: str) -> Send:
    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            MutableHeaders(scope=message)["X-Trace-Id"] = trace_id
        await send(message)

    return wrapped


class TraceContextMiddleware:
    """纯 ASGI 中间件，刻意不用 `BaseHTTPMiddleware`。

    `BaseHTTPMiddleware` 会把下游应用放进另一个 task 执行：contextvars 的传播
    依赖 Starlette 的内部实现细节，且响应体会被它缓冲——Phase 10 的
    `/stream` 要边算边推 SSE，被缓冲等于功能不成立。
    纯 ASGI 中间件与下游同处一个 task，这两个问题都不存在。
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # lifespan / websocket 不生成请求级 trace_id
            await self._app(scope, receive, send)
            return

        trace_id = new_trace_id()
        scope.setdefault("state", {})["trace_id"] = trace_id
        bind_context(trace_id=trace_id)
        try:
            await self._app(scope, receive, _with_trace_header(send, trace_id))
        finally:
            # 每个请求由 ASGI 服务器放在独立 task 中执行，此处清理是为了
            # 不把 trace_id 留给同一 task 内后续产生的日志。
            clear_context()
