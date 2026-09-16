"""OTel 初始化、span 工具与跨进程 trace 传递（开发流程 6.3 施工项 5）。

**这个模块刻意不 import 任何 Web 框架相关的东西**。FastAPI 的自动埋点
（`opentelemetry-instrumentation-fastapi`）会连带 import fastapi，
而 `app/worker.py` 要 import 本模块——只要这里引了它，Worker 进程就会把
fastapi 一起加载进内存，L2「Worker 不得依赖 Web 框架」在实质上被破坏，
而分层检查器只看 `app/worker.py` 自己的 import，**查不出来**。
因此 FastAPI 埋点放在 `app/main.py` 里单独调用，不放在这儿。

**跨进程 trace 为什么是这一阶段的重头**：详细设计 19.4.1 要求从
`/api/agent/chat` 到最终 `final_answer` 的所有 span 属于同一个 trace。
API 与 Worker 是两个进程、中间隔着 Redis 队列，OTel 的进程内上下文
到这里就断了。唯一的办法是在投递时把上下文序列化进任务参数
（`capture_trace_context`），Worker 领到任务后再恢复（`restore_trace_context`）。
这件事必须在 Phase 1.5 做——等 Phase 7 图写完再补，届时所有节点都按
「自己就是根」写好了，等于整条链路返工。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

from opentelemetry import baggage, context, trace
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, Tracer

from app.core.config import Settings

logger = logging.getLogger(__name__)

#: baggage 里携带业务 trace_id 的键名。OTel 的 trace 是 32 位十六进制，
#: 本项目对外暴露的是 `trc_` 前缀的 ULID 风格 ID（详细设计 19.4.1），
#: 两者都要跨进程传：前者用于 span 父子关系，后者用于日志检索。
BAGGAGE_TRACE_ID: Final[str] = "trc_id"

_INSTRUMENTATION_NAME: Final[str] = "app"

#: 模块级而非走全局 provider：OTel 的全局 provider 只能设置一次
#: （`trace.set_tracer_provider` 有一次性保护），而测试需要在同一个进程里
#: 反复换 exporter。持有自己的 provider 同时让代码不依赖全局可变状态。
_provider: TracerProvider | None = None


def setup_observability(settings: Settings, *, service_name: str) -> None:
    """进程启动时调用一次。`OTEL_ENABLED=false` 时是空操作。

    采样策略用 `ParentBased` 包住比例采样，而不是裸的 `TraceIdRatioBased`：
    裸比例采样下，API 侧被采样的那条 trace，它的 Worker 侧子 span 会**各自独立**
    再抽一次签，于是同一任务在追踪后端上只剩下半条链。ParentBased 的语义正是
    「有父就跟着父」，这才是跨进程链路能连起来的前提。
    """
    global _provider

    if not settings.otel_enabled:
        logger.info("OTel 未启用（OTEL_ENABLED=false），跳过初始化")
        return
    if not settings.otlp_endpoint:
        # 配置模型已校验过这一条，这里兜底是因为 setup 可能被单独调用
        raise ValueError("OTEL_ENABLED=true 时必须提供 OTLP_ENDPOINT")

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": settings.app_env,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(settings.otel_sample_rate)),
    )
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint))
    )
    _provider = provider
    trace.set_tracer_provider(provider)
    logger.info(
        "OTel 已初始化：%s -> %s", resource.attributes["service.name"], settings.otlp_endpoint
    )


def configure_for_testing(provider: TracerProvider) -> None:
    """测试用：换成本地 provider（通常是 `InMemorySpanExporter`）。

    不碰全局 provider——那是一次性的，第一个用例设过之后其余用例就换不动了。
    """
    global _provider
    _provider = provider


def reset_for_testing() -> None:
    global _provider
    _provider = None


def get_tracer() -> Tracer:
    if _provider is not None:
        return _provider.get_tracer(_INSTRUMENTATION_NAME)
    return trace.get_tracer(_INSTRUMENTATION_NAME)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """业务 span。属性里的 None 会被丢掉——OTel 接受 None 但会渲染成 `"None"`。"""
    with get_tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def setup_error_tracking(settings: Settings) -> None:
    """Sentry 初始化。没配 DSN 时是空操作。

    `send_default_pii=False` 是硬性的：本项目禁止把 Token、口令、
    SQL 原始行数据带出进程（CLAUDE.md 的脱敏纪律），而 Sentry 默认会带上
    请求头与用户信息，那里面就有 Authorization。
    """
    if not settings.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.app_env,
        release=None,
        traces_sample_rate=0.0,  # 链路追踪走 OTel，不让 Sentry 再采一份
        send_default_pii=False,
    )
    logger.info("Sentry 已初始化")


# --------------------------------------------------------------- 跨进程传递
def capture_trace_context(trace_id: str | None = None) -> dict[str, str]:
    """把当前 trace 上下文序列化成可放进队列参数的普通 dict。

    返回值里通常有 `traceparent`（W3C 标准头，决定 span 的父子关系）
    与 `baggage`（携带业务 trace_id）。队列参数必须是 JSON 可序列化的，
    因此这里刻意返回 `dict[str, str]` 而不是 OTel 内部的 carrier 对象。

    `inject` **只调用一次**：它同时写 traceparent 与 baggage 两个键，
    分两次调用会让第二次把第一次写好的 baggage 覆盖掉，
    表现是「链路连上了，但 Worker 侧拿不到业务 trace_id」。
    """
    carrier: dict[str, str] = {}
    ctx = context.get_current()
    if trace_id:
        ctx = baggage.set_baggage(BAGGAGE_TRACE_ID, trace_id, context=ctx)
    inject(carrier, context=ctx)
    return carrier


@contextmanager
def restore_trace_context(carrier: dict[str, str]) -> Iterator[None]:
    """Worker 侧恢复上下文。必须在任务体的**整个**执行期间保持。

    用 `with` 包住任务体，退出时自动 detach——不 detach 的话上下文会泄漏到
    同一 Worker 进程里下一个任务上，表现为「两个任务的 span 串成一条链」，
    比断链更难发现。
    """
    token = context.attach(extract(carrier or {}))
    try:
        yield
    finally:
        context.detach(token)


def business_trace_id_from_context() -> str | None:
    """从 baggage 里取回本项目的 `trc_` ID（Worker 侧绑定日志上下文用）。"""
    value = baggage.get_baggage(BAGGAGE_TRACE_ID)
    return str(value) if value else None


__all__ = [
    "BAGGAGE_TRACE_ID",
    "business_trace_id_from_context",
    "capture_trace_context",
    "configure_for_testing",
    "get_tracer",
    "reset_for_testing",
    "restore_trace_context",
    "setup_error_tracking",
    "setup_observability",
    "span",
]
