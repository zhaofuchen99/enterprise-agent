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
from contextlib import contextmanager, suppress
from typing import Any, Final, cast

from opentelemetry import baggage, context, metrics, trace
from opentelemetry.metrics import Counter, Histogram, Meter, Observation, UpDownCounter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
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

#: 与 `_provider` 同形的 meter 侧单例。**两者必须被同一套幂等守卫与
#: `shutdown_observability` 成对管理**（各自说明见那两处）。
_meter_provider: MeterProvider | None = None

#: 按 `(种类, 名字)` 缓存的仪表。**provider 换了必须清空**（见 `_instrument`）：
#: 否则在 setup 之前创建的仪表会被永久绑在 no-op meter 上。
_instruments: dict[tuple[str, str], Any] = {}

#: 队列深度的最近一次采样值。`ObservableGauge` 的回调是**同步**的
#: （OTel 在导出线程里调它），不能在回调里 await 一次 `ZCARD`——
#: 所以由抽样方定期写入这里（见 `set_queue_depth`）。
_queue_depth: int = 0


def setup_observability(settings: Settings, *, service_name: str) -> None:
    """进程启动时调用一次。`OTEL_ENABLED=false` 时是空操作。

    采样策略用 `ParentBased` 包住比例采样，而不是裸的 `TraceIdRatioBased`：
    裸比例采样下，API 侧被采样的那条 trace，它的 Worker 侧子 span 会**各自独立**
    再抽一次签，于是同一任务在追踪后端上只剩下半条链。ParentBased 的语义正是
    「有父就跟着父」，这才是跨进程链路能连起来的前提。

    **幂等**：`worker.run_forever` 在依赖抖动时会先 `on_shutdown` 再 `on_startup`，
    也就是**重新走一遍这里**。不挡住的话，每一轮重启都会给 provider 叠加一个
    `BatchSpanProcessor` 与一个 `PeriodicExportingMetricReader`——症状是导出量
    随重启次数线性增长，而没有任何报错（`trace.set_tracer_provider` 那个
    "只能设一次"的警告只挡得住 tracer 本身，挡不住我们自己 add 的 processor）。

    ⚠️ 这个守卫与 `shutdown_observability` **必须成对**：shutdown 之后若不清空
    单例，下一次启动命中的就是"已初始化 → 提前返回"，而 provider 已经被关掉了
    ——span 与指标从此**永久静默**，比重复叠加更难发现（叠加至少还导出得出东西）。
    """
    global _provider, _meter_provider, _instruments

    if not settings.otel_enabled:
        logger.info("OTel 未启用（OTEL_ENABLED=false），跳过初始化")
        return
    if not settings.otlp_endpoint:
        # 配置模型已校验过这一条，这里兜底是因为 setup 可能被单独调用
        raise ValueError("OTEL_ENABLED=true 时必须提供 OTLP_ENDPOINT")
    if _provider is not None:
        logger.info("OTel 已初始化，跳过重复初始化")
        return

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
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=_otlp_endpoint(settings.otlp_endpoint, "/v1/traces"))
        )
    )
    _provider = provider
    trace.set_tracer_provider(provider)

    if settings.observability.metrics_enabled:
        _meter_provider = _build_meter_provider(settings, resource)
        metrics.set_meter_provider(_meter_provider)
        # **清空仪表缓存**：setup 之前创建的仪表绑在 no-op meter 上，而缓存键
        # 里没有 provider 的身份——不清的话它们的值永远出不去，而调用点
        # 看起来完全正常（见 `_instrument`）
        _instruments = {}

    logger.info(
        "OTel 已初始化：%s -> %s", resource.attributes["service.name"], settings.otlp_endpoint
    )


def _otlp_endpoint(base: str | None, path: str) -> str | None:
    """把 base URL 与 OTLP 的版本路径拼成**完整 URL**。

    ⚠️ **这一步不能省**：`OTLPSpanExporter(endpoint=...)` 收的是完整 URL，
    它**不会**替你补 `/v1/traces`。会替调用方补路径的是环境变量
    `OTEL_EXPORTER_OTLP_ENDPOINT` 那条路，而本实现走的是参数。
    不补的症状是**导出恒 404 而任务一切正常**：`OTEL_ENABLED=false` 时
    完全看不出来，一旦打开就是"接了后端却什么也看不到"，而报错只躺在
    应用日志的一角（`Failed to export span batch code: 404`）。

    实测（2026-09-29，`otel/opentelemetry-collector:0.161.0`）：
    `POST http://127.0.0.1:4318/` → **404**，`POST .../v1/traces` → 200。
    这个缺陷自 Phase 1.5 写在这里起就存在——**OTel 默认关闭，所以从未被跑过**。

    `OTLP_ENDPOINT` 的语义因此是 **base URL**（与 OTel 标准里那个
    `OTEL_EXPORTER_OTLP_ENDPOINT` 一致），两个 exporter 各自补自己的路径。
    """
    if not base:
        return None
    return f"{base.rstrip('/')}{path}"


def _build_meter_provider(settings: Settings, resource: Resource) -> MeterProvider:
    """构造 MeterProvider，并注册需要**采集时回调取值**的那几个仪表。

    OTLP 的 HTTP exporter 与 trace 侧共用同一个 `OTLP_ENDPOINT`：它是 base URL，
    两个 exporter 各自补默认路径（`/v1/traces` 与 `/v1/metrics`）——
    所以这里不用推导，也**不必再加第二个配置项**。
    """
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    provider = MeterProvider(
        resource=resource,
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=_otlp_endpoint(settings.otlp_endpoint, "/v1/metrics")),
                export_interval_millis=settings.observability.metric_export_interval_ms,
            )
        ],
    )
    # 队列深度是**唯一一个"值由别人提供、采集时拉取"的仪表**：它没有
    # "记录一次"这个动作（深度是个瞬时量），只能被观测
    provider.get_meter(_INSTRUMENTATION_NAME).create_observable_gauge(
        "agent.queue.depth",
        callbacks=[_observe_queue_depth],
        unit="{job}",
        description="待领取的任务数（19.4.3 点名它是一线故障信号）",
    )
    return provider


def _observe_queue_depth(options: Any) -> Iterator[Observation]:
    """`ObservableGauge` 的回调。**必须是同步的**：OTel 在导出线程里调它，
    在这里 await 一次 `ZCARD` 会直接抛错，而那个错发生在导出线程里、
    不会冒到调用点——症状是"指标一直不出现"，排查方向会跑到 collector 上去。
    取值的责任因此落在抽样方（`set_queue_depth`）。
    """
    yield Observation(_queue_depth)


def shutdown_observability() -> None:
    """进程收尾：冲刷并关闭两个 provider，**并清空单例**。

    清空是承重的一步，理由见 `setup_observability` 的幂等说明。
    `suppress(Exception)`：依赖抖动时（正是重启外壳会走到这里的那个场景）
    flush 本来就会失败，而那时任务已经在收尾——抛出去只会掩盖真正的原因。
    """
    global _provider, _meter_provider, _instruments

    for provider in (_meter_provider, _provider):
        if provider is not None:
            with suppress(Exception):
                provider.shutdown()
    _provider = None
    _meter_provider = None
    _instruments = {}


def configure_for_testing(provider: TracerProvider) -> None:
    """测试用：换成本地 provider（通常是 `InMemorySpanExporter`）。

    不碰全局 provider——那是一次性的，第一个用例设过之后其余用例就换不动了。

    **签名刻意不动**（meter 侧另有一个 `configure_metrics_for_testing`）：
    它已被 `test_observability.py` 的 `Process` 夹具与 `tools/sql/conftest.py`
    调用，为加一个用不上的参数而改那两处不值得。
    """
    global _provider
    _provider = provider


def configure_metrics_for_testing(reader: MetricReader) -> None:
    """测试用：换成本地 MeterProvider（通常是 `InMemoryMetricReader`）。

    **不清 `_instruments`** 是有意的：那由 `reset_for_testing` 负责。在这里清
    会让"配置 A → 记录 → 配置 B → 记录"这类用例丢掉第一段，而症状看起来
    只是"第一个 reader 没有数据"。
    """
    global _meter_provider
    _meter_provider = MeterProvider(metric_readers=[reader])


def reset_for_testing() -> None:
    """清掉**全部**模块级单例（含 meter 与仪表缓存）。

    `test_observability.py` 的 autouse 夹具会调它。漏清 meter 会让前一个用例的
    reader 继续收后面用例记的指标——那正是"断言绿得莫名其妙"的典型来源。
    """
    global _provider, _meter_provider, _instruments
    _provider = None
    _meter_provider = None
    _instruments = {}


def get_tracer() -> Tracer:
    if _provider is not None:
        return _provider.get_tracer(_INSTRUMENTATION_NAME)
    return trace.get_tracer(_INSTRUMENTATION_NAME)


def get_meter() -> Meter:
    """当前 meter。**没有 provider 时返回 no-op meter**——与 `get_tracer` 同形，
    因此调用点永远不需要写 `if enabled`，也不会因为开关状态而分叉出第二条路径。
    """
    if _meter_provider is not None:
        return _meter_provider.get_meter(_INSTRUMENTATION_NAME)
    return metrics.get_meter(_INSTRUMENTATION_NAME)


def counter(name: str, *, unit: str = "", description: str = "") -> Counter:
    """取一个 Counter（同名同类型只建一次）。"""
    return cast(Counter, _instrument("counter", name, unit, description))


def histogram(name: str, *, unit: str = "", description: str = "") -> Histogram:
    return cast(Histogram, _instrument("histogram", name, unit, description))


def up_down_counter(name: str, *, unit: str = "", description: str = "") -> UpDownCounter:
    """可增可减的计数。**SSE 连接数用它**：连接是会减下去的量，而 `Counter`
    只能往上加——那样"此刻有几个连接"就永远读不出来（只剩一个累计值）。
    """
    return cast(UpDownCounter, _instrument("up_down_counter", name, unit, description))


def set_queue_depth(value: int) -> None:
    """喂给 `agent.queue.depth` 那个 `ObservableGauge`（见 `_queue_depth`）。"""
    global _queue_depth
    _queue_depth = value


def _instrument(kind: str, name: str, unit: str, description: str) -> Any:
    """按 `(种类, 名字)` 缓存仪表。

    缓存省下的是每次打点的对象创建，而**它的正确性依赖"provider 换了就清空"**
    （`setup_observability` 与 `reset_for_testing` 各清一次）：缓存键里没有
    provider 的身份，所以在 setup 之前创建的仪表会被永久绑在 no-op meter 上
    ——症状是"指标代码明明在跑，后端一条都收不到"，而调用点看起来完全正常。
    """
    key = (kind, name)
    existing = _instruments.get(key)
    if existing is not None:
        return existing
    meter = get_meter()
    factories: dict[str, Any] = {
        "counter": meter.create_counter,
        "histogram": meter.create_histogram,
        "up_down_counter": meter.create_up_down_counter,
    }
    created: Any = factories[kind](name, unit=unit, description=description)
    _instruments[key] = created
    return created


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
    "configure_metrics_for_testing",
    "counter",
    "get_meter",
    "get_tracer",
    "histogram",
    "reset_for_testing",
    "restore_trace_context",
    "set_queue_depth",
    "setup_error_tracking",
    "setup_observability",
    "shutdown_observability",
    "span",
    "up_down_counter",
]
