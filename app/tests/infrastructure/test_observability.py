"""可观测：span 工具与跨进程 trace 上下文（开发流程 6.3 施工项 5）。

**这个文件守的是 Phase 1.5 门禁里最难事后补救的一条**：
从 `/api/agent/chat` 到最终 `final_answer` 的所有 span 必须属于同一个 trace
（详细设计 19.4.1）。API 与 Worker 是两个进程、中间隔着 Redis 队列，
OTel 的进程内上下文在那里天然断掉，唯一的接法是**投递时序列化、领取时恢复**。
等 Phase 7 图写完再补，届时所有节点都按「自己就是根」写好了，等于整条链路返工。

测试用两个各自独立的 `TracerProvider` 模拟两个进程：
如果只用同一个 provider，「跨进程」这件事根本没被验证到。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.infrastructure import observability
from app.infrastructure.observability import (
    BAGGAGE_TRACE_ID,
    business_trace_id_from_context,
    capture_trace_context,
    configure_for_testing,
    reset_for_testing,
    restore_trace_context,
    span,
)


class Process:
    """一个「进程」：独立的 provider 与 exporter，互不共享内存状态。"""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def activate(self) -> None:
        configure_for_testing(self.provider)

    @property
    def spans(self) -> list[object]:
        return list(self.exporter.get_finished_spans())


@pytest.fixture(autouse=True)
def _clean_tracer() -> Iterator[None]:
    # 模块级 provider 是全局状态，用例之间必须清干净，
    # 否则前一个用例的 exporter 会跟着后面的用例一起增长。
    reset_for_testing()
    yield
    reset_for_testing()


@pytest.fixture
def api_process() -> Process:
    return Process()


@pytest.fixture
def worker_process() -> Process:
    return Process()


# ------------------------------------------------------------------ span 工具
def test_span_records_attributes(api_process: Process) -> None:
    api_process.activate()

    with span("api.create_task", task_id="tsk_1", worker_id=None) as current:
        assert current.is_recording()

    (finished,) = api_process.spans
    assert finished.name == "api.create_task"  # type: ignore[attr-defined]
    attributes = finished.attributes  # type: ignore[attr-defined]
    assert attributes["task_id"] == "tsk_1"
    # None 属性被丢掉：写进去会渲染成字符串 "None"，比不写更误导人
    assert "worker_id" not in attributes


def test_spans_nest_within_one_process(api_process: Process) -> None:
    api_process.activate()

    with span("outer"), span("inner"):
        pass

    by_name = {s.name: s for s in api_process.spans}  # type: ignore[attr-defined]
    assert by_name["inner"].parent is not None  # type: ignore[attr-defined]
    assert by_name["inner"].parent.span_id == by_name["outer"].context.span_id  # type: ignore[attr-defined]


# --------------------------------------------------- 跨进程链路（本阶段门禁）
def test_worker_span_joins_the_api_trace(api_process: Process, worker_process: Process) -> None:
    """门禁：API 与 Worker 的 span 属于同一条 trace，链路不断。"""
    api_process.activate()
    with span("api.http") as api_span:
        carrier = capture_trace_context("trc_0000000001AAAAAAAAAAAA")

    # ---- 进程边界：上下文只能靠这个 dict 过去 ----
    worker_process.activate()
    with restore_trace_context(carrier), span("worker.task_body"):
        pass

    (worker_span,) = worker_process.spans
    assert worker_span.context.trace_id == api_span.context.trace_id  # type: ignore[attr-defined]


def test_worker_span_is_a_child_of_the_api_span(
    api_process: Process, worker_process: Process
) -> None:
    """只是「同一条 trace」还不够：父子关系断了，瀑布图上会排成两个并列的根。"""
    api_process.activate()
    with span("api.http") as api_span:
        carrier = capture_trace_context()

    worker_process.activate()
    with restore_trace_context(carrier), span("worker.task_body"):
        pass

    (worker_span,) = worker_process.spans
    assert worker_span.parent is not None  # type: ignore[attr-defined]
    assert worker_span.parent.span_id == api_span.context.span_id  # type: ignore[attr-defined]


def test_business_trace_id_survives_the_process_boundary(
    api_process: Process, worker_process: Process
) -> None:
    """OTel 的 trace 是 32 位十六进制，本项目对外的是 `trc_` ID，两者都要带过去。

    业务 trace_id 走 baggage：日志检索、`agent_task.trace_id` 都用它，
    丢了它跨进程链路即使连上也查不到对应的任务。
    """
    api_process.activate()
    with span("api.http"):
        carrier = capture_trace_context("trc_0000000001AAAAAAAAAAAA")

    worker_process.activate()
    with restore_trace_context(carrier):
        assert business_trace_id_from_context() == "trc_0000000001AAAAAAAAAAAA"


def test_baggage_key_is_inside_the_carrier(api_process: Process) -> None:
    api_process.activate()
    with span("api.http"):
        carrier = capture_trace_context("trc_x")

    assert "traceparent" in carrier
    assert BAGGAGE_TRACE_ID in carrier["baggage"]


def test_context_does_not_leak_to_the_next_task(
    api_process: Process, worker_process: Process
) -> None:
    """一个 Worker 会连续跑很多任务，上下文泄漏会让两个任务的 span 串成一条链——
    这比断链更难发现，因为图上看不出异常。"""
    api_process.activate()
    with span("api.http"):
        carrier = capture_trace_context("trc_1")

    worker_process.activate()
    with restore_trace_context(carrier), span("first_task"):
        pass
    with span("second_task"):
        pass

    by_name = {s.name: s for s in worker_process.spans}  # type: ignore[attr-defined]
    first = by_name["first_task"]
    second = by_name["second_task"]
    assert first.context.trace_id != second.context.trace_id  # type: ignore[attr-defined]
    assert second.parent is None  # type: ignore[attr-defined]


def test_empty_carrier_starts_a_new_trace(worker_process: Process) -> None:
    """投递时没有活动 span（比如补偿扫描重投）不该让 Worker 崩溃或复用别人的上下文。"""
    worker_process.activate()

    with restore_trace_context({}), span("worker.task_body"):
        pass

    (finished,) = worker_process.spans
    assert finished.parent is None  # type: ignore[attr-defined]


# ------------------------------------------------------------------ 初始化
def test_setup_is_a_noop_when_disabled(settings: object) -> None:
    """OTEL_ENABLED=false 时不该建 provider、也不该去连 OTLP 端点。"""
    observability.setup_observability(settings, service_name="api")  # type: ignore[arg-type]

    assert observability._provider is None


def test_error_tracking_is_a_noop_without_a_dsn(settings: object) -> None:
    observability.setup_error_tracking(settings)  # type: ignore[arg-type]
