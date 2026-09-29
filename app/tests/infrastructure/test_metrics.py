"""OTel Metrics 的装配与两条静默失效（详细设计 19.4.3）。

这一层钉的是**"指标真的被送出去了"**，不是某个业务指标算得对不对——
后者的用例在各自埋点旁边（`test_tracing.py` / `test_task_runner.py` …）。

两条最容易静默失效的性质，都在这里：

1. **幂等**：`worker.run_forever` 在依赖抖动时会先 `on_shutdown` 再 `on_startup`，
   也就是**重走一遍** `setup_observability`。不挡住的症状是导出量随重启次数
   线性增长，而没有任何报错。
2. **shutdown 之后清空单例**：守卫看的是"`_provider` 是不是 None"，而 shutdown
   之后 provider 已经关掉了。不清空的话，下一次 setup 命中"已初始化 → 提前返回"，
   span 与指标从此**永久静默**——比重复导出更难发现。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from app.core.config import Settings
from app.infrastructure import model_gateway, observability
from app.infrastructure.model_gateway import TokenUsage

#: 一个**不可达**的 collector 地址。用例只需要把 exporter 建起来（构造不连接），
#: 真正会推的周期是 60s，而每个用例结束前都会 `shutdown_observability` 收掉线程。
_UNREACHABLE = "http://127.0.0.1:14318"


@pytest.fixture(autouse=True)
def _clean_singletons() -> Iterator[None]:
    """用例前后各清一次模块级单例——同 `test_observability.py` 的夹具。

    漏清的症状是前一个用例的 exporter / reader 继续收后面用例的东西，
    而那表现为"某个断言绿得莫名其妙"。
    """
    observability.reset_for_testing()
    yield
    observability.reset_for_testing()


def _enabled(settings: Settings) -> Settings:
    return settings.model_copy(update={"otel_enabled": True, "otlp_endpoint": _UNREACHABLE})


def _value(reader: InMemoryMetricReader, name: str) -> float:
    """把 reader 里的某个指标读成一个数。

    ⚠️ Counter 是**累积**语义：同一进程内多次 `add` 会相加，所以断言写的是
    一个确切值、不是"大于等于"——后者会让"记了两次"与"记了一次"同形。
    """
    data = reader.get_metrics_data()
    assert data is not None
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    # `Metric` 只是外壳，数据点挂在它的 `data` 上
                    # （`Sum` / `Gauge` 的 point 是 `.value`，`Histogram` 的是 `.sum`）
                    point = metric.data.data_points[0]
                    return float(getattr(point, "value", getattr(point, "sum", 0)))
    raise AssertionError(f"reader 里没有指标 {name}")


def test_meter_is_a_noop_before_any_setup() -> None:
    """没装 provider 时取仪表**不报错**（返回 no-op）。

    这是"调用点不写 `if enabled`"的前提。分叉出两条路径的话，漏掉一处不会有
    任何症状——那个指标永远不出现，而代码看起来完全正常。
    """
    observability.get_meter()
    observability.counter("agent.test.noop").add(1)


def test_a_second_setup_is_ignored(settings: Settings) -> None:
    """连续调两次 `setup_observability` 不重建 provider（幂等）。

    `worker.run_forever` 在依赖抖动时会重走一遍启动流程。不挡住的话，每一轮
    重启都会多一个 `BatchSpanProcessor` 与一个 `PeriodicExportingMetricReader`，
    而 `trace.set_tracer_provider` 那个"只能设一次"的警告**只挡得住 tracer 本身**，
    挡不住我们自己 add 上去的 processor。
    """
    enabled = _enabled(settings)

    observability.setup_observability(enabled, service_name="api")
    first_tracer = observability._provider
    first_meter = observability._meter_provider
    assert first_tracer is not None
    assert first_meter is not None

    observability.setup_observability(enabled, service_name="api")

    assert observability._provider is first_tracer
    assert observability._meter_provider is first_meter

    observability.shutdown_observability()


def test_shutdown_clears_the_singletons_so_a_restart_works(settings: Settings) -> None:
    """`shutdown_observability` 之后**必须清空单例**，否则再也开不起来。

    这是幂等守卫的反面：守卫看的是"`_provider` 是不是 None"，而 shutdown 之后
    provider 已经被关掉了。不清空的话下一次 setup 命中"已初始化 → 提前返回"，
    指标与 span 从此永久静默，而启动日志写着"已初始化"。
    """
    enabled = _enabled(settings)

    observability.setup_observability(enabled, service_name="worker")
    observability.shutdown_observability()

    assert observability._provider is None
    assert observability._meter_provider is None

    # 重启外壳的完整一轮：关掉之后还能再开起来
    observability.setup_observability(enabled, service_name="worker")

    assert observability._provider is not None
    assert observability._meter_provider is not None

    observability.shutdown_observability()


def test_counters_and_histograms_reach_the_reader() -> None:
    """记录的指标能被 reader 读到，且**属性跟着一起到**。

    属性是这批指标的全部价值所在——`agent.tool.calls` 不带 `tool` / `status`
    的话只是一堆无法归因的数字。
    """
    reader = InMemoryMetricReader()
    observability.configure_metrics_for_testing(reader)

    observability.counter("agent.test.calls", unit="{call}").add(2, {"tool": "sql"})
    observability.histogram("agent.test.duration", unit="ms").record(120, {"node": "sql"})

    assert _value(reader, "agent.test.calls") == 2
    assert _value(reader, "agent.test.duration") == 120

    data = reader.get_metrics_data()
    assert data is not None
    points = [
        point
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        for point in metric.data.data_points
    ]
    assert any(dict(point.attributes or {}).get("tool") == "sql" for point in points)


def test_instruments_are_rebound_after_a_reset() -> None:
    """`reset_for_testing` 之后新建的仪表绑到**新的** reader 上。

    这是"provider 换了要清仪表缓存"那条规矩的正面断言：不清的话，第二个 reader
    一条都收不到，而调用点看起来完全正常（指标代码在跑、后端没有数据）。
    """
    first, second = InMemoryMetricReader(), InMemoryMetricReader()

    observability.configure_metrics_for_testing(first)
    observability.counter("agent.test.rebound").add(1)

    observability.reset_for_testing()
    observability.configure_metrics_for_testing(second)
    observability.counter("agent.test.rebound").add(1)

    assert _value(second, "agent.test.rebound") == 1


def _names(reader: InMemoryMetricReader) -> set[str]:
    data = reader.get_metrics_data()
    assert data is not None
    return {
        metric.name
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def test_cost_is_absent_when_the_price_is_unset(settings: Settings) -> None:
    """单价未配置时**只记 token，不记成本**。

    记一条 `cost=0` 是把"不知道单价"写成了"免费"——而它在任何报表上都
    看不出来（同"引用拿不到就留空、不填 0"的取舍）。token 计数照记：
    那个数是确定的，与知不知道单价无关。
    """
    reader = InMemoryMetricReader()
    observability.configure_metrics_for_testing(reader)

    model_gateway._record_usage(TokenUsage(prompt_tokens=100, completion_tokens=50), "m", settings)

    assert _value(reader, "llm.tokens") == 100
    assert "llm.cost" not in _names(reader)


def test_cost_is_recorded_when_the_price_is_set(settings: Settings) -> None:
    """配了单价就按 **prompt / completion 分别折算**（两者不同价是常态）。"""
    reader = InMemoryMetricReader()
    observability.configure_metrics_for_testing(reader)
    observability_settings = settings.observability.model_copy(
        update={
            "model_prompt_price_per_million": 1.0,
            "model_completion_price_per_million": 2.0,
        }
    )
    priced = settings.model_copy(update={"observability": observability_settings})

    model_gateway._record_usage(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000), "m", priced
    )

    assert _value(reader, "llm.cost") == 3.0


def test_a_tracer_provider_does_not_turn_metrics_on() -> None:
    """只有 trace provider 时，`get_meter` 仍然返回 no-op（不抛）。

    `configure_for_testing` 与 `configure_metrics_for_testing` 是**两条独立的
    测试钩子**：`tools/sql/conftest.py` 只调前者，而工具内部会记指标——
    那条路径必须能安全地跑（记录被丢掉，不报错）。
    """
    observability.configure_for_testing(TracerProvider())

    assert observability._meter_provider is None
    observability.counter("agent.test.tracer_only").add(1)
