"""ModelGateway 契约测试（开发流程 6.5、详设 21.6）。

结构沿用 `test_storage_contract.py` 的三段式：**基本契约 / 越界防护 / 依赖行为**。

**同一批断言跑两个实现**，这是本文件的核心设计：

| 实现 | 覆盖什么 |
|---|---|
| `HttpModelGateway` + `MockTransport` | 请求构造、超时、状态码分类、重试编排、JSON 解析 |
| `FakeModelGateway` | 调用方（Phase 4/6/7 的节点）依赖的那部分契约 |

两个实现共用一个夹具，因此「替身和生产实现行为不一致」不会拖到线上才暴露
（这是 `fakes.py` 里那条纪律的反向保障：能用真实实现就别写替身，
而既然写了替身，就必须让它和真实实现受同一份约束）。

`MockTransport` 是关键：它让 `HttpModelGateway` 在**无网络**下跑完整实现，
所以「网关自己有没有重试」「状态码怎么分类」这些逻辑不靠替身假装。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from app.agent.prompts import PromptTemplate
from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.infrastructure.model_gateway import (
    EmbeddingDimensionError,
    HttpModelGateway,
    ModelGateway,
    build_json_contract,
)
from app.tests.fakes import FakeFailure, FakeModelGateway

#: 契约测试统一的向量维度。**必须固定**——真实实现的维度来自 `EMBEDDING_DIM`，
#: 替身则来自构造参数，两者不一致时形状断言就会在两个实现上给出不同答案。
_CONTRACT_EMBEDDING_DIM = 8

#: 契约测试用的模板。变量名刻意为 `question`，`render` 的变量校验路径因此被真实走过。
_TEMPLATE = PromptTemplate(
    name="contract",
    version="v-test",
    template="请回答：{question}",
)


class Verdict(BaseModel):
    """形状足够简单的判据模型：标量 + 数组 + 可选，覆盖示例生成的三条分支。"""

    decision: str
    confidence: int
    reasons: list[str]
    note: str | None = None


_VALID_VERDICT: dict[str, Any] = {"decision": "SUFFICIENT", "confidence": 5, "reasons": ["r"]}


def _chat_body(
    content: str, *, prompt_tokens: int = 7, completion_tokens: int = 3
) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _embeddings_body(count: int) -> dict[str, Any]:
    return {
        "data": [
            {"index": index, "embedding": [0.1] * _CONTRACT_EMBEDDING_DIM} for index in range(count)
        ]
    }


def _default_handler(request: httpx.Request) -> httpx.Response:
    """默认「一切正常」的服务端，让 happy-path 用例零配置即可跑两个实现。"""
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json=_embeddings_body(len(json.loads(request.content)["input"])))
    return httpx.Response(200, json=_chat_body(json.dumps(_VALID_VERDICT)))


def _settings(settings: object, *, tuning: dict[str, Any] | None = None) -> Settings:
    """在测试 Settings 上派生一份，固定向量维度并按需覆盖重试预算。

    刻意不走环境变量。契约测试关心的是「同一份配置下网关的行为」，
    而 `MODEL_TUNING__*` 前缀写错会**静默走默认值**（见 config.py 的说明）——
    那类问题由 `app/tests/core/test_config.py` 覆盖，在这里重复只会让
    一个拼错的前缀表现为「配置没生效」的假象。

    嵌套项也要显式重建 `model_tuning`：`model_copy(update=...)` 是浅更新，
    直接写 `model_copy(update={"model_tuning": "..."})` 会把整个嵌套模型换掉。
    """
    assert isinstance(settings, Settings)
    update: dict[str, Any] = {"embedding_dim": _CONTRACT_EMBEDDING_DIM}
    if tuning is not None:
        update["model_tuning"] = settings.model_tuning.model_copy(update=tuning)
    return settings.model_copy(update=update)


# --------------------------------------------------------------------- 夹具
@pytest.fixture
def handler_box() -> dict[str, Any]:
    """可替换的 MockTransport 处理函数。

    `httpx.MockTransport` 在构造时绑定 handler，而用例需要在**运行中**换掉
    服务端行为（先 429 再 200），所以垫一层间接。
    """
    return {"handler": _default_handler}


@pytest.fixture
def make_gateway(settings: object, handler_box: dict[str, Any]) -> Callable[..., ModelGateway]:
    """契约测试的**唯一入口**——两个实现都从这里构造。"""

    def _make(
        kind: str = "http", *, tuning: dict[str, Any] | None = None, **overrides: Any
    ) -> ModelGateway:
        current = _settings(settings, tuning=tuning).model_copy(update=overrides)
        if kind == "fake":
            return FakeModelGateway(embedding_dim=_CONTRACT_EMBEDDING_DIM)
        transport = httpx.MockTransport(lambda request: handler_box["handler"](request))
        return HttpModelGateway(current, transport=transport)

    return _make


@pytest.fixture
def gateway(make_gateway: Callable[..., ModelGateway]) -> ModelGateway:
    return make_gateway()


def _set_handler(
    handler_box: dict[str, Any], fn: Callable[[httpx.Request], httpx.Response]
) -> None:
    handler_box["handler"] = fn


def _ok(content: str) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, json=_chat_body(content))


def _script(gateway: ModelGateway, *replies: Any) -> None:
    """给替身排脚本。真实实现由 `handler_box` 驱动，这里什么都不做。"""
    if isinstance(gateway, FakeModelGateway):
        gateway.responses.extend(replies)


# ================================================================ 基本契约
async def test_returns_validated_model_instance(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """返回的必须是**校验过的 Pydantic 实例**，不是裸 dict。

    开发流程 5.3 的硬性约束：所有进入 LangGraph State 的 LLM 输出必须先经
    Pydantic 校验，禁止把任意 dict 直接写入 State。契约测试是这条约束的第一道门。
    """
    payload = {"decision": "EXPAND", "confidence": 3, "reasons": ["缺口"]}
    _set_handler(handler_box, _ok(json.dumps(payload)))
    _script(gateway, Verdict(**payload))
    result = await gateway.invoke_structured(_TEMPLATE, Verdict, question="华东区下滑原因？")

    assert isinstance(result.value, Verdict)
    assert result.value.decision == "EXPAND"
    assert result.prompt_version == _TEMPLATE.version
    assert result.attempts == 1
    assert result.duration_ms >= 0


async def test_usage_is_reported(gateway: ModelGateway) -> None:
    """用量必须随结果返回。

    `agent_tool_call` 与 `agent_trace_event` 都没有 token 列，用量只能由调用方
    取用——网关丢了它就再也补不回来。
    """
    _script(gateway, _VALID_VERDICT)
    result = await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0


async def test_prompt_is_rendered_and_carries_json_contract(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """发出去的 prompt 里变量被替换、且带着 JSON 输出契约。

    DeepSeek 的 JSON 模式要求 prompt 含字面量 "json" 并给出格式示例，
    缺了服务端不保证返回合法 JSON。这条断言守住的就是那个要求。
    """
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=_chat_body(json.dumps(_VALID_VERDICT)))

    _set_handler(handler_box, handler)
    _script(gateway, _VALID_VERDICT)
    await gateway.invoke_structured(_TEMPLATE, Verdict, question="华东区下滑原因？")

    if isinstance(gateway, FakeModelGateway):
        rendered = gateway.calls[0].rendered
        assert "华东区下滑原因？" in rendered
    else:
        content = sent["messages"][0]["content"]
        assert "华东区下滑原因？" in content
        assert "json" in content
        assert '"decision"' in content
        assert sent["response_format"] == {"type": "json_object"}
        # 思考模式必须显式关闭（DeepSeek V4 服务端默认开启）
        assert sent["thinking"] == {"type": "disabled"}


async def test_embed_shape(gateway: ModelGateway) -> None:
    vectors = await gateway.embed(["华东", "渠道折扣"])
    assert len(vectors) == 2
    assert all(len(vector) == _CONTRACT_EMBEDDING_DIM for vector in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


async def test_embed_follows_returned_index_order(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """向量必须按 `index` 归位，不能依赖返回顺序。

    OpenAI 兼容端点不保证 `data` 与输入同序。直接按返回顺序取用会让
    「第二段文本」拿到「第一段文本」的向量——检索结果全错，且不报任何错。
    """
    gateway = make_gateway()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0] * _CONTRACT_EMBEDDING_DIM},
                    {"index": 0, "embedding": [1.0] * _CONTRACT_EMBEDDING_DIM},
                ]
            },
        )

    _set_handler(handler_box, handler)
    vectors = await gateway.embed(["第一段", "第二段"])

    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 2.0


async def test_embed_empty_input_makes_no_call(gateway: ModelGateway) -> None:
    """空输入直接返回，不打服务——多数端点对空 input 的行为是未定义的。"""
    assert await gateway.embed([]) == []
    if isinstance(gateway, FakeModelGateway):
        assert gateway.embed_calls == []


async def test_aclose_is_idempotent(gateway: ModelGateway) -> None:
    await gateway.aclose()
    await gateway.aclose()


# --------------------------------------------------------------------- 重排
#
# 重排的**服务形状与 chat / embeddings 不是同一套**：cross-encoder 没有
# OpenAI 兼容端点，业界事实标准是 Cohere 那套 `POST /rerank`
# （`{model, query, documents[], top_n}` → `results[].index / relevance_score`）。
# 因此这一段的断言逐条对着那个形状写，而不是套用上面的 chat 用例。
#
# **两个实现都跑**：替身脚本与 MockTransport 的返回由同一份 `_RERANK_SCORES`
# 驱动，于是"替身与真实实现不一致"不会拖到线上才暴露。

#: 服务端"真相"：三条候选的分数。
_RERANK_SCORES = [0.9, 0.1, 0.5]

#: 打开重排所需的配置。**三项都必填**（`Settings` 的交叉校验要求），
#: 少一项的报错由 `app/tests/core/test_config.py` 覆盖。
_RERANK_ON: dict[str, Any] = {
    "reranker_enabled": True,
    "reranker_model": "contract-reranker",
    "reranker_base_url": "https://rerank.invalid/v1",
    "reranker_api_key": "test-key",
}


def _rerank_handler(
    scores: list[float], *, order: Sequence[int] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """按 Cohere 形状返回结果。`order` 用来模拟"服务端按分数降序返回"。"""

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["documents"])
        indices = list(order) if order is not None else list(range(count))
        return httpx.Response(
            200,
            json={
                "results": [{"index": index, "relevance_score": scores[index]} for index in indices]
            },
        )

    return handler


async def test_rerank_returns_scores_aligned_to_the_input(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """返回与输入**同序同长**的分数。

    网关不排序也不剔除：那是 `tools/rag/reranker.py` 的策略（阈值、Top-K、
    并列怎么破恰恰是评测要调的东西）。网关给排好序的结果，等于把策略的一半
    塞进基础设施层。
    """
    gateway = make_gateway("http", **_RERANK_ON)
    _set_handler(handler_box, _rerank_handler(_RERANK_SCORES))

    scores = await gateway.rerank("华东渠道折扣上限", ["甲", "乙", "丙"])

    assert scores == _RERANK_SCORES

    fake = FakeModelGateway(rerank_scores=[_RERANK_SCORES])
    assert await fake.rerank("华东渠道折扣上限", ["甲", "乙", "丙"]) == _RERANK_SCORES


async def test_rerank_follows_returned_index_order(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """`results` 按分数降序返回（`index` 是乱序的），必须**按 `index` 归位**。

    直接按返回顺序取用会让「第 1 条候选」拿到「最相关那条」的分数——
    分数全对、配对全错，而且不报任何错。
    """
    gateway = make_gateway("http", **_RERANK_ON)
    _set_handler(handler_box, _rerank_handler(_RERANK_SCORES, order=[1, 2, 0]))

    assert await gateway.rerank("q", ["甲", "乙", "丙"]) == _RERANK_SCORES


async def test_rerank_posts_the_documented_request(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """请求体：`{model, query, documents, top_n}`，路径 `/rerank`，带 Bearer 头。

    **`top_n` 必须等于候选数**：不传时部分服务端只回前若干条，而"没回来的"
    与"没算过的"在 `results` 里长得一样。

    路径断言写全 `/v1/rerank`：`base_url` 里带着 `/v1`（硅基流动等厂商都是
    这个形状），`rerank` 才是这里拼上去的那一段。
    """
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        sent["path"] = request.url.path
        sent["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    gateway = make_gateway("http", **_RERANK_ON)
    _set_handler(handler_box, handler)
    await gateway.rerank("华东渠道折扣上限", ["甲"])

    assert sent["path"] == "/v1/rerank"
    assert sent["model"] == "contract-reranker"
    assert sent["query"] == "华东渠道折扣上限"
    assert sent["documents"] == ["甲"]
    assert sent["top_n"] == 1
    assert sent["authorization"] == "Bearer test-key"


async def test_rerank_missing_scores_is_loud(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """少一条就报错，**绝不补 0**。

    补 0 等于断言"这条与问题无关"——那是个我们并不知道的结论（它压根没被评分）。
    报错之后调用方会整次降级回 RRF 序，那是个正确的结果。
    """
    gateway = make_gateway("http", **_RERANK_ON)
    _set_handler(
        handler_box,
        lambda request: httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.9}]}
        ),
    )

    with pytest.raises(AgentError) as failure:
        await gateway.rerank("q", ["甲", "乙"])

    assert failure.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert "不足" in failure.value.message


async def test_rerank_server_error_is_upstream_unavailable(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """5xx 与 429 都归 `UPSTREAM_UNAVAILABLE`，且**不带响应体**。

    不带体是脱敏纪律：服务端报错时经常把请求回显在 body 里，而请求头上有
    `Authorization`。
    """
    gateway = make_gateway("http", **_RERANK_ON)
    _set_handler(handler_box, lambda request: httpx.Response(503, json={"echo": "secret-key"}))

    with pytest.raises(AgentError) as failure:
        await gateway.rerank("q", ["甲"])

    assert failure.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert "secret-key" not in str(failure.value.details)


async def test_rerank_without_configuration_is_loud(gateway: ModelGateway) -> None:
    """`RERANKER_ENABLED=false` 时调用重排直接报错，**不静默返回假分数**。

    静默返回一个"全都相关"的分数，会让一次错误的排序看起来像是重排器的判断结果。
    """
    with pytest.raises(AgentError, match="未配置"):
        await gateway.rerank("q", ["甲"])


async def test_rerank_empty_input_makes_no_call(gateway: ModelGateway) -> None:
    """空输入不打服务（与 `embed` 同）：没有候选就没有可排的东西。"""
    assert await gateway.rerank("q", []) == []


# ================================================================ 越界防护
async def test_timeout_is_reported_as_upstream_unavailable(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """详设 21.6 的四类异常用例之一：超时。

    超时归入 TRANSIENT（与「网络抖动」从客户端看不可区分），重试耗尽后必须落成
    `UPSTREAM_UNAVAILABLE`——**不是 `TASK_TIMEOUT`**，后者的语义是「任务总预算
    已耗尽」，用它会让客户端误判为任务已经终结。
    """
    if isinstance(gateway, FakeModelGateway):
        gateway.failure = FakeFailure.UNAVAILABLE
        expected_code = ErrorCode.UPSTREAM_UNAVAILABLE
    else:

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        _set_handler(handler_box, handler)
        expected_code = ErrorCode.UPSTREAM_UNAVAILABLE

    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert excinfo.value.code is expected_code
    assert excinfo.value.retryable is True


async def test_rate_limited_is_mapped_to_its_own_code(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """四类异常之二：429。

    429 有专属错误码 `MODEL_RATE_LIMITED`，不能混进通用的 `UPSTREAM_UNAVAILABLE`——
    两者对客户端的含义不同（一个是「配额用完了」、一个是「服务坏了」）。
    本用例同时是 `MODEL_RATE_LIMITED` 这个码的**首个产生点**回归。
    """
    if isinstance(gateway, FakeModelGateway):
        gateway.failure = FakeFailure.RATE_LIMITED
        expected_attempts = 1
    else:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429, json={"error": "rate limited"})

        _set_handler(handler_box, handler)
        expected_attempts = 2  # 1 次原始 + 1 次重试

    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert excinfo.value.code is ErrorCode.MODEL_RATE_LIMITED
    assert excinfo.value.details["attempts"] == expected_attempts


async def test_malformed_json_is_reported_as_invalid_output(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """四类异常之三：坏 JSON。服务是好的，问题在这一次的输出上。"""
    if isinstance(gateway, FakeModelGateway):
        gateway.failure = FakeFailure.INVALID_OUTPUT
    else:
        _set_handler(handler_box, _ok("这不是 JSON，是一段散文。"))

    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert excinfo.value.code is ErrorCode.MODEL_OUTPUT_INVALID


async def test_schema_mismatch_is_reported_as_invalid_output(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """四类异常之四：Schema 不匹配（合法 JSON，但字段对不上）。"""
    if isinstance(gateway, FakeModelGateway):
        gateway.failure = FakeFailure.INVALID_OUTPUT
    else:
        _set_handler(handler_box, _ok(json.dumps({"unexpected": True})))

    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert excinfo.value.code is ErrorCode.MODEL_OUTPUT_INVALID
    assert excinfo.value.retryable is False


async def test_empty_content_is_reported_as_invalid_output(
    gateway: ModelGateway, handler_box: dict[str, Any]
) -> None:
    """空 content。

    DeepSeek 的 JSON 模式**官方承认会偶发返回空内容**——这条用例不是假想的。
    """
    if isinstance(gateway, FakeModelGateway):
        gateway.failure = FakeFailure.INVALID_OUTPUT
    else:
        _set_handler(handler_box, _ok(""))

    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert excinfo.value.code is ErrorCode.MODEL_OUTPUT_INVALID


async def test_missing_prompt_variable_is_rejected_before_any_call(
    gateway: ModelGateway,
) -> None:
    """模板变量写错必须在**调用之前**炸，而不是把残缺 prompt 发给模型。

    发给模型的表现是「答非所问」，排查成本比一个异常高一个数量级。
    """
    with pytest.raises(ValueError, match="缺少变量"):
        await gateway.invoke_structured(_TEMPLATE, Verdict, wrong_name="x")


async def test_embedding_dimension_mismatch_is_loud(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """维度与 `EMBEDDING_DIM` 不符时必须炸，不能静默返回。

    维度对不上时向量库照样能建、能写入，却永远检索不到正确结果——
    属于「跑得通但结果是错的」，正是最该在写入前挡住的那类。
    """
    gateway = make_gateway()
    assert isinstance(gateway, HttpModelGateway)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 4}]})

    _set_handler(handler_box, handler)
    with pytest.raises(EmbeddingDimensionError, match="EMBEDDING_DIM"):
        await gateway.embed(["华东"])


# ================================================================ 依赖行为
async def test_transient_failure_is_retried_up_to_budget(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """TRANSIENT 预算：先 503 再 200，应当成功且 `attempts == 2`。

    只断言「最终成功」不够——那样即使一次都没重试也能通过。
    """
    gateway = make_gateway()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_chat_body(json.dumps(_VALID_VERDICT)))

    _set_handler(handler_box, handler)
    result = await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert calls["n"] == 2
    assert result.attempts == 2


async def test_transport_retry_budget_is_bounded(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """传输重试次数必须受配置约束，不能无界重试把任务卡满 `TASK_TIMEOUT`。"""
    gateway = make_gateway(tuning={"max_transport_retries": 0})
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": "unavailable"})

    _set_handler(handler_box, handler)
    with pytest.raises(AgentError):
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert calls["n"] == 1  # 预算为 0 时只发一次


async def test_repair_retry_does_not_consume_transport_budget(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """**两条预算相互独立、不可借用**。

    这里让第一次返回坏 JSON（消耗 VALIDATION 预算），第二次返回 503
    （消耗 TRANSIENT 预算），第三次成功。若两条预算共用一个计数器，
    传输重试会在第二次就被判定为「已耗尽」——总次数会掉到 2。
    """
    gateway = make_gateway()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_chat_body("半截 JSON"))
        if calls["n"] == 2:
            return httpx.Response(503, json={"error": "unavailable"})
        return httpx.Response(200, json=_chat_body(json.dumps(_VALID_VERDICT)))

    _set_handler(handler_box, handler)
    result = await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert calls["n"] == 3
    assert result.attempts == 3


async def test_repair_retry_sends_back_the_reason(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """修复重试必须把错因回灌，而不是原样重发同一个 prompt。

    原样重发在温度不为 0 时纯属碰运气；把「哪个字段不对」告诉模型才是修复。
    """
    gateway = make_gateway()
    contents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        contents.append(json.loads(request.content)["messages"][0]["content"])
        if len(contents) == 1:
            return httpx.Response(200, json=_chat_body(json.dumps({"wrong": 1})))
        return httpx.Response(200, json=_chat_body(json.dumps(_VALID_VERDICT)))

    _set_handler(handler_box, handler)
    await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert len(contents) == 2
    assert contents[1] != contents[0]
    # Pydantic 的字段路径应当出现在第二次的 prompt 里
    assert "decision" in contents[1]


async def test_non_retryable_4xx_is_not_retried(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """401 不能重试。

    密钥配错了重试一万次也是同样的结果，只会白白拖长任务。这条同时是
    「`_NonRetryableError` 必须是 `_TransientError` 的兄弟而不是子类」的回归保护——
    做成父子时调用点的 `except` 会把 401 一起接住。
    """
    gateway = make_gateway()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    _set_handler(handler_box, handler)
    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert calls["n"] == 1
    assert excinfo.value.retryable is False


async def test_api_key_never_leaks_into_error_or_logs(
    make_gateway: Callable[..., ModelGateway],
    handler_box: dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """开发流程 6.5 门禁：「模型错误不泄露密钥」。

    这里刻意让服务端**把密钥回显在响应体里**——真实世界里网关前置代理、
    或服务商的错误页都可能这么干。网关必须不把响应体带进异常或日志。
    """
    secret = "sk-contract-test-must-not-leak"
    gateway = make_gateway(model_api_key=secret)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid key: {secret}"})

    _set_handler(handler_box, handler)

    with caplog.at_level("DEBUG"), pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert secret not in str(excinfo.value)
    assert secret not in str(excinfo.value.details)
    assert secret not in caplog.text


async def test_bad_output_error_details_carry_no_model_output(
    make_gateway: Callable[..., ModelGateway], handler_box: dict[str, Any]
) -> None:
    """`MODEL_OUTPUT_INVALID` 的 `details` 不得夹带模型的原始输出。

    `details` 会进日志与 Trace。把整段原始输出塞进去，既可能夹带业务数据，
    也会让异常日志从几行膨胀到几百行。
    """
    gateway = make_gateway()
    noisy = "x" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(json.dumps({"noise": noisy})))

    _set_handler(handler_box, handler)
    with pytest.raises(AgentError) as excinfo:
        await gateway.invoke_structured(_TEMPLATE, Verdict, question="q")

    assert noisy not in json.dumps(excinfo.value.details, ensure_ascii=False)


# ================================================================ 纯函数
def test_json_contract_includes_nested_and_optional_fields() -> None:
    """格式示例要覆盖嵌套、数组与可选字段，否则模型只照着顶层填。"""

    class Inner(BaseModel):
        label: str

    class Outer(BaseModel):
        items: list[Inner]
        maybe: str | None = None

    contract = build_json_contract(Outer)
    assert '"items"' in contract
    assert '"label"' in contract  # 嵌套模型通过 $ref 解析出来
    assert '"maybe"' in contract
    assert "json" in contract
