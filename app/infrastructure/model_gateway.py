"""统一模型出口（开发流程 6.5、详细设计 4.1 技术栈表的「LLM 适配」行）。

**为什么放 `infrastructure/`**：这里 import 了 `httpx`。需求 8.4 的分层要求
「禁止在路由层直接执行 SQL 或调用模型」，`infrastructure/` 的职责正是把外部
组件隔离在一个文件内——换实现（换成 OpenAI 官方 SDK、换成 Anthropic 格式端点、
换成带原生 structured output 的服务）的成本应当只落在这个文件里，
而不泄漏到任何调用方。

**这个模块不得 import fastapi**：它会被 `app/agent/nodes/**` 使用，
而分层检查器的 L2 禁止 Worker 侧依赖 Web 框架。检查器只看
`app/worker.py` / `app/agent/**` **自己的** import，查不出「引了一个引了
fastapi 的模块」，所以这条纪律得靠这里自觉。

## 一个实现，不做 provider 分支

需求 8.6 要求「模型通过统一 Gateway 适配兼容 OpenAI Chat 风格的服务，
但当前运行只配置一个主模型」。云端的 `deepseek-flash` 与本地的
`bge-m3`（Ollama）恰好都提供 OpenAI 兼容端点，因此**生产实现只有一个**：
差别只是 `base_url` / `model` / `api_key` 三个配置值。
`model_provider` 保留为标签进 Trace，不作为分支条件。

**重排是这条纪律的一个例外，但例外只在协议形状上**：cross-encoder 没有
OpenAI 兼容端点（业界事实标准是 Cohere 的 `POST /rerank`），所以
`rerank` 的请求体与 chat / embeddings 不同。**「一个实现」本身没有被破坏**——
Jina、硅基流动、Cohere 都兼容那一套形状，换服务商仍然只改三个配置值，
调用方看到的也只是 `rerank(query, documents) -> list[float]`。

## 两条独立的失败预算

详细设计 9.4 的错误分类表把可重试错误分成两类，这里逐类实现：

| 类别 | 触发 | 预算与动作 |
|---|---|---|
| `TRANSIENT` | 429 / 5xx / 连接失败 / 超时 | `MAX_TRANSPORT_RETRIES`，指数退避 |
| `VALIDATION` | HTTP 200 但 JSON 解析或 Schema 失败 | `MAX_REPAIR_RETRIES`，错因回灌 prompt |

**两条预算相互独立、不可借用**——与 `loop` 的四类预算同一条纪律。
耗尽传输预算不会消耗修复预算，反之亦然。

> 关于超时归入 `TRANSIENT`：详设 9.4 的 `TIMEOUT` 行「可降级则继续，否则失败」
> 是 **Tool 级**语义（指的是 SQL 执行这类工具超时）；单次模型 HTTP 调用超时
> 从客户端看与「网络抖动」不可区分，故按 `TRANSIENT` 处理。

**网关只负责抛 `AgentError`，不负责降级。** 需求 FR-PLAN-004 要求
「反思模型不可用或输出无法解析时，降级为确定性规则……不得静默跳过反思」——
降级是节点的决定（详设 6.3 已有范式：`decision = assessment.decision if assessment else None`），
网关把它吞掉反而会让「模型在降级」这件事不可观测。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import Any, Final, Protocol

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.infrastructure.observability import span

logger = logging.getLogger(__name__)

#: 生成 JSON 格式示例时允许的最大嵌套深度。
#: 存在的理由是**递归模型会无限展开**（`Node.children: list[Node]`），
#: 没有上限的展开会把 prompt 撑爆。超过深度就返回空对象占位——
#: 示例只是给模型看的引导，不是校验依据（真正的校验是 Pydantic）。
_MAX_SCHEMA_DEPTH: Final[int] = 5

#: 可重试的 HTTP 状态码：408 请求超时、429 限流、5xx 服务端故障。
#: **401/403 刻意不在其中**——那是密钥或权限配错了，重试一万次也是同样的结果。
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})

#: JSON 输出契约。**必须包含字面量 "json" 并给出格式示例**——
#: 这不是我们的偏好，是 DeepSeek JSON 模式的服务端要求：官方文档明确
#: 「在 system 或 user prompt 中包含 json 一词」并「提供期望的 JSON 格式示例」，
#: 否则不保证返回合法 JSON。放在网关而不是各模板里：这是**服务方的协议要求**，
#: 换一个原生支持 structured output 的服务时只改这一处。
_JSON_CONTRACT: Final[str] = (
    "\n\n要求：只输出一个 json 对象，不要输出任何解释文字，"
    "也不要包裹 markdown 代码块。\n"
    "json 的字段名与类型必须与下面的示例完全一致"
    "（示例中的值是占位符，请按实际内容填写）：\n"
    "{example}\n"
)

#: 修复重试时追加在原文案后的提示。同样含 "json" 字样。
_REPAIR_HINT: Final[str] = (
    "\n\n上一次的输出无法被接受，原因：\n{reason}\n请重新输出一个完整的 json 对象。"
)


class EmbeddingDimensionError(ValueError):
    """服务返回的向量维度与 `EMBEDDING_DIM` 配置不一致。

    继承内建 `ValueError`：这是**配置与实现的错配**，不是运行期故障
    （与 `storage.py` 的 `InvalidObjectKeyError` 同类）。
    刻意不映射成 `AgentError`——它不该被降级路径吞掉，而要立刻炸给开发者看：
    维度对不上时向量库照样能建、能写入，却永远检索不到正确结果，
    属于「跑得通但结果是错的」那类故障。
    """


class TokenUsage(BaseModel):
    """一次调用的用量。落点见 `StructuredResult` 的说明。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StructuredResult[T: BaseModel](BaseModel):
    """结构化调用的完整结果。

    **为什么连用量一起返回，而不是只返回解析好的对象**：详细设计 19.4.2
    把「Token 与成本」归到工程追踪层，而 `agent_tool_call` 与
    `agent_trace_event` 两张表都**没有 token 列**（见
    `app/infrastructure/models/task.py` 与 `evidence.py`）。因此用量只能由
    调用方在埋点/写 Trace 时取用——网关不替它决定落到哪，但也绝不能把
    这个信息丢掉。

    Attributes:
        value: 通过 Pydantic 校验的对象。**这是唯一允许进入 LangGraph State 的形态**
            （开发流程 5.3：禁止把任意 dict 直接写入 State）。
        usage: token 用量。
        model: 实际调用的模型名。
        prompt_version: 模板版本号。与 `model` 必须同时出现才谈得上归因
            （详细设计 22.9：改动 prompt 或模型都必须记录版本）。
        duration_ms: 本次调用的墙钟耗时，**含全部重试**。
        attempts: 实际发出的 HTTP 请求次数。重试是否发生、发生几次靠它可观测——
            只记「成功/失败」看不出模型正在抖动。
    """

    value: T
    usage: TokenUsage
    model: str
    prompt_version: str
    duration_ms: int
    attempts: int


class PromptSource(Protocol):
    """网关对 prompt 的最小要求。

    只声明它真正用到的三样东西。`app/agent/prompts/base.py` 的
    `PromptTemplate` 结构上满足它，**但不显式继承**——这样 `agent` 侧
    不必 import `infrastructure`，依赖方向保持单向。

    `name` / `version` 声明成**只读属性**而不是普通变量：模板是
    `frozen=True` 的 dataclass（版本号不该在运行期被改），而 Protocol 里
    写成可写变量会要求实现也是可写的，于是 frozen dataclass 反而不满足它。
    只读属性的约束更弱、正好匹配。
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def render(self, **variables: Any) -> str: ...


class ModelGateway(Protocol):
    """模型出口契约。生产实现见 `HttpModelGateway`，测试替身见 `app/tests/fakes.py`。

    三个方法对应三类**形状完全不同**的服务：`invoke_structured` 走 OpenAI 兼容的
    chat（JSON 模式 + 两条重试预算），`embed` 走 OpenAI 兼容的 embeddings，
    `rerank` 走 Cohere 那套 `/rerank`（检索侧 cross-encoder 的事实标准）。
    它们共处一个 Protocol 是因为**调用方不该知道服务商是谁**——
    换服务商的成本必须落在这一个文件里。
    """

    async def invoke_structured[T: BaseModel](
        self, prompt: PromptSource, schema: type[T], /, **variables: Any
    ) -> StructuredResult[T]: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------- 失败信号
class _TransportError(Exception):
    """传输类失败的共同基类。不对外暴露，由 `invoke_structured` 转成 `AgentError`。"""

    def __init__(self, *, status_code: int | None, reason: str) -> None:
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason

    def to_agent_error(self, *, model: str, attempts: int) -> AgentError:
        """转成对外的 `AgentError`。

        **刻意不带响应体**：模型服务报错时常把请求内容回显在 body 里，
        而我们的请求头带着 `Authorization`——把 body 记进日志等于把密钥
        写进日志（开发流程 5.5 / 需求 8.3 的脱敏纪律）。
        要排查细节请看服务商自己的控制台，那里本来就有完整请求。
        """
        code = (
            ErrorCode.MODEL_RATE_LIMITED
            if self.status_code == 429
            else ErrorCode.UPSTREAM_UNAVAILABLE
        )
        return AgentError(
            code,
            "模型服务暂时不可用，请稍后重试",
            details={
                "model": model,
                "status_code": self.status_code,
                "attempts": attempts,
                "reason": self.reason,
            },
        )


class _TransientError(_TransportError):
    """可重试：429 / 5xx / 连接失败 / 超时。"""


class _NonRetryableError(_TransportError):
    """不可重试：401/403 这类「重试一万次也是同样结果」的错误。

    与 `_TransientError` 是**兄弟**而不是父子关系。做成父子会让调用点的
    `except _TransientError` 连它一起接住，于是拿一个配错的密钥反复重试——
    而这正是引入这个类型想要避免的事。分开成两个类型后，
    「可不可重试」由类型本身表达，调用点不可能漏判。
    """

    def to_agent_error(self, *, model: str, attempts: int) -> AgentError:
        # 错误码仍用 UPSTREAM_UNAVAILABLE（服务确实没能完成任务），
        # 但可重试标记要按 errors.py 允许的例外显式覆盖掉：
        # 让客户端重试一个配错的密钥没有意义。
        return AgentError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "模型服务调用失败，请联系管理员检查模型配置",
            details={"model": model, "status_code": self.status_code, "attempts": attempts},
            retryable=False,
        )


class _InvalidOutputError(Exception):
    """内部信号：HTTP 200 但内容不可用。走 `VALIDATION` 预算，不计入传输预算。"""


# ------------------------------------------------------------------------ 网关
class HttpModelGateway:
    """OpenAI 兼容端点上的模型网关（云 chat + 本地 embedding）。

    依赖注入的是 **transport 而不是 client**：测试用 `httpx.MockTransport`
    注入时，客户端仍由本类自己构造（超时、请求头、base_url 全部走生产路径），
    因此单测覆盖到的是真实代码路径，而不是一个被替换掉的壳。

    生命周期归调用方：自己建的连接池由 `aclose()` 释放，
    **绝不 close 别人传进来的东西**。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._model = settings.model_name
        timeout = float(settings.model_timeout_seconds)
        # httpx 的 timeout 覆盖 connect/read/write/pool 四个阶段，
        # **其中 pool 覆盖了「等待连接池空位」**——这正是 Redis 侧必须额外套
        # `asyncio.wait_for` 的原因（见 services/rate_limit.py 的说明），
        # 这里不需要再加一层。
        self._chat = httpx.AsyncClient(
            base_url=settings.model_base_url,
            headers={"Authorization": f"Bearer {settings.model_api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._embeddings = httpx.AsyncClient(
            base_url=settings.embedding_base_url or settings.model_base_url,
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
            timeout=timeout,
            transport=transport,
        )
        # 重排器：**没开就不建客户端**。建一个 base_url 指向聊天服务商的客户端
        # 不会有任何症状，直到开关被打开而 base_url 忘了填——那时报的是 404，
        # 排查方向会跑到"这个服务商不支持 rerank"上去。
        # 配置校验（`Settings` 的交叉校验）已经把"开了却没配齐"挡在启动时，
        # 这里的 `None` 是"没开"的正常态。
        self._reranker: httpx.AsyncClient | None = None
        if settings.reranker_enabled and settings.reranker_base_url:
            self._reranker = httpx.AsyncClient(
                base_url=settings.reranker_base_url,
                headers={"Authorization": f"Bearer {settings.reranker_api_key or ''}"},
                timeout=float(settings.reranker_timeout_seconds),
                transport=transport,
            )

    # -------------------------------------------------------------- 结构化调用
    async def invoke_structured[T: BaseModel](
        self, prompt: PromptSource, schema: type[T], /, **variables: Any
    ) -> StructuredResult[T]:
        started = time.monotonic()
        tuning = self._settings.model_tuning
        # 模板正文 + 服务方要求的 JSON 契约，一次拼好；修复重试时在它后面追加错因
        base_content = prompt.render(**variables) + build_json_contract(schema)
        content = base_content

        transport_left = tuning.max_transport_retries
        repair_left = tuning.max_repair_retries
        attempts = 0

        # 这个循环必然终止：每次 `continue` 恰好消耗一个预算，两个预算都有界
        # 且只减不增，故最多执行 `1 + max_transport_retries + max_repair_retries` 次。
        with span(
            "llm.chat",
            **{
                "llm.provider": self._settings.model_provider,
                "llm.model": self._model,
                "llm.prompt_name": prompt.name,
                "llm.prompt_version": prompt.version,
                "llm.schema": schema.__name__,
            },
        ) as current:
            while True:
                attempts += 1
                try:
                    body = await self._post_chat(content)
                    usage, raw = self._parse_chat_response(body)
                    value = schema.model_validate(raw)
                except _NonRetryableError as failure:
                    current.set_attribute("llm.outcome", "failed_transport")
                    current.set_attribute("llm.attempts", attempts)
                    raise failure.to_agent_error(model=self._model, attempts=attempts) from failure
                except _TransientError as failure:
                    if transport_left <= 0:
                        current.set_attribute("llm.outcome", "failed_transport")
                        current.set_attribute("llm.attempts", attempts)
                        raise failure.to_agent_error(
                            model=self._model, attempts=attempts
                        ) from failure
                    transport_left -= 1
                    # 指数退避：第 n 次重试等 base * 2**(n-1)。不用 tenacity 的
                    # 装饰器——要按类别分预算、还要能断言 attempts 计数。
                    await asyncio.sleep(tuning.backoff_base_seconds * 2 ** (attempts - 1))
                    continue
                except (ValidationError, _InvalidOutputError) as invalid:
                    reason = (
                        _summarize_validation_error(invalid)
                        if isinstance(invalid, ValidationError)
                        else str(invalid)
                    )
                    if repair_left <= 0:
                        current.set_attribute("llm.outcome", "invalid_output")
                        current.set_attribute("llm.attempts", attempts)
                        raise AgentError(
                            ErrorCode.MODEL_OUTPUT_INVALID,
                            "模型返回的内容不符合预期结构，请重试或换个问法",
                            details={
                                "model": self._model,
                                "prompt_version": prompt.version,
                                "schema": schema.__name__,
                                "attempts": attempts,
                                "reason": reason,
                            },
                        ) from invalid
                    repair_left -= 1
                    # 只回灌错因（字段名 + 错误类型），**不回灌模型自己吐出的原文**：
                    # 原文可能几百行，重发一次的代价比这次调用的全部成功调用还贵，
                    # 而定位错误并不需要它。
                    content = base_content + _REPAIR_HINT.format(reason=reason)
                    continue
                else:
                    duration_ms = int((time.monotonic() - started) * 1000)
                    current.set_attribute("llm.outcome", "ok")
                    current.set_attribute("llm.attempts", attempts)
                    current.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                    current.set_attribute("llm.completion_tokens", usage.completion_tokens)
                    current.set_attribute("llm.duration_ms", duration_ms)
                    return StructuredResult[T](
                        value=value,
                        usage=usage,
                        model=self._model,
                        prompt_version=prompt.version,
                        duration_ms=duration_ms,
                        attempts=attempts,
                    )

    # ------------------------------------------------------------------ 向量化
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            # 空输入直接返回，不打服务：多数端点对空 input 的行为未定义
            return []
        expected = self._settings.embedding_dim
        with span(
            "llm.embed",
            **{
                "llm.provider": self._settings.model_provider,
                "llm.model": self._settings.embedding_model,
                "llm.input_count": len(texts),
            },
        ) as current:
            try:
                response = await self._embeddings.post(
                    "/embeddings",
                    json={"model": self._settings.embedding_model, "input": list(texts)},
                )
            except httpx.TransportError as exc:
                raise AgentError(
                    ErrorCode.UPSTREAM_UNAVAILABLE,
                    "向量化服务暂时不可用，请稍后重试",
                    details={
                        "model": self._settings.embedding_model,
                        "reason": type(exc).__name__,
                    },
                ) from exc

            if response.status_code >= 400:
                # 同上：不记响应体，避免服务端回显把密钥带进日志
                raise AgentError(
                    ErrorCode.UPSTREAM_UNAVAILABLE,
                    "向量化服务返回错误，请稍后重试",
                    details={
                        "model": self._settings.embedding_model,
                        "status_code": response.status_code,
                    },
                )

            data = response.json().get("data") or []
            # **按 index 排序，不依赖返回顺序**：OpenAI 兼容端点不保证 `data`
            # 与输入同序，直接取用会让「第三段文本」拿到「第一段文本」的向量——
            # 检索结果全错，且不报任何错。
            vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
            current.set_attribute("llm.embedding_dim", expected)

        if vectors and len(vectors[0]) != expected:
            raise EmbeddingDimensionError(
                f"embedding 维度与配置不符：EMBEDDING_MODEL={self._settings.embedding_model!r} "
                f"返回 {len(vectors[0])} 维，而 EMBEDDING_DIM={expected}。"
                "向量库按 EMBEDDING_DIM 建表，不一致时能写入却永远检索不到，请对齐其中之一。"
            )
        return vectors

    # -------------------------------------------------------------------- 重排
    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """逐条给 `documents` 打相关性分，**返回与输入同序同长**的分数。

        契约取「同序同长」而不是"直接给出排好序的下标"：排序与剔除是**策略**
        （阈值、Top-K、并列怎么破），属于 `tools/rag/reranker.py`；
        网关只负责把模型说的话原样带回来。返回排好序的结果会把策略的一半
        塞进基础设施层，而那一半恰恰是评测要调的东西。

        **失败一律抛 `AgentError`，不在这里降级**：降级成"保留原顺序"是检索侧
        的决定（那条路径与"没开重排"共用同一段代码），网关把它吞掉会让
        「重排这次没生效」变成一件不可观测的事。

        **不重试**（与 `embed` 同形）。重排是增强步骤，失败的处置是退回 RRF 序
        ——那本来就是一个正确的结果；重试只会把一次可以忽略的故障拖长。
        9.4 的两条重试预算管的是 chat（那是任务的必经路径）。
        """
        if not documents:
            return []
        client = self._reranker
        if client is None:
            # 走到这里说明调用方没看 `RERANKER_ENABLED` 就调了重排。
            # **报错而不是静默返回**：静默返回一个"全都相关"的假分数，
            # 会让错误的排序看起来是重排器的判断结果。
            raise AgentError(
                ErrorCode.UPSTREAM_UNAVAILABLE,
                "重排服务未配置（RERANKER_ENABLED=false）",
                details={"model": self._settings.reranker_model},
            )

        with span(
            "llm.rerank",
            **{
                "llm.provider": self._settings.model_provider,
                "llm.model": self._settings.reranker_model,
                "llm.input_count": len(documents),
            },
        ) as current:
            try:
                response = await client.post(
                    "/rerank",
                    json={
                        "model": self._settings.reranker_model,
                        "query": query,
                        "documents": list(documents),
                        # **要全部候选的分数**：`top_n` 不传时部分服务端只回前若干条，
                        # 而少掉的那些会被读成"没有分"——那与我们没算过它长得一样。
                        "top_n": len(documents),
                    },
                )
            except httpx.TransportError as exc:
                raise AgentError(
                    ErrorCode.UPSTREAM_UNAVAILABLE,
                    "重排服务暂时不可用，请稍后重试",
                    details={
                        "model": self._settings.reranker_model,
                        "reason": type(exc).__name__,
                    },
                ) from exc

            if response.status_code >= 400:
                # 同 `embed`：不记响应体，避免服务端回显把 Authorization 带进日志
                raise AgentError(
                    ErrorCode.UPSTREAM_UNAVAILABLE,
                    "重排服务返回错误，请稍后重试",
                    details={
                        "model": self._settings.reranker_model,
                        "status_code": response.status_code,
                    },
                )

            results = response.json().get("results")
            scores = _scores_by_index(results, expected=len(documents))
            current.set_attribute("llm.outcome", "ok")
        return scores

    async def aclose(self) -> None:
        await self._chat.aclose()
        await self._embeddings.aclose()
        if self._reranker is not None:
            await self._reranker.aclose()

    # ---------------------------------------------------------------- 内部实现
    async def _post_chat(self, content: str) -> dict[str, Any]:
        """发一次对话请求，把传输类失败分类抛成上面两个信号。

        只在这里做**一次**网络调用——重试的编排在上层，
        这样「重试了几次」与「每次为什么失败」是分开可测的。
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "max_tokens": self._settings.model_tuning.max_tokens,
        }
        if not self._settings.model_tuning.thinking_enabled:
            # DeepSeek V4 **服务端默认开启思考模式**（effort=high），必须显式关掉。
            # 开着不会报错，只会更贵更慢、且 `temperature` 静默失效——
            # 属于「默认值错了但看不出来」的那类坑。
            payload["thinking"] = {"type": "disabled"}

        try:
            response = await self._chat.post("/chat/completions", json=payload)
        except httpx.TransportError as exc:
            # 连接失败、读超时、协议错误都落这里
            raise _TransientError(status_code=None, reason=type(exc).__name__) from exc

        if response.status_code >= 400:
            reason = f"HTTP {response.status_code}"
            failure_type = (
                _TransientError if response.status_code in _RETRYABLE_STATUS else _NonRetryableError
            )
            raise failure_type(status_code=response.status_code, reason=reason)

        try:
            body = response.json()
        except ValueError as exc:
            # 200 但 body 不是 JSON：网关前面挂了代理时偶发，按瞬时故障重试
            raise _TransientError(status_code=None, reason="invalid_json_body") from exc
        if not isinstance(body, dict):
            raise _TransientError(status_code=None, reason="unexpected_body_shape")
        return body

    def _parse_chat_response(self, body: dict[str, Any]) -> tuple[TokenUsage, Any]:
        """取用量与解析后的 JSON 内容。内容不可用时报 `_InvalidOutputError`。

        内容不可用**不算传输失败**：HTTP 是 200、服务是好的，问题在这一次的
        输出上，按详设 9.4 应走 `VALIDATION` 预算。DeepSeek 官方文档明确提到
        JSON 模式**可能返回空内容**——这正是这条路径的现实来源，
        也是「解析失败要能重试」不是杞人忧天的原因。
        """
        usage = TokenUsage.model_validate(body.get("usage") or {})
        choices = body.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        content = (message or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise _InvalidOutputError("模型返回了空内容")
        try:
            return usage, json.loads(content)
        except json.JSONDecodeError as exc:
            raise _InvalidOutputError(f"模型返回的不是合法 JSON：{exc.msg}") from exc


# ------------------------------------------------------------------ 契约构造
def _scores_by_index(results: Any, *, expected: int) -> list[float]:
    """把 `/rerank` 的 `results` 摊成与输入同序的分数列表。

    **按 `index` 归位，不依赖返回顺序**（同 `embed` 的理由）：服务端把结果
    按分数降序返回，直接取用会让「第 1 条候选」拿到「最相关那条」的分数——
    分数全对、配对全错，而且不报任何错。

    **少一条就报错，不补 0**：补 0 等于断言"这条与问题无关"，那是个我们
    并不知道的结论（它压根没被评分）。缺条只能说明服务端没按 `top_n` 返回，
    处置是整次降级回 RRF 序——那由调用方决定，这里只把事实说清楚。
    """
    if not isinstance(results, list):
        raise AgentError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            "重排服务的返回里没有 results 字段",
            details={"expected": expected},
        )
    scores: list[float | None] = [None] * expected
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not isinstance(score, (int, float)):
            continue
        if 0 <= index < expected:
            scores[index] = float(score)
    missing = [index for index, score in enumerate(scores) if score is None]
    if missing:
        raise AgentError(
            ErrorCode.UPSTREAM_UNAVAILABLE,
            f"重排服务返回的候选数不足：应有 {expected} 条，缺 {len(missing)} 条",
            details={"expected": expected, "returned": len(results)},
        )
    return [float(score) for score in scores if score is not None]


def build_json_contract(schema: type[BaseModel]) -> str:
    """由 Schema 生成 JSON 输出契约（含格式示例）。

    示例由 `model_json_schema()` 反向构造而不是手写：手写的示例一旦与 Schema
    漂移，模型就会照着错的示例输出、然后被 Pydantic 拒绝——表现为
    「模型不听话」，实际是我们给错了引导。
    """
    raw = schema.model_json_schema()
    example = _example_from_schema(raw, raw.get("$defs", {}))
    return _JSON_CONTRACT.format(example=json.dumps(example, ensure_ascii=False, indent=2))


def _example_from_schema(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    """从 JSON Schema 递归构造一个「形状正确」的示例值。

    值本身没有意义（`0` / `""` / `false` 都是占位符），有意义的是
    **字段名与嵌套结构**——模型照着它填，结构就不会跑偏。
    """
    if depth > _MAX_SCHEMA_DEPTH or not isinstance(node, dict):
        return {}

    if ref := node.get("$ref"):
        # Pydantic 把嵌套模型放进 `$defs`，用 `#/$defs/Name` 引用
        return _example_from_schema(defs.get(str(ref).rsplit("/", 1)[-1], {}), defs, depth + 1)

    if node.get("enum"):
        # Literal / Enum：取第一个取值，模型据此知道合法取值长什么样
        return node["enum"][0]

    # `Optional[X]` 在 Pydantic 里生成 `anyOf: [X, {type: null}]`。
    #
    # **可空字段的示例给 `null`，不给 X 构造出来的具体值。** 这条规则是按
    # 「模型照抄示例的后果」定的，不是按"示例要有内容"定的：
    #
    # - **照抄 `null` 得到正确语义**（这个字段可以没有），
    # - 照抄一个具体值得到的是**看起来合法、实际错误**的值。
    #
    # 实测（2026-09-20）：`time_range` 的示例原本是 `{"start": "", "end": ""}`，
    # 模型把"这个问题没有时间"原样写成那个空串 → 校验失败（整条任务挂）。
    # 把示例改成合法日期之后，**它开始照抄那个日期** → 意图里凭空多出
    # "2025-01-01 到 2025-01-01"，会真的去过滤文档生效期、把 v2.0 滤掉，
    # 而且**没有任何报错**。间歇发生，比稳定报错危险得多。
    for key in ("anyOf", "oneOf"):
        if key in node:
            if any(option.get("type") == "null" for option in node[key]):
                return None
            return _example_from_schema(node[key][0], defs, depth + 1)

    node_type = node.get("type")
    if node_type == "string":
        # **示例必须是"这个字段合法取值"，否则模型会照着它填出一个非法值。**
        # 实测踩到（2026-09-20）：`date` 字段的示例原来是空串 `""`，
        # 于是模型把「没有时间」原样写成 `{"start": "", "end": ""}`，
        # Pydantic 判非法 → `MODEL_OUTPUT_INVALID` → 整条任务失败。
        # 症状具有极强的误导性：**只有不含明确年份的问题会挂**
        # （带"2025年Q3"的问题模型会填真日期，看起来一切正常）。
        if node.get("format") == "date":
            return "2025-01-01"
        if node.get("format") == "date-time":
            return "2025-01-01T00:00:00Z"
        # 同理：声明了非空下限的字段给空串也是**注定非法**
        return "示例" if int(node.get("minLength") or 0) > 0 else ""
    if node_type == "object" or "properties" in node:
        return {
            key: _example_from_schema(value, defs, depth + 1)
            for key, value in node.get("properties", {}).items()
        }
    if node_type == "array":
        return [_example_from_schema(node.get("items", {}), defs, depth + 1)]
    if node_type == "integer":
        return 0
    if node_type == "number":
        return 0.0
    if node_type == "boolean":
        return False
    return None


def _summarize_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 报错压成几行「字段 + 原因」，用于回灌给模型。

    只用字段路径与错误描述，**不带输入值**——`errors()` 默认会带上 `input`，
    那里面就是模型自己的输出，既长又可能含业务数据。
    """
    items = exc.errors(include_url=False, include_input=False)
    lines = [f"- {'.'.join(str(p) for p in item['loc'])}: {item['msg']}" for item in items]
    return "\n".join(lines[:10])


def build_model_gateway(settings: Settings) -> ModelGateway:
    """装配点用的工厂。命名沿用 `build_memory_repositories` / `build_job_queue` 的惯例。"""
    return HttpModelGateway(settings)


__all__ = [
    "EmbeddingDimensionError",
    "HttpModelGateway",
    "ModelGateway",
    "PromptSource",
    "StructuredResult",
    "TokenUsage",
    "build_json_contract",
    "build_model_gateway",
]
