"""测试替身。放在 `app/tests/` 而不是生产模块里。

`FakeRateLimiter` 曾经写在 `services/rate_limit.py`，后来删掉了——
限流现在由 fakeredis 跑真实的 Lua 脚本，替身反而让测试覆盖不到真实路径。
**能跑真实实现时就别写替身**：替身只会验证「调用方按我设想的方式调用了」，
真实实现才验证「它真的做对了」。

队列是例外：arq 的 `ArqRedis` 需要真实连接才能 `enqueue_job`，
在单元测试里跑不起（integration 用例可以）。而投递行为本身需要被断言
（「创建任务后有没有投递」「投递失败时接口是否仍然 202」），因此保留替身。

模型是第二个例外，理由见 `FakeModelGateway` 的 docstring。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.infrastructure.model_gateway import (
    ModelGateway,
    PromptSource,
    StructuredResult,
    TokenUsage,
)
from app.infrastructure.queue import JobQueue
from app.repositories import Repositories
from app.repositories.conversation_repo import InMemoryConversationRepository
from app.repositories.knowledge_repo import InMemoryKnowledgeDocumentRepository
from app.repositories.task_repo import InMemoryTaskRepository
from app.repositories.user_repo import InMemoryUserRepository, seed_demo_users
from app.repositories.vocab_repo import InMemoryVocabRepository
from app.tools.sql.schemas import SqlExecutionResult, ValidatedSql


@dataclass
class EnqueuedJob:
    task_id: str
    trace_id: str
    trace_context: dict[str, str]


@dataclass
class FakeJobQueue:
    """记录投递，并可按需模拟投递失败。

    `fail_with` 模拟 Redis 不可达：`ArqJobQueue.enqueue` 在那种情况下
    返回 False 而不是抛异常，替身必须复现这个契约，
    否则「投递失败时接口仍返回 202」的用例测的是替身的行为，不是代码的。
    """

    jobs: list[EnqueuedJob] = field(default_factory=list)
    #: 置 True 时模拟队列不可用
    fail_with: bool = False
    #: 已经「在队列里」的 job id。补偿扫描靠 `is_pending` 区分排队慢与真丢了，
    #: 用例通过直接改这个集合来构造两种场景。
    present: set[str] = field(default_factory=set)
    closed: bool = False

    async def enqueue(self, *, task_id: str, trace_id: str, trace_context: dict[str, str]) -> bool:
        if self.fail_with:
            return False
        if task_id in self.present:
            # arq 在 job_id 已存在时返回 None，即「没有真的入队」
            return False
        self.present.add(task_id)
        self.jobs.append(
            EnqueuedJob(task_id=task_id, trace_id=trace_id, trace_context=trace_context)
        )
        return True

    async def is_pending(self, task_id: str) -> bool:
        return task_id in self.present

    async def aclose(self) -> None:
        self.closed = True


def build_job_queue() -> JobQueue:
    """类型标注成协议，让用例里的替身与生产实现受同一份契约约束。"""
    return FakeJobQueue()


class _FakeSession:
    """就绪探针只用到 `execute`。"""

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        return None


class FakeSessionFactory:
    """`app.state.sessions` 的替身，供不连 MySQL 的单元测试使用。

    就绪探针会拿它执行一次 `SELECT 1` 来判断 MySQL 是否可用。单元测试没有
    真实库，这里给出一个「执行成功」的替身；**探针本身的行为**（含 503 分支）
    由 `make test-integration` 下的用例对着真实 MySQL 验证——
    替身只保证「有这个对象、调用形状对」，不保证探针真的能判活。
    """

    @asynccontextmanager
    async def __call__(self) -> AsyncIterator[_FakeSession]:
        yield _FakeSession()


class FakeReply(StrEnum):
    """脚本里的特殊条目：这一次调用不返回内容，而是模拟「输出不可用」。

    对应真实网关的 `MODEL_OUTPUT_INVALID` —— 例如模型返回空 content
    （DeepSeek 的 JSON 模式官方承认会偶发）或吐出半截 JSON。
    详设 22.10.1 的「模型输出非法 decision」终止性用例靠它构造。
    """

    INVALID_OUTPUT = "invalid_output"


class FakeFailure(StrEnum):
    """整体失败开关：进入这个状态后**每次**调用都抛错。

    对应「模型持续不可用」——详设 22.10.1 要求验证节点在这种情况下
    会走确定性降级而不是把任务挂死。

    取值刻意与错误码同名一一对应，且 `AgentError` 的 `retryable` 取自
    `DEFAULT_RETRYABLE`（不覆盖），因此替身抛出的错误与真实网关**同源**。
    """

    RATE_LIMITED = "MODEL_RATE_LIMITED"
    UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    INVALID_OUTPUT = "MODEL_OUTPUT_INVALID"


@dataclass
class FakeModelCall:
    """一次调用的记录，供「有没有调用过模型」「用哪个模板、传了什么变量」这类断言。"""

    prompt_name: str
    prompt_version: str
    schema: str
    variables: dict[str, Any]
    rendered: str


@dataclass
class FakeModelGateway:
    """按脚本返回固定响应的模型替身（开发流程 6.5 施工项 2）。

    **为什么这个替身是必要的**（本文件开头的头号纪律是「能跑真实实现时就别写替身」）：
    真实网关要打外部 HTTP 服务，单元测试既不能依赖网络、也不能依赖某一次
    模型输出的随机性。注意 `HttpModelGateway` 自身的 HTTP 路径**并未因此
    失去覆盖**——契约测试用 `httpx.MockTransport` 让它在无网络下跑完整实现，
    两个实现受同一批断言约束。

    必需的第二层理由：详设 22.10.1 的终止性用例要求「Fake Model 固定返回
    EXPAND」「反复提出目标相同的步骤」「返回未知枚举或坏 JSON」——
    这些断言的对象是**调用方的行为**，必须能精确控制返回值，真模型做不到。

    Attributes:
        responses: 按序弹出的脚本。每项可以是 schema 实例、dict（校验后返回）
            或 JSON 字符串。**脚本用尽后重复最后一项**——这正是
            「固定返回 EXPAND」的写法：只放一项即可。
        failure: 置位后每次调用都抛对应的 `AgentError`，用于验证调用方的降级路径。
        calls: 调用记录（顺序即调用顺序）。
        embedding_dim: `embed` 返回的向量维度，同时也是维度不匹配用例的开关。
        embeddings: 显式指定的向量表；为空时按文本长度生成**确定性**占位向量。
        embed_calls: 每次 `embed` 收到的文本批次。
        closed: 是否已被 `aclose()` 关闭。
    """

    responses: list[Any] = field(default_factory=list)
    failure: FakeFailure | None = None
    calls: list[FakeModelCall] = field(default_factory=list)
    embedding_dim: int = 8
    embeddings: list[list[float]] = field(default_factory=list)
    embed_calls: list[list[str]] = field(default_factory=list)
    closed: bool = False
    _cursor: int = 0

    async def invoke_structured[T: BaseModel](
        self, prompt: PromptSource, schema: type[T], /, **variables: Any
    ) -> StructuredResult[T]:
        # 先渲染再记录：模板变量写错时**必须在调用前炸**，与真实网关一致。
        # 替身如果跳过这一步，「节点传错变量名」这类缺陷就只有在接真模型时
        # 才暴露——而那正是最不该出问题的时候。
        rendered = prompt.render(**variables)
        self.calls.append(
            FakeModelCall(
                prompt_name=prompt.name,
                prompt_version=prompt.version,
                schema=schema.__name__,
                variables=dict(variables),
                rendered=rendered,
            )
        )
        if self.failure is not None:
            raise _failure_error(self.failure)

        reply = self._next_reply()
        if reply is FakeReply.INVALID_OUTPUT:
            raise _failure_error(FakeFailure.INVALID_OUTPUT)

        value = (
            reply
            if isinstance(reply, schema)
            else schema.model_validate_json(reply)
            if isinstance(reply, str)
            else schema.model_validate(reply)
        )
        # attempts 恒为 1：替身不做重试。重试编排是 `HttpModelGateway` 的职责，
        # 由契约测试用 MockTransport 覆盖——替身假装重试过反而会掩盖真实缺陷。
        return StructuredResult[T](
            value=value,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="fake-model",
            prompt_version=prompt.version,
            duration_ms=0,
            attempts=1,
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        # **`failure` 对 `embed` 同样生效**：它的类注释写的是"每次调用都抛错"，
        # 而 Phase 5 之前没有任何用例用到向量化，这条分支就一直没被走。
        # 少了它，"向量化服务不可用"的降级用例会**静默通过**——
        # 替身老老实实返回占位向量，用例断言的事情一件都没发生。
        if self.failure is not None:
            raise _failure_error(self.failure)
        if self.embeddings:
            return list(self.embeddings)
        # 确定性占位向量：**不承载语义**，只保证形状与可复现。
        # 需要真实相似度的用例（Phase 5 的 RRF）应显式传 `embeddings`。
        return [[float(len(text))] + [0.0] * (self.embedding_dim - 1) for text in texts]

    async def aclose(self) -> None:
        self.closed = True

    def _next_reply(self) -> Any:
        if not self.responses:
            raise AssertionError(
                "FakeModelGateway 没有脚本可返回：请在用例里显式设置 responses。"
                "刻意不提供默认响应——否则「忘了配脚本」会表现为一个看似通过的用例。"
            )
        reply = self.responses[min(self._cursor, len(self.responses) - 1)]
        self._cursor += 1
        return reply


def _failure_error(failure: FakeFailure) -> AgentError:
    """构造与真实网关**同源**的 `AgentError`（错误码取自枚举值，不覆盖 retryable）。"""
    code = ErrorCode(failure.value)
    messages = {
        ErrorCode.MODEL_RATE_LIMITED: "模型服务暂时不可用，请稍后重试",
        ErrorCode.UPSTREAM_UNAVAILABLE: "模型服务暂时不可用，请稍后重试",
        ErrorCode.MODEL_OUTPUT_INVALID: "模型返回的内容不符合预期结构，请重试或换个问法",
    }
    return AgentError(code, messages[code], details={"model": "fake-model"})


def build_model_gateway() -> ModelGateway:
    """类型标注成协议，让用例里的替身与生产实现受同一份契约约束。"""
    return FakeModelGateway()


@dataclass
class FakeSqlRunner:
    """`SqlRunner` 的替身：按脚本返回执行结果或抛错。

    **为什么执行器要替身而校验器不要**（本文件开头的头号纪律是「能跑真实实现时
    就别写替身」）：`SqlValidator` 是纯函数，跑真实实现的成本几乎为零，
    替身它只会让「校验规则写错了」这类缺陷逃过测试。而执行器连的是真实 MySQL，
    单元测试既不能依赖它，也无法用它构造「第 3 次修复时返回除零错误」这类时序。

    真实 `SqlExecutor` 的 SQL 路径并未因此失去覆盖：`make test-integration`
    下的用例对着真实业务库跑同一条链路，两条路径各有各的验证。

    Attributes:
        results: 按序弹出的脚本。每项是 `SqlExecutionResult` 或 `Exception` 实例；
            **用尽后重复最后一项**，与 `FakeModelGateway` 同一约定。
        calls: 收到的 `ValidatedSql`，用于断言「执行的是重写后的 SQL 而不是原文」。
        closed: 是否已被 `aclose()`。
    """

    results: list[Any] = field(default_factory=list)
    calls: list[ValidatedSql] = field(default_factory=list)
    closed: bool = False
    _cursor: int = 0

    async def execute(self, validated: ValidatedSql) -> SqlExecutionResult:
        self.calls.append(validated)
        if not self.results:
            raise AssertionError(
                "FakeSqlRunner 没有脚本可返回：请在用例里显式设置 results。"
                "刻意不提供默认成功——否则「忘了配脚本」会表现为一个看似通过的用例。"
            )
        reply = self.results[min(self._cursor, len(self.results) - 1)]
        self._cursor += 1
        if isinstance(reply, Exception):
            raise reply
        assert isinstance(reply, SqlExecutionResult)  # 替身脚本写错了要立刻炸
        return reply

    async def aclose(self) -> None:
        self.closed = True


def build_memory_repositories(settings: Settings) -> Repositories:
    """四个仓储的内存实现（Pydantic 版）。

    与 MySQL 实现受同一份 `Protocol` 约束——仓储契约测试
    （`app/tests/repositories/`）会把同一批断言同时跑在两个实现上，
    因此「内存实现和 SQL 实现行为不一致」不会拖到线上才暴露。
    """
    return Repositories(
        users=InMemoryUserRepository(seed_demo_users(settings)),
        conversations=InMemoryConversationRepository(),
        tasks=InMemoryTaskRepository(),
        vocab=InMemoryVocabRepository(),
        documents=InMemoryKnowledgeDocumentRepository(),
    )
