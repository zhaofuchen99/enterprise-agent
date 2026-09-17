"""`rag_retrieve` 工具（详细设计 9.1 / 9.2 的 Tool 契约，11.7 的在线检索）。

## 空结果：与 SQL 侧**故意不一样**

SQL 的 0 行返回 `SUCCEEDED`（`tools/sql/tool.py` 的 `_success_result`），
理由写得很清楚：「SQL 正确执行了，只是没有匹配的行」——那是一条关于数据的
事实，分析节点可以照实说"没查到"。

RAG 的"没有相关知识"**不能照此处理**，它是 `FAILED` + `NO_RELEVANT_KNOWLEDGE`：

- 11.8 第 1 条原文是「检索为空时显式返回 `NO_RELEVANT_KNOWLEDGE`」，
  19.1 的错误码表里这个码就是为它准备的（422 / 不可重试 / 降级、补证或澄清）。
  标成 `SUCCEEDED` 的话，`payload.chunks` 是空列表，下游只看得到"成功、零条"——
  那与"工具没跑到"长得一模一样；
- 更实际的理由：把"语料里没有这条制度"当成一次成功检索，
  生成节点就有机会把空结果补写成"公司暂无相关规定"。**那正是 11.8 要防的幻觉**，
  而一个非零的错误码是这条纪律在接口上的落点。

`error_class` 取 `EMPTY_RESULT`（9.4 定义的类别），Reviewer 由此仍然可以
按 9.4 的分工决定补证、澄清还是受限回答——**FAILED 不等于终止**，
它只是让"没有证据"这件事无法被误读成"有一条空证据"。

## 与 SQL Tool 一致的三处

- `ctx.permission_scope` 是**唯一**的数据权限来源，工具不读用户仓储；
- 工具不自行决定重试，只返回错误类别（9.3）；
- 结果先进 Pydantic 模型再进 `payload`（开发流程 5.3）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence
from app.infrastructure.cache import VersionedCache
from app.infrastructure.model_gateway import ModelGateway
from app.infrastructure.observability import span
from app.infrastructure.storage import ObjectStorage
from app.infrastructure.vector_store import VectorStore
from app.repositories.vocab_repo import VocabRepository
from app.tools.base import ToolContext, ToolError, ToolName, ToolResult
from app.tools.rag.evidence import build_document_evidence
from app.tools.rag.retriever import Retriever
from app.tools.rag.schemas import (
    TOOL_NAME,
    RagQueryArgs,
    RagToolResult,
    RetrievalOutcome,
)
from app.tools.rag.tokenizer import Tokenizer
from app.tools.rag.vocabulary import load_vocabulary


class RagRetrieveTool:
    """企业知识库检索（11.7）。

    依赖由构造函数注入，装配见 `build_rag_retrieve_tool`。
    """

    def __init__(self, *, settings: Settings, retriever: Retriever) -> None:
        self.name: ToolName = TOOL_NAME
        self._settings = settings
        self._retriever = retriever

    async def execute(self, args: RagQueryArgs, ctx: ToolContext) -> ToolResult:
        """详设 9.1 的 `BaseTool.execute`。

        `ctx.permission_scope` 是**唯一**的数据权限来源——同 SQL Tool，
        工具不读用户仓储、也不接受调用方另传一份范围。
        """
        started_at = datetime.now(UTC)
        call_id = new_id(IdPrefix.TOOL_CALL)

        with span(
            "tool.rag_retrieve",
            **{
                "tool.name": TOOL_NAME,
                "tool.call_id": call_id,
                "task.id": ctx.task_id,
                "rag.as_of": args.as_of.isoformat() if args.as_of else None,
                "rag.filtered": bool(args.document_types or args.departments),
            },
        ) as current:
            try:
                outcome = await self._retriever.retrieve(args, scope=ctx.permission_scope)
            except AgentError as exc:
                return _failure_result(call_id, started_at, exc)

            current.set_attribute("rag.candidates", outcome.candidate_count)
            current.set_attribute("rag.best_dense_score", round(outcome.best_dense_score, 4))
            current.set_attribute("rag.rewrite_degraded", outcome.rewrite_degraded)

            if outcome.no_relevant_knowledge:
                current.set_attribute("rag.no_relevant_knowledge", True)
                return _no_knowledge_result(call_id, started_at, outcome)

            evidence = build_document_evidence(outcome.candidates, question=args.question)
            current.set_attribute("rag.evidence", len(evidence))
            return _success_result(call_id, started_at, outcome, evidence)

    async def aclose(self) -> None:
        """空操作。

        `RagRetrieveTool` 自己持有零个需要关闭的资源——向量库与网关是
        长生命周期的共享资源（Phase 6/7 的其它节点也要用），由装配点管理。
        留着这个方法是为了与 `SqlQueryTool.aclose` 同形状，
        免得调用方要记住"哪个工具有 aclose"。
        """
        return None


def _result_payload(outcome: RetrievalOutcome, call_id: str) -> RagToolResult:
    return RagToolResult(
        call_id=call_id,
        queries=outcome.queries,
        rewrite_degraded=outcome.rewrite_degraded,
        candidate_count=outcome.candidate_count,
        chunks=outcome.candidates,
        relevance_threshold=outcome.relevance_threshold,
        best_dense_score=outcome.best_dense_score,
        unseen_topics=outcome.unseen_topics,
        duration_ms=outcome.duration_ms,
        warnings=_warnings(outcome),
    )


def _warnings(outcome: RetrievalOutcome) -> tuple[str, ...]:
    """降级信号。**改变过行为的事必须能被调用方看到**。

    改写降级尤其需要：检索结果变差时，第一件要排除的就是
    "这次用的是原问题还是改写后的查询"，而它在结果里看不出来。
    """
    if not outcome.rewrite_degraded:
        return ()
    return ("查询改写未生效（模型不可用或输出不合规），已退化为原问题检索",)


def _success_result(
    call_id: str,
    started_at: datetime,
    outcome: RetrievalOutcome,
    evidence: Sequence[Evidence],
) -> ToolResult:
    payload = _result_payload(outcome, call_id)
    return ToolResult(
        call_id=call_id,
        tool=TOOL_NAME,
        status="SUCCEEDED",
        started_at=started_at,
        finished_at=datetime.now(UTC),
        summary=f"命中 {len(evidence)} 条文档证据（候选 {payload.candidate_count} 条）",
        payload=payload.as_payload(),
        evidence=list(evidence),
    )


def _no_knowledge_result(
    call_id: str, started_at: datetime, outcome: RetrievalOutcome
) -> ToolResult:
    """没有相关知识（11.8 第 1 条）。

    `evidence` 是**空列表**——`payload` 里也不带候选。见模块 docstring：
    这条纪律靠的是"没有东西可写"，不是靠调用方自觉。
    """
    payload = _result_payload(outcome, call_id)
    return ToolResult(
        call_id=call_id,
        tool=TOOL_NAME,
        status="FAILED",
        started_at=started_at,
        finished_at=datetime.now(UTC),
        summary=(
            f"知识库中没有与问题相关的制度或报告（最高相似度 {payload.best_dense_score:.4f}）"
        ),
        payload=payload.as_payload(),
        error=ToolError(
            code=ErrorCode.NO_RELEVANT_KNOWLEDGE.value,
            message="知识库中没有可支撑该问题的内容",
            error_class="EMPTY_RESULT",
            # 不可重试：换个说法再查一次得到的还是同一批候选，
            # 该做的是澄清或让 Reviewer 去补证据源（9.4 的 EMPTY_RESULT 一行）
            retryable=False,
            # `safe_detail` 面向排查且必须已脱敏（9.2）。**两个判据都要报**：
            # 它们的处置完全不同——"语料没见过这个词"要去确认是不是问错了，
            # "余弦太低"要去查阈值是不是调高了
            safe_detail=(
                f"判据一｜主题词未登录："
                f"{'、'.join(payload.unseen_topics) if payload.unseen_topics else '无'}；"
                f"判据二｜最高余弦 {payload.best_dense_score:.4f} "
                f"(阈值 {payload.relevance_threshold})；"
                f"实际使用的查询：{'|'.join(payload.queries)}"
            ),
        ),
    )


def _failure_result(call_id: str, started_at: datetime, exc: AgentError) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        tool=TOOL_NAME,
        status="FAILED",
        started_at=started_at,
        finished_at=datetime.now(UTC),
        summary=exc.message,
        error=ToolError(
            code=exc.code.value,
            message=exc.message,
            error_class="INTERNAL" if exc.code is ErrorCode.INTERNAL_ERROR else "VALIDATION",
            # 重试与否由 Graph 路由按错误码决定（9.3），这里只传递事实
            retryable=exc.retryable,
            safe_detail=str(exc.details) if exc.details else None,
        ),
    )


async def build_retriever(
    settings: Settings,
    gateway: ModelGateway,
    *,
    vector_store: VectorStore,
    storage: ObjectStorage,
    vocab: VocabRepository,
    cache: VersionedCache,
) -> Retriever:
    """装配 `Retriever`（**与 Tool 分开**）。

    分开的理由是**有两个消费方**：`rag_retrieve` 工具，以及
    `make verify-corpus`——后者要直接跑检索来断言"失效版本不被召回"与
    "无答案提问被拒答"，而它不需要 `ToolResult` 那层封装（证据、call_id、
    错误码都不是它要的）。让 `verify` 去 `tool._retriever` 里掏，
    就是让一个命令依赖另一个类的私有属性。

    **词表必须从快照装载**（`load_vocabulary`），不能按需去 MySQL 现算：
    查询侧与入库侧的 IDF 必须来自同一份冻结快照，否则同一个词在两侧的
    稀疏权重不同，点积就不再可比（11.6.4）。快照缺失时它报错提示重跑
    `make vocab`，**不降级**。
    """
    vocabulary = await load_vocabulary(
        vocab,
        cache,
        storage,
        ttl_seconds=settings.rag.vocab_cache_ttl_seconds,
    )
    return Retriever(
        settings=settings,
        gateway=gateway,
        vector_store=vector_store,
        tokenizer=Tokenizer.from_settings(settings),
        vocabulary=vocabulary,
    )


async def build_rag_retrieve_tool(
    settings: Settings,
    gateway: ModelGateway,
    *,
    vector_store: VectorStore,
    storage: ObjectStorage,
    vocab: VocabRepository,
    cache: VersionedCache,
) -> RagRetrieveTool:
    """装配点。参数与 `build_retriever` 相同，见那里的说明。"""
    retriever = await build_retriever(
        settings,
        gateway,
        vector_store=vector_store,
        storage=storage,
        vocab=vocab,
        cache=cache,
    )
    return RagRetrieveTool(settings=settings, retriever=retriever)


__all__ = ["RagRetrieveTool", "build_rag_retrieve_tool", "build_retriever"]
