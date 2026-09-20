"""在线检索与相关证据（详细设计 11.7 / 11.8 / 13.1）。

**用真实的 `InMemoryVectorStore`**（它真算余弦、点积与 RRF，不是打桩），
只有模型网关是替身——它要连外部服务。这样断言的是检索**行为**
（召回谁、拦下谁），而不是"调用了几次"。

向量只取 4 维：这条链路的正确性与维度无关，而小维度让"哪条该被召回"
一眼能看出来——1024 维的随机向量之间没有可读的关系。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.errors import AgentError, ErrorCode
from app.domain.knowledge import DocumentStatus, SourceKind
from app.domain.user import PermissionScope, UserRole
from app.infrastructure.model_gateway import PromptSource, StructuredResult
from app.infrastructure.vector_store import InMemoryVectorStore, VectorPoint
from app.tests.fakes import FakeFailure, FakeModelGateway
from app.tools.base import ToolContext
from app.tools.rag.evidence import build_document_evidence
from app.tools.rag.metadata import ChunkMetadata
from app.tools.rag.reranker import SKIP_DISABLED, SKIP_INCOMPLETE, SKIP_UNAVAILABLE, Reranker
from app.tools.rag.retriever import Retriever, _keeps_numbers
from app.tools.rag.schemas import QueryRewrite, RagQueryArgs
from app.tools.rag.tokenizer import Tokenizer, Vocabulary, build_sparse
from app.tools.rag.tool import RagRetrieveTool

DIM = 4

#: 语料里"见过"的词。**未登录词判据的全部依据就是它**——
#: 词表里没有的词 = 语料从未出现过。因此这里的词表要覆盖测试问题里的主题词。
#: `规定` / `管理` 也放进来：它们是通用动词，测试问题里到处都在用，
#: 留着未登录会让"未登录词判据"的用例断言不出它真正要断的东西。
_KNOWN = (
    "华东",
    "归口",
    "区域",
    "渠道折扣",
    "政策",
    "退货",
    "金额",
    "口径",
    "净销售额",
    "规定",
    "管理",
)


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary(
        {token: index for index, token in enumerate(_KNOWN)},
        document_count=100,
        df=dict.fromkeys(_KNOWN, 10),
    )


def _metadata(
    index: int,
    *,
    title: str = "华东区域渠道折扣政策",
    logical_key: str = "policy/east-china-channel-discount",
    version: str = "v2.0",
    section: tuple[str, ...] = ("华东区域渠道折扣政策", "第二章 折扣标准"),
    classification: str = "INTERNAL",
    source_kind: SourceKind = SourceKind.INTERNAL,
    document_type: str = "POLICY",
    effective_from: date | None = None,
    effective_to: date | None = None,
    status: DocumentStatus = DocumentStatus.ACTIVE,
    is_table: bool = False,
    table_caption: str | None = None,
) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=f"chk_{index:04d}",
        document_id=f"{logical_key}@{version}",
        logical_key=logical_key,
        document_version=version,
        title=title,
        document_type=document_type,
        section_path=section,
        page_no=2,
        char_start=index * 10,
        char_end=index * 10 + 9,
        effective_from=effective_from,
        effective_to=effective_to,
        status=status,
        is_table=is_table,
        table_caption=table_caption,
        classification=classification,
        source_kind=source_kind,
        allowed_roles=("ADMIN",) if classification == "CONFIDENTIAL" else ("ANALYST", "ADMIN"),
        checksum=f"{index:064d}",
    )


def _point(
    index: int,
    *,
    text: str,
    dense: list[float],
    sparse: dict[int, float] | None = None,
    **meta: object,
) -> VectorPoint:
    metadata = _metadata(index, **meta)  # type: ignore[arg-type]
    return VectorPoint(
        chunk_id=metadata.chunk_id,
        text=text,
        dense=dense,
        sparse=sparse if sparse is not None else {0: 1.0},
        payload=metadata.payload(),
    )


def _one_query_gateway(settings: Settings, question: str, vector: list[float]) -> FakeModelGateway:
    """只发一路查询的替身网关（大多数用例用它）。

    改写脚本给的就是原问题本身，`_rewrite` 去重后**只剩原问题一条**——
    于是 `embed` 只会收到一条文本，`embeddings` 只需一项。
    **不传空脚本**：`FakeModelGateway` 对空脚本抛的是 `AssertionError`
    （它刻意不让"忘了配脚本"表现为一个看似通过的用例），
    而 `_rewrite` 只捕获 `AgentError`——两者对不上，用例会以"测试写错了"
    的形式失败，而不是落到降级路径上。要验降级路径请显式设 `failure`。
    """
    return FakeModelGateway(
        responses=[QueryRewrite(queries=(question,))],
        embedding_dim=DIM,
        embeddings=[vector],
    )


def _retriever(
    settings: Settings,
    store: InMemoryVectorStore,
    gateway: FakeModelGateway,
    vocabulary: Vocabulary,
) -> Retriever:
    """按 `settings` 装配检索器。

    `settings.reranker_enabled` 决定重排是否生效——**默认的测试配置里它是关的**，
    因此绝大多数用例走的仍是"RRF 序直接取 Top-K"那条路径；
    要验重排的用例自己把开关打开（见 `_with_reranker`）。
    """
    return Retriever(
        settings=settings,
        gateway=gateway,
        vector_store=store,
        tokenizer=Tokenizer.from_settings(settings),
        vocabulary=vocabulary,
        reranker=Reranker(settings=settings, gateway=gateway),
    )


def _with_reranker(settings: Settings, *, threshold: float = 0.2) -> Settings:
    """打开重排的配置副本。

    用 `model_copy` 而不是造一份完整 Settings：这里要改的只有"开不开"与
    "阈值多少"，其余（Top-K、召回条数、词表路径）必须与线上那份逐字相同——
    另造一份等于让用例测的是一个不存在的配置。
    """
    return settings.model_copy(
        update={
            "reranker_enabled": True,
            "reranker_model": "fake-reranker",
            "reranker_base_url": "https://rerank.invalid/v1",
            "reranker_api_key": "test-key",
            "rag": settings.rag.model_copy(update={"rerank_score_threshold": threshold}),
        }
    )


def _scope(role: UserRole = UserRole.ANALYST) -> PermissionScope:
    return PermissionScope(role=role)


# ------------------------------------------------------------------ 改写


def test_keeps_numbers_rejects_a_rewrite_that_dropped_the_year() -> None:
    """11.7 第 2 步的「必须保留实体和时间」——**由代码判定，不靠 prompt 自觉**。

    「2025 年华东 Q3」被改写成「三季度区域业绩」时，检索照常跑、照常返回结果，
    只是不再是我们问的那件事，没有任何报错。
    """
    assert _keeps_numbers("2025 年华东 Q3 的净销售额", "2025年华东Q3净销售额")
    assert not _keeps_numbers("2025 年华东 Q3 的净销售额", "三季度区域业绩")


def test_keeps_numbers_accepts_the_chinese_numeral_form() -> None:
    """归一化之后再比：中文数词与阿拉伯数字是同一个数。

    不归一化的话，「2025 年第三季度」与「2025 年 Q3」会被判成不同的时间，
    而它们指的是同一段时间——那会把一条完全正确的改写误杀。
    """
    assert _keeps_numbers("2025年第三季度净销售额", "2025年Q3净销售额")


class _RerankUnavailable(FakeModelGateway):
    """只让**重排**失败，改写与向量化照常。

    用 `failure=FakeFailure.UNAVAILABLE` 一把全关掉测的是另一条路径：
    向量化挂掉时检索**必须抛异常**（不能伪装成"语料里没有"，见
    `test_embedding_outage_is_not_reported_as_no_knowledge`），
    根本走不到重排那一步。
    """

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        raise AgentError(ErrorCode.UPSTREAM_UNAVAILABLE, "重排服务不可用（测试构造）")


class _RewriteUnavailable(FakeModelGateway):
    """只让**对话接口**失败，向量化照常。

    真实的故障就是这样**部分**的：主模型服务商限流时，本机 Ollama 上的
    `bge-m3` 好好的。用 `FakeModelGateway.failure` 一把全关掉，测的就不是
    "改写失败怎么办"，而是"整个网关都挂了怎么办"——那是另一条路径
    （见 `test_embedding_outage_is_not_reported_as_no_knowledge`）。
    """

    # 签名与基类逐字对齐（`mypy --strict` 会查覆写兼容性）：
    # 用 `*args: Any, **kwargs: Any` 之类糊过去的话，哪天基类签名变了，
    # 这个替身会静默地不再被调用到，而用例仍然"通过"。
    async def invoke_structured[T: BaseModel](
        self, prompt: PromptSource, schema: type[T], /, **variables: Any
    ) -> StructuredResult[T]:
        raise AgentError(ErrorCode.UPSTREAM_UNAVAILABLE, "改写服务不可用（测试构造）")


async def test_rewrite_failure_degrades_to_the_original_question(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """改写失败 → 退化成原问题检索，不抛给调用方。

    这是开发流程对基础设施 Phase 要求的降级路径：**"改不动"不等于"查不了"**。
    一次模型抖动不该让整个检索步骤失败。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = _RewriteUnavailable(responses=[], embedding_dim=DIM, embeddings=[[1.0, 0, 0, 0]])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.rewrite_degraded
    assert outcome.queries == ("华东区域渠道折扣政策",)
    assert outcome.candidates  # 检索照常完成


async def test_embedding_outage_is_not_reported_as_no_knowledge(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """向量化服务挂了 → **抛出**，绝不返回"没有相关知识"。

    这两件事的处置完全相反：服务不可用要重试或降级，
    而"语料里没有"是要告诉用户"公司没有这条制度"。
    把前者伪装成后者，等于在故障时对着用户编一个结论出来——
    而它看起来完全正常（`NO_RELEVANT_KNOWLEDGE` 是个合法错误码）。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    gateway.failure = FakeFailure.UNAVAILABLE

    with pytest.raises(AgentError):
        await _retriever(settings, store, gateway, vocabulary).retrieve(
            RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
        )


async def test_rewrites_are_added_but_the_original_is_never_replaced(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """改写是**补充**召回，不是替换。

    改写模型把时间写歪时（这里模拟成"丢了数字"），只拿改写结果去查会让
    整次检索问错问题；而原问题无损。所以原问题永远排在第一路。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = FakeModelGateway(
        responses=[
            QueryRewrite(
                queries=(
                    "2025年华东渠道折扣制度",
                    "丢掉了时间的三季度业绩",
                    "2025年华东渠道折扣政策规定",
                )
            ),
        ],
        embedding_dim=DIM,
        # 三路：原问题 + 两条保留了 2025 的改写
        embeddings=[[1.0, 0, 0, 0]] * 3,
    )

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="2025年华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.queries[0] == "2025年华东区域渠道折扣政策"
    assert "2025年华东渠道折扣制度" in outcome.queries
    # 丢了数字的那条**被丢掉**，而不是"留着但排后面"
    assert "丢掉了时间的三季度业绩" not in outcome.queries
    assert not outcome.rewrite_degraded


# ------------------------------------------------------------------ 门禁


async def test_low_similarity_yields_no_relevant_knowledge(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """语料里没有相关内容 → 不返回任何候选（11.8 第 1 条）。

    **候选清空而不是"打个标记"**：只要候选还在，下游就有机会把它当成证据用，
    而"不让生成节点补写制度"这条纪律靠的是没有东西可写。
    """
    store = InMemoryVectorStore()
    # 分块向量与查询向量正交 → 余弦 0
    await store.upsert([_point(0, text="华东渠道折扣", dense=[0.0, 1.0, 0, 0])])
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.no_relevant_knowledge
    assert outcome.candidates == ()
    assert outcome.best_dense_score < settings.rag.score_threshold
    # 融合本身照常跑了（候选数不为 0）——被拦下的是"判定"，不是"检索没做"
    assert outcome.candidate_count > 0


async def test_unseen_topic_is_rejected_even_when_similarity_passes(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """**语料从未出现过的主题词** → 拒答，哪怕余弦高于阈值。

    这是 19 条实测里真实发生的情形：「公司食堂管理规定」的最高余弦是 0.6961，
    高于阈值，是"食堂"这个词不在词表里（88 篇语料一次都没出现过）把它拦下的。
    单靠余弦的门禁会把它连同另外 3 条一起放行。
    """
    store = InMemoryVectorStore()
    # 余弦 1.0，远高于阈值——只有未登录词判据能拦下它
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = _one_query_gateway(settings, "食堂管理规定", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="食堂管理规定"), scope=_scope()
    )

    assert outcome.best_dense_score == pytest.approx(1.0, abs=1e-6)
    assert outcome.unseen_topics == ("食堂",)
    assert outcome.no_relevant_knowledge
    assert outcome.candidates == ()

    # 对照：把"食堂"换成词表里有的词，同样高的余弦 → 正常召回。
    # 两组只差一个词，说明拦下它的是**词表事实**而不是分数
    store2 = InMemoryVectorStore()
    await store2.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway2 = _one_query_gateway(settings, "华东区域管理规定", [1.0, 0, 0, 0])
    ok = await _retriever(settings, store2, gateway2, vocabulary).retrieve(
        RagQueryArgs(question="华东区域管理规定"), scope=_scope()
    )
    assert ok.unseen_topics == ()
    assert not ok.no_relevant_knowledge


async def test_a_compound_containing_a_known_term_is_not_unseen(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """**语料认识的词被包在更大的词里时，不算"语料没见过"。**

    实测踩到的：用户写「华东**地区**」，而语料一律写「华东**区域**」，
    jieba 把「华东地区」切成一个整词——于是 `id_of()` 是 None，
    一条余弦 0.71 的**真问题**被判成"没有相关知识"。
    `covers` 只查了"问题词被语料词包含"（归口 ⊂ 归口管理部门），
    缺了镜像方向。这条钉住它。
    """
    assert vocabulary.covers("华东地区") is True  # 华东 ⊂ 华东地区
    assert vocabulary.covers("归口") is True  # 归口 ⊂ 归口管理部门（另一方向）
    assert vocabulary.covers("碳积分") is False  # 两个方向都不沾


async def test_demonstratives_do_not_count_as_unseen_topics(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """指示代词同样是 OOV 的常客，且同样不携带主题。

    语料是陈述性的，从不用「**这个**口径怎么定」这种指代写法——
    而用户会很自然地这么问。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策这个口径", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策这个口径"), scope=_scope()
    )

    assert outcome.unseen_topics == ()
    assert not outcome.no_relevant_knowledge


async def test_interrogatives_do_not_count_as_unseen_topics(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """疑问词不算主题词。

    语料是陈述性文本（制度、报告、口径说明），从不用疑问句式写句子，
    于是「怎么」「哪些」在词表里一律未登录。不排除它们的话，
    **任何问句**都会被判成"语料没见过"，真问题会被一起挡掉。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策怎么规定的", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策怎么规定的"), scope=_scope()
    )

    assert outcome.unseen_topics == ()
    assert not outcome.no_relevant_knowledge


# ------------------------------------------------------------------ 标量过滤


async def test_analyst_cannot_see_confidential_chunks(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """权限过滤（11.7 第 3 步 / 21.8 验收 2）。

    机密件的 `allowed_roles` 里没有 ANALYST，检索必须把它整条挡在外面——
    这靠的是 payload 上的标量过滤，与 SQL 侧的 `region_ids` 是两套机制。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(0, text="机密折扣红线", dense=[1.0, 0, 0, 0], classification="CONFIDENTIAL"),
            _point(1, text="华东渠道折扣", dense=[1.0, 0, 0, 0]),
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    retriever = _retriever(settings, store, gateway, vocabulary)

    analyst = await retriever.retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope(UserRole.ANALYST)
    )
    admin = await retriever.retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope(UserRole.ADMIN)
    )

    assert [c.chunk_id for c in analyst.candidates] == ["chk_0001"]
    assert {c.chunk_id for c in admin.candidates} == {"chk_0000", "chk_0001"}


async def test_expired_version_is_not_retrieved(settings: Settings, vocabulary: Vocabulary) -> None:
    """同名制度的失效版本不能被召回（VERSION_PAIR 的机制基础）。

    两版都在库里、都是 ACTIVE，由**生效区间**决定谁进候选。
    不按 as_of 过滤的话，检索会同时拿到两段相互矛盾的规定。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(
                0,
                text="旧版折扣标准",
                dense=[1.0, 0, 0, 0],
                version="v1.0",
                effective_to=date(2025, 6, 30),
            ),
            _point(
                1,
                text="新版折扣标准",
                dense=[1.0, 0, 0, 0],
                version="v2.0",
                effective_from=date(2025, 7, 1),
            ),
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    retriever = _retriever(settings, store, gateway, vocabulary)

    later = await retriever.retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策", as_of=date(2025, 9, 1)),
        scope=_scope(),
    )

    assert [c.metadata.document_version for c in later.candidates] == ["v2.0"]


async def test_document_type_filter_reaches_the_store(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """Intent 提取出来的文档类型要真的下推到标量过滤（11.7 第 1、3 步）。"""
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0], document_type="POLICY"),
            _point(1, text="华东月度经营情况", dense=[1.0, 0, 0, 0], document_type="REPORT"),
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策", document_types=("REPORT",)),
        scope=_scope(),
    )

    assert [c.chunk_id for c in outcome.candidates] == ["chk_0001"]


# ------------------------------------------------------------------ 融合与截断


async def test_top_k_bounds_the_evidence(settings: Settings, vocabulary: Vocabulary) -> None:
    """11.7 第 7 步：最终只取 Top 8（配置值 `rerank_top_k`）。

    **这条走的是重排关闭的那条路径**（默认配置）：按 RRF 序取前 K 条。
    重排生效时的 Top-K 在 `test_reranker_takes_top_k_only_after_reordering`
    ——那里要证明的是"先重排再截断"，顺序反过来就会把候选范围预先砍掉。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(index, text=f"华东渠道折扣第{index}条", dense=[1.0, 0, 0, 0])
            for index in range(15)
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert len(outcome.candidates) == settings.rag.rerank_top_k
    assert [c.rank for c in outcome.candidates] == list(range(1, 9))
    # 融合分单调不增。**只对召回来的那些断言**：第 ⑧ 步补回来的行块
    # 没有参与过检索，因而 `fusion_score` 是 `None`（填 0 会被读成"垫底"）
    scores = [c.fusion_score for c in outcome.candidates if c.fusion_score is not None]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ 证据


async def test_evidence_locator_carries_everything_needed_to_cite(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """证据必须可定位（13.1 / 22.3 的「引用 chunk_id 与原文一致」）。

    定位信息少一个字段**不会报错**，症状是引用打得开但定位不到原文——
    而"这段话出自哪份文件的哪一节"正是文档证据存在的意义。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(
                0,
                text="华东区域渠道折扣政策 > 第二章 折扣标准\n直营渠道折扣上限 12%",
                dense=[1.0, 0, 0, 0],
                effective_from=date(2025, 7, 1),
                effective_to=date(2025, 12, 31),
            )
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    evidence = build_document_evidence(outcome.candidates, question="渠道折扣上限")
    assert len(evidence) == 1
    item = evidence[0]
    locator = item.locator

    assert item.source_type == "DOCUMENT"
    assert locator["chunk_id"] == "chk_0000"
    assert locator["logical_key"] == "policy/east-china-channel-discount"
    assert locator["document_version"] == "v2.0"
    assert locator["section_path"] == ["华东区域渠道折扣政策", "第二章 折扣标准"]
    assert locator["page_no"] == 2
    assert str(locator["checksum"]).startswith("0000")
    assert locator["source_kind"] == "INTERNAL"
    # claim 是**原文**，不是转述
    assert item.claim.startswith("华东区域渠道折扣政策 > 第二章 折扣标准")
    assert item.content_hash == hashlib.sha256(outcome.candidates[0].text.encode()).hexdigest()
    assert item.access_level == "INTERNAL"


async def test_evidence_reliability_follows_13_2(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """可靠性按用途判（13.2）：制度是规范性的，外部材料只能补充。

    两组取值不能混：内部报告与外部报告都是 `REPORT`，
    但 13.2 第 4 条约束的是**来源性质**，「外部不得覆盖内部事实」
    （FR-SEARCH-001）正是靠这个等级差在 Phase 9 成立的。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0], document_type="POLICY"),
            _point(1, text="华东渠道折扣", dense=[1.0, 0, 0, 0], document_type="REPORT"),
            _point(
                2,
                text="华东渠道折扣",
                dense=[1.0, 0, 0, 0],
                document_type="REPORT",
                source_kind=SourceKind.EXTERNAL,
            ),
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    # **不能按 document_type 建索引**：chk_0001 与 chk_0002 都是 REPORT，
    # 按类型索引会让后者覆盖前者，断言就变成了"只测了其中一条"
    evidence = {
        item.locator["chunk_id"]: item
        for item in build_document_evidence(outcome.candidates, question="渠道折扣")
    }

    assert evidence["chk_0000"].reliability == "HIGH"  # 内部制度：规范性
    assert evidence["chk_0001"].reliability == "MEDIUM"  # 内部报告：叙述性
    # 外部报告不因为也叫 REPORT 就拿到内部报告的地位（13.2 第 4 条约束的是来源性质）
    assert evidence["chk_0002"].reliability == "LOW"


async def test_event_time_is_none_when_the_range_is_open(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """只有起始日（长期有效）的制度**不给 `event_time`**。

    `TimeRange` 是闭开区间、两端都不能为空（SQL 侧「Q3 = [7-01, 10-01)」那条
    论证就建立在此）。给长期有效的制度编一个终止日就是凭空造一个事实，
    而 13.4 会拿它去比 TIME 冲突。原始日期仍然进 locator，不丢信息。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0], effective_from=date(2025, 7, 1)),
            _point(
                1,
                text="华东渠道折扣",
                dense=[1.0, 0, 0, 0],
                effective_from=date(2025, 7, 1),
                effective_to=date(2025, 12, 31),
            ),
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )
    evidence = {
        e.locator["chunk_id"]: e for e in build_document_evidence(outcome.candidates, question="q")
    }

    assert evidence["chk_0000"].event_time is None
    assert evidence["chk_0000"].locator["effective_to"] is None
    closed = evidence["chk_0001"].event_time
    assert closed is not None
    # 终止日按**当天结束**取，与检索侧的闭区间语义一致
    assert closed.end.date() == date(2025, 12, 31)
    assert closed.contains(datetime(2025, 12, 31, 12, 0, tzinfo=UTC))


# ------------------------------------------------------------------ Tool


def _context() -> ToolContext:
    return ToolContext(user_id="usr_test", task_id="tsk_test")


async def test_tool_returns_evidence_on_success(settings: Settings, vocabulary: Vocabulary) -> None:
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣上限 12%", dense=[1.0, 0, 0, 0])])
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    tool = RagRetrieveTool(
        settings=settings, retriever=_retriever(settings, store, gateway, vocabulary)
    )

    result = await tool.execute(RagQueryArgs(question="华东区域渠道折扣政策"), _context())

    assert result.status == "SUCCEEDED"
    assert result.tool == "rag_retrieve"
    assert len(result.evidence) == 1
    assert result.evidence[0].source_type == "DOCUMENT"
    assert result.payload is not None and result.payload["chunks"]


async def test_tool_maps_no_knowledge_to_a_distinct_error_code(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """⑩：`NO_RELEVANT_KNOWLEDGE`，且**一条证据都不产出**。

    与 SQL 的空结果**故意不一样**：SQL 的 0 行返回 SUCCEEDED（那是关于数据的
    事实），而 RAG 的空结果必须是非零错误码——标成 SUCCEEDED 的话，
    `payload.chunks` 是空列表，下游只看得到"成功、零条"，与"工具没跑到"
    长得一模一样，生成节点就有机会把空结果补写成"公司暂无相关规定"。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[0.0, 1.0, 0, 0])])
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])
    tool = RagRetrieveTool(
        settings=settings, retriever=_retriever(settings, store, gateway, vocabulary)
    )

    result = await tool.execute(RagQueryArgs(question="华东区域渠道折扣政策"), _context())

    assert result.status == "FAILED"
    assert result.evidence == []
    assert result.error is not None
    assert result.error.code == ErrorCode.NO_RELEVANT_KNOWLEDGE.value
    # EMPTY_RESULT 而不是 SECURITY / VALIDATION：9.4 要求 Reviewer 据此
    # 判断补证、澄清还是受限回答——FAILED 不等于终止
    assert result.error.error_class == "EMPTY_RESULT"
    assert result.error.retryable is False
    # 「阈值调高了」与「语料没见过这个词」处置完全不同，两个判据都要报出来
    assert "判据一" in (result.error.safe_detail or "")
    assert "判据二" in (result.error.safe_detail or "")


async def test_tool_exposes_rewrite_degradation_as_a_warning(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """降级过的事必须能被调用方看到。

    检索结果变差时，第一件要排除的就是"这次用的是原问题还是改写后的查询"，
    而它在结果里看不出来。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东渠道折扣", dense=[1.0, 0, 0, 0])])
    gateway = _RewriteUnavailable(responses=[], embedding_dim=DIM, embeddings=[[1.0, 0, 0, 0]])
    tool = RagRetrieveTool(
        settings=settings, retriever=_retriever(settings, store, gateway, vocabulary)
    )

    result = await tool.execute(RagQueryArgs(question="华东区域渠道折扣政策"), _context())

    assert result.status == "SUCCEEDED"
    assert result.payload is not None
    assert result.payload["rewrite_degraded"] is True
    assert result.payload["warnings"]


async def test_sparse_only_candidates_have_no_dense_score(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """只被稀疏路召回的候选 `dense_score` 是 `None`，不是 0.0。

    它压根没被稠密路评估过，而 0.0 读起来是"语义完全不相关"——
    那是个我们并不知道的结论，下游一旦读到就会把它当成事实。
    """
    store = InMemoryVectorStore()
    # 24 条与查询同向（余弦 1.0）且不携带任何 token；
    # 1 条与查询正交（余弦 0.0）但携带查询里的 token。
    # 稠密路取 Top 20，正交那条排在第 25 名、**进不了稠密结果集**——
    # 于是它只被稀疏路带回，没有稠密分可言。
    await store.upsert(
        [
            _point(index, text=f"华东渠道折扣第{index}条", dense=[1.0, 0, 0, 0], sparse={})
            for index in range(24)
        ]
    )
    await store.upsert(
        [_point(24, text="华东渠道折扣的邻接条款", dense=[0.0, 1.0, 0, 0], sparse={0: 1.0})]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert not outcome.no_relevant_knowledge, "整体余弦很高，门禁不该拦"
    sparse_only = [c for c in outcome.candidates if c.dense_score is None]
    assert len(sparse_only) == 1
    assert sparse_only[0].chunk_id == "chk_0024"
    cited = {
        e.locator["chunk_id"]: e
        for e in build_document_evidence(outcome.candidates, question="华东")
    }
    assert cited["chk_0024"].locator["dense_score"] is None


def test_sparse_drops_tokens_outside_the_vocabulary(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """词表里没有的 token 直接丢弃，**入库侧与查询侧对称地丢**。

    这是 11.6.4 固定 IDF 方案的已知代价：未登录词分配不到 id，于是它在
    稀疏路彻底消失。对称很重要——只丢一边会让点积恒为 0，表现为
    "检索永远召回不到东西"，而两边各自的代码看起来都对。

    整块 token 全部未登录的情形由入库侧的 `_check_sparse_coverage` 挡住
    （它会以"先跑 make vocab"报错），不靠这里兜。
    """
    vector = build_sparse(["华东", "碳积分"], vocabulary, dim=settings.rag.sparse_dim)

    assert set(vector) == {vocabulary.id_of("华东")}
    assert all(weight > 0.0 for weight in vector.values())


# ------------------------------------------------------------------ 重排（① 11.7 第 ⑥⑦ 步）


def _rerank_gateway(
    settings: Settings, question: str, vector: list[float], scores: list[float]
) -> FakeModelGateway:
    """带重排脚本的替身网关。

    `scores` 与**候选同序**（候选按 RRF 序给出，本文件里通常就是 chunk_id 序——
    所有点的向量相同时融合分会并列，`_fuse` 按 chunk_id 兜底排序）。
    """
    return FakeModelGateway(
        responses=[QueryRewrite(queries=(question,))],
        embedding_dim=DIM,
        embeddings=[vector],
        rerank_scores=[scores],
    )


async def _three_candidates() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    await store.upsert(
        [_point(index, text=f"华东渠道折扣第{index}条", dense=[1.0, 0, 0, 0]) for index in range(3)]
    )
    return store


async def test_reranker_reorders_by_semantic_score(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """重排分高的排前面，**名次由重排重编**（不是 RRF 的名次）。

    `rank` 的语义是"最终第几条"。重排之后还留着 RRF 的名次，读的人会以为
    那就是融合序——而两者在这条用例里恰好相反。

    分数全部高于阈值，因此这条用例只看**排序**；剔除是下一条的事。
    """
    store = await _three_candidates()
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], [0.25, 0.5, 0.9])

    outcome = await _retriever(_with_reranker(settings), store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.rerank_applied is True
    assert outcome.rerank_skipped_reason is None
    assert [c.chunk_id for c in outcome.candidates] == ["chk_0002", "chk_0001", "chk_0000"]
    assert [c.rank for c in outcome.candidates] == [1, 2, 3]
    assert [c.rerank_score for c in outcome.candidates] == [0.9, 0.5, 0.25]
    assert outcome.best_rerank_score == 0.9


async def test_reranker_prunes_candidates_below_the_threshold(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """11.7 第 ⑦ 步：低于校准阈值的候选**剔除**，不是排到后面。

    这是重排器存在的第二个理由（第一个是排序）：稀疏路独有的候选没有稠密分，
    逐候选的语义判定只有它有资格做（见 `reranker.py` 的模块说明）。
    """
    store = await _three_candidates()
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], [0.9, 0.05, 0.3])

    outcome = await _retriever(
        _with_reranker(settings, threshold=0.2), store, gateway, vocabulary
    ).retrieve(RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope())

    assert [c.chunk_id for c in outcome.candidates] == ["chk_0000", "chk_0002"]
    assert outcome.rerank_pruned == 1
    assert outcome.rerank_threshold == 0.2
    assert outcome.no_relevant_knowledge is False


async def test_all_candidates_pruned_is_reported_as_no_knowledge(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """一条都没留下 → `NO_RELEVANT_KNOWLEDGE`，且**候选清空**。

    重排生效时这就是"语料里没有"的判据：cross-encoder 把问题与候选一起过了一遍
    模型，它给出的分数正是那两条代偿规则（词表、余弦）想近似的东西。
    """
    store = await _three_candidates()
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], [0.01, 0.05, 0.1])

    outcome = await _retriever(
        _with_reranker(settings, threshold=0.5), store, gateway, vocabulary
    ).retrieve(RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope())

    assert outcome.rerank_applied is True
    assert outcome.no_relevant_knowledge is True
    assert outcome.candidates == ()
    # 剔空了也要留下"离阈值多远"：只报一个布尔量的话，
    # 「阈值调高了」与「语料真没有」又分不出来了
    assert outcome.best_rerank_score == 0.1
    assert outcome.rerank_pruned == 3


async def test_reranking_can_overrule_the_unseen_topic_criterion(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """重排生效时，**未登录词不再单独拒答**（这是本次切换的核心语义）。

    「报备」在语料里一次都没出现（制度写的是「备案」），词表里自然没有它，
    于是旧的词汇判据把一条余弦 0.74 的真问题判成了"语料没见过"
    （CLAUDE.md 约定 30 记的漏网类，金标 rag-06）。纯词汇规则区分不了
    "语料没讲过这件事"与"语料用了另一个说法"——而 cross-encoder 能。

    词表与余弦**仍然是诊断信息**，所以这里一并断言它们还在。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="制度要求事前备案，未备案不得开展", dense=[1.0, 0, 0, 0])])
    gateway = _rerank_gateway(settings, "报备流程怎么规定", [1.0, 0, 0, 0], [0.85])

    outcome = await _retriever(_with_reranker(settings), store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="报备流程怎么规定"), scope=_scope()
    )

    assert "报备" in outcome.unseen_topics, "这条用例的前提是词汇判据会触发"
    assert outcome.no_relevant_knowledge is False
    assert [c.chunk_id for c in outcome.candidates] == ["chk_0000"]


async def test_reranker_sends_the_original_question_not_the_rewrite(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """11.7 第 ⑥ 步：对**原问题**与候选重排。

    改写是为了召回（它可能把「碳积分」换成更通用的说法），而"这条候选
    到底回没回答用户问的那件事"要以原问题为准。送错一个，排序会整体偏向
    改写后的措辞，而且没有任何报错。
    """
    store = await _three_candidates()
    gateway = FakeModelGateway(
        responses=[QueryRewrite(queries=("华东区域渠道折扣政策", "华东渠道折扣上限"))],
        embedding_dim=DIM,
        embeddings=[[1.0, 0, 0, 0], [1.0, 0, 0, 0]],
        rerank_scores=[[0.5, 0.5, 0.5]],
    )

    await _retriever(_with_reranker(settings), store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert gateway.rerank_calls[0][0] == "华东区域渠道折扣政策"
    # 候选**先全部交给重排**（≤ 30 条），不是先截到 Top 8 再排
    assert len(gateway.rerank_calls[0][1]) == 3


async def test_rerank_passage_is_truncated_to_the_configured_limit(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """超长块按 `rerank_max_chars` 截断。**它是护栏不是调参项**：
    超长输入只会让 cross-encoder 变慢，而超出模型窗口的部分本来也会被丢弃。
    """
    store = InMemoryVectorStore()
    await store.upsert([_point(0, text="华东" * 500, dense=[1.0, 0, 0, 0])])
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], [0.9])
    tiny = _with_reranker(settings).model_copy(
        update={"rag": settings.rag.model_copy(update={"rerank_max_chars": 120})}
    )

    await _retriever(tiny, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert len(gateway.rerank_calls[0][1][0]) == 120


async def test_reranker_outage_falls_back_to_the_rrf_order(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """重排服务不可用时**照常出证据**，只是顺序退回 RRF，并留下原因。

    把一次重排超时升级成检索失败，代价与收益完全不相称：重排是增强步骤，
    它不在的时候系统必须照常工作——而"照常"的那条路径与"没开重排"是同一条。
    """
    store = await _three_candidates()
    gateway = _RerankUnavailable(
        responses=[QueryRewrite(queries=("华东区域渠道折扣政策",))],
        embedding_dim=DIM,
        embeddings=[[1.0, 0, 0, 0]],
    )

    outcome = await _retriever(_with_reranker(settings), store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.rerank_applied is False
    assert outcome.rerank_skipped_reason == f"{SKIP_UNAVAILABLE}:UPSTREAM_UNAVAILABLE"
    assert [c.chunk_id for c in outcome.candidates] == ["chk_0000", "chk_0001", "chk_0002"]
    # **失败时不写 `rerank_score`**：那是"模型给的分数"，没算过就没有值。
    # 填一个假的进去，读的人会把 RRF 名次当成模型的判断
    assert all(c.rerank_score is None for c in outcome.candidates)
    assert outcome.no_relevant_knowledge is False


async def test_disabled_reranker_keeps_the_rrf_order(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """默认关闭：不调重排服务、不写分数，原因与"调用失败"**必须不同**。

    "召回质量下降了"是排查时的第一类问题，而"没开"与"打了服务但失败了"
    是两条完全不同的路径，合成一个"没生效"就分不出来了。
    """
    store = await _three_candidates()
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], [0.9, 0.9, 0.9])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.rerank_applied is False
    assert outcome.rerank_skipped_reason == SKIP_DISABLED
    assert outcome.rerank_threshold is None
    assert gateway.rerank_calls == [], "没开重排就不该调它"


class _ShortRerank(FakeModelGateway):
    """只返回一条分数的重排替身，用来验"条数不符"这条路径。

    真实网关会在这里之前就报错（`_scores_by_index` 见缺条即抛），这条兜底是
    给替身与将来的实现留的：**长度对不上时按顺序硬配是最危险的做法**——
    每个候选会拿到别人的分数，而排序看起来完全正常。
    """

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        return [0.9]


async def test_incomplete_rerank_scores_fall_back_instead_of_misaligning(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    store = await _three_candidates()
    gateway = _ShortRerank(
        responses=[QueryRewrite(queries=("华东区域渠道折扣政策",))],
        embedding_dim=DIM,
        embeddings=[[1.0, 0, 0, 0]],
    )

    outcome = await _retriever(_with_reranker(settings), store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert outcome.rerank_applied is False
    assert outcome.rerank_skipped_reason == SKIP_INCOMPLETE
    assert [c.chunk_id for c in outcome.candidates] == ["chk_0000", "chk_0001", "chk_0002"]


async def test_reranker_takes_top_k_only_after_reordering(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """先重排、再取 Top-K。**顺序反过来就等于把重排器的判断范围预先砍掉**：
    RRF 排在第 9 位的候选若其实是唯一真正相关的，先截断就没有它了。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(index, text=f"华东渠道折扣第{index}条", dense=[1.0, 0, 0, 0])
            for index in range(15)
        ]
    )
    # 分数与"RRF 顺序"完全相反：0 号最高、14 号最低
    scores = [1.0 - index * 0.05 for index in range(15)]
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], scores)

    outcome = await _retriever(_with_reranker(settings), store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert len(gateway.rerank_calls[0][1]) == 15, "候选要先全部交给重排"
    assert len(outcome.candidates) == settings.rag.rerank_top_k
    assert [c.chunk_id for c in outcome.candidates] == [
        f"chk_{index:04d}" for index in range(settings.rag.rerank_top_k)
    ]
    assert [c.rank for c in outcome.candidates] == list(range(1, settings.rag.rerank_top_k + 1))


async def test_tool_reports_the_rerank_criterion_in_safe_detail(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """重排生效时拒答，`safe_detail` 要能看出**是它判的**。

    三条判据的处置完全不同：「语料没见过这个词」要去确认是不是问错了，
    「余弦太低」要去查阈值，「重排分都不够」要去查重排阈值或语料。
    只报前两条的话，读的人会去查一个这次根本没参与判定的数。
    """
    store = await _three_candidates()
    gateway = _rerank_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0], [0.02, 0.01, 0.03])
    tool = RagRetrieveTool(
        settings=settings,
        retriever=_retriever(_with_reranker(settings, threshold=0.5), store, gateway, vocabulary),
    )

    result = await tool.execute(RagQueryArgs(question="华东区域渠道折扣政策"), _context())

    assert result.status == "FAILED"
    detail = (result.error.safe_detail if result.error else "") or ""
    assert "判据三" in detail
    assert "重排分最高 0.0300" in detail


# ---------------------------------------------------------------- ⑧ 表格行定位


async def test_table_rows_carry_their_position_in_the_whole_table(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """表格按行分块，所以「这张表还有别的行」必须由检索侧补上。

    没有它，召回 8 行与召回整张表在证据列表上完全同形——实测踩到过：
    模型拿 8 行求和当成区域合计（7,146.22 万 vs 真值 13,249.31 万）。

    场景：一张 4 行的表，**只有第 1 行进得了候选**（其余三行与查询正交、
    又不携带任何 token，被 21 条更相关的干扰块挤出稠密 Top 20）。
    数行数要回到存储层去数，因此断言的是"总共 4 行"而不是"召回了 4 条"。
    """
    store = InMemoryVectorStore()
    # 同一张表的 4 行：char_start 递增，故行号就是入块顺序
    await store.upsert(
        [
            _point(
                index,
                text=f"华南 | 渠道{index} | 智能家居 | {index}.00 | 1.00",
                dense=[1.0, 0, 0, 0] if index == 10 else [0.0, 1.0, 0, 0],
                sparse={},
                is_table=True,
                table_caption="分区域分渠道分产品线净销售额明细",
                section=("2025年第三季度经营分析", "五、风险提示"),
            )
            for index in range(10, 14)
        ]
    )
    # 21 条干扰块：比那三行更像查询，把稠密路塞满
    await store.upsert(
        [
            _point(
                index,
                text=f"华东渠道折扣第{index}条",
                dense=[0.99, 0.1, 0, 0],
                sparse={},
                logical_key="report/other",
            )
            for index in range(100, 121)
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    rows = {c.chunk_id: c.table_row for c in outcome.candidates if c.metadata.is_table}
    assert rows, "表格行块应当被召回"
    # 行号与总行数都要来自**存储层**，不是来自召回集：这张表 4 行、
    # 只召回了 1 行，而第 ⑧ 步把其余 3 行补了回来（表装得下）
    assert set(rows.values()) == {(1, 4), (2, 4), (3, 4), (4, 4)}
    assert sorted(rows) == ["chk_0010", "chk_0011", "chk_0012", "chk_0013"]
    # 非表格块没有位置可言——`None` 表示"不是表格的一部分"，
    # 不是"表有 0 行"，两者混起来会让下游把正文块也算进表格缺口
    assert all(c.table_row is None for c in outcome.candidates if not c.metadata.is_table), (
        "非表格块不该带表格位置"
    )


async def test_table_expansion_marks_where_each_row_came_from(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """补回来的行块与召回的行块**必须分得开**。

    三件事同时钉住：`expanded` 说明来源、`fusion_score` 是 `None`
    而不是 0（它压根没被检索过，填 0 会被读成"排名垫底"）、
    `rerank_score` 同样没有值——补块发生在重排**之后**，这正是 11.7
    把第 ⑧ 步排在第 ⑦ 步之后的原因：拿去重排会按相关性被再剔一次，
    而它们存在的理由恰恰是"分数低但缺不得"。
    """
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(
                index,
                text=f"华东 | 渠道{index} | 智能家居 | {index}.00 | 1.00",
                dense=[1.0, 0, 0, 0] if index == 0 else [0.0, 1.0, 0, 0],
                sparse={},
                is_table=True,
                table_caption="分区域分渠道分产品线净销售额明细",
                section=("2025年第三季度经营分析", "五、风险提示"),
            )
            for index in range(3)
        ]
    )
    await store.upsert(
        [
            _point(
                index,
                text=f"华东渠道折扣第{index}条",
                dense=[0.99, 0.1, 0, 0],
                sparse={},
                logical_key="report/other",
            )
            for index in range(100, 121)
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    expanded = [c for c in outcome.candidates if c.expanded]
    retrieved = [c for c in outcome.candidates if not c.expanded]
    assert len(expanded) == 2, "3 行的表只召回了 1 行，其余 2 行应当补回来"
    assert all(c.fusion_score is None for c in expanded)
    assert all(c.rerank_score is None for c in expanded)
    assert all(c.fusion_score is not None for c in retrieved), "召回的那些必须有融合分"


async def test_table_expansion_skips_a_table_it_cannot_afford(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """装不下的表**不补**——这是有意的边界，不是没做完。

    语料里那张 80 行的明细表要 80 个块才装得下，会撑爆分析节点的证据预算；
    而这类聚合问题的正确来源本来就是数据库。此时仍要**标注位置**
    （下游据此如实说明覆盖不足），只是不把行块补进来。
    """
    big = settings.model_copy(
        update={"rag": settings.rag.model_copy(update={"table_expand_max_rows": 2})}
    )
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(
                index,
                text=f"华东 | 渠道{index} | 智能家居 | {index}.00 | 1.00",
                dense=[1.0, 0, 0, 0] if index == 0 else [0.0, 1.0, 0, 0],
                sparse={},
                is_table=True,
                table_caption="一张装不下的表",
                section=("2025年第三季度经营分析", "五、风险提示"),
            )
            for index in range(4)
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(big, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )

    assert not [c for c in outcome.candidates if c.expanded]
    marked = [c for c in outcome.candidates if c.metadata.is_table]
    assert marked and marked[0].table_row == (1, 4), "不补块也要标注：下游靠它说清覆盖不足"


async def test_table_position_reaches_the_evidence_locator(
    settings: Settings, vocabulary: Vocabulary
) -> None:
    """位置必须一路走到证据上：分析节点读的是 `Evidence`，不是检索候选。"""
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _point(
                index,
                text=f"华南 | 渠道{index} | 智能家居 | {index}.00 | 1.00",
                dense=[1.0, 0, 0, 0] if index == 0 else [0.0, 1.0, 0, 0],
                sparse={},
                is_table=True,
                table_caption="分区域分渠道分产品线净销售额明细",
                section=("2025年第三季度经营分析", "五、风险提示"),
            )
            for index in range(3)
        ]
    )
    await store.upsert(
        [
            _point(
                index,
                text=f"华东渠道折扣第{index}条",
                dense=[0.99, 0.1, 0, 0],
                sparse={},
                logical_key="report/other",
            )
            for index in range(100, 121)
        ]
    )
    gateway = _one_query_gateway(settings, "华东区域渠道折扣政策", [1.0, 0, 0, 0])

    outcome = await _retriever(settings, store, gateway, vocabulary).retrieve(
        RagQueryArgs(question="华东区域渠道折扣政策"), scope=_scope()
    )
    evidence = build_document_evidence(outcome.candidates, question="华东区域渠道折扣政策")

    table_evidence = [e for e in evidence if e.locator["is_table"]]
    assert table_evidence and table_evidence[0].locator["table_row"] == [1, 3]
