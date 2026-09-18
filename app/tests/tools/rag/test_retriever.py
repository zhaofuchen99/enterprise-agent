"""在线检索与相关证据（详细设计 11.7 / 11.8 / 13.1）。

**用真实的 `InMemoryVectorStore`**（它真算余弦、点积与 RRF，不是打桩），
只有模型网关是替身——它要连外部服务。这样断言的是检索**行为**
（召回谁、拦下谁），而不是"调用了几次"。

向量只取 4 维：这条链路的正确性与维度无关，而小维度让"哪条该被召回"
一眼能看出来——1024 维的随机向量之间没有可读的关系。
"""

from __future__ import annotations

import hashlib
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
    return Retriever(
        settings=settings,
        gateway=gateway,
        vector_store=store,
        tokenizer=Tokenizer.from_settings(settings),
        vocabulary=vocabulary,
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
    """11.7 第 7 步：最终只取 Top 8（配置值 `rerank_top_k`）。"""
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
    # 融合分单调不增
    scores = [c.fusion_score for c in outcome.candidates]
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
