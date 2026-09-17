"""入库流程（详细设计 11.1 的九步 / 11.9 的幂等与发布）。

**全部用 `tmp_path` 现造文件，不读 `data/corpus`**：语料是 `make corpus` 的产物、
`data/` 已 gitignore，CI 上没有它。依赖生成产物的用例会在本机全绿、在 CI 全红，
而那时读的人会先怀疑入库逻辑。

四个替身都是真实实现：`InMemoryVectorStore` 真算余弦/点积/RRF，
`InMemoryKnowledgeDocumentRepository` 真落实两个唯一键，
`LocalObjectStorage` 真落盘。只有模型网关是桩——它要连外部服务。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from app.core.config import Settings, get_settings
from app.core.errors import AgentError, ErrorCode
from app.domain.knowledge import DocumentStatus, SourceKind
from app.infrastructure.storage import LocalObjectStorage
from app.infrastructure.vector_store import ChunkFilter, InMemoryVectorStore
from app.repositories.knowledge_repo import InMemoryKnowledgeDocumentRepository
from app.tests.fakes import FakeFailure, FakeModelGateway
from app.tools.rag.ingestion import (
    DocumentMetadata,
    IngestionReport,
    ingest_document,
    original_key,
    report_key,
)
from app.tools.rag.tokenizer import Tokenizer
from app.tools.rag.vocabulary import build_vocabulary

#: 一篇最小的制度：小标题 + 表格，够走完解析、分块、表格序列化三条路。
_DOC = """# 华东区域渠道折扣政策

## 第一章 总则

本政策适用于华东区域全部直营与经销渠道，折扣审批以归口管理部门为准。

## 第二章 折扣标准

| 渠道 | 折扣上限 |
| --- | --- |
| 直营 | 12% |
| 经销 | 8% |

## 第三章 附则

本政策自 2025-01-01 起施行，由销售运营部负责解释。
"""


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def storage(tmp_path: Path) -> LocalObjectStorage:
    return LocalObjectStorage(tmp_path / "storage")


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def documents() -> InMemoryKnowledgeDocumentRepository:
    return InMemoryKnowledgeDocumentRepository()


@pytest.fixture
def gateway(settings: Settings) -> FakeModelGateway:
    """向量维度取 `settings.embedding_dim`：入库会用它建 collection，
    替身给别的维度会让"维度不匹配"这类真实故障被测试自己制造出来。"""
    return FakeModelGateway(embedding_dim=settings.embedding_dim)


@pytest.fixture
async def vocabulary(settings: Settings) -> object:
    """用本文档现建一份词表。**不读 `configs/rag_user_dict.txt` 以外的产物**：
    词表快照是 `make vocab` 的产物，落在对象存储里，CI 上没有。"""
    tokenizer = Tokenizer.from_settings(settings)
    result = build_vocabulary([_DOC], tokenizer=tokenizer, sparse_dim=settings.rag.sparse_dim)
    return result.vocabulary


def _write(tmp_path: Path, text: str = _DOC, name: str = "policy.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _metadata(**overrides: object) -> DocumentMetadata:
    base: dict[str, object] = {
        "logical_key": "policy/east-china-channel-discount",
        "version": "v1.0",
        "title": "华东区域渠道折扣政策",
        "document_type": "POLICY",
        "source_kind": SourceKind.INTERNAL,
        "department": "销售运营部",
        "effective_from": date(2025, 1, 1),
    }
    return DocumentMetadata.model_validate({**base, **overrides})


async def _ingest(
    path: Path,
    *,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
    metadata: DocumentMetadata | None = None,
    force: bool = False,
) -> IngestionReport:
    return await ingest_document(
        path,
        metadata or _metadata(),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        tokenizer=Tokenizer.from_settings(settings),
        vocabulary=vocabulary,  # type: ignore[arg-type]
        force=force,
    )


# ------------------------------------------------------------------ 正常路径


async def test_ingest_publishes_and_is_retrievable(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """走完九步之后：文档 ACTIVE、chunk 数对得上、而且**查得到**。

    最后一条是重点。前两条都能在"向量压根没写进去"的情况下成立：
    记录是入库代码自己写的，块数是分块代码数的，只有检索这一条会真的去
    碰向量库并带上过滤条件。
    """
    report = await _ingest(
        _write(tmp_path),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    assert report.ok
    assert report.status is DocumentStatus.ACTIVE
    assert report.chunk_count > 0
    assert report.published_count == report.chunk_count
    # 冒烟三条必须全中：一条不中的意思是"发布出去了但召回不到自己"
    assert [item.matched for item in report.smoke] == [True] * len(report.smoke)
    assert (
        await vector_store.count(ChunkFilter(status="ACTIVE", document_ids=(report.document_id,)))
        == report.chunk_count
    )
    stored = await documents.get("policy/east-china-channel-discount", "v1.0")
    assert stored is not None and stored.status is DocumentStatus.ACTIVE


async def test_original_and_report_are_archived(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """11.9：原文件与入库报告都要归档。**原文件是 collection 可重建的唯一依据**，
    报告是"当初这一版是怎么进来的"的唯一记录。"""
    report = await _ingest(
        _write(tmp_path),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    assert await storage.exists(original_key("policy/east-china-channel-discount", "v1.0", "md"))
    assert report.report_path is not None
    assert await storage.exists(report_key("policy/east-china-channel-discount", "v1.0"))
    assert await storage.get(report.report_path)  # 报告内容非空


async def test_payload_carries_metadata_needed_for_filtering(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """11.7 第 3 步的"权限、ACTIVE 状态和有效期先作为标量过滤"，
    前提是这些字段**真的进了 payload**。少一个，过滤条件就恒不命中——
    表现为"这份文档查不到"，而它明明入库成功了。
    """
    await _ingest(
        _write(tmp_path),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
        metadata=_metadata(classification="CONFIDENTIAL", source_kind=SourceKind.EXTERNAL),
    )

    hits = await vector_store.search_dense(
        [0.1] * settings.embedding_dim,
        limit=1,
        chunk_filter=ChunkFilter(document_ids=("policy/east-china-channel-discount@v1.0",)),
    )
    payload = hits[0].payload

    assert payload["classification"] == "CONFIDENTIAL"
    # 密级决定可见角色（`ROLES_BY_CLASSIFICATION`）：
    # 机密件只有 ADMIN 看得见，落成 ANALYST 就是一次越权可见
    assert payload["allowed_roles"] == ["ADMIN"]
    assert payload["source_kind"] == "EXTERNAL"
    assert payload["effective_from"] == "2025-01-01"
    assert payload["logical_key"] == "policy/east-china-channel-discount"
    assert payload["document_version"] == "v1.0"
    assert payload["status"] == "ACTIVE"


async def test_confidential_chunks_are_invisible_to_analyst(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """21.8 验收 2「权限过滤正确」的入库侧那一半：机密件的 payload 必须
    **在检索时真的把 ANALYST 挡在外面**，而不只是字段填对了。"""
    await _ingest(
        _write(tmp_path),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
        metadata=_metadata(classification="CONFIDENTIAL"),
    )
    doc = ("policy/east-china-channel-discount@v1.0",)

    async def visible(role: str) -> int:
        hits = await vector_store.search_dense(
            [0.1] * settings.embedding_dim,
            limit=50,
            chunk_filter=ChunkFilter(document_ids=doc, roles=(role,)),
        )
        return len(hits)

    assert await visible("ADMIN") > 0
    assert await visible("ANALYST") == 0


async def test_expired_version_is_filtered_out(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """失效版本（VERSION_PAIR 的 v1.0）入库要成功，但**查不到**。

    这正是"同名制度跨版本"这条缺陷注入想要的形态：两版都在库里，
    由生效区间决定谁被召回。若入库时把失效版本也标成可见，
    检索侧会同时拿到 v1.0 与 v2.0 两段相互矛盾的规定。
    """
    await _ingest(
        _write(tmp_path),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
        metadata=_metadata(effective_to=date(2025, 6, 30)),
    )

    async def at(moment: date) -> int:
        hits = await vector_store.search_dense(
            [0.1] * settings.embedding_dim,
            limit=50,
            chunk_filter=ChunkFilter(
                document_ids=("policy/east-china-channel-discount@v1.0",),
                effective_at=moment,
            ),
        )
        return len(hits)

    assert await at(date(2025, 6, 30)) > 0  # 闭区间：最后一天仍有效
    assert await at(date(2025, 7, 1)) == 0


# ------------------------------------------------------------------ 幂等（11.9）


async def test_same_file_twice_is_skipped(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """11.9：相同指纹重复上传**直接返回现有结果**，不重新入库。

    "跳过"要能被观测到，否则它和"重新跑了一遍、结果碰巧一样"无法区分——
    而后者意味着每次重跑都白花一份 embedding 的钱，
    且向量被覆盖一次（同一个 point id，内容相同）。
    """
    path = _write(tmp_path)
    first = await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )
    calls_after_first = len(gateway.embed_calls)

    second = await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    assert second.skipped
    assert second.chunk_count == first.chunk_count
    # **没有重新向量化**：这是"跳过"的唯一硬证据
    assert len(gateway.embed_calls) == calls_after_first


async def test_same_version_with_new_content_is_rejected(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """同版本号、不同内容 → 拒绝。**这是唯一一种"静默覆盖会造成不可逆损坏"的情形**：
    `chunk_id` 由 `logical_key@version` 派生，覆盖后此前引用过 `chk_xxx` 的证据
    会指向另一段文字，而引用本身仍然打得开。
    """
    path = _write(tmp_path)
    await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )
    path.write_text(_DOC + "\n## 第四章 补充\n\n新增一条按月复核的要求。\n", encoding="utf-8")

    with pytest.raises(AgentError) as caught:
        await _ingest(
            path,
            settings=settings,
            storage=storage,
            vector_store=vector_store,
            documents=documents,
            gateway=gateway,
            vocabulary=vocabulary,
        )

    assert caught.value.code is ErrorCode.INVALID_ARGUMENT
    assert "v1.0" in caught.value.message


async def test_force_rebuild_replaces_content(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """`force=True` 承认这是一次重建：新内容生效，报告里留下重建前的指纹。"""
    path = _write(tmp_path)
    first = await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )
    path.write_text(_DOC + "\n## 第四章 补充\n\n新增一条按月复核的要求。\n", encoding="utf-8")

    rebuilt = await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
        force=True,
    )

    assert rebuilt.ok
    assert rebuilt.previous_checksum == first.checksum
    assert rebuilt.checksum != first.checksum
    # **旧版的 chunk 不能留下**：留下的话它们仍是 ACTIVE，
    # 检索会把"已删除的旧条文"和"新条文"一起召回
    assert (
        await vector_store.count(ChunkFilter(status="ACTIVE", document_ids=(rebuilt.document_id,)))
        == rebuilt.chunk_count
    )


async def test_duplicate_content_under_another_key_is_rejected(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """同一份内容挂两个 `logical_key` → 拒绝。

    允许的话，库里会有两份一模一样的向量并列被召回，
    在证据链上看起来像"两个来源相互印证"，而它们其实是同一份文件。
    """
    path = _write(tmp_path)
    await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    with pytest.raises(AgentError) as caught:
        await _ingest(
            path,
            settings=settings,
            storage=storage,
            vector_store=vector_store,
            documents=documents,
            gateway=gateway,
            vocabulary=vocabulary,
            metadata=_metadata(logical_key="policy/another-policy", title="另一份制度"),
        )

    assert caught.value.code is ErrorCode.INVALID_ARGUMENT
    assert "another-policy" in str(caught.value.details.get("incoming"))


# ------------------------------------------------------------------ 失败路径


async def test_document_without_text_marks_failed(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """扫描件走的是"**有结论的失败**"而不是异常（11.2 明确允许标记不支持）。

    这条边界要划清楚：返回 FAILED 报告的用例可以被批处理跳过并继续，
    抛异常的用例会中断整批——而语料里注定有 2 份扫描件，
    让它们中断整批等于另外 86 篇永远入不了库。
    """
    report = await _ingest(
        _write(tmp_path, "", "scanned.md"),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    assert not report.ok
    assert report.status is DocumentStatus.FAILED
    assert report.chunk_count == 0
    assert report.error_summary is not None and "文本层" in report.error_summary
    # 原文件仍然归档：它是日后补 OCR 的出发点
    assert await storage.exists(original_key("policy/east-china-channel-discount", "v1.0", "md"))
    stored = await documents.get("policy/east-china-channel-discount", "v1.0")
    assert stored is not None and stored.status is DocumentStatus.FAILED


async def test_unsupported_suffix_is_rejected_before_any_write(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """格式校验要**在动任何存储之前**失败，否则库里会留下一条 PROCESSING 的僵尸记录。"""
    path = tmp_path / "policy.xlsx"
    path.write_bytes(b"not really a spreadsheet")

    with pytest.raises(AgentError) as caught:
        await _ingest(
            path,
            settings=settings,
            storage=storage,
            vector_store=vector_store,
            documents=documents,
            gateway=gateway,
            vocabulary=vocabulary,
        )

    assert caught.value.code is ErrorCode.INVALID_ARGUMENT
    assert await vector_store.count() == 0
    assert await documents.list_documents() == []


async def test_content_not_matching_suffix_is_rejected(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """文件头与扩展名不符要挡住（施工项 5 的 MIME 检查）。

    拦的是"把文本存成 .pdf"这类错误：放过去的话它会解析出 0 个块，
    然后以"空文档"的名义失败——而真实原因是文件根本不是 PDF。
    """
    path = tmp_path / "policy.pdf"
    path.write_text(_DOC, encoding="utf-8")

    with pytest.raises(AgentError) as caught:
        await _ingest(
            path,
            settings=settings,
            storage=storage,
            vector_store=vector_store,
            documents=documents,
            gateway=gateway,
            vocabulary=vocabulary,
        )

    assert "文件头" in caught.value.message


async def test_empty_sparse_vector_is_rejected(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
) -> None:
    """词表没覆盖这份文档 → 报错，不是打日志。

    稀疏路对未登录 token 的处理是丢弃（11.6.4 固定 IDF 的已知代价），
    整块全丢时这块**永远无法被稀疏路召回**，而稠密路照常工作——
    检索结果看起来"大部分正常"，只有精确词查询会莫名其妙少几篇。
    这是本项目最不想留下的那类"部分可用"的降级。
    """
    with pytest.raises(AgentError) as caught:
        await ingest_document(
            _write(tmp_path),
            _metadata(),
            settings=settings,
            storage=storage,
            vector_store=vector_store,
            documents=documents,
            gateway=gateway,
            tokenizer=Tokenizer.from_settings(settings),
            # 空词表：任何 token 都分配不到 id
            vocabulary=build_vocabulary(
                [],
                tokenizer=Tokenizer.from_settings(settings),
                sparse_dim=settings.rag.sparse_dim,
            ).vocabulary,
        )

    assert "make vocab" in caught.value.message
    assert await vector_store.count() == 0
    stored = await documents.get("policy/east-china-channel-discount", "v1.0")
    assert stored is not None and stored.status is DocumentStatus.FAILED


# --- 失败回滚：两条路径必须分开，否则 --force 重建会把已发布版本删掉


class _PublishFails(InMemoryVectorStore):
    """发布那一步失败，用来验证"本轮写过向量库"的回滚路径。

    **打桩打在 `set_status` 上而不是让它自然失败**：真正要测的是
    "upsert 之后出错时留下的 PROCESSING point 有没有被清掉"，
    而这一点只在 upsert 成功、发布失败时才成立。
    """

    async def set_status(self, chunk_ids: Sequence[str], status: str) -> None:
        raise RuntimeError("发布失败（测试构造）")


async def test_failure_after_upsert_removes_staging_points(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """已经写进向量库但没发布成功 → 整批删掉，不留 PROCESSING 残渣。

    状态位只挡得住**读者**，挡不住"下一版入库时 chunk_id 撞上这些残留"——
    同一个 `chunk_id` 必然得到同一个 point id，残留会被下一次 upsert 覆盖，
    但如果下一次的内容更少，多出来的那些就会以 PROCESSING 长期躺在库里。
    """
    store = _PublishFails()
    with pytest.raises(RuntimeError):
        await _ingest(
            _write(tmp_path),
            settings=settings,
            storage=storage,
            vector_store=store,
            documents=documents,
            gateway=gateway,
            vocabulary=vocabulary,
        )

    # 连 PROCESSING 都不该有：整批已删
    assert await store.count(ChunkFilter(status=None)) == 0
    stored = await documents.get("policy/east-china-channel-discount", "v1.0")
    assert stored is not None and stored.status is DocumentStatus.FAILED


async def test_failed_rebuild_keeps_the_published_version(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """`--force` 重建失败时**必须保住上一版**。

    这是回滚逻辑里最容易写错的一处：把"删掉这一版的 point"当成无条件动作，
    重建一篇已发布文档时只要解析或 embedding 失败，就会把上一次成功发布的
    那批 point 一起删掉——而本轮一个 point 都没写。库里那行随后置成 FAILED，
    于是检索侧少了一版，且没有任何地方会报错。

    判据是 `state.upserted`（本轮有没有写过向量库），不是"失败了就清理"。
    """
    store = InMemoryVectorStore()
    path = _write(tmp_path)
    published = await _ingest(
        path,
        settings=settings,
        storage=storage,
        vector_store=store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    # 改内容后重建，但让向量化在写库之前就失败
    path.write_text(_DOC + "\n## 第四章 补充\n\n按月复核。\n", encoding="utf-8")
    # `UNAVAILABLE` 对应"向量化服务连不上"，是入库最常见的整批失败原因
    # （本机的 Ollama 不在 compose 里，最容易踩）
    broken = FakeModelGateway(embedding_dim=settings.embedding_dim, failure=FakeFailure.UNAVAILABLE)
    with pytest.raises(AgentError):
        await _ingest(
            path,
            settings=settings,
            storage=storage,
            vector_store=store,
            documents=documents,
            gateway=broken,
            vocabulary=vocabulary,
            force=True,
        )

    # 上一版必须原封不动，而且**仍然查得到**
    assert (
        await store.count(ChunkFilter(status="ACTIVE", document_ids=(published.document_id,)))
        == published.chunk_count
    )
    stored = await documents.get("policy/east-china-channel-discount", "v1.0")
    assert stored is not None
    assert stored.status is DocumentStatus.ACTIVE
    assert stored.checksum == published.checksum


# ------------------------------------------------------------------ 元数据映射


async def test_payload_round_trips_through_chunk_metadata(
    tmp_path: Path,
    settings: Settings,
    storage: LocalObjectStorage,
    vector_store: InMemoryVectorStore,
    documents: InMemoryKnowledgeDocumentRepository,
    gateway: FakeModelGateway,
    vocabulary: object,
) -> None:
    """`ChunkMetadata.payload()` → `from_payload()` 必须无损。

    检索侧就是靠这一对函数把 payload 还原成证据引用的。丢掉字段不会报错，
    症状是**引用里的章节/页码变空**——引用仍然打得开，只是定位信息没了，
    没有人会把它当成 bug 报。
    """
    from app.tools.rag.metadata import ChunkMetadata

    await _ingest(
        _write(tmp_path),
        settings=settings,
        storage=storage,
        vector_store=vector_store,
        documents=documents,
        gateway=gateway,
        vocabulary=vocabulary,
    )

    hits = await vector_store.search_dense(
        [0.1] * settings.embedding_dim,
        limit=1,
        chunk_filter=ChunkFilter(document_ids=("policy/east-china-channel-discount@v1.0",)),
    )
    restored = ChunkMetadata.from_payload(hits[0].payload)

    assert restored.chunk_id == hits[0].chunk_id
    assert restored.title == "华东区域渠道折扣政策"
    assert restored.section_path  # 标题路径不能是空的：它决定"这段话出自哪一节"
    assert restored.effective_from == date(2025, 1, 1)
    assert restored.checksum and len(restored.checksum) == 64
    assert restored.status is DocumentStatus.ACTIVE
