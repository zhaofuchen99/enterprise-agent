"""知识文档版本仓储契约（详细设计 16.8 / 11.9）。

**同一份断言跑两个实现**（同 `test_vocab_repo.py` 的规矩）。

这里最要紧的是两个唯一键的**语义差别**，它们是 11.9 幂等设计的全部依据：

- `(logical_key, version)` 撞了 → **覆盖同一行**，且 `id` 与 `created_at` 不变；
  每次重建都新插一行的话，"这一版是谁什么时候建进来的"会随重试次数漂移；
- `(checksum, version)` 撞了 → 同一份内容不许出现在两个 `logical_key` 下，
  否则库里会有两份一模一样的向量，检索时并列出现，
  看起来像"两个来源相互印证"。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, date, datetime

import pytest

from app.domain.knowledge import DocumentStatus, KnowledgeDocumentRecord, SourceKind
from app.repositories.knowledge_repo import (
    InMemoryKnowledgeDocumentRepository,
    KnowledgeDocumentRepository,
    SqlKnowledgeDocumentRepository,
    touch,
)


@pytest.fixture(
    params=[
        pytest.param("memory", id="memory"),
        # 连真实 MySQL 的变体。标记写在 param 上而不是用例上——
        # 否则内存变体会被一起排除掉，`make test` 就什么都测不到了。
        pytest.param("sql", id="sql", marks=pytest.mark.integration),
    ]
)
async def make_repo(
    request: pytest.FixtureRequest,
) -> AsyncIterator[Callable[[], KnowledgeDocumentRepository]]:
    if request.param == "memory":
        yield InMemoryKnowledgeDocumentRepository
        return

    from app.tests.db import sql_sessions

    async with sql_sessions() as sessions:
        yield lambda: SqlKnowledgeDocumentRepository(sessions)


_NOW = datetime(2026, 9, 17, 10, 0, tzinfo=UTC)


def _record(
    *,
    record_id: str = "doc_0000000000000000000001",
    logical_key: str = "policy/east-china-channel-discount",
    version: str = "v1.0",
    checksum: str = "a" * 64,
    status: DocumentStatus = DocumentStatus.PROCESSING,
    title: str = "华东区域渠道折扣政策",
    effective_to: date | None = None,
    record_at: datetime = _NOW,
) -> KnowledgeDocumentRecord:
    return KnowledgeDocumentRecord(
        id=record_id,
        logical_key=logical_key,
        version=version,
        title=title,
        document_type="POLICY",
        department="销售运营部",
        storage_path=f"knowledge/{logical_key}/{version}/original.pdf",
        checksum=checksum,
        effective_from=date(2024, 1, 1),
        effective_to=effective_to,
        classification="INTERNAL",
        source_kind=SourceKind.INTERNAL,
        allowed_roles=("ANALYST", "ADMIN"),
        status=status,
        parser_version="rag-1",
        embedding_version="bge-m3",
        created_at=record_at,
        updated_at=record_at,
    )


# ------------------------------------------------------------------ 基本读写


async def test_get_returns_none_for_unknown_version(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    repo = make_repo()
    assert await repo.get("policy/unknown", "v1.0") is None


async def test_save_then_get_round_trips(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    repo = make_repo()
    saved = await repo.save(_record())

    assert await repo.get("policy/east-china-channel-discount", "v1.0") == saved
    assert saved.source_kind is SourceKind.INTERNAL
    assert saved.status is DocumentStatus.PROCESSING


async def test_versions_of_one_logical_key_coexist(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    """同名制度的两个版本必须能同时存在——VERSION_PAIR 缺陷注入的机制基础。

    16.8 的 `(logical_key, version)` 唯一键要挡的是"同一版存两行"，
    不是"一份制度只能有一版"。挡错了的话，语料里 2 组同名跨版本制度
    只能入进去一份，而"版本过滤生效"这条断言会在**检索侧**才失败。
    """
    repo = make_repo()
    await repo.save(
        _record(
            record_id="doc_0000000000000000000001",
            version="v1.0",
            checksum="a" * 64,
            effective_to=date(2025, 6, 30),
        )
    )
    # **两版必须是两个 id**：`id` 也是唯一键，复用会让 upsert 因主键冲突
    # 而把 v1.0 那行当成更新目标——v2.0 根本没插进去，而语句本身不报错。
    await repo.save(
        _record(record_id="doc_0000000000000000000002", version="v2.0", checksum="b" * 64)
    )

    assert len(await repo.list_documents()) == 2
    assert (await repo.get("policy/east-china-channel-discount", "v2.0")) is not None


# ------------------------------------------------------- 两个唯一键的语义差别


async def test_same_version_overwrites_and_keeps_identity(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    """同一版重入（失败重试 / `--force` 重建）落在**同一行**，`id` 与 `created_at` 不变。

    每次重建新插一行的话，`created_by` 与首次入库时间会随重试次数漂移，
    而没有任何地方会因此报错——这一行看起来只是"有一版被建了很多次"。

    **入参带一个不同的 id 也要落回原行**：这一行的身份由库决定，不由调用方决定。
    要做成"id 不匹配就报错"的话，`--force` 重建那条路径会在每次重试时炸，
    而那正是最需要它幂等的时候。
    """
    repo = make_repo()
    first = await repo.save(_record(record_id="doc_AAAA", status=DocumentStatus.FAILED))
    again = await repo.save(
        _record(record_id="doc_BBBB", checksum="a" * 64, status=DocumentStatus.ACTIVE)
    )

    assert again.id == first.id
    assert again.created_at == first.created_at
    assert again.status is DocumentStatus.ACTIVE
    assert len(await repo.list_documents()) == 1


async def test_id_owned_by_another_row_is_rejected(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    """`id` 已被**别的** (logical_key, version) 占用 → 必须报错。

    这是 ON DUPLICATE KEY UPDATE 最容易造成静默损坏的一种：语句会因主键冲突
    去更新那一行，于是"插入新版"变成"改写另一份文档的标题、路径与状态"，
    而那一行的 `logical_key` 不变——库里从此有一行"名字是 A、内容是 B"。
    """
    repo = make_repo()
    await repo.save(_record(record_id="doc_AAAA"))

    with pytest.raises(ValueError, match="已属于"):
        await repo.save(_record(record_id="doc_AAAA", version="v2.0", checksum="b" * 64))


async def test_same_checksum_under_another_key_is_rejected(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    """同一份内容不许出现在两个 `logical_key` 下（`(checksum, version)` 唯一）。

    允许的话，库里会有两份一模一样的向量同时被检索到，
    证据链上表现为"两个来源相互印证"，而它们其实是同一份文件。
    """
    repo = make_repo()
    await repo.save(_record(checksum="c" * 64))

    with pytest.raises(ValueError, match="checksum"):
        await repo.save(
            _record(
                record_id="doc_0000000000000000000002",
                logical_key="policy/another-policy",
                checksum="c" * 64,
            )
        )


async def test_find_by_checksum_is_independent_of_logical_key(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    """11.9 的指纹去重按 `(checksum, version)` 找，不按 `logical_key`。

    与 `get` 是两条独立查询：`get` 回答"这一版现在什么状态"，
    `find_by_checksum` 回答"这份内容是不是已经作为这一版入过了"。
    """
    repo = make_repo()
    await repo.save(_record(checksum="d" * 64))

    found = await repo.find_by_checksum("d" * 64, "v1.0")
    assert found is not None
    assert await repo.find_by_checksum("d" * 64, "v9.9") is None
    assert await repo.find_by_checksum("e" * 64, "v1.0") is None


# ------------------------------------------------------------------ 状态推进


async def test_touch_advances_status_and_refreshes_updated_at(
    make_repo: Callable[[], KnowledgeDocumentRepository],
) -> None:
    """改状态必须顺带刷 `updated_at`——它是排查"入库卡住了"时第一个要看的东西。

    直接 `model_copy(update={"status": ...})` 会漏掉它，而漏掉之后
    一行 PROCESSING 会看起来"刚刚还在动"。
    """
    repo = make_repo()
    saved = await repo.save(_record())

    advanced = touch(saved, DocumentStatus.ACTIVE, chunk_count=12)
    await repo.save(advanced)
    stored = await repo.get(saved.logical_key, saved.version)

    assert stored is not None
    assert stored.status is DocumentStatus.ACTIVE
    assert stored.chunk_count == 12
    assert stored.updated_at > saved.updated_at
