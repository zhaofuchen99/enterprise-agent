"""向量库的**契约测试**（开发流程 6.7 施工项 6 的验证命令）。

参数化是有意的，同 `test_storage_contract.py` 的理由：`QdrantVectorStore` 是
唯一的真实实现，`InMemoryVectorStore` 是单测替身，两者跑**同一套断言**——
替身只有在通过契约时才配叫做"替身"，否则它只是一组恰好让测试变绿的桩。

**替身不是打桩**：它真算余弦相似度、点积与 RRF 融合。因此这里断言的是检索**行为**
（谁被召回、谁被过滤掉），不是"调用了几次"——后者在实现被改坏时往往照样通过。

真连的那一档要 `make up`（Qdrant 服务端），打 `integration` 标记，
`make test` 默认跳过、`make test-integration` 覆盖。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest

from app.infrastructure.vector_store import (
    ChunkFilter,
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorPoint,
    VectorStore,
    chunk_point_id,
)

#: 契约测试用的 collection 名。**不叫 `enterprise_knowledge_chunks_v1`**：
#: 那会与本地开发库撞名，跑一次测试就把演示数据删了。
_TEST_COLLECTION = "contract_test_chunks_v1"

DIM = 4


@pytest.fixture(params=["memory", pytest.param("qdrant", marks=pytest.mark.integration)])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[VectorStore]:
    """两个实现共用的构造入口。加第三个实现时，在这里加一个分支即自动被覆盖。"""
    if request.param == "memory":
        impl: VectorStore = InMemoryVectorStore()
        await impl.ensure_collection(dim=DIM)
    else:
        from app.core.config import get_settings

        qdrant = QdrantVectorStore(get_settings().qdrant_url, _TEST_COLLECTION)
        if await qdrant._client.collection_exists(_TEST_COLLECTION):
            await qdrant._client.delete_collection(_TEST_COLLECTION)
        await qdrant.ensure_collection(dim=DIM)
        impl = qdrant
    yield impl
    await impl.aclose()


def _point(
    index: int,
    *,
    dense: list[float],
    sparse: dict[int, float],
    status: str = "ACTIVE",
    document_id: str = "doc_a",
    effective_from: str | None = "2025-01-01",
    effective_to: str | None = "2025-12-31",
    allowed_roles: list[str] | None = None,
    document_type: str = "POLICY",
) -> VectorPoint:
    return VectorPoint(
        chunk_id=f"chk_{index:04d}",
        text=f"第 {index} 条分块",
        dense=dense,
        sparse=sparse,
        payload={
            "document_id": document_id,
            "document_type": document_type,
            "status": status,
            "department": "销售部",
            "classification": "INTERNAL",
            "effective_from": effective_from,
            "effective_to": effective_to,
            "allowed_roles": allowed_roles if allowed_roles is not None else ["ANALYST"],
        },
    )


async def _seed(store: VectorStore) -> None:
    """三条语义分明的分块，让"召回哪一条"本身就能说明问题。"""
    await store.upsert(
        [
            # 稠密最像 query，且稀疏命中 token 1
            _point(0, dense=[1.0, 0.0, 0.0, 0.0], sparse={1: 1.0}),
            # 只被稀疏路命中
            _point(1, dense=[0.0, 0.0, 0.0, 1.0], sparse={2: 1.0}),
            # 两路都沾一点
            _point(2, dense=[0.7, 0.7, 0.0, 0.0], sparse={1: 0.5, 2: 0.5}),
        ]
    )


# ------------------------------------------------------------------ point id


def test_point_id_is_deterministic() -> None:
    """派生必须纯函数——否则 11.9 的"重复入库覆盖同一条"会变成"产生重复证据"，
    且只在第二次入库时才显形。"""
    assert chunk_point_id("chk_0001") == chunk_point_id("chk_0001")


def test_point_id_is_unsigned_and_fits_uint64() -> None:
    """Qdrant 的 point id 是**无符号** 64 位。

    用 signed 解释摘要会产出负数，负 id 到 Qdrant 那里是非法值——
    而它只在部分 chunk_id 上出现，属于典型的"大部分测试都过"的坑。
    """
    for cid in ("chk_0001", "chk_9999", "doc_abc", ""):
        pid = chunk_point_id(cid)
        assert 0 <= pid < 2**64


def test_point_id_differs_for_different_chunks() -> None:
    ids = {chunk_point_id(f"chk_{i:04d}") for i in range(1000)}
    assert len(ids) == 1000


# ------------------------------------------------------------------ 基本检索


async def test_dense_search_returns_most_similar_first(store: VectorStore) -> None:
    await _seed(store)

    hits = await store.search_dense([1.0, 0.0, 0.0, 0.0], limit=3, chunk_filter=ChunkFilter())

    assert hits[0].chunk_id == "chk_0000"
    assert hits[0].score > hits[-1].score


async def test_sparse_search_finds_exact_token_match(store: VectorStore) -> None:
    """稀疏路存在的意义就是制度编号、产品型号这类**精确词**。

    这里断言 token 2 只被 chk_0001 与 chk_0002 携带，而稠密路完全找不到
    chk_0001（它与 query 正交）——两路互补，这正是 11.7 要混合检索的原因。
    """
    await _seed(store)

    hits = await store.search_sparse({2: 1.0}, limit=3, chunk_filter=ChunkFilter())

    assert "chk_0001" in {h.chunk_id for h in hits}


async def test_hybrid_rrf_recalls_both_branches(store: VectorStore) -> None:
    """融合后应该同时包含"稠密路的最优"和"稀疏路的最优"。

    只断言包含关系、不断言次序：RRF 的分数由**排名**决定，
    两路都命中的候选会被加成到前列，具体位次依赖两路各自的排名分布，
    写死次序的断言会在无关改动下随机失败。
    """
    await _seed(store)

    hits = await store.hybrid_rrf(
        [1.0, 0.0, 0.0, 0.0], {2: 1.0}, limit=3, chunk_filter=ChunkFilter()
    )
    ids = {h.chunk_id for h in hits}

    assert "chk_0000" in ids  # 稠密路最优
    assert "chk_0001" in ids  # 稀疏路最优


async def test_empty_store_returns_empty(store: VectorStore) -> None:
    """空库必须返回空列表而不是报错——11.8 要求空结果走到 `NO_RELEVANT_KNOWLEDGE`，
    如果这里抛异常，那条路径永远走不到。"""
    hits = await store.search_dense([1.0, 0.0, 0.0, 0.0], limit=5, chunk_filter=ChunkFilter())

    assert hits == []


async def test_limit_is_respected(store: VectorStore) -> None:
    await _seed(store)

    hits = await store.search_dense([1.0, 0.0, 0.0, 0.0], limit=2, chunk_filter=ChunkFilter())

    assert len(hits) == 2


# ------------------------------------------------------------------ 标量过滤


async def test_filter_excludes_non_active_by_default(store: VectorStore) -> None:
    """11.9：失败版本不得被在线查询命中。

    **默认过滤条件就是 `status=ACTIVE`**，所以"不传 status 的调用方"
    自动是安全的——把这条纪律交给调用方自觉是设计错误。
    """
    await _seed(store)
    await store.set_status(["chk_0000"], "PROCESSING")

    ids = {
        h.chunk_id
        for h in await store.search_dense([1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter())
    }

    assert "chk_0000" not in ids


async def test_filter_can_include_non_active_explicitly(store: VectorStore) -> None:
    """`status=None` 表示不过滤——运维脚本与 `verify-corpus` 需要看到 staging 态。"""
    await _seed(store)
    await store.set_status(["chk_0000"], "PROCESSING")

    ids = {
        h.chunk_id
        for h in await store.search_dense(
            [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(status=None)
        )
    }

    assert "chk_0000" in ids


async def test_effective_range_is_closed_on_both_ends(store: VectorStore) -> None:
    """生效区间是**闭区间**：制度写"有效期至 2025-12-31"，那一天本身仍有效。

    这是最容易写错的一处：真实实现走 Qdrant 的 `DatetimeRange`，
    替身走 Python 的日期比较，**两者必须对边界给出一致答案**，
    否则契约测试会在两个实现上分别通过/失败，看起来像偶发。
    """
    await _seed(store)

    on_last_day = await store.search_dense(
        [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(effective_at=date(2025, 12, 31))
    )
    day_after = await store.search_dense(
        [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(effective_at=date(2026, 1, 1))
    )

    assert len(on_last_day) == 3
    assert day_after == []


async def test_missing_effective_dates_mean_always_valid(store: VectorStore) -> None:
    """没有起始/终止日的文档视为长期有效，不能被区间过滤误伤。"""
    await store.upsert(
        [_point(7, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, effective_from=None, effective_to=None)]
    )

    hits = await store.search_dense(
        [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(effective_at=date(2030, 1, 1))
    )

    assert [h.chunk_id for h in hits] == ["chk_0007"]


async def test_role_filter_needs_intersection(store: VectorStore) -> None:
    """权限过滤：`allowed_roles` 与用户角色**有交集**才可见（11.4）。"""
    await store.upsert(
        [
            _point(0, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, allowed_roles=["ANALYST"]),
            _point(1, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, allowed_roles=["ADMIN"]),
        ]
    )

    analyst = await store.search_dense(
        [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(roles=("ANALYST",))
    )
    nobody = await store.search_dense(
        [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(roles=("VIEWER",))
    )

    assert [h.chunk_id for h in analyst] == ["chk_0000"]
    assert nobody == []


async def test_document_type_filter(store: VectorStore) -> None:
    await store.upsert(
        [
            _point(0, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, document_type="POLICY"),
            _point(1, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, document_type="REPORT"),
        ]
    )

    hits = await store.search_dense(
        [1.0, 0, 0, 0], limit=9, chunk_filter=ChunkFilter(document_types=("REPORT",))
    )

    assert [h.chunk_id for h in hits] == ["chk_0001"]


async def test_document_filter_is_version_scoped(store: VectorStore) -> None:
    """按 `document_id` 过滤必须**按版本**，不能按 `logical_key` 串味。

    同名制度的 v1.0 与 v2.0 共用 `logical_key`，内容却有意不同——
    这正是 VERSION_PAIR 缺陷注入要测的东西。若这里的 `document_id`
    退化成 `logical_key`，v1.0 与 v2.0 的 chunk 会互相被召回，
    而"版本过滤生效"这条断言会在**另一处**（检索侧）才失败。
    """
    await store.upsert(
        [
            _point(0, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, document_id="policy/ec@v1.0"),
            _point(1, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, document_id="policy/ec@v2.0"),
        ]
    )

    v1 = await store.search_dense(
        [1.0, 0, 0, 0],
        limit=9,
        chunk_filter=ChunkFilter(document_ids=("policy/ec@v1.0",)),
    )

    assert [h.chunk_id for h in v1] == ["chk_0000"]
    assert await store.count(ChunkFilter(document_ids=("policy/ec@v1.0",))) == 1


async def test_document_filter_sees_staging_before_publish(store: VectorStore) -> None:
    """入库冒烟要在**发布之前**看到自己刚写的那批（11.1 的 Retrieval Smoke Test）。

    这是唯一一处必须显式覆盖默认 `status=ACTIVE` 的读路径：
    烟测跑在 `status=PROCESSING` 阶段，用默认过滤条件会一条都查不到，
    于是"冒烟通过"变成"什么都没测"。
    """
    await store.upsert([_point(0, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, status="PROCESSING")])

    assert await store.count(ChunkFilter(document_ids=("doc_a",))) == 0
    assert await store.count(ChunkFilter(status="PROCESSING", document_ids=("doc_a",))) == 1


# ------------------------------------------------------------------ 写入语义


async def test_upsert_same_chunk_id_overwrites(store: VectorStore) -> None:
    """11.9 幂等的基础：同一个 `chunk_id` 重复入库是**覆盖**，不是追加。

    依赖的是 `chunk_point_id` 的确定性——改成自增序号会让这条断言失败，
    而在真实语料上重跑一次入库就产生一份重复证据。
    """
    await _seed(store)
    # 覆盖前：token 3 无人携带，稀疏路一条都召回不了
    assert await store.search_sparse({3: 1.0}, limit=9, chunk_filter=ChunkFilter()) == []

    await store.upsert([_point(0, dense=[0.0, 1.0, 0.0, 0.0], sparse={3: 1.0})])

    assert await store.count() == 3  # 覆盖而非追加
    hits = await store.search_sparse({3: 1.0}, limit=9, chunk_filter=ChunkFilter())
    assert [h.chunk_id for h in hits] == ["chk_0000"]
    assert hits[0].score > 0.0


async def test_delete_document_removes_only_that_document(store: VectorStore) -> None:
    await _seed(store)
    await store.upsert([_point(9, dense=[1.0, 0, 0, 0], sparse={1: 1.0}, document_id="doc_b")])

    await store.delete_document("doc_a")

    assert await store.count() == 1
    assert await store.count(ChunkFilter(status=None)) == 1


async def test_ensure_collection_rejects_dimension_change(store: VectorStore) -> None:
    """11.5：更换向量模型必须**新建 collection**，不得在原 collection 上混用维度。

    自动重建看起来很贴心，实际是把"换了 embedding 模型"的后果从
    "启动时报错、需要人工决策"降级成"老数据被静默删掉"。
    """
    with pytest.raises(ValueError, match="维度"):
        await store.ensure_collection(dim=DIM + 1)


async def test_payload_round_trips(store: VectorStore) -> None:
    """`text` 与 `chunk_id` 必须能取回来：证据要引用原文，正文不在 MySQL 里。"""
    await _seed(store)

    hits = await store.search_dense([1.0, 0, 0, 0], limit=1, chunk_filter=ChunkFilter())

    assert hits[0].chunk_id == "chk_0000"
    assert hits[0].payload["text"] == "第 0 条分块"
    assert hits[0].payload["document_id"] == "doc_a"


# ------------------------------------------------------------------ fetch


async def test_fetch_returns_chunks_not_scores(store: VectorStore) -> None:
    """`fetch` 是**按定位取块**，不是检索：没有查询、没有打分。

    所以它返回 `ChunkRecord` 而不是 `ScoredPoint`——后者的 `score` 填 0.0
    会被读成"和问题完全无关"，而这一路压根没有相关性可言。
    """
    await _seed(store)

    records = await store.fetch(ChunkFilter(document_ids=("doc_a",)), limit=10)

    assert sorted(r.chunk_id for r in records) == ["chk_0000", "chk_0001", "chk_0002"]
    assert records[0].text.startswith("第 ")
    assert records[0].payload["document_id"] == "doc_a"


async def test_fetch_honours_the_limit_without_paging(store: VectorStore) -> None:
    """`limit` 是**截断**不是分页：给几条就是几条，不返回游标。"""
    await _seed(store)

    records = await store.fetch(ChunkFilter(document_ids=("doc_a",)), limit=2)

    assert len(records) == 2


async def test_fetch_applies_the_same_filters_as_search(store: VectorStore) -> None:
    """过滤条件与检索**共用同一套语义**——两条路各写一份必然漂移。

    这里用状态位：默认的 `ChunkFilter` 只要 ACTIVE，PROCESSING 的必须取不到。
    """
    await store.upsert(
        [_point(9, dense=[1.0, 0.0, 0.0, 0.0], sparse={1: 1.0}, status="PROCESSING")]
    )

    default = await store.fetch(ChunkFilter(document_ids=("doc_a",)), limit=10)
    staging = await store.fetch(ChunkFilter(document_ids=("doc_a",), status="PROCESSING"), limit=10)

    assert "chk_0009" not in [r.chunk_id for r in default]
    assert [r.chunk_id for r in staging] == ["chk_0009"]


async def test_fetch_requires_a_document_scope(store: VectorStore) -> None:
    """**没有文档定位就取不到块**——这是它与"枚举全库"的分界线。

    空过滤在两个实现里都跑得通（Qdrant 侧就是全库扫描），
    所以这条只能靠显式拒绝来守；靠文档约定守不住，因为不报错。
    """
    with pytest.raises(ValueError, match="document_ids"):
        await store.fetch(ChunkFilter(), limit=10)
