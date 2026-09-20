"""向量库接入层（详细设计 11.5 在线检索 / 23.1.1 选型依据）。

**这一层存在的意义是把 Qdrant 关在一个文件里。** 详设 discipline 要求
「`infrastructure/` 层必须把外部组件隔离干净，使换实现的成本控制在一个文件内」——
TBC-05 的选型实测（四个候选项、两条被否决的路线）已经把这条纪律用掉了一次，
下一次换库时改的应该还是这一个文件，而不是散在 retriever / ingestion / 测试里的
几十处 `import qdrant_client`。

## 三个实现

| 实现 | 用途 | 特点 |
|---|---|---|
| `QdrantVectorStore` | 开发与生产（**服务端形态**） | 唯一的真实实现 |
| `InMemoryVectorStore` | 单元测试 | 真实执行 RRF 融合与标量过滤，不是打桩 |

**为什么替身要真算 RRF**：稀疏与稠密两路的融合排序是 11.7 的核心，
用一个「返回固定列表」的桩去测，测的只是「代码调了它」，
而融合逻辑本身（哪一路该排前面、阈值怎么截断）完全没被验证。
`InMemoryVectorStore` 用真实公式算，与 Qdrant 的排序在正常输入下一致。

**没有本地模式的实现**：`qdrant-client` 的 `path=` 本地模式单进程独占存储目录，
而本项目 `make run` 是 api + worker 双进程、`make ingest` 又是第三个。
把它做成一个可选项等于给后来者留一个必然踩的坑，所以这里不提供——
要本地跑就用 `make up` 起容器，它只占 100MB 量级。

## point id：为什么是确定性派生而不是自增

Qdrant 的 point id **只接受 uint64 或 UUID**，而 11.4 的 `chunk_id` 是
`chk_` 前缀的字符串，形状对不上，必须有一个映射。

选 `BLAKE2b(chunk_id, digest_size=8)` 而不是自增序号或随机 UUID，是因为它让
**「同一个 chunk 重复入库」天然幂等**：相同 `chunk_id` 必然得到相同 `point_id`，
`upsert` 直接覆盖同一条，不需要先查后写。这是 11.9 幂等设计的一部分，
不只是 id 生成方式——改掉它，重复入库就会产生重复证据。

碰撞概率：4000 个 chunk 下，64 位空间的生日碰撞概率约 4e-13，可忽略。
**用 BLAKE2b 而不是 MD5/SHA1**：不是为了抗攻击（这里不涉及安全），
而是它原生支持指定摘要长度，不必先算 32 字节再截断。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings

#: 稠密向量在 collection 里的字段名（11.5 的「命名向量」）
DENSE_VECTOR = "dense"
#: 稀疏向量字段名
SPARSE_VECTOR = "sparse"

#: 由 `chunk_id` 派生的 point id 是 BLAKE2b 摘要解释成的无符号整数
_POINT_ID_BYTES = 8


def chunk_point_id(chunk_id: str) -> int:
    """`chunk_id` → Qdrant 的 uint64 point id。

    **必须是纯函数**：同一个 `chunk_id` 在任何进程、任何时间都要得到同一个 id，
    否则 11.9 的「重复入库直接覆盖」会退化成「重复入库产生重复证据」，
    而且这种错误只在**第二次**入库时才显形，第一次测试完全看不出来。
    """
    digest = hashlib.blake2b(chunk_id.encode("utf-8"), digest_size=_POINT_ID_BYTES).digest()
    # Qdrant 的 point id 是无符号 64 位：必须用 unsigned 解释摘要，
    # 用 signed 解释会出现负数 id，而负数在 Qdrant 里会被当成非法值拒绝。
    return int.from_bytes(digest, byteorder="big", signed=False)


class VectorPoint(BaseModel):
    """一条待写入的 chunk（11.5 的字段设计的写入侧形态）。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    dense: list[float]
    #: `{token_id: weight}`，与 11.6 的稀疏表示一致
    sparse: dict[int, float]
    #: 标量字段（11.4 的 ChunkMetadata 里参与过滤的那些）
    payload: dict[str, Any] = Field(default_factory=dict)


class ScoredPoint(BaseModel):
    """检索结果的一条。`score` 的含义随检索方式变化（余弦 / 内积 / RRF 融合分）。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    score: float
    payload: dict[str, Any]


class ChunkRecord(BaseModel):
    """按条件取回的一块**本身**（不含分数、不含向量）。

    与 `ScoredPoint` 分开而不是复用它：那个模型的 `score` 是"这次检索给它的
    相关性"，而这里根本没有检索。填 `0.0` 会被读成"和问题完全无关"——
    那是个我们并不知道的结论（同 `RetrievedChunk.dense_score` 可空的理由）。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    payload: dict[str, Any]


class ChunkFilter(BaseModel):
    """检索前的标量过滤条件（11.7 第 3 步 / 11.4 的 ChunkMetadata）。

    **做成显式模型而不是直接暴露 Qdrant 的 `Filter` 对象**：过滤条件还要被
    `InMemoryVectorStore` 解释一遍，如果调用方直接构造 Qdrant 的 Filter，
    替身就没法执行同一套断言，契约测试也就退化成只测真实实现。
    这里的每个字段都对应 11.7 明确要求的一条过滤。

    Attributes:
        status: 文档状态。**默认就是 `ACTIVE`**——11.9 要求"失败版本不得被
            在线查询过滤条件命中"，把默认值设成不过滤等于把这条纪律交给调用方自觉。
        document_ids: 只在这些文档版本里找。值是 `logical_key@version`（见 11.5 的
            落地记录），**按版本而不是按 `logical_key` 过滤**——同名制度的两个版本
            共用 `logical_key`，按它过滤等于让 v1.0 与 v2.0 互相串味。
            三个调用方：入库冒烟（只找刚写进去的那批）、失败回滚（只删这一版）、
            11.7 第 8 步的邻近块扩展（只在同文档内扩）。
        document_types: 文档类型白名单，空表示不限。
        departments: 部门白名单，空表示不限。
        effective_at: 按生效区间过滤的基准日（`effective_from <= d <= effective_to`）。
            None 表示不限——**但入库阶段必须显式传**，见 11.7 第 3 步。
        roles: 用户角色。与 chunk 的 `allowed_roles` 求交集，空表示不限。
        classifications: 密级白名单（TBC-07 的 INTERNAL / CONFIDENTIAL），空表示不限。
    """

    model_config = ConfigDict(frozen=True)

    status: str | None = "ACTIVE"
    document_ids: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    departments: tuple[str, ...] = ()
    effective_at: date | None = None
    roles: tuple[str, ...] = ()
    classifications: tuple[str, ...] = ()

    def matches(self, payload: dict[str, Any]) -> bool:
        """替身实现用：判断一条 payload 是否通过全部条件。

        真实实现把这个判断下推给 Qdrant，这里保留一份 Python 版是为了让
        `InMemoryVectorStore` 与真实实现**对同一组用例给出同一组结果**。
        """
        if self.status is not None and payload.get("status") != self.status:
            return False
        if self.document_ids and payload.get("document_id") not in self.document_ids:
            return False
        if self.document_types and payload.get("document_type") not in self.document_types:
            return False
        if self.departments and payload.get("department") not in self.departments:
            return False
        if self.classifications and payload.get("classification") not in self.classifications:
            return False
        if self.effective_at is not None and not _in_effective_range(payload, self.effective_at):
            return False
        if self.roles:
            allowed = payload.get("allowed_roles") or []
            if not set(allowed) & set(self.roles):
                return False
        return True


def _in_effective_range(payload: dict[str, Any], moment: date) -> bool:
    """生效区间是**闭区间** `[effective_from, effective_to]`。

    与 `domain.evidence.TimeRange` 的半开区间**故意不同**：制度写的是
    "有效期至 2025-12-31"，那一天本身仍然有效；而 SQL 里的季度区间
    `[Q3_start, Q4_start)` 是半开的，否则 9/30 会被算两次。
    两处的语义各由各自的业务决定，不能为了"统一"而统一。
    """
    start = payload.get("effective_from")
    end = payload.get("effective_to")
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    if start is not None and moment < start:
        return False
    return not (end is not None and moment > end)


class VectorStore(Protocol):
    """向量库接口（11.5 / 11.7）。

    **没有「列出全部」这类方法**：需要枚举 chunk 时应查 MySQL 的
    `knowledge_document`，把它当目录服务用是错的方向（同 `ObjectStorage` 的理由）。
    """

    async def ensure_collection(self, *, dim: int) -> None:
        """建 collection（幂等）。`dim` 不匹配时**必须报错而不是重建**，见实现说明。"""
        ...

    async def upsert(self, points: Sequence[VectorPoint]) -> None: ...

    async def search_dense(
        self, query: Sequence[float], *, limit: int, chunk_filter: ChunkFilter
    ) -> list[ScoredPoint]: ...

    async def search_sparse(
        self, query: dict[int, float], *, limit: int, chunk_filter: ChunkFilter
    ) -> list[ScoredPoint]: ...

    async def hybrid_rrf(
        self,
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
        *,
        limit: int,
        chunk_filter: ChunkFilter,
        rrf_k: int = 60,
    ) -> list[ScoredPoint]: ...

    async def set_status(self, chunk_ids: Sequence[str], status: str) -> None:
        """批量改状态。这是 11.9「原子发布」的落点（staging → ACTIVE）。"""
        ...

    async def delete_document(self, document_id: str) -> None:
        """删掉某文档**版本**的全部 chunk。失败回滚、`--force` 重建都要用它。

        参数名沿用 11.4 的 `document_id`，但值是 `logical_key@version`
        （见 11.5 的落地记录）。**按版本而不是按 `logical_key` 删**是必须的：
        11.9 要删的是"这一版没发布成功的那批 point"，
        按 `logical_key` 删会顺手把同一制度的其它已发布版本一起删掉，
        而那个版本的文档记录仍然是 ACTIVE——检索侧从此少了一版，
        没有任何地方会报错。
        """
        ...

    async def fetch(self, chunk_filter: ChunkFilter, *, limit: int) -> list[ChunkRecord]:
        """按标量条件把块**取回来**——不是检索：没有查询、没有打分、没有阈值。

        存在的理由只有一个：11.7 第 ⑧ 步要「按定位取同文档同章节的相邻块」，
        而那句话的前提是先知道有哪些块。

        它与上面那句「没有『列出全部』这类方法」并不冲突：`chunk_filter` 是
        **必填**的，且实现要求 `document_ids` 非空——调用方必须先定位到一份文档，
        拿不到文档就一块也取不回来。把过滤做成可选，它就退化成"枚举全库"，
        而那件事该由 MySQL 的 `knowledge_document` 回答。

        ⚠️ `limit` 是**截断**不是分页。调用方要么给够，要么如实接受不完整结果——
        分页会引入"翻到第几页了"的状态，而这个接口的用途没有这个需求。
        """
        ...

    async def count(self, chunk_filter: ChunkFilter | None = None) -> int:
        """按条件计数（None 表示全库）。

        **它是接口的一部分，不是实现细节**：11.1 的入库冒烟要断言
        "这一版发布了几条"，`verify-corpus` 要断言缺陷注入的份数对得上。
        没有它，那两处就只能改成"检索一次看返回几条"，而检索受 limit 与
        阈值影响，断言会变得又脆又看不出意图。
        """
        ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------- Qdrant


class QdrantVectorStore:
    """Qdrant 服务端实现（TBC-05 选定的形态）。

    **只连服务端**：构造参数是 URL，连不上就报错，不会静默退化成嵌入式——
    静默降级在本地开发时很方便，到多进程部署时会变成一个只在
    `make run` 下才复现的诡异故障（见模块 docstring）。
    """

    def __init__(self, url: str, collection: str) -> None:
        self._url = url
        self._collection = collection
        self._client = AsyncQdrantClient(url=url)
        self._dim: int | None = None

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self, *, dim: int) -> None:
        """建 collection（幂等），并为过滤字段建 payload 索引。

        **维度不一致时直接报错，绝不重建**：11.5 明写"更换向量模型必须新建
        collection 全量重建，禁止混用维度或模型版本"。自动重建看起来贴心，
        实际是把「换了 embedding 模型」这件事的后果从"启动失败、需要人工决策"
        降级成"老数据被静默删掉"——而后者要等到检索结果变差很久以后才会被发现。
        """
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            existing = _dense_dim_of(info)
            if existing is not None and existing != dim:
                raise ValueError(
                    f"collection {self._collection} 的 dense 维度是 {existing}，"
                    f"而当前 EMBEDDING_DIM 是 {dim}。按详细设计 11.5，"
                    "更换向量模型必须**新建 collection** 并全量重建，"
                    "不得在原 collection 上混用维度。"
                )
            self._dim = existing or dim
            return

        await self._client.create_collection(
            self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        self._dim = dim
        await self._ensure_payload_indexes()

    async def _ensure_payload_indexes(self) -> None:
        """给参与过滤的标量字段建索引（11.4 的 ChunkMetadata）。

        没有索引 Qdrant 也能过滤，只是退化成全量扫描。80+ 文档时两者都够快，
        但**建索引的代价是一次性的，而漏建的表现是"语料涨上去以后检索变慢"**——
        那时没人会想到是这里。
        """
        keyword_fields = (
            "status",
            "document_type",
            "department",
            "classification",
            "document_id",
        )
        for field in keyword_fields:
            await self._client.create_payload_index(
                self._collection, field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        # 生效区间用 DatetimeRange 过滤，见 `_build_filter`
        await self._client.create_payload_index(
            self._collection, "effective_from", field_schema=models.PayloadSchemaType.DATETIME
        )
        await self._client.create_payload_index(
            self._collection, "effective_to", field_schema=models.PayloadSchemaType.DATETIME
        )

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        await self._client.upsert(
            self._collection,
            points=[_to_point_struct(p) for p in points],
            wait=True,
        )

    async def search_dense(
        self, query: Sequence[float], *, limit: int, chunk_filter: ChunkFilter
    ) -> list[ScoredPoint]:
        result = await self._client.query_points(
            self._collection,
            query=list(query),
            using=DENSE_VECTOR,
            limit=limit,
            query_filter=_build_filter(chunk_filter),
            with_payload=True,
        )
        return _to_scored(result.points)

    async def search_sparse(
        self, query: dict[int, float], *, limit: int, chunk_filter: ChunkFilter
    ) -> list[ScoredPoint]:
        result = await self._client.query_points(
            self._collection,
            query=_to_sparse(query),
            using=SPARSE_VECTOR,
            limit=limit,
            query_filter=_build_filter(chunk_filter),
            with_payload=True,
        )
        return _to_scored(result.points)

    async def hybrid_rrf(
        self,
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
        *,
        limit: int,
        chunk_filter: ChunkFilter,
        rrf_k: int = 60,
    ) -> list[ScoredPoint]:
        """双路召回 + RRF 融合（11.7 第 4–5 步）。

        **两路的 limit 由调用方从配置给足**（默认各 20），融合后的 `limit`
        才是最终候选数——把两路都设成最终值会让"每路多召回一些、由融合来筛"
        这个前提失效，融合就只是在两个已经很窄的集合里挑。
        """
        result = await self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(
                    query=list(dense_query),
                    using=DENSE_VECTOR,
                    limit=limit,
                    filter=_build_filter(chunk_filter),
                ),
                models.Prefetch(
                    query=_to_sparse(sparse_query),
                    using=SPARSE_VECTOR,
                    limit=limit,
                    filter=_build_filter(chunk_filter),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return _to_scored(result.points)

    async def set_status(self, chunk_ids: Sequence[str], status: str) -> None:
        if not chunk_ids:
            return
        await self._client.set_payload(
            self._collection,
            payload={"status": status},
            points=[chunk_point_id(cid) for cid in chunk_ids],
            wait=True,
        )

    async def delete_document(self, document_id: str) -> None:
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def count(self, chunk_filter: ChunkFilter | None = None) -> int:
        """按过滤条件计数。入库冒烟与 `verify-corpus` 用它断言"发布了几条"。"""
        result = await self._client.count(
            self._collection,
            count_filter=_build_filter(chunk_filter) if chunk_filter else None,
            exact=True,
        )
        return result.count

    async def fetch(self, chunk_filter: ChunkFilter, *, limit: int) -> list[ChunkRecord]:
        _require_document_scope(chunk_filter)
        records, _ = await self._client.scroll(
            self._collection,
            scroll_filter=_build_filter(chunk_filter),
            limit=limit,
            # 取 payload 不取向量：调用方要的是正文与标量字段，而向量
            # 每次都是 1024 维浮点——白搬一趟。
            with_payload=True,
            with_vectors=False,
        )
        return [_to_record(dict(item.payload or {})) for item in records]

    async def aclose(self) -> None:
        await self._client.close()

    def __repr__(self) -> str:
        # 不带 URL：它可能含凭据，而 repr 会进日志（19.4 脱敏纪律）
        return f"QdrantVectorStore(collection={self._collection!r})"


def _require_document_scope(chunk_filter: ChunkFilter) -> None:
    """`fetch` 的前置条件：必须定位到文档。**两个实现共用这一条**。

    不放在 `fetch` 的文档里当约定，是因为约定不会报错：一个空的
    `ChunkFilter()` 在 Qdrant 侧就是"全库扫描"，而它跑起来完全正常，
    只是把"枚举全库"这件事悄悄做成了——那正是 Protocol 明写不要的方向。
    """
    if not chunk_filter.document_ids:
        raise ValueError(
            "fetch 必须带 document_ids：它是「按定位取块」，不是枚举接口。"
            "要枚举 chunk 请查 MySQL 的 knowledge_document。"
        )


def _to_record(payload: dict[str, Any]) -> ChunkRecord:
    """payload → `ChunkRecord`。`chunk_id` / `text` 写在 payload 里（见 `_to_point_struct`）。"""
    return ChunkRecord(
        chunk_id=str(payload.get("chunk_id", "")),
        text=str(payload.get("text", "")),
        payload=payload,
    )


def _to_point_struct(point: VectorPoint) -> models.PointStruct:
    payload = dict(point.payload)
    payload["chunk_id"] = point.chunk_id
    payload["text"] = point.text
    return models.PointStruct(
        id=chunk_point_id(point.chunk_id),
        vector={
            DENSE_VECTOR: list(point.dense),
            SPARSE_VECTOR: _to_sparse(point.sparse),
        },
        payload=payload,
    )


def _to_sparse(values: dict[int, float]) -> models.SparseVector:
    """`{token_id: weight}` → Qdrant 的稀疏向量。

    **按 token_id 排序再拆分**：Qdrant 要求 indices 严格递增，乱序会被拒。
    字典的插入顺序取决于分词顺序，直接 `list(values)` 在多数情况下碰巧有序，
    但那只是巧合——一旦哪个环节改成并行分词就会变成间歇性报错。
    """
    indices = sorted(values)
    return models.SparseVector(indices=indices, values=[values[i] for i in indices])


def _to_scored(points: Sequence[Any]) -> list[ScoredPoint]:
    out: list[ScoredPoint] = []
    for p in points:
        payload = dict(p.payload or {})
        out.append(
            ScoredPoint(
                chunk_id=str(payload.get("chunk_id", "")),
                score=float(p.score),
                payload=payload,
            )
        )
    return out


def _build_filter(chunk_filter: ChunkFilter) -> models.Filter:
    """`ChunkFilter` → Qdrant 的 Filter。

    `effective_at` 用一个 `DatetimeRange` 表达"这一刻在生效区间内"，
    但**区间两端不是同一种条件**，所以拆成两条 must：
      - `effective_from <= 基准日`：空值视为**不限制**（制度没有起始日 = 一直有效）
      - `effective_to >= 基准日 或 为空`：空值 = 无终止日

    用 `should`（或）表达"为空"这一支，是 Qdrant 里表达可空字段范围过滤的常规写法。
    """
    must: list[models.Condition] = []
    if chunk_filter.status is not None:
        must.append(
            models.FieldCondition(key="status", match=models.MatchValue(value=chunk_filter.status))
        )
    for key, values in (
        ("document_id", chunk_filter.document_ids),
        ("document_type", chunk_filter.document_types),
        ("department", chunk_filter.departments),
        ("classification", chunk_filter.classifications),
    ):
        if values:
            must.append(models.FieldCondition(key=key, match=models.MatchAny(any=list(values))))
    if chunk_filter.roles:
        # allowed_roles 是数组字段，`MatchAny` 在数组上表示"任一无素命中"
        must.append(
            models.FieldCondition(
                key="allowed_roles", match=models.MatchAny(any=list(chunk_filter.roles))
            )
        )
    if chunk_filter.effective_at is not None:
        moment = _as_datetime(chunk_filter.effective_at)
        # **区间两端都要带上"为空即无界"这一支**：制度可能只写起始日、
        # 也可能只写终止日，还可能两个都不写（长期有效）。
        # 直接用 `DatetimeRange` 会在字段缺失时判 false，把"长期有效"的文档
        # 误伤成"已过期"——这个错误只在有文档缺日期时显形，
        # 而缺日期恰恰是真实语料里最常见的形态。
        # 这条分歧也是契约测试在两个实现上跑同一套断言时暴露的。
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="effective_from", is_null=True),
                    models.FieldCondition(
                        key="effective_from", range=models.DatetimeRange(lte=moment)
                    ),
                ]
            )
        )
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="effective_to", is_null=True),
                    models.FieldCondition(
                        key="effective_to", range=models.DatetimeRange(gte=moment)
                    ),
                ]
            )
        )
    return models.Filter(must=must or None)


def _as_datetime(day: date) -> str:
    """日期 → Qdrant 的 `DatetimeRange` 要的 RFC3339 字符串。

    **取当天零点**：`effective_to = 2025-12-31` 在 Qdrant 里若按零点比较，
    "12-31 当天"会因为 `00:00 > 00:00` 不成立而被判为已过期——
    而这里用的是 `gte`，零点相等成立，所以当天仍然有效，与 `_in_effective_range`
    的闭区间语义一致。两者必须一起改，否则真实实现与替身会对同一天给出不同答案。
    """
    return f"{day.isoformat()}T00:00:00Z"


def _dense_dim_of(info: Any) -> int | None:
    """从 collection 信息里读出 dense 的维度。

    Qdrant 的返回结构随版本变化（`vectors` 可能是 dict 也可能是对象），
    这里做防御性读取；读不到就返回 None，由调用方决定是否报错——
    **比猜一个值然后写进断言强**。
    """
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict):
        dense = vectors.get(DENSE_VECTOR)
        return getattr(dense, "size", None)
    return None


# ------------------------------------------------------------------ 内存实现


class InMemoryVectorStore:
    """单元测试用的替身，**真实执行检索与融合**。

    它不是打桩：稠密相似度、稀疏点积、RRF 融合、标量过滤都在这里真算。
    这样契约测试断言的是"检索行为"，而不是"调用次数"——后者在实现被改坏时
    往往照样通过。

    排序与 Qdrant 的一致性：余弦相似度、点积、RRF 三者的排序在无并列时一致；
    **有并列时的次序可能不同**，因此断言应写"命中了什么"，不写"第几条是什么"。
    """

    def __init__(self) -> None:
        self._points: dict[int, VectorPoint] = {}
        self._dim: int | None = None

    async def ensure_collection(self, *, dim: int) -> None:
        if self._dim is not None and self._dim != dim:
            raise ValueError(f"维度不一致：已有 {self._dim}，传入 {dim}")
        self._dim = dim

    async def upsert(self, points: Sequence[VectorPoint]) -> None:
        for point in points:
            # 用与真实实现同一个派生函数，保证"重复 chunk_id 覆盖同一条"的行为一致
            self._points[chunk_point_id(point.chunk_id)] = point

    async def search_dense(
        self, query: Sequence[float], *, limit: int, chunk_filter: ChunkFilter
    ) -> list[ScoredPoint]:
        scored = [
            ScoredPoint(chunk_id=p.chunk_id, score=_cosine(query, p.dense), payload=_payload(p))
            for p in self._candidates(chunk_filter)
        ]
        return _top(scored, limit)

    async def search_sparse(
        self, query: dict[int, float], *, limit: int, chunk_filter: ChunkFilter
    ) -> list[ScoredPoint]:
        """稀疏检索**只返回至少共享一个 token 的候选**。

        这不是优化，是必须与真实实现对齐的行为：Qdrant 的稀疏路走倒排索引，
        没有共同 token 的文档**根本不在候选集里**，所以不会返回零分项。
        第一版替身把全部点都算一遍再排序，于是"零分项也返回"，
        稠密路与稀疏路在**候选集合**上就不一致了——
        这个分歧是契约测试在两个实现上跑同一套断言时暴露出来的。
        """
        scored = [
            ScoredPoint(chunk_id=p.chunk_id, score=score, payload=_payload(p))
            for p in self._candidates(chunk_filter)
            if (score := _dot(query, p.sparse)) > 0.0
        ]
        return _top(scored, limit)

    async def hybrid_rrf(
        self,
        dense_query: Sequence[float],
        sparse_query: dict[int, float],
        *,
        limit: int,
        chunk_filter: ChunkFilter,
        rrf_k: int = 60,
    ) -> list[ScoredPoint]:
        """与 Qdrant 的 `Fusion.RRF` 同公式：`score = Σ 1/(k + rank)`，rank 从 1 起。"""
        dense_hits = await self.search_dense(dense_query, limit=limit, chunk_filter=chunk_filter)
        sparse_hits = await self.search_sparse(sparse_query, limit=limit, chunk_filter=chunk_filter)
        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                payloads[hit.chunk_id] = hit.payload
        fused = [
            ScoredPoint(chunk_id=cid, score=score, payload=payloads[cid])
            for cid, score in scores.items()
        ]
        return _top(fused, limit)

    async def fetch(self, chunk_filter: ChunkFilter, *, limit: int) -> list[ChunkRecord]:
        _require_document_scope(chunk_filter)
        # 顺序按 point id 排：两个实现的次序不必一致（同 `_top` 的说明），
        # 但**同一个实现重复调用要一致**——不一致会让"这一批取回来的第几条"
        # 变成不可复现的事实，而调用方会拿它当定位用。
        records = [
            _to_record(_payload(p))
            for pid, p in sorted(self._points.items())
            if chunk_filter.matches(_payload(p))
        ]
        return records[:limit]

    async def set_status(self, chunk_ids: Sequence[str], status: str) -> None:
        for chunk_id in chunk_ids:
            pid = chunk_point_id(chunk_id)
            point = self._points.get(pid)
            if point is not None:
                self._points[pid] = point.model_copy(
                    update={"payload": {**point.payload, "status": status}}
                )

    async def delete_document(self, document_id: str) -> None:
        doomed = [
            pid for pid, p in self._points.items() if p.payload.get("document_id") == document_id
        ]
        for pid in doomed:
            del self._points[pid]

    async def count(self, chunk_filter: ChunkFilter | None = None) -> int:
        if chunk_filter is None:
            return len(self._points)
        return len(self._candidates(chunk_filter))

    async def aclose(self) -> None:
        return None

    def _candidates(self, chunk_filter: ChunkFilter) -> list[VectorPoint]:
        return [p for p in self._points.values() if chunk_filter.matches(_payload(p))]


def _payload(point: VectorPoint) -> dict[str, Any]:
    """替身里的 payload **必须与真实实现写入的形状一致**，否则过滤断言测的是幻觉。"""
    return {**point.payload, "chunk_id": point.chunk_id, "text": point.text}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)


def _dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(token, 0.0) for token, weight in a.items())


def _top(scored: list[ScoredPoint], limit: int) -> list[ScoredPoint]:
    """按分数降序取前 `limit`。

    **并列时按 chunk_id 兜底排序**：只按分数排的话，并列项的顺序取决于
    字典遍历顺序，同一个测试会时过时不过。补一个确定性的次序，
    让"不稳定"这件事在测试里立刻可见，而不是偶发地闪。
    """
    return sorted(scored, key=lambda s: (-s.score, s.chunk_id))[:limit]


def build_vector_store(settings: Settings) -> VectorStore:
    """装配点。**唯一一处把配置与具体实现绑在一起的地方。**"""
    return QdrantVectorStore(settings.qdrant_url, settings.rag.collection)


__all__ = [
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "ChunkFilter",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "ScoredPoint",
    "VectorPoint",
    "VectorStore",
    "build_vector_store",
    "chunk_point_id",
]
