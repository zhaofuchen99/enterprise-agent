"""词表构建与装载（详细设计 11.6.3 / 11.6.4）。

`tokenizer.py` 里的 `Vocabulary` 是**数据结构**（token 映射 + 冻结 IDF）；
这里的 `build_vocabulary` 是**构建过程**：扫一遍语料、统计 df、给新词分配 id。

## 为什么构建是独立的一步，而不是"入库时顺手加"

11.6.4 选了**固定 IDF**：稀疏向量在入库时就被固定下来，而 BM25 的 IDF
依赖全库统计。两者要同时成立，只有一条路——**先把词表冻结，再入库**。
如果每入一篇文档就顺手把新词加进去、顺手改 df，那么：

- 先入库的 chunk 用的是当时那份 df，后入库的用新的，
  两批向量的权重不可比，而它们在同一次检索里会被一起打分；
- 更糟的是这个偏差**不报错**。它表现为"后入库的文档莫名其妙更容易被召回"，
  排查会先怀疑 embedding 模型，再怀疑分词，最后才想到 df。

所以构建是显式的一步（`make vocab`），产物是冻结的快照。
增量入库遇到未登录词时**照常分配新 id**（`existing` 参数）——
只增不改的语义保证已发布的 chunk 不受影响，这正是 11.6.3 的设计。

## 分配顺序为什么是"高频优先"

新词的 id 按 `(-df, token)` 排序后依次分配。排序本身是硬要求——
不排序的话两次构建会给出两套 id，而**词表是入库与查询共用的映射**，
两边拿到不同的 id，点积恒为 0，表现为"检索永远召回不到东西"，
而两边各自的代码看起来都对。

在"确定性"之外再按 df 降序，是为了让 `ORDER BY token_id` 大致按重要性排列：
排查词表时先看到的是"渠道折扣"而不是某个只出现一次的噪声词。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from app.core.errors import AgentError, ErrorCode
from app.domain.vocab import VocabEntry
from app.infrastructure.cache import VersionedCache
from app.infrastructure.storage import ObjectStorage
from app.repositories.vocab_repo import VocabRepository
from app.tools.rag.tokenizer import SNAPSHOT_FORMAT_VERSION, Tokenizer, Vocabulary


@dataclass(frozen=True, slots=True)
class BuildResult:
    """一次构建的结果。

    **是 dataclass 而不是 Pydantic 模型**：它装着 `Vocabulary`，
    而 `Vocabulary` 有预计算的 IDF 字典、不适合序列化，也不该被序列化——
    它是内存里的检索结构，落到磁盘上的形态是 `VocabSnapshot`。
    做成 Pydantic 模型会诱使别人把它当传输对象用，然后在
    "为什么 IDF 没跟着走"上花掉半天。

    Attributes:
        vocabulary: 合并了历史词条与本次新增的完整词表（入库/查询都用它）。
        added: **本次新增**的词条，只有这些需要落库。
        total_chunks: 参与统计的 chunk 数，即 IDF 的分母。
    """

    vocabulary: Vocabulary
    added: tuple[VocabEntry, ...]
    total_chunks: int


def build_vocabulary(
    texts: Iterable[str],
    *,
    tokenizer: Tokenizer,
    sparse_dim: int,
    existing: Sequence[VocabEntry] = (),
) -> BuildResult:
    """扫一遍文本、统计 df、把新词接到已有的 id 序列后面。

    Args:
        texts: 待统计的文本（本阶段是 chunk 的 `text`）。
            **同一段文本内重复出现的 token 只计一次 df**——
            df 是"含该词的单元数"，不是词频。
        tokenizer: 分词器。**必须与入库、查询用的是同一个实例配置**：
            词典不同则切分不同，词表统计的就不是检索时真正会用的那批 token。
        sparse_dim: 稀疏向量维度上限。超出即报错而不是截断，
            理由见 `tokenizer.build_sparse`。
        existing: 已发布的词条。它们的 id 与 df **原样沿用**（只增不改）。

    Raises:
        ValueError: 新词分配出的 id 超过 `sparse_dim`。
    """
    known = {entry.token: entry for entry in existing}
    df: Counter[str] = Counter()
    total = 0

    for text in texts:
        total += 1
        # set() 而不是逐次累加：df 数的是"含该词的单元数"，
        # 一个词在同一 chunk 里出现十次仍然只算一个单元
        df.update(set(tokenizer.cut(text)))

    new_tokens = [token for token in df if token not in known]
    next_id = max((entry.token_id for entry in known.values()), default=-1) + 1
    # 排序是硬要求（见模块 docstring）：先按 df 降序，同频再按词形保证稳定
    ordered = sorted(new_tokens, key=lambda token: (-df[token], token))

    added: list[VocabEntry] = []
    for offset, token in enumerate(ordered):
        token_id = next_id + offset
        if token_id >= sparse_dim:
            raise ValueError(
                f"词表超出稀疏向量维度 {sparse_dim}（token={token!r} 分配到 {token_id}）。"
                "rag.sparse_dim 需要扩容——扩容后历史 chunk 的稀疏向量无需重算，"
                "因为已分配的 token_id 不变（11.6.3）。"
            )
        added.append(VocabEntry(token=token, token_id=token_id, df=df[token]))

    merged = {**known}
    for entry in added:
        merged[entry.token] = entry
    vocabulary = Vocabulary(
        {entry.token: entry.token_id for entry in merged.values()},
        document_count=total,
        df={entry.token: entry.df for entry in merged.values()},
    )
    return BuildResult(vocabulary=vocabulary, added=tuple(added), total_chunks=total)


class VocabSnapshot(BaseModel):
    """快照的 Pydantic 形态。

    `Vocabulary.to_snapshot()` 返回的是裸 dict——它能直接写文件，
    但不能交给 `VersionedCache`（它要求 `BaseModel` 才能做读回校验）。
    多这一层的收益是：Redis 里的老快照配上新代码时，
    **校验失败会被当成缓存未命中**而不是一路带进检索（见 `cache.py` 的说明）。
    """

    model_config = ConfigDict(extra="ignore")

    format_version: int = SNAPSHOT_FORMAT_VERSION
    document_count: int
    tokens: dict[str, dict[str, int]]


def snapshot_of(vocabulary: Vocabulary) -> VocabSnapshot:
    return VocabSnapshot.model_validate(vocabulary.to_snapshot())


# ------------------------------------------------------------------ 快照的存与取


def snapshot_key(version: str) -> str:
    """快照在对象存储里的路径。

    `system/` 前缀把它与知识文档（`knowledge/{logical_key}/{version}/`）分开：
    词表不是某一份文档的附件，它是**整套索引的口径**。
    混在 `knowledge/` 下会让"按 logical_key 清理某份文档"误删词表快照。
    """
    return f"system/vocab/{version}/vocab.json"


async def export_snapshot(storage: ObjectStorage, vocabulary: Vocabulary, *, version: str) -> str:
    """把词表快照写到对象存储，返回它的 key。

    **快照是运行时的读取入口**，这一点值得说清楚：MySQL 的 `rag_vocab`
    是词条账本（只增不改的 id 序列 + df），但 IDF 的分母（chunk 总数）
    不在那张表里——11.6.3 定义的列就是 `token / token_id / df / created_at`。
    硬要从表里推分母，只能拿"当前 chunk 数"顶替，而词表冻结之后再入库新文档，
    两者必然分叉，于是查询侧与入库侧的 IDF 不可比。

    所以读路径是 **Redis → 快照**，MySQL 只在构建与增量分配时用到。
    快照缺失就报错提示重跑 `make vocab`，**不做"尽力而为"的降级**：
    用一个猜出来的分母建词表，得到的是一套不报错但分数不可比的向量。
    """
    key = snapshot_key(version)
    # 走 `VocabSnapshot` 而不是 `json.dumps(vocabulary.to_snapshot())`：
    # 写出去的形态必须与 `_read_snapshot` 校验的形态是同一个模型，
    # 否则"写的时候多一个字段、读的时候少一个"这类不一致只能等运行时报出来
    payload = snapshot_of(vocabulary).model_dump_json()
    await storage.put(key, payload.encode("utf-8"), content_type="application/json")
    return key


async def load_vocabulary(
    repository: VocabRepository,
    cache: VersionedCache,
    storage: ObjectStorage,
    *,
    ttl_seconds: int,
) -> Vocabulary:
    """装载当前词表。命中顺序：Redis → 对象存储快照。

    **不读 MySQL**：见 `export_snapshot` 的说明。`repository` 只用来取版本指纹，
    它是"这次该用哪份快照"的判据——快照的路径里带着版本，
    版本变了自然指向新文件，这正是 4.4 纪律 1 要的"按版本键自然失效"。
    """
    version = await repository.version()
    snapshot = await cache.get_or_set(
        "vocab",
        version,
        lambda: _read_snapshot(storage, version),
        model=VocabSnapshot,
        ttl_seconds=ttl_seconds,
    )
    return Vocabulary.from_snapshot(snapshot.model_dump())


async def _read_snapshot(storage: ObjectStorage, version: str) -> VocabSnapshot:
    key = snapshot_key(version)
    try:
        raw = await storage.get(key)
    except Exception as exc:
        raise AgentError(
            ErrorCode.INTERNAL_ERROR,
            "知识库词表快照缺失，无法建立检索索引",
            details={"key": key, "error": type(exc).__name__},
        ) from exc
    return VocabSnapshot.model_validate_json(raw)


__all__ = [
    "BuildResult",
    "VocabSnapshot",
    "build_vocabulary",
    "export_snapshot",
    "load_vocabulary",
    "snapshot_key",
    "snapshot_of",
]
