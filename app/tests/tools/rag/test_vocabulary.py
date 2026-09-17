"""词表构建与装载（详细设计 11.6.3 / 11.6.4）。

构建这一层最容易出的错**不报错**：两次构建给出两套 `token_id`、
df 被后一次覆盖、IDF 的分母在入库与查询两侧不同——三者都表现为
"某些查询召回不到东西"，而两边的代码看起来都对。
下面的用例基本是冲着这几种安静失败去的。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import jieba
import pytest

from app.core.errors import AgentError, ErrorCode
from app.domain.vocab import VocabEntry
from app.infrastructure.cache import VersionedCache
from app.infrastructure.storage import LocalObjectStorage
from app.repositories.vocab_repo import InMemoryVocabRepository
from app.tools.rag.tokenizer import Tokenizer, Vocabulary, build_sparse
from app.tools.rag.vocabulary import (
    build_vocabulary,
    export_snapshot,
    load_vocabulary,
    snapshot_key,
    snapshot_of,
)


@pytest.fixture(autouse=True)
def _isolate_jieba() -> Iterator[None]:
    """恢复 jieba 的进程级词典（理由见 `test_tokenizer.py` 的同名夹具）。"""
    saved = dict(jieba.dt.FREQ)
    saved_paths = set(Tokenizer._loaded_paths)
    yield
    jieba.dt.FREQ.clear()
    jieba.dt.FREQ.update(saved)
    Tokenizer._loaded_paths.clear()
    Tokenizer._loaded_paths.update(saved_paths)


def _tokenizer() -> Tokenizer:
    """用**仓库里那份业务词典**，与 `make vocab` 走同一条链路。

    不自己现造一份小词典：词表的 token 就是分词切出来的东西，
    两边用不同的词典，用例验的就不是真实链路了。
    """
    return Tokenizer(user_dict_path="configs/rag_user_dict.txt")


# ------------------------------------------------------------------ 构建


def test_build_assigns_contiguous_ids_from_zero() -> None:
    result = build_vocabulary(
        ["华东区域的渠道折扣政策", "华南区域的渠道折扣政策"],
        tokenizer=_tokenizer(),
        sparse_dim=1000,
    )

    ids = sorted(entry.token_id for entry in result.added)
    assert ids == list(range(len(result.added)))


def test_build_is_deterministic() -> None:
    """同样的输入必须给出同样的映射。

    **这是最要紧的一条**：词表是入库与查询共用的映射。
    两次构建给出两套 id 的话，查询侧算出的稀疏向量与入库侧的对不上——
    点积恒为 0，表现为"检索永远召回不到东西"，而两边代码看起来都对。
    """
    texts = ["华东区域的渠道折扣政策", "华南区域的渠道折扣政策"]

    first = build_vocabulary(texts, tokenizer=_tokenizer(), sparse_dim=1000)
    second = build_vocabulary(texts, tokenizer=_tokenizer(), sparse_dim=1000)

    assert first.vocabulary.to_snapshot() == second.vocabulary.to_snapshot()


def test_df_counts_chunks_not_occurrences() -> None:
    """df 数的是**含该词的 chunk 数**，不是词频。

    同一个词在一段里出现十次仍然只算一个单元。按出现次数统计的话，
    一句反复强调的话会让它的 df 虚高、IDF 虚低——而那些"强调"恰恰是
    作者认为重要的内容。
    """
    result = build_vocabulary(
        ["渠道折扣渠道折扣渠道折扣", "别的内容"],
        tokenizer=_tokenizer(),
        sparse_dim=1000,
    )

    entries = {entry.token: entry.df for entry in result.added}
    assert entries["渠道折扣"] == 1


def test_total_chunks_is_the_idf_denominator() -> None:
    result = build_vocabulary(
        ["甲的内容", "乙的内容", "丙的内容"], tokenizer=_tokenizer(), sparse_dim=1000
    )

    assert result.total_chunks == 3
    assert result.vocabulary.document_count == 3


def test_existing_entries_are_kept_untouched() -> None:
    """只增不改：已有词条的 id 与 df 原样沿用。

    这条守住的是 11.6.3 的核心承诺——**词表扩容后历史 chunk 无需重算**。
    """
    existing = [VocabEntry(token="渠道折扣", token_id=7, df=99)]

    result = build_vocabulary(
        ["渠道折扣与新的术语"],
        tokenizer=_tokenizer(),
        sparse_dim=1000,
        existing=existing,
    )

    assert result.vocabulary.id_of("渠道折扣") == 7
    assert result.vocabulary.idf("渠道折扣") == Vocabulary(
        {"渠道折扣": 7}, document_count=1, df={"渠道折扣": 99}
    ).idf("渠道折扣")
    assert all(entry.token != "渠道折扣" for entry in result.added)


def test_new_ids_start_after_the_highest_existing() -> None:
    """新 id 从**最大已用 id + 1** 开始，而不是从 0。

    从 0 开始的话新词会把老词的 id 抢走（或撞上唯一约束被跳过）——
    两种结果都让"扩容不重算"失效。
    """
    existing = [
        VocabEntry(token="华东", token_id=3, df=1),
        VocabEntry(token="华南", token_id=9, df=1),
    ]

    result = build_vocabulary(
        ["华东与华南与西南"], tokenizer=_tokenizer(), sparse_dim=1000, existing=existing
    )

    assert [entry.token for entry in result.added] == ["西南"]
    assert result.added[0].token_id == 10


def test_entry_counts_do_not_decrease() -> None:
    """增量构建之后，老 chunk 的 id **仍然全部存在**。

    这是"扩容不重算"的直接断言：取老向量用到的 id 集合，
    在新词表下逐个查——有一个消失，那些 chunk 就永久召回不到了。
    """
    old = build_vocabulary(["华东区域的渠道折扣政策"], tokenizer=_tokenizer(), sparse_dim=1000)
    old_sparse = build_sparse(["华东", "渠道折扣"], old.vocabulary, dim=1000)

    grown = build_vocabulary(
        ["华东区域的渠道折扣政策", "全新的产品线术语"],
        tokenizer=_tokenizer(),
        sparse_dim=1000,
        existing=list(old.added),
    )

    assert all(tid in grown.vocabulary._token_ids.values() for tid in old_sparse)


def test_exceeding_sparse_dim_raises() -> None:
    """词表超出维度时**报错**，不静默丢弃。

    静默丢弃的表现是"这个词检索不到"，排查方向会先跑到分词上去。
    """
    with pytest.raises(ValueError, match="稀疏向量维度"):
        build_vocabulary(["甲乙丙丁戊己庚辛"], tokenizer=_tokenizer(), sparse_dim=1)


# ------------------------------------------------------------------ 快照


def test_snapshot_round_trips_through_the_model() -> None:
    result = build_vocabulary(["华东区域的渠道折扣政策"], tokenizer=_tokenizer(), sparse_dim=1000)

    restored = Vocabulary.from_snapshot(snapshot_of(result.vocabulary).model_dump())

    assert restored.to_snapshot() == result.vocabulary.to_snapshot()


def test_snapshot_key_is_namespaced_away_from_documents() -> None:
    """快照放在 `system/` 下，不混进 `knowledge/`。

    混进去的话，「按 logical_key 清理某一份文档」会连词表一起删掉——
    而词表不是任何一份文档的附件，它是整套索引的口径。
    """
    assert snapshot_key("3-2") == "system/vocab/3-2/vocab.json"


async def test_export_then_load_round_trips(tmp_path: Path) -> None:
    storage = LocalObjectStorage(tmp_path)
    result = build_vocabulary(["华东区域的渠道折扣政策"], tokenizer=_tokenizer(), sparse_dim=1000)
    repository = InMemoryVocabRepository(result.added)
    version = await repository.version()

    await export_snapshot(storage, result.vocabulary, version=version)
    loaded = await load_vocabulary(
        repository, VersionedCache(_NoCache(), default_ttl_seconds=60), storage, ttl_seconds=60
    )

    assert loaded.to_snapshot() == result.vocabulary.to_snapshot()


async def test_load_fails_loudly_when_the_snapshot_is_missing(tmp_path: Path) -> None:
    """快照缺失时**报错**，不做"尽力而为"的降级。

    降级的做法是拿"当前 chunk 数"顶替 IDF 的分母，而词表冻结之后再入库新文档，
    两者必然分叉——得到的是一套不报错、但查询侧与入库侧分数不可比的向量。
    """
    storage = LocalObjectStorage(tmp_path)

    with pytest.raises(AgentError) as excinfo:
        await load_vocabulary(
            InMemoryVocabRepository(),
            VersionedCache(_NoCache(), default_ttl_seconds=60),
            storage,
            ttl_seconds=60,
        )

    assert excinfo.value.code == ErrorCode.INTERNAL_ERROR


class _NoCache:
    """永远未命中的 Redis 替身。

    用真 `fakeredis` 也行，但这里要验的是**装载链路**而不是缓存行为，
    未命中替身让"每次都回源"成为确定的事实——否则用例的结果会取决于
    上一次跑留下的键。
    """

    async def get(self, name: str) -> None:
        return None

    async def set(self, name: str, value: str, *, ex: int | None = None) -> None:
        return None
