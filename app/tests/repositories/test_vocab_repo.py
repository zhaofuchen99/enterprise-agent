"""词表仓储契约（详细设计 11.6.3）。

**同一份断言跑两个实现**，与任务仓储同样的规矩：只测 SQL 实现的话内存实现会
悄悄烂掉，只测内存实现的话真正上线的那份没被验证过。

这里最要紧的不是"能读能写"，而是**只增不改**：`add` 对已存在的 token
必须一个字段都不动。违反它的后果不是报错，是历史 chunk 的稀疏向量
悄悄指向别的词——检索结果漂移，而数据本身看不出任何异常。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import pytest

from app.domain.vocab import VocabEntry
from app.repositories.vocab_repo import (
    InMemoryVocabRepository,
    SqlVocabRepository,
    VocabRepository,
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
) -> AsyncIterator[Callable[[], VocabRepository]]:
    if request.param == "memory":
        yield InMemoryVocabRepository
        return

    from app.tests.db import sql_sessions

    async with sql_sessions() as sessions:
        yield lambda: SqlVocabRepository(sessions)


def _entry(token: str, token_id: int, df: int = 1) -> VocabEntry:
    return VocabEntry(token=token, token_id=token_id, df=df)


# ------------------------------------------------------------------ 基本读写


async def test_add_then_load_round_trips(make_repo: Callable[[], VocabRepository]) -> None:
    repo = make_repo()
    await repo.add([_entry("渠道折扣", 0, 12), _entry("华东", 1, 40)])

    assert await repo.load() == [_entry("渠道折扣", 0, 12), _entry("华东", 1, 40)]


async def test_load_is_empty_for_a_fresh_table(make_repo: Callable[[], VocabRepository]) -> None:
    assert await make_repo().load() == []


async def test_add_returns_the_number_of_new_entries(
    make_repo: Callable[[], VocabRepository],
) -> None:
    """返回值是**真正新增**的条数，不是入参的长度。

    入库报告与 `make vocab` 的输出都读它；把入参长度当新增数，
    重复跑一次就会报"新增 2781 条"，而库里一条都没多——
    那个数字会被拿去判断"要不要重跑校准"，错得很安静。
    """
    repo = make_repo()
    await repo.add([_entry("华东", 0)])

    assert await repo.add([_entry("华东", 0), _entry("华南", 1)]) == 1


async def test_add_is_idempotent(make_repo: Callable[[], VocabRepository]) -> None:
    repo = make_repo()
    await repo.add([_entry("华东", 0, 5)])

    assert await repo.add([_entry("华东", 0, 5)]) == 0
    assert await repo.load() == [_entry("华东", 0, 5)]


async def test_add_of_nothing_is_a_no_op(make_repo: Callable[[], VocabRepository]) -> None:
    assert await make_repo().add([]) == 0


# ------------------------------------------------------------------ 只增不改


async def test_existing_token_id_is_never_reassigned(
    make_repo: Callable[[], VocabRepository],
) -> None:
    """**同一个 token 再报一次新 id，也必须保留原 id。**

    这是 11.6.3 的核心：已入库 chunk 的稀疏向量是 `{token_id: weight}`，
    id 一改，那些向量就指向了别的词。

    注意这条用例与 `test_add_is_idempotent` 的区别——前者传的 id 相同，
    这条传的是**不同的 id**。上游算错（比如没读到历史词表就重新分配）
    时正是这种情况，而它的表现是"新词把老词挤掉了"，
    比重复插入危险得多。
    """
    repo = make_repo()
    await repo.add([_entry("华东", 7)])

    await repo.add([_entry("华东", 0)])

    assert await repo.load() == [_entry("华东", 7)]


async def test_existing_df_is_frozen(make_repo: Callable[[], VocabRepository]) -> None:
    """`df` 同样不改。

    IDF 是入库时算进稀疏向量里的固定值；事后更新 df 会让老向量与新向量
    不可比，而它们在同一次检索里会被一起打分。11.6.4 的"固定 IDF"
    与"df 冻结"是同一件事的两面。
    """
    repo = make_repo()
    await repo.add([_entry("华东", 0, df=3)])

    await repo.add([_entry("华东", 0, df=99)])

    assert (await repo.load())[0].df == 3


async def test_duplicate_token_id_with_a_different_token_is_skipped(
    make_repo: Callable[[], VocabRepository],
) -> None:
    """`token_id` 上有唯一约束：撞了不能让整批写入失败。

    上游把 id 算错（复用了一个已分配的 id）时，正确行为是**跳过这一条**，
    而不是让另外那几十条也一起丢掉。丢掉的表现是"词表少了一批词"，
    而入库那边完全不知情。
    """
    repo = make_repo()
    await repo.add([_entry("华东", 5)])

    added = await repo.add([_entry("华南", 5), _entry("华北", 6)])

    assert added == 1
    assert {entry.token for entry in await repo.load()} == {"华东", "华北"}


# ------------------------------------------------------------------ 版本指纹


async def test_version_changes_when_entries_are_added(
    make_repo: Callable[[], VocabRepository],
) -> None:
    """版本变了 → 缓存键变了 → 新词表自然生效（4.4 纪律 1）。"""
    repo = make_repo()
    before = await repo.version()

    await repo.add([_entry("华东", 0)])

    assert await repo.version() != before


async def test_version_is_stable_when_nothing_changes(
    make_repo: Callable[[], VocabRepository],
) -> None:
    """没有新增时版本**必须不动**。

    否则每跑一次 `make vocab`（哪怕一条都没加）都会让缓存失效，
    而失效的代价是每个进程重新拉一次全量词表。
    """
    repo = make_repo()
    await repo.add([_entry("华东", 0)])
    version = await repo.version()

    await repo.add([_entry("华东", 0)])

    assert await repo.version() == version


async def test_version_is_safe_for_object_keys(make_repo: Callable[[], VocabRepository]) -> None:
    """版本值会进缓存键与快照的对象 key，必须是路径安全的。

    冒号在 S3 key 里合法，落到本地文件系统或 Windows 上就是非法字符——
    这类问题只在换存储后端时才暴露。
    """
    repo = make_repo()
    await repo.add([_entry("华东", 0)])

    version = await repo.version()

    assert all(ch.isalnum() or ch in "-_.@" for ch in version), version
