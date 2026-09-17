"""稀疏检索词表仓储（详细设计 11.6.3 / 16.8 的 `rag_vocab` 表）。

## 一条铁律：只增不改

`token_id` 一旦分配就不再变动，`df` 也一样。理由不是"少写点代码"，
而是**已发布的 chunk 带走了当时的稀疏向量**（`{token_id: weight}`）：

- 复用或重排 `token_id` → 老向量指向别的词。检索结果悄悄漂移，
  而数据本身看不出任何异常；
- 更新 `df` → 老向量里的权重与它当时的 IDF 对不上，
  新旧 chunk 的相关性分数不可比。11.6.4 选的就是"固定 IDF"这条路线，
  代价明写在设计里（新增大量文档后 IDF 不再最优）。

所以 `add` 对已存在的 token **不更新任何列**。

## 为什么不是 `INSERT IGNORE`

`INSERT IGNORE` 会把**所有**错误降级成警告——包括数据超长被截断、
类型不匹配。那意味着一个 200 字的 token 会被静默截成 128 字存进去，
而它在表里看起来完全正常，只是永远匹配不上任何查询。

`ON DUPLICATE KEY UPDATE token = token`（自我赋值，无实际变更）只处理
唯一键冲突这一种情况，其余错误照常抛出。

**不用它的 affected-rows 来数新增条数**（那是很自然的顺手写法）：
MySQL 的这个数字取决于连接上的 `CLIENT_FOUND_ROWS` 标志，
换驱动或改连接参数就会变，而它要显示在入库报告里。新增条数改为
"先查已存在的、取差集"，与驱动行为无关，见 `SqlVocabRepository.add`。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.vocab import VocabEntry
from app.infrastructure.db import session_scope
from app.infrastructure.models.knowledge import RagVocab


class VocabRepository(Protocol):
    """词表持久化。

    **没有 `update` 也没有 `delete`**：见模块 docstring。接口不存在，
    调用方就没法在某个"特殊情况"下绕过这条铁律。
    """

    async def load(self) -> list[VocabEntry]: ...

    async def add(self, entries: Sequence[VocabEntry]) -> int:
        """追加词条，返回**真正新增**的条数（已存在的原样跳过）。"""
        ...

    async def version(self) -> str:
        """词表当前的版本指纹，用作缓存键。

        见 `SqlVocabRepository.version` 的实现说明：因为只增不改，
        "条数 + 最大 id"足以标识任意一次变更。
        """
        ...


class SqlVocabRepository:
    """MySQL 实现（`rag_vocab` 表）。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load(self) -> list[VocabEntry]:
        async with session_scope(self._sessions) as session:
            rows = (
                await session.execute(
                    select(RagVocab.token, RagVocab.token_id, RagVocab.df).order_by(
                        RagVocab.token_id
                    )
                )
            ).all()
        return [VocabEntry(token=row.token, token_id=row.token_id, df=row.df) for row in rows]

    async def add(self, entries: Sequence[VocabEntry]) -> int:
        """追加词条，返回**真正新增**的条数。

        ## 新增条数是"先查出来"的，不是 `rowcount` 报的

        `INSERT ... ON DUPLICATE KEY UPDATE` 的 affected-rows **取决于连接上的
        `CLIENT_FOUND_ROWS` 标志**：同一批语句、同一个 MySQL，标志不同就是
        "新插入 1 行"与"命中并更新 2 行"两种结果。驱动或连接参数一换，
        这个数字就变了，而它是要显示在入库报告里的。

        所以这里先查出已存在的 token，只有不在其中的才写，条数由差集得到——
        与驱动行为无关，两个实现也能写成同一套语义。

        ## 为什么仍然带 `ON DUPLICATE KEY UPDATE`

        先查后写在**并发**下是有窗口的：两条 `make vocab` 同时跑，都查到
        "这些 token 不存在"，然后都去插。这时 `ON DUPLICATE KEY UPDATE`
        是兜底——语句本身不会因为撞唯一键而失败，只是那一条不生效。
        代价是并发下返回值可能**多报**；它只用于展示，而重复写入本身无害
        （见模块 docstring：已存在的什么都不改）。
        """
        if not entries:
            return 0
        now = datetime.now(UTC).replace(tzinfo=None)
        async with session_scope(self._sessions) as session:
            tokens = [entry.token for entry in entries]
            token_ids = [entry.token_id for entry in entries]
            # **两个唯一键都要查**。只查 token 的话，"上游把 id 算错、
            # 让新词抢了已分配的 id"这种情况会被数成新增，而它其实被
            # `uk_vocab_token_id` 挡掉了——报告说加了、库里没有，
            # 而这份报告正是判断"要不要重跑校准"的依据。
            known_tokens = set(
                (await session.execute(select(RagVocab.token).where(RagVocab.token.in_(tokens))))
                .scalars()
                .all()
            )
            taken_ids = set(
                (
                    await session.execute(
                        select(RagVocab.token_id).where(RagVocab.token_id.in_(token_ids))
                    )
                )
                .scalars()
                .all()
            )
            fresh = [
                entry
                for entry in entries
                if entry.token not in known_tokens and entry.token_id not in taken_ids
            ]
            if not fresh:
                return 0
            rows = [
                {
                    "token": entry.token,
                    "token_id": entry.token_id,
                    "df": entry.df,
                    "created_at": now,
                }
                for entry in fresh
            ]
            statement = mysql_insert(RagVocab).values(rows)
            # 自我赋值：语义上什么都没改，但让这条语句变成"冲突即跳过"，
            # 而不是把冲突报成错误（见模块 docstring 为何不用 INSERT IGNORE）
            await session.execute(statement.on_duplicate_key_update(token=statement.inserted.token))
        return len(fresh)

    async def version(self) -> str:
        """`"{条数}-{最大 token_id}"`。

        **为什么它足以标识一次变更**：本表只增不改、不删，
        所以任何一次写入都必然改变条数或最大 id，两者都不变的唯一可能就是
        没有写入。这比"用时间戳"可靠——时间戳会随一次没有新增任何词条的
        构建而改变，让缓存无谓失效；也比"算全表哈希"便宜得多。

        **分段符用 `-` 而不是 `:`**：这个值会进缓存键，也会进快照的对象 key
        （`system/vocab/{version}/vocab.json`）。冒号在 S3 key 里合法，
        但落到本地文件系统（`STORAGE_BACKEND=local`）或 Windows 上
        就是一个文件名非法字符——那类问题只在换存储后端时才暴露。
        """
        async with session_scope(self._sessions) as session:
            row = (
                await session.execute(
                    select(func.count(), func.coalesce(func.max(RagVocab.token_id), 0))
                )
            ).one()
        return f"{int(row[0])}-{int(row[1])}"


class InMemoryVocabRepository:
    """进程内实现，供不连 MySQL 的单元测试使用。

    与 SQL 实现受同一份 `Protocol` 约束，`add` 的"只增不改"语义在这里
    也要成立——否则用它写的用例会给出与生产不同的结论。
    """

    def __init__(self, entries: Sequence[VocabEntry] = ()) -> None:
        self._entries: dict[str, VocabEntry] = {entry.token: entry for entry in entries}
        self._taken_ids: set[int] = {entry.token_id for entry in self._entries.values()}

    async def load(self) -> list[VocabEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.token_id)

    async def add(self, entries: Sequence[VocabEntry]) -> int:
        added = 0
        for entry in entries:
            # **两个唯一约束都要落实**，不只是 token。SQL 实现靠
            # `uk_vocab_token_id` 挡住"不同 token 抢同一个 id"，
            # 这里不挡的话，同一份用例在两个实现上会给出不同结论——
            # 而契约测试存在的全部意义就是不让这种情况发生。
            if entry.token in self._entries or entry.token_id in self._taken_ids:
                continue
            self._entries[entry.token] = entry
            self._taken_ids.add(entry.token_id)
            added += 1
        return added

    async def version(self) -> str:
        if not self._entries:
            return "0-0"
        return f"{len(self._entries)}-{max(e.token_id for e in self._entries.values())}"
