"""知识文档版本仓储（详细设计 16.8 的 `knowledge_document`）。

## 这张表是入库的"账本"，不是索引

正文与向量在向量库，这里只记**版本级的事实**：这一版是什么内容（`checksum`）、
发布到哪一步了（`status`）、当时用什么解析器和模型建的（`parser_version` /
`embedding_version`）。

因此有两个动作刻意**没有**进接口：

- **没有 `delete`**：版本记录是证据链的一环（哪一版进过检索、引用指向它）。
  11.9 的"失败版本不得被在线查询命中"靠的是状态位与向量侧的整批删除，
  不是把这一行抹掉——抹掉之后"这份文档到底入过没有"就没有答案了。
- **没有"按 id 改状态"之外的批量操作**：入库是逐篇推进的，
  批量的地方（`make ingest` 的汇总）在调用侧循环，不在这层。

## `save` 是 upsert，且**不覆盖 `id`**

`(logical_key, version)` 上有唯一键，同一版重复入库（失败重试、`--force` 重建）
必须落到**同一行**上。如果每次重建都新插一行，`created_by` / 第一次入库时间
这些"这一版是谁什么时候建进来的"就会随重试次数漂移，而没有人会注意到。

## `ON DUPLICATE KEY UPDATE` 会在**任意**唯一键上触发，所以要显式挡冲突

这张表有三个唯一键：主键 `id`、`(logical_key, version)`、`(checksum, version)`。
MySQL 的 `ON DUPLICATE KEY UPDATE` **不区分是哪一个撞了**——三者中任意一个命中，
它都会把那一行当成"要更新的目标"。后果不是报错，而是**静默的错误写入**：

- `id` 撞了（比如重建时新生成一个 ULID，而库里已有同一版）：语句变成更新那一行，
  新行根本没插进去，调用方还以为写成功了；
- `(checksum, version)` 撞了（同一份内容换了 `logical_key` 再入一次）：
  语句会去更新**另一个 `logical_key` 的那一行**，把它的标题、路径、状态
  改成这一份的——而它自己的 `logical_key` 没变。库里从此有一行
  "名字是 A、内容是 B"的记录。

两种都是契约测试的 SQL 变体抓出来的，内存实现当时并不复现（它不看唯一键）。
所以 `save` 在 upsert 之前显式查一遍这两个键，撞了就抛 `ValueError`——
**这条路径必须报错而不是"让数据库去管"**，因为数据库管的方式正是静默更新。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.knowledge import DocumentStatus, KnowledgeDocumentRecord
from app.infrastructure.db import session_scope
from app.infrastructure.models.knowledge import KnowledgeDocument
from app.repositories._mapping import knowledge_from_row, knowledge_to_row_values

#: upsert 时**允许被覆盖**的列。白名单而不是黑名单：
#: 新增列时默认"不可覆盖"（先不动存量语义），要覆盖必须显式加进来。
#: `id` 与 `created_at` 不在其中——它们标识"这一行是怎么来的"，
#: 被重试覆盖掉之后就没有任何地方记得原始值了。
_UPDATABLE_COLUMNS: tuple[str, ...] = (
    "title",
    "type",
    "department",
    "storage_path",
    "checksum",
    "effective_from",
    "effective_to",
    "classification",
    "source_kind",
    "allowed_roles_json",
    "status",
    "parser_version",
    "embedding_version",
    "chunk_count",
    "error_summary",
    "updated_at",
)


class KnowledgeDocumentRepository(Protocol):
    """文档版本持久化（16.8）。"""

    async def get(self, logical_key: str, version: str) -> KnowledgeDocumentRecord | None: ...

    async def find_by_checksum(self, checksum: str, version: str) -> KnowledgeDocumentRecord | None:
        """按 `(checksum, version)` 查——11.9 的幂等指纹去重走这条。

        **与 `get` 是两条独立的查询**，不能互相顶替：
        `get` 回答"这一版现在是什么状态"（同版本不同内容时要能发现），
        `find_by_checksum` 回答"这份内容是不是已经作为这一版入过了"
        （换了个 `logical_key` 重复上传时，靠它挡住重复入库）。
        """
        ...

    async def save(self, record: KnowledgeDocumentRecord) -> KnowledgeDocumentRecord:
        """写入或更新一行，返回**落库后**的记录。"""
        ...

    async def list_documents(self) -> list[KnowledgeDocumentRecord]:
        """全量列出。入库汇总、`verify-corpus` 与重建索引都要枚举文档。"""
        ...


class SqlKnowledgeDocumentRepository:
    """MySQL 实现。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get(self, logical_key: str, version: str) -> KnowledgeDocumentRecord | None:
        async with session_scope(self._sessions) as session:
            row = await self._select_one(session, logical_key, version)
        return knowledge_from_row(row) if row is not None else None

    async def find_by_checksum(self, checksum: str, version: str) -> KnowledgeDocumentRecord | None:
        async with session_scope(self._sessions) as session:
            row = (
                await session.execute(
                    select(KnowledgeDocument).where(
                        KnowledgeDocument.checksum == checksum,
                        KnowledgeDocument.version == version,
                    )
                )
            ).scalar_one_or_none()
        return knowledge_from_row(row) if row is not None else None

    async def save(self, record: KnowledgeDocumentRecord) -> KnowledgeDocumentRecord:
        """写入或更新，并**回读落库结果**后返回。

        回读是必须的，不是讲究：upsert 命中已有行时，`id` 与 `created_at`
        保持库里那份（见白名单），于是入参里的这两项**不成立**。
        直接把入参返回出去，调用方拿到的就是一个"库里并不存在的 id"，
        而它会一路写进入库报告与日志——直到有人拿这个 id 去查库才发现。
        """
        values = knowledge_to_row_values(record)
        async with session_scope(self._sessions) as session:
            await self._reject_conflicts(session, record)
            statement = mysql_insert(KnowledgeDocument).values(**values)
            # 冲突时只更新白名单里的列。**不用 `INSERT IGNORE`**：
            # 它会把数据超长截断之类的错误一并降级成警告（同 `vocab_repo` 的理由）。
            await session.execute(
                statement.on_duplicate_key_update(
                    **{name: statement.inserted[name] for name in _UPDATABLE_COLUMNS}
                )
            )
            row = await self._select_one(session, record.logical_key, record.version)
        if row is None:  # pragma: no cover - upsert 之后必定存在
            raise RuntimeError(f"upsert 之后读不到 {record.logical_key}@{record.version}")
        return knowledge_from_row(row)

    @staticmethod
    async def _reject_conflicts(session: AsyncSession, record: KnowledgeDocumentRecord) -> None:
        """挡住两种"ODKU 会静默改错行"的冲突（见模块 docstring）。

        **判据是"这次带的 id 是不是已经属于别的行"，不是"id 与库里那行是否相同"。**
        区别很重要：`(logical_key, version)` 相同而 id 不同时，upsert 会
        命中那个唯一键、更新那一行、**id 保持库里那份**——这正是 upsert 该有的
        语义（这一行的身份由库决定，不由调用方决定）。而 `record.id` 若已经
        落在**另一行**身上，语句会因为主键冲突去改那一行：既没插入、又改了无关的记录。

        **先查后写有并发窗口**，这一点不掩饰：同一版被两个进程同时首次入库时，
        两边都可能查到"不冲突"。那一条由 `(logical_key, version)` 唯一键 + ODKU
        正确收敛（落成同一行），不需要这里挡；而这里挡的两种冲突都不是并发产生的
        （id 由调用方给、checksum 来自文件），所以先查后写在这里是够的。
        """
        by_id = await session.get(KnowledgeDocument, record.id)
        if by_id is not None and (by_id.logical_key, by_id.version) != (
            record.logical_key,
            record.version,
        ):
            raise ValueError(
                f"id {record.id} 已属于 {by_id.logical_key}@{by_id.version}，"
                f"而本次要写的是 {record.logical_key}@{record.version}。"
                "带一个已被占用的 id 会让 upsert 因主键冲突去改那一行："
                "既没插入新行，又覆盖了无关记录的元数据。"
            )
        clash = (
            await session.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.checksum == record.checksum,
                    KnowledgeDocument.version == record.version,
                )
            )
        ).scalar_one_or_none()
        if clash is not None and (clash.logical_key, clash.version) != (
            record.logical_key,
            record.version,
        ):
            raise ValueError(
                f"(checksum, version) 冲突：{record.checksum[:12]}… 已作为 "
                f"{clash.logical_key}@{clash.version} 入过库。同一份内容不允许挂两个 "
                "logical_key——那会让同一批向量在检索里并列出现，看起来像互相印证。"
            )

    async def list_documents(self) -> list[KnowledgeDocumentRecord]:
        async with session_scope(self._sessions) as session:
            rows = (
                (
                    await session.execute(
                        select(KnowledgeDocument).order_by(
                            KnowledgeDocument.logical_key, KnowledgeDocument.version
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [knowledge_from_row(row) for row in rows]

    @staticmethod
    async def _select_one(
        session: AsyncSession, logical_key: str, version: str
    ) -> KnowledgeDocument | None:
        return (
            await session.execute(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.logical_key == logical_key,
                    KnowledgeDocument.version == version,
                )
            )
        ).scalar_one_or_none()


class InMemoryKnowledgeDocumentRepository:
    """进程内实现，供不连 MySQL 的单元测试使用。

    受同一份 `Protocol` 约束，并且**落实同一组唯一键**：
    `(logical_key, version)` 覆盖写、`(checksum, version)` 撞了要报错。
    少落实后者的话，"同一份内容以两个 logical_key 重复入库"这个用例
    会在这里悄悄通过，而到了 MySQL 上直接抛 IntegrityError。
    """

    def __init__(self, records: Sequence[KnowledgeDocumentRecord] = ()) -> None:
        self._rows: dict[tuple[str, str], KnowledgeDocumentRecord] = {}
        for record in records:
            self.save_sync(record)

    async def get(self, logical_key: str, version: str) -> KnowledgeDocumentRecord | None:
        return self._rows.get((logical_key, version))

    async def find_by_checksum(self, checksum: str, version: str) -> KnowledgeDocumentRecord | None:
        for record in self._rows.values():
            if record.checksum == checksum and record.version == version:
                return record
        return None

    async def save(self, record: KnowledgeDocumentRecord) -> KnowledgeDocumentRecord:
        return self.save_sync(record)

    def save_sync(self, record: KnowledgeDocumentRecord) -> KnowledgeDocumentRecord:
        """同步版，供构造函数与断言使用。

        **两种冲突都要挡，与 SQL 实现逐条对齐**（见模块 docstring）：
        SQL 那边 ON DUPLICATE KEY UPDATE 对"主键撞别行"的处理是"改那一行"，
        这里若不挡，同一份用例会在两个实现上给出不同结论——
        而契约测试存在的全部意义就是不让这种情况发生。

        判据同 SQL 实现：**id 是否已被别的行占用**，不是"id 与库里那行是否相同"。
        """
        owner = next((row for row in self._rows.values() if row.id == record.id), None)
        if owner is not None and (owner.logical_key, owner.version) != (
            record.logical_key,
            record.version,
        ):
            raise ValueError(
                f"id {record.id} 已属于 {owner.logical_key}@{owner.version}，"
                f"而本次要写的是 {record.logical_key}@{record.version}"
            )
        existing = self._rows.get((record.logical_key, record.version))
        clash = next(
            (
                row
                for key, row in self._rows.items()
                if key != (record.logical_key, record.version)
                and row.checksum == record.checksum
                and row.version == record.version
            ),
            None,
        )
        if clash is not None:
            raise ValueError(
                f"(checksum, version) 冲突：{record.checksum[:12]}… 已作为 "
                f"{clash.logical_key}@{clash.version} 入过库"
            )
        if existing is not None:
            # 与 SQL 实现的 `_UPDATABLE_COLUMNS` 对齐：`id` 与 `created_at` 保持不变
            record = record.model_copy(
                update={"id": existing.id, "created_at": existing.created_at}
            )
        self._rows[(record.logical_key, record.version)] = record
        return record

    async def list_documents(self) -> list[KnowledgeDocumentRecord]:
        return sorted(self._rows.values(), key=lambda r: (r.logical_key, r.version))


def touch(
    record: KnowledgeDocumentRecord, status: DocumentStatus, **changes: Any
) -> KnowledgeDocumentRecord:
    """按状态推进产出新记录，并刷新 `updated_at`。

    **这是唯一改状态的入口**：直接 `model_copy(update={"status": ...})` 会漏掉
    `updated_at`，而"最后动过是什么时候"正是排查入库卡住时第一个要看的东西。
    """
    now = datetime.now(UTC)
    return record.model_copy(update={"status": status, "updated_at": now, **changes})


__all__ = [
    "InMemoryKnowledgeDocumentRepository",
    "KnowledgeDocumentRepository",
    "SqlKnowledgeDocumentRepository",
    "touch",
]
