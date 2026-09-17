"""知识文档与稀疏检索词表（详细设计 16.8）。

**分块正文不在这两张表里**——chunk 与其向量存在向量库，
MySQL 只保留文档级的元数据与版本。`storage_path` 指向对象存储中的原文件，
用于索引重建时取回原文。
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import BigInteger, Date, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base
from app.infrastructure.models.common import json_list_opt, ulid_pk, utc_dt


class KnowledgeDocument(Base):
    """文档版本与入库状态（16.8）。

    `(logical_key, version)` 唯一使「同一个制度的多个版本」能共存，
    检索时按生效区间过滤——这是 TIME 冲突与跨版本串味的根治手段。
    """

    __tablename__ = "knowledge_document"

    id: Mapped[ulid_pk]
    #: 逻辑标识：同一份文档的不同版本共用它
    logical_key: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(255))
    #: POLICY / REPORT / PRODUCT / METRIC / OTHER（11.4 的 document_type）
    type: Mapped[str] = mapped_column(String(16))
    department: Mapped[str | None] = mapped_column(String(64))
    #: 对象存储中的相对路径：knowledge/{logical_key}/{version}/original.{ext}
    storage_path: Mapped[str] = mapped_column(String(512))
    checksum: Mapped[str] = mapped_column(String(64))
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    #: INTERNAL / CONFIDENTIAL
    classification: Mapped[str] = mapped_column(String(16))
    #: INTERNAL / EXTERNAL —— 文档来自内部制度体系，还是外部材料（行业协会、券商、媒体）。
    #:
    #: **为什么必须单独成列**：FR-SEARCH-001 要求「外部信息与内部制度冲突时不得覆盖
    #: 内部事实」，22.6 的 SOURCE 冲突用例、16.11.2 的「外部称增长、内部在下降」
    #: 缺陷注入都依赖这个判定。它会与两种已有字段混淆：
    #:   - `classification` 是**密级**轴（INTERNAL/CONFIDENTIAL），与来源内外正交；
    #:   - `type` 是封闭 5 值枚举，外部行业报告与内部经营报告会同落 `REPORT`，
    #:     区分不出来。靠 `department` 自由文本约定编码，则会把一个语义轴
    #:     藏在字符串里，日后改动取值会静默破掉门禁。
    #: 检索侧也要按它过滤（「只召回内部来源」），故它是可断言的独立字段。
    source_kind: Mapped[str] = mapped_column(String(16), server_default="INTERNAL")
    allowed_roles_json: Mapped[json_list_opt]
    #: PROCESSING / ACTIVE / FAILED / ARCHIVED，**只有 ACTIVE 进入检索**
    status: Mapped[str] = mapped_column(String(16))
    #: 解析器与 embedding 版本。换模型必须全量重建，所以版本要落库可查。
    parser_version: Mapped[str | None] = mapped_column(String(32))
    embedding_version: Mapped[str | None] = mapped_column(String(32))
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    #: 入库失败原因摘要（如扫描件不支持 OCR）
    error_summary: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(String(26))
    created_at: Mapped[utc_dt]
    updated_at: Mapped[utc_dt]

    __table_args__ = (
        UniqueConstraint("logical_key", "version", name="uk_document_logical_version"),
        # 重复文件检测：同一份内容以同一版本重复入库应被挡住
        UniqueConstraint("checksum", "version", name="uk_document_checksum_version"),
        # 检索时的生效区间过滤走这条索引
        Index("idx_document_status_effective", "status", "effective_from", "effective_to"),
    )


class RagVocab(Base):
    """稀疏检索词表（16.8 / 11.6.3）。

    **`token_id` 只增不改**：新增词分配新 ID，已发布 chunk 的稀疏向量因此
    无需重算。删词或复用 ID 会让历史向量悄悄指向别的词，
    表现为「检索结果莫名漂移」，且无法从数据上察觉。

    主键是 `token` 而非 ULID——这是详细设计唯一明确指定非 ULID 主键的表。
    """

    __tablename__ = "rag_vocab"

    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    #: 自增序列分配；唯一约束保证不重复分配
    token_id: Mapped[int] = mapped_column(BigInteger)
    #: 文档频率，用于 IDF 计算（11.6.4 的固定 IDF 方案）
    df: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[utc_dt]

    __table_args__ = (UniqueConstraint("token_id", name="uk_vocab_token_id"),)
