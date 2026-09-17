"""知识文档版本记录（详细设计 16.8 的 `knowledge_document`）。

**chunk 不在这张表里**：正文与其向量住在向量库，MySQL 只保留文档级的元数据与版本。
这一行的作用是回答三件事：

1. **这一版是什么**（`logical_key` + `version` + `checksum`）——11.9 的幂等指纹；
2. **它发布成功了没有**（`status`）——只有 ACTIVE 进入检索；
3. **它当时是怎么被切/被向量化的**（`parser_version` / `embedding_version`）——
   换解析规则或换 embedding 模型必须全量重建，而"要不要重建"这个判断
   只能来自"库里这批是什么版本建的"。

## 三个状态位与两种"不完整"要分清

| 状态 | 含义 | 向量库里有没有它 |
|---|---|---|
| PROCESSING | 正在入库，读者不可见 | **有**，但整批是 PROCESSING |
| ACTIVE | 已发布 | 有，且是 ACTIVE |
| FAILED | 入库失败 | 没有（失败时已整批删除） |
| ARCHIVED | 被新版本取代，明确下架 | 保留但改状态，用于「上一版写了什么」 |

PROCESSING 与 FAILED 都会让文档"查不到"，但处置方式相反：前者要继续等或回收，
后者要重做。两者混成一个状态，入库中断后就没法判断"该清掉还是该重跑"。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.user import UserRole


class DocumentStatus(StrEnum):
    """16.8 的四个状态。**只有 ACTIVE 进入检索**（11.9）。"""

    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class SourceKind(StrEnum):
    """文档来自内部制度体系还是外部材料（FR-SEARCH-001 的判定依据）。

    **没有默认值是有意的**：CLAUDE.md 的临时约定 12 写明「必须由入库侧显式指定，
    不要依赖列的 `server_default='INTERNAL'`」——把外部材料误标成 INTERNAL，
    会让 SOURCE 冲突判定静默失效。列上的默认值只是给存量行的安全兜底。
    """

    INTERNAL = "INTERNAL"
    EXTERNAL = "EXTERNAL"


#: 密级 → 可见角色。**访问策略只有这一处**。
#:
#: 放在领域层而不是入库脚本里，是因为它同时约束入库（写 `allowed_roles`）
#: 与检索（`ChunkFilter.roles` 与它求交集）两侧。散在两处的话，
#: 「入库放开了、检索没放开」与「检索放开了、入库没放开」的症状完全不同
#: （前者是看不见，后者是越权可见），却都源自同一个不一致。
ROLES_BY_CLASSIFICATION: dict[str, tuple[str, ...]] = {
    "INTERNAL": (UserRole.ANALYST.value, UserRole.ADMIN.value),
    "CONFIDENTIAL": (UserRole.ADMIN.value,),
}


class KnowledgeDocumentRecord(BaseModel):
    """`knowledge_document` 的一行（16.8）。

    做成不可变模型：入库是一次"建行 → 改状态"的推进，
    每一步都产出一个新值而不是就地改字段。就地改的话，失败路径上的
    「该写 FAILED 却还留着 PROCESSING 的旧值」这类问题要等查库才发现。
    """

    model_config = ConfigDict(frozen=True)

    id: str
    #: 逻辑标识：同一份文档的不同版本共用它（`policy/east-china-channel-discount`）
    logical_key: str
    version: str
    title: str
    #: POLICY / REPORT / PRODUCT / METRIC / OTHER
    document_type: str
    department: str | None = None
    #: 对象存储中的相对路径：`knowledge/{logical_key}/{version}/original.{ext}`
    storage_path: str
    #: 原文件 SHA-256。**幂等指纹的一半**（另一半是元数据），见 11.9
    checksum: str
    effective_from: date | None = None
    effective_to: date | None = None
    classification: str = "INTERNAL"
    source_kind: SourceKind = Field(description="必须显式指定，见 SourceKind")
    allowed_roles: tuple[str, ...] = ()
    status: DocumentStatus = DocumentStatus.PROCESSING
    parser_version: str | None = None
    embedding_version: str | None = None
    chunk_count: int | None = None
    #: 失败原因摘要（如扫描件不支持 OCR）。**这是给人看的**，
    #: 所以写的是"为什么不行"，不是异常类名。
    error_summary: str | None = None
    created_by: str | None = None
    #: 两个时间戳**不给默认值**：库里是 NOT NULL，而"忘了填"如果在这里被
    #: 默认值兜住，就会一路走到 INSERT 才报 IntegrityError——
    #: 那时报错信息里只有列名，看不出是哪一次入库忘了填。
    created_at: datetime
    updated_at: datetime

    def roles_for(self) -> tuple[str, ...]:
        """本行的可见角色。取不到显式配置时按密级回退。"""
        return self.allowed_roles or ROLES_BY_CLASSIFICATION.get(self.classification, ())


def roles_for_classification(classification: str) -> tuple[str, ...]:
    """密级 → 可见角色。未知密级**返回空集而不是全集**。

    空集在检索侧落成"谁都看不见"（`ChunkFilter.roles` 与它求交集为空），
    这是安全的失败方向；反过来给全集，一次密级拼写错误就会变成一次越权可见。
    """
    return ROLES_BY_CLASSIFICATION.get(classification, ())


__all__ = [
    "ROLES_BY_CLASSIFICATION",
    "DocumentStatus",
    "KnowledgeDocumentRecord",
    "SourceKind",
    "roles_for_classification",
]
