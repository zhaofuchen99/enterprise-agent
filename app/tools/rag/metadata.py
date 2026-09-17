"""分块元数据与向量库 payload 的映射（详细设计 11.4 / 11.5）。

## 为什么单独一个模块

`ChunkMetadata` 有**两个方向的使用者**：入库侧把它拼成 payload 写进 Qdrant
（11.1 的「Metadata Validation」），检索侧从 payload 读回来还原成证据引用
（11.7 的输出、13.1 的 `DocumentEvidence`）。放进 `ingestion.py` 会让检索侧
import 入库模块，两边一旦各自维护一份字段清单，症状是**证据里的章节/页码
悄悄变空**——引用仍然打得开，只是定位信息没了，没人会当成 bug 报。

## 与 11.4 的两处偏离（都回写在 CLAUDE.md 的临时约定里）

**1. `status` 的取值不是 11.4 写的 `DRAFT / ACTIVE / ARCHIVED`。**
11.1 的 staging 态写的是 `payload.status = PROCESSING`，11.9 要求
"失败版本不得被在线查询过滤条件命中"，过滤条件就是拿这个字段比。
11.4 那三个值是**文档生命周期**的写法，与发布态不是一回事：
DRAFT 在入库流程里根本不存在（草稿不落向量库）。
两套取值并存必然漂移——过滤条件只认一个字符串，而枚举有两个。
因此这里**复用 `domain.knowledge.DocumentStatus`**，行与 payload 同一个枚举。

**2. 多了 `logical_key` / `char_start` / `char_end` / `is_table` / `table_caption`。**
`logical_key` 理论上能从 `document_id`（`logical_key@version`）切出来，
但**按分隔符反解一个复合键是错的**：`logical_key` 里本来就有 `/`，
哪天有人在版本号里放个 `@`，反解就静默切错。存一份显式的，代价是一个字段。
后四个来自 11.3 的「页码、章节和原文件偏移写入 metadata」——
偏移是引用定位的最后一环（"这段话在原文的哪个位置"），
而 `is_table` / `table_caption` 让检索侧能对表格块做11.3 的表格序列化还原。

## payload 里的日期是字符串

`date` / `datetime` 不是 JSON 类型，`payload` 是 `dict[str, Any]` 且要经
Qdrant 序列化，塞 `date` 对象进去会在写入时才炸。统一走 `isoformat()`：
日期 `YYYY-MM-DD`、时间戳带 `Z`。**实测 Qdrant 的 `DatetimeRange` 接受
不带时间的 `YYYY-MM-DD`**（契约测试的生效区间用例就是这么建的），
所以不额外补 `T00:00:00Z`。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.domain.knowledge import DocumentStatus, SourceKind
from app.tools.rag.chunker import Chunk


def document_key(logical_key: str, version: str) -> str:
    """`logical_key@version` —— 一个文档版本的稳定身份。

    **三处必须用同一个值**：`chunk_id` 的派生种子（`chunker.chunk_id_for`）、
    payload 的 `document_id`、以及向量库里按文档删除的选择器。
    三者一旦不一致，"删掉这一版没发布成功的 point"就会删到别处去，
    而删除是**不会报错**的操作——它只是什么也没删。

    分隔符用 `@`：`logical_key` 里已经用了 `/`（`policy/east-china-...`），
    再用 `/` 就分不出边界了。`@` 在 S3 key 与文件系统里都合法。
    """
    return f"{logical_key}@{version}"


class ChunkMetadata(BaseModel):
    """11.4 的 `ChunkMetadata`，加上述两处偏离。

    `frozen=True`：它是写完就不再变的事实快照。入库流程里会派生很多份
    （每块一份），可变的话"改了一份、另一份没改"只会表现为某一条证据的
    页码对不上，而条目数量完全正常。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    #: `logical_key@version`，见 `document_key()`
    document_id: str
    #: 显式存一份，不从 `document_id` 反解（见模块 docstring）
    logical_key: str
    document_version: str
    title: str
    #: POLICY / REPORT / PRODUCT / METRIC / OTHER（11.4 的封闭 5 值）
    document_type: str
    department: str | None = None
    section_path: tuple[str, ...] = ()
    page_no: int | None = None
    char_start: int = 0
    char_end: int = 0
    is_table: bool = False
    table_caption: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    published_at: datetime | None = None
    #: 发布态（PROCESSING / ACTIVE / FAILED / ARCHIVED），见模块 docstring 的偏离 1
    status: DocumentStatus = DocumentStatus.PROCESSING
    classification: str = "INTERNAL"
    #: INTERNAL / EXTERNAL。**没有默认值**——同 `SourceKind` 的理由，
    #: 必须由入库侧显式指定
    source_kind: SourceKind
    allowed_roles: tuple[str, ...] = ()
    #: 原文件 SHA-256。让证据能回答"引用的是哪一份文件"，
    #: 而不是"哪一份标题相同的文件"（同名制度多版本时这两者不同）
    checksum: str

    def payload(self) -> dict[str, Any]:
        """→ 写进向量库的标量部分（11.5 的 payload）。

        `chunk_id` / `text` 不在这里：它们由 `VectorPoint` 单独携带，
        真实实现写 point 时再合并（见 `vector_store._to_point_struct`）。
        在这里也塞一份的话，两处会各写各的，而 Qdrant 的 payload 是覆盖语义——
        后写的那份赢，且没有任何提示。

        `mode="json"` 一把梭做序列化转换：`date` / `datetime` / `tuple`
        都由 Pydantic 转成 JSON 形态，不必手写 isoformat。
        **用它而不是手写，是为了让"哪些字段需要转换"这件事跟着模型走**——
        加一个日期字段时，手写版本会漏掉它，而漏掉的症状是写入时 500。
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChunkMetadata:
        """从 payload 还原。**多余字段直接忽略**（`extra="ignore"`）。

        向量库里可能存着上一个版本的代码写下的 payload（本模块还在演进中），
        严格模式会让"加了个新字段"变成"老数据全部读不出来"。
        少字段则照常报错——那是另一回事：说明这条 payload 是残缺的，
        用它做证据会给出一个定位不到原文的引用。
        """
        return cls.model_validate(payload)


#: `ChunkMetadata.payload()` 会写进向量库的键。**入库冒烟与契约断言用它**：
#: Qdrant 的 payload 是自由的 dict，没人拦得住往里面多写一个键；
#: 而多写的键会让"payload 里到底有哪些字段"这个问题在两边给出不同答案。
PAYLOAD_KEYS: tuple[str, ...] = tuple(ChunkMetadata.model_fields)


def metadata_for(
    chunk: Chunk,
    *,
    logical_key: str,
    version: str,
    title: str,
    document_type: str,
    source_kind: SourceKind,
    classification: str,
    allowed_roles: tuple[str, ...],
    checksum: str,
    department: str | None = None,
    effective_from: date | None = None,
    effective_to: date | None = None,
    published_at: datetime | None = None,
    status: DocumentStatus = DocumentStatus.PROCESSING,
) -> ChunkMetadata:
    """`Chunk` + 文档级元数据 → 一块的 `ChunkMetadata`。

    存在的意义是**把文档级的字段只传一次**：88 篇、1800+ 块，
    逐块拼 dict 的写法里，"把 `source_kind` 写成了 `classification`"
    会让外部材料被当成内部制度，而 SOURCE 冲突判定就此静默失效
    （FR-SEARCH-001 就是靠它成立的）。
    """
    return ChunkMetadata(
        chunk_id=chunk.chunk_id,
        document_id=document_key(logical_key, version),
        logical_key=logical_key,
        document_version=version,
        title=title,
        document_type=document_type,
        department=department,
        section_path=chunk.section_path,
        page_no=chunk.page_no,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        is_table=chunk.is_table,
        table_caption=chunk.table_caption,
        effective_from=effective_from,
        effective_to=effective_to,
        published_at=published_at,
        status=status,
        classification=classification,
        source_kind=source_kind,
        allowed_roles=allowed_roles,
        checksum=checksum,
    )


__all__ = ["PAYLOAD_KEYS", "ChunkMetadata", "document_key", "metadata_for"]
