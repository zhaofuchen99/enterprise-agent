"""文档入库（详细设计 11.1 的九步 / 11.9 的幂等与发布）。

```text
文件校验 → 归档原文 → 解析 → 清洗 → 分块 → metadata 校验
  → Dense + Sparse 表示 → 写入 collection（status=PROCESSING）
  → 抽样检索 → 原子切换 status=ACTIVE（发布）
```

## 三个不显然的决定

### 1. staging 用 payload 状态位，发布不是一次元数据操作

11.1 的落地记录已经说明：Qdrant 没有用户可见的 partition，所以"暂存 → 发布"
= 对这批 point 做一次批量 payload 更新。**这一步不是事务**，兑现的是 11.9 的
另一句话："失败版本不得被在线查询过滤条件命中"。读者恒定带 `status=ACTIVE`
（`ChunkFilter` 的默认值），于是：

- 失败版本整批仍是 PROCESSING，从头到尾不可见 —— 这条是**真的**；
- 翻转窗口内（单次 RPC，实测毫秒级）读者可能看到同一版本的部分 chunk ——
  这条**不是原子保证**，如实写在这里，不假装 Qdrant 有事务。

代价可接受：影响的是"新文档是否完整可见"，而不是"失败版本能不能被查到"。
真要严格，得把状态位提到文档级并让查询先查文档再过滤，那是另一套设计。

### 2. 向量只在万事俱备之后才动

`delete_document` + `upsert` 排在**解析、分块、向量化全部成功之后**。
反过来（先清理再处理）看起来更"干净"，代价是：`--force` 重建一篇已发布的文档时，
只要解析或 embedding 失败，线上就少了一版——而库里那一行仍写着 ACTIVE。
**先破坏再重建的顺序，把一次可恢复的失败变成了不可恢复的**。

### 3. 幂等指纹是"文件内容 + 版本"，不是"文件内容"

11.9 写的是"文件 SHA-256 + 业务元数据形成唯一版本指纹"。两者缺一不可：

- 只看 SHA-256：同一份内容被当成两个 `logical_key` 上传时会被当成新文档，
  库里于是有两份一模一样的向量，检索时并列出现，看起来像"两个来源相互印证"；
- 只看 `(logical_key, version)`：内容改了但版本号没改，重复入库直接命中缓存，
  新内容永远进不去——而调用方以为自己更新成功了。

**同版本不同内容**因此是一个显式错误（不是静默覆盖）：要改内容就必须升版本号，
或者用 `force=True` 承认这是一次重建。理由是 `chunk_id` 由 `logical_key@version`
确定性派生——静默覆盖会让此前引用过 `chk_xxx` 的那些证据指向另一段文字，
而引用本身仍然打得开，没有任何症状。

## 词表必须先进库

稀疏向量是 `{token_id: weight}`，`token_id` 来自冻结的词表快照（11.6.3/11.6.4）。
**入库前必须先跑 `make vocab`**：词表没覆盖到的 token 会被 `build_sparse` 丢弃，
症状是"某些查询永远召回不到"，且不指向词表。因此这里对"稀疏向量整块为空"
直接报错而不是打日志——见 `_check_sparse_coverage`。

## 中断会留下一行 PROCESSING

入库进行到一半被 Ctrl-C 或进程被杀，库里会留下一行 `PROCESSING` 的记录，
而向量库里什么也没有（或者只有半批——那时 `state.upserted` 还没来得及置位）。
**不需要额外的孤儿回收**：下次对同一份文件跑 `make ingest` 时，
`(logical_key, version)` 命中且内容相同，`_resume_or_reject` 只在
`status == ACTIVE` 时跳过，失败与处理中的状态都会走完整流程，
而流程开头就会 `delete_document` 清掉半批残留。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.core.ids import IdPrefix, new_id
from app.domain.knowledge import (
    DocumentStatus,
    KnowledgeDocumentRecord,
    SourceKind,
    roles_for_classification,
)
from app.infrastructure.model_gateway import ModelGateway
from app.infrastructure.storage import ObjectStorage
from app.infrastructure.vector_store import ChunkFilter, VectorPoint, VectorStore
from app.repositories.knowledge_repo import KnowledgeDocumentRepository, touch
from app.tools.rag.chunker import Chunk, chunk_document
from app.tools.rag.metadata import document_key, metadata_for
from app.tools.rag.parser import PARSER_VERSION, SUFFIX_TO_FORMAT, parse_document
from app.tools.rag.tokenizer import Tokenizer, Vocabulary, build_sparse

logger = logging.getLogger(__name__)

#: 扩展名 → 存储时用的规范扩展名与 MIME。
#:
#: **按格式而不是按原文件名取扩展名**：语料里同一格式有 `.md` 与 `.markdown`
#: 两种写法，原样保留会让同一批文档的归档路径长得不一样（`original.md` 与
#: `original.markdown`），而 `storage_path` 是要落库、要被人拿去下载的。
_FORMAT_META: dict[str, tuple[str, str]] = {
    "pdf": (".pdf", "application/pdf"),
    "docx": (
        ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "md": (".md", "text/markdown"),
    "txt": (".txt", "text/plain"),
}

#: 文件头 → 格式。**做的是"内容与扩展名是否一致"的检查**，不是完整 MIME 嗅探。
#:
#: 没有引入 `python-magic`：它要装系统库 `libmagic`，而这里要拦的情况只有一类——
#: 把 `.docx` 存成 PDF、把扫描件存成文本。这类错误用文件头足够认出来，
#: 而"装一个系统依赖来认文件类型"对本项目的收益不成比例。
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF", "pdf"),
    (b"PK\x03\x04", "docx"),
)

#: 文本格式里出现这些字节几乎一定是二进制内容（被改了扩展名的图/压缩包）。
_BINARY_SAMPLE = 4096


class DocumentMetadata(BaseModel):
    """入库侧的文档级元数据（11.4 的文档级部分 / 16.8 的列）。

    **`source_kind` 没有默认值**（CLAUDE.md 临时约定 12）：外部材料被误标成
    INTERNAL 会让 SOURCE 冲突判定静默失效，而 FR-SEARCH-001 就靠它成立。
    列上的 `server_default='INTERNAL'` 是给存量行的兜底，不是新文档的默认值。
    """

    model_config = ConfigDict(frozen=True)

    logical_key: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=32)
    title: str = Field(min_length=1, max_length=255)
    #: POLICY / REPORT / PRODUCT / METRIC / OTHER
    document_type: str = Field(min_length=1, max_length=16)
    source_kind: SourceKind
    classification: Literal["INTERNAL", "CONFIDENTIAL"] = "INTERNAL"
    department: str | None = Field(default=None, max_length=64)
    effective_from: date | None = None
    effective_to: date | None = None
    published_at: datetime | None = None
    created_by: str | None = None

    @property
    def document_id(self) -> str:
        return document_key(self.logical_key, self.version)

    @property
    def allowed_roles(self) -> tuple[str, ...]:
        """按密级推出的可见角色（`domain.knowledge.ROLES_BY_CLASSIFICATION`）。"""
        return roles_for_classification(self.classification)


class SmokeQuery(BaseModel):
    """抽样检索的一条结果（11.1 的 Retrieval Smoke Test）。"""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    query: str
    #: 1 起的名次；`None` 表示没进 TopK
    rank: int | None = None
    score: float | None = None

    @property
    def matched(self) -> bool:
        return self.rank is not None


class IngestionReport(BaseModel):
    """入库报告，归档到 `knowledge/{logical_key}/{version}/ingestion_report.json`。

    11.9 要求它与原文件一起归档——原文件让 collection 可重建，
    报告让"当初这一版是怎么进来的"可回答（用哪个解析器、多少块、冒烟过没过、
    失败原因是什么）。两者缺一，事后就只能靠猜。

    做成 Pydantic 模型而不是 dict：它要被序列化进对象存储，也要被 CLI 读出来打印，
    字段名写错在两边都会静默变成"少了一行"。
    """

    model_config = ConfigDict(frozen=True)

    document_id: str
    record_id: str
    logical_key: str
    version: str
    title: str
    source_file: str
    format: str
    checksum: str
    status: DocumentStatus
    #: 幂等命中：本次没有重新入库，直接返回了现有结果（11.9）
    skipped: bool = False
    #: **文件本身不可入库**（11.2 的"扫描件走 OCR 或明确标记不支持"），
    #: 与"入库过程出了故障"分开记。
    #:
    #: 这个区分决定 `make ingest` 的退出码：语料里 2 份扫描件**注定**失败，
    #: 若它们也算命令失败，整条命令就永远返回非零——而 Phase 5 的门禁
    #: 恰恰要求"明确标记不支持"算是通过。反过来把两者并成一类，
    #: "Ollama 没起导致 88 篇全挂"也会安静地返回 0。
    unsupported: bool = False
    chunk_count: int = 0
    page_count: int = 0
    #: 解析阶段清洗掉的内容。**必须出现在报告里**：清洗是"正确时无声、
    #: 错误时也无声"的操作，多丢一行不会有任何症状（见 `make chunk`）
    dropped: list[str] = Field(default_factory=list)
    #: 稀疏向量整块为空的 chunk 数。>0 表示词表没覆盖这份文档（见模块 docstring）
    empty_sparse_chunks: int = 0
    smoke: list[SmokeQuery] = Field(default_factory=list)
    published_count: int = 0
    storage_path: str
    report_path: str | None = None
    error_summary: str | None = None
    parser_version: str = PARSER_VERSION
    embedding_version: str = ""
    #: 上一次这一版的内容指纹。**只在内容真的变了（`--force` 重建）时非空**，
    #: 用来回答"重建前的那些引用指向的是哪一份文件"。
    #:
    #: 失败重试（同一份文件再入一次）**不算重建**：那一次 `existing.checksum`
    #: 与本次相同，把它记成"重建"会让报告读起来像"内容换过了"，
    #: 而重建与重试要做的排查完全不同。
    previous_checksum: str | None = None
    started_at: datetime
    finished_at: datetime

    @property
    def ok(self) -> bool:
        """**这一次尝试**成没成。与 `status` 是两个问题。

        `status` 是库里那一行**推进之后**的状态，`ok` 是本次尝试的结果——
        两者在 `--force` 重建失败时不相等，且都值得如实记下：

        - 重建一篇已发布的文档，解析阶段就失败 → `ok=False`（这次没成），
          但 `status` 仍是 `ACTIVE`（库里那一版还原封不动，检索照常查得到）。
          只记 `status` 的话，报告看起来像"什么都没发生"，
          而实际上有人试过重建、失败了，原因就在 `error_summary` 里。
        """
        return self.error_summary is None


# --------------------------------------------------------------------- 校验


def validate_format(path: Path) -> str:
    """扩展名白名单（11.2）。**复用 `SUFFIX_TO_FORMAT`，不另立一份清单**：
    两处清单迟早会不同步，而症状是"入库放行了但下游解析不了"。"""
    fmt = SUFFIX_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"不支持的文件格式：{path.suffix or '（无扩展名）'}",
            details={"path": str(path), "supported": "、".join(sorted(SUFFIX_TO_FORMAT))},
        )
    return fmt


def validate_size(path: Path, settings: Settings) -> None:
    """大小上限（施工项 5 的 ≤50MB，取自 `rag.max_file_bytes`）。"""
    size = path.stat().st_size
    limit = settings.rag.max_file_bytes
    if size > limit:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"文件超过大小上限（{size} > {limit} 字节）",
            details={"path": str(path), "size": size, "limit": limit},
        )


def validate_content(path: Path, fmt: str) -> None:
    """文件头与扩展名是否一致（施工项 5 的 MIME 检查）。

    **不能只看扩展名**：入库最怕的一类错误是"扫描件被当成文本 PDF 塞进来"——
    它解析得出 0 个块，于是以"空文档"的名义失败，而真实原因是文件本身是图片。
    先把这类挡在解析之前，失败信息才指向真正的问题。
    """
    head = path.read_bytes()[:_BINARY_SAMPLE]
    for magic, actual in _MAGIC:
        if head.startswith(magic):
            if actual != fmt:
                raise AgentError(
                    ErrorCode.INVALID_ARGUMENT,
                    f"文件内容与扩展名不符：扩展名说是 {fmt}，文件头是 {actual}",
                    details={"path": str(path)},
                )
            return
    if fmt in ("pdf", "docx"):
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"文件内容不是合法的 {fmt}（缺少文件头标记）",
            details={"path": str(path)},
        )
    # 文本格式：要求是合法的 UTF-8，且不要有 NUL 字节
    if b"\x00" in head:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            "文本文件里出现 NUL 字节，内容可能是二进制",
            details={"path": str(path)},
        )
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"文本文件不是合法的 UTF-8：{exc.reason}",
            details={"path": str(path), "offset": exc.start},
        ) from exc


def sha256_file(path: Path) -> str:
    """原文件指纹。分块读，不整个吃进内存——上限虽然是 50MB，但没必要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------- 归档


def original_key(logical_key: str, version: str, fmt: str) -> str:
    """原文件在对象存储里的路径（11.9）。"""
    return f"knowledge/{logical_key}/{version}/original{_FORMAT_META[fmt][0]}"


def report_key(logical_key: str, version: str) -> str:
    return f"knowledge/{logical_key}/{version}/ingestion_report.json"


# ------------------------------------------------------------------ 主流程


async def ingest_document(
    path: Path,
    metadata: DocumentMetadata,
    *,
    settings: Settings,
    storage: ObjectStorage,
    vector_store: VectorStore,
    documents: KnowledgeDocumentRepository,
    gateway: ModelGateway,
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
    force: bool = False,
    now: datetime | None = None,
) -> IngestionReport:
    """把一份文件入库（11.1 的九步）。

    Args:
        path: 原文件。
        metadata: 文档级元数据。**调用方负责保证它是对的**——
            这一层能校验形状（枚举取值、长度），校验不了语义
            （"这份文件真的属于这个 department 吗"）。
        settings: 分块、批大小、大小上限等参数。
        storage: 原文件与入库报告的归档位置。
        vector_store: 写入与发布。
        documents: 版本记录与幂等判据。
        gateway: Dense Embedding。
        tokenizer: 分词。**必须与建词表时是同一个实例配置**，否则切分不同，
            统计出来的 token 与词表对不上，表现为大量未登录 token。
        vocabulary: 冻结的词表快照（11.6.3）。
        force: 允许在"同版本不同内容"时重建。默认拒绝，见模块 docstring。
        now: 便于测试注入时间。

    Returns:
        入库报告。**文档本身没问题但不可入库**（扫描件无文本层）时返回
        `status=FAILED` 的报告而不抛异常——11.2 允许"明确标记不支持"，
        这属于预期内的结果，不是错误。

    Raises:
        AgentError: 校验失败、同版本内容冲突、词表未覆盖、或基础设施不可用。
            这些是**调用方需要处理**的问题，与上面那条区分开。
    """
    started = now or datetime.now(UTC)
    fmt = validate_format(path)
    validate_size(path, settings)
    validate_content(path, fmt)
    checksum = sha256_file(path)
    key = metadata.document_id

    existing = await documents.get(metadata.logical_key, metadata.version)
    resumed = _resume_or_reject(existing, checksum, metadata, force=force)
    if resumed is not None:
        logger.info("入库命中幂等指纹，跳过：%s", key)
        return resumed.model_copy(update={"started_at": started, "finished_at": started})

    # 同一份内容挂在两个 logical_key 下要在这里拦住。**仓储层也会拦**
    # （`(checksum, version)` 唯一键，见 `knowledge_repo` 的模块说明），
    # 但那条路径抛的是 ValueError，而这是运维最可能撞上的一次误操作
    # （改错了 `logical_key` 就把同一份文件又入了一遍），
    # 值得给一条能直接照着做的错误信息。
    duplicate = await documents.find_by_checksum(checksum, metadata.version)
    if duplicate is not None and duplicate.logical_key != metadata.logical_key:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"这份文件（SHA-256 {checksum[:12]}…）已作为 "
            f"{duplicate.logical_key}@{duplicate.version} 入库，"
            f"不能再用 {metadata.logical_key} 入一遍。同一份内容挂两个 logical_key "
            "会让同一批向量在检索里并列出现，看起来像两个来源互相印证。",
            details={
                "checksum": checksum,
                "existing": f"{duplicate.logical_key}@{duplicate.version}",
                "incoming": metadata.document_id,
            },
        )

    # 原文件先归档：它让 Qdrant collection 成为可重建的派生数据（11.9）。
    # 即使后面解析失败也留着——"扫描件没有文本层"这个结论，
    # 日后补上 OCR 时要从这里重新出发。
    storage_path = original_key(metadata.logical_key, metadata.version, fmt)
    await _archive_original(storage, path, storage_path, fmt)

    record = await documents.save(
        _new_record(
            existing,
            metadata,
            checksum=checksum,
            storage_path=storage_path,
            embedding_version=settings.embedding_model,
            started=started,
        )
    )

    state = _RunState(
        path=path,
        fmt=fmt,
        metadata=metadata,
        key=key,
        checksum=checksum,
        storage_path=storage_path,
        record=record,
        started=started,
        previous_checksum=(
            existing.checksum if existing is not None and existing.checksum != checksum else None
        ),
    )
    try:
        await _run_pipeline(
            state,
            existing=existing,
            settings=settings,
            vector_store=vector_store,
            documents=documents,
            gateway=gateway,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
        )
    except Exception as exc:
        # **已知错误与未知错误走同一条收尾路径**：词表未覆盖（AgentError）
        # 与 Ollama 连不上（httpx 异常）对"库里该是什么状态"的要求是一样的。
        # 分成两条路径写，迟早会有一条漏掉 `_settle_failure`。
        summary = str(exc) if isinstance(exc, AgentError) else f"{type(exc).__name__}: {exc}"
        await _settle_failure(state, documents, vector_store, existing, summary)
        # 失败也要写报告：它正是"为什么没进来"的唯一记录，
        # 而"扫描件不支持"与"Ollama 没开"在日志里都是一行 traceback
        await _finalize(state, storage)
        if not isinstance(exc, AgentError):
            logger.exception("入库失败：%s", state.key)
        raise
    return await _finalize(state, storage)


class _RunState:
    """一次入库的进行态。**只在本模块内使用**。

    做成一个可变对象而不是一路往下传十几个局部变量：失败路径要用到
    "已经写到哪一步"（比如 `chunk_ids` 有没有值，决定回滚时删不删）。
    拆成参数的话，异常处理那里就得靠嵌套作用域去够那些变量，
    而异常处理代码恰恰是最少被读、最容易漏掉一半的那部分。
    """

    __slots__ = (
        "checksum",
        "chunk_ids",
        "dropped",
        "empty_sparse_chunks",
        "error_summary",
        "fmt",
        "key",
        "metadata",
        "page_count",
        "path",
        "previous_checksum",
        "published_count",
        "record",
        "smoke",
        "started",
        "status",
        "storage_path",
        "unsupported",
        "upserted",
    )

    def __init__(
        self,
        *,
        path: Path,
        fmt: str,
        metadata: DocumentMetadata,
        key: str,
        checksum: str,
        storage_path: str,
        record: KnowledgeDocumentRecord,
        started: datetime,
        previous_checksum: str | None,
    ) -> None:
        self.path = path
        self.fmt = fmt
        self.metadata = metadata
        self.key = key
        self.checksum = checksum
        self.storage_path = storage_path
        self.record = record
        self.started = started
        self.previous_checksum = previous_checksum
        self.status = DocumentStatus.PROCESSING
        #: **本轮有没有往向量库里写过**。它是回滚的判据，不是日志字段，见 `_settle_failure`
        self.upserted = False
        self.unsupported = False
        self.chunk_ids: list[str] = []
        self.dropped: list[str] = []
        self.smoke: list[SmokeQuery] = []
        self.page_count = 0
        self.empty_sparse_chunks = 0
        self.published_count = 0
        self.error_summary: str | None = None


async def _run_pipeline(
    state: _RunState,
    *,
    existing: KnowledgeDocumentRecord | None,
    settings: Settings,
    vector_store: VectorStore,
    documents: KnowledgeDocumentRepository,
    gateway: ModelGateway,
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
) -> None:
    """`_RunState` 从 PROCESSING 推进到 ACTIVE。

    失败时自己收尾（`_settle_failure`）后再返回，不抛异常——
    除了 `_vectorize` 里那些"该不该继续"必须由调用方决定的错误。
    """
    parsed = parse_document(state.path)
    state.page_count = parsed.page_count
    state.dropped = list(parsed.dropped)

    chunks = chunk_document(
        parsed,
        document_key=state.key,
        settings=settings,
        title=state.metadata.title,
    )
    if not chunks:
        # 扫描件走到这里（11.2：走 OCR 或**明确标记不支持**）。
        # 切片内不做 OCR，所以这是一个**有结论的失败**，不是异常——
        # 没有异常可抛，收尾要自己做一次。
        # `unsupported` 与"入库故障"分开记：前者是设计内已知的限制，
        # 后者是这次跑坏了。命令的退出码靠这个区分（见 `IngestionReport.unsupported`）。
        state.unsupported = True
        await _settle_failure(
            state,
            documents,
            vector_store,
            existing,
            "无文本层：扫描件 PDF 不支持（切片内不做 OCR）",
        )
        return

    points = await _vectorize(
        state,
        chunks,
        settings=settings,
        gateway=gateway,
        tokenizer=tokenizer,
        vocabulary=vocabulary,
    )
    state.chunk_ids = [point.chunk_id for point in points]

    # 到这里才动向量库，见模块 docstring 决定 2
    await vector_store.ensure_collection(dim=settings.embedding_dim)
    await vector_store.delete_document(state.key)
    await vector_store.upsert(points)
    state.upserted = True

    state.smoke = await _smoke_test(state, points, settings=settings, vector_store=vector_store)
    await vector_store.set_status(state.chunk_ids, DocumentStatus.ACTIVE.value)

    # **发布必须回读验证**：`set_status` 是"发了就算"的调用，Qdrant 不会告诉你
    # 有多少 point 真的被改了（id 对不上时它只是改了个寂寞）。
    # 发布生效的判据只有一条：带着默认过滤条件去数，数得出来才算发布成功。
    state.published_count = await vector_store.count(
        ChunkFilter(status=DocumentStatus.ACTIVE.value, document_ids=(state.key,))
    )
    if state.published_count != len(points):
        summary = f"发布后可见条数不符：期望 {len(points)}，实际 {state.published_count}"
        await _settle_failure(state, documents, vector_store, existing, summary)
        return

    state.status = DocumentStatus.ACTIVE
    state.record = await documents.save(
        touch(
            state.record,
            DocumentStatus.ACTIVE,
            chunk_count=len(points),
            error_summary=None,
        )
    )


async def _vectorize(
    state: _RunState,
    chunks: Sequence[Chunk],
    *,
    settings: Settings,
    gateway: ModelGateway,
    tokenizer: Tokenizer,
    vocabulary: Vocabulary,
) -> list[VectorPoint]:
    """分块 → 向量点（Dense + Sparse + payload）。

    Dense 与 Sparse 是**两条独立的路线**（11.7 第 4 步），因此它们对
    "同一个 chunk 该长什么样"的判断也不同：Dense 看正文，Sparse 看分词。
    这里分两段算，任一段的结果都不影响另一段。
    """
    texts = [chunk.text for chunk in chunks]
    dense = await _embed_in_batches(texts, settings=settings, gateway=gateway)

    points: list[VectorPoint] = []
    empty_sparse: list[str] = []
    for chunk, vector in zip(chunks, dense, strict=True):
        sparse = build_sparse(tokenizer.cut(chunk.text), vocabulary, dim=settings.rag.sparse_dim)
        if not sparse:
            empty_sparse.append(chunk.chunk_id)
        meta = metadata_for(
            chunk,
            logical_key=state.metadata.logical_key,
            version=state.metadata.version,
            title=state.metadata.title,
            document_type=state.metadata.document_type,
            source_kind=state.metadata.source_kind,
            classification=state.metadata.classification,
            allowed_roles=state.metadata.allowed_roles,
            checksum=state.checksum,
            department=state.metadata.department,
            effective_from=state.metadata.effective_from,
            effective_to=state.metadata.effective_to,
            published_at=state.metadata.published_at,
            status=DocumentStatus.PROCESSING,
        )
        points.append(
            VectorPoint(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                dense=vector,
                sparse=sparse,
                payload=meta.payload(),
            )
        )

    state.empty_sparse_chunks = len(empty_sparse)
    _check_sparse_coverage(state, empty_sparse, vocabulary)
    return points


def _check_sparse_coverage(
    state: _RunState, empty_sparse: Sequence[str], vocabulary: Vocabulary
) -> None:
    """有 chunk 的稀疏向量是空的 → 报错，而不是打一条日志。

    **为什么值得一次硬失败**：`build_sparse` 对未登录 token 的处理是**丢弃**，
    这是 11.6.4 固定 IDF 方案的已知代价，本身没错。但整块 token 全部未登录
    时，这块的稀疏向量就是 `{}`，它将**永远无法被稀疏路召回**——
    而稠密路照常工作，所以检索结果看起来"大部分正常"，
    只有精确词查询会莫名其妙地少几篇。这类"部分可用"的降级最难被发现。

    报错信息里给出前几个 chunk_id 与词表规模：这个失败几乎总是因为
    改了语料却没重跑 `make vocab`（CLAUDE.md 临时约定 13）。
    """
    if not empty_sparse:
        return
    raise AgentError(
        ErrorCode.INVALID_ARGUMENT,
        f"{len(empty_sparse)} 个 chunk 的稀疏向量为空，词表未覆盖该文档。"
        "先跑 `make vocab` 重建词表快照再入库。",
        details={
            "document_id": state.key,
            "empty_chunks": list(empty_sparse[:5]),
            "vocabulary_size": len(vocabulary),
        },
    )


async def _embed_in_batches(
    texts: Sequence[str], *, settings: Settings, gateway: ModelGateway
) -> list[list[float]]:
    """分批向量化。

    条数对不上就报错：网关少返回一条时，`zip(strict=True)` 会在下游炸在
    "长度不匹配"上，而那个报错完全指不出是**哪一批**少了。
    在这里按批校验，报错信息直接给出批的区间。
    """
    batch_size = settings.rag.embed_batch_size
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        produced = await gateway.embed(batch)
        if len(produced) != len(batch):
            raise AgentError(
                ErrorCode.INTERNAL_ERROR,
                f"向量化返回条数不符：第 {start}–{start + len(batch) - 1} 块"
                f"请求 {len(batch)} 条，返回 {len(produced)} 条",
                details={"start": start, "requested": len(batch), "returned": len(produced)},
            )
        vectors.extend(produced)
    return vectors


async def _smoke_test(
    state: _RunState,
    points: Sequence[VectorPoint],
    *,
    settings: Settings,
    vector_store: VectorStore,
) -> list[SmokeQuery]:
    """发布前的抽样检索（11.1 的 Retrieval Smoke Test）。

    用 chunk 自己的正文去查，期望它出现在自己的文档版本里。这能一次抓到
    三类失败：向量压根没写进去、payload 的状态位/文档键写错、
    稀疏与稠密两路的 id 空间对不上（词表不一致时正是这个症状）。

    **过滤条件显式写 `status=PROCESSING`**：烟测跑在发布之前，
    用默认的 ACTIVE 过滤会一条都查不到，于是"冒烟通过"变成"什么都没测"
    （契约测试里有一条专门钉住这个）。
    过滤里再加 `document_ids`：烟测要回答的是"**这一版**写进去了吗"，
    不限定文档时，其它文档的相似块会把名次挤掉，冒烟会偶发地假失败。
    """
    if settings.rag.publish_smoke_queries <= 0 or not points:
        return []

    chunk_filter = ChunkFilter(
        status=DocumentStatus.PROCESSING.value,
        document_ids=(state.key,),
    )
    results: list[SmokeQuery] = []
    for point in _spread(points, settings.rag.publish_smoke_queries):
        hits = await vector_store.hybrid_rrf(
            point.dense,
            point.sparse,
            limit=settings.rag.rerank_top_k,
            chunk_filter=chunk_filter,
            rrf_k=settings.rag.rrf_k,
        )
        rank = next(
            (i for i, hit in enumerate(hits, start=1) if hit.chunk_id == point.chunk_id),
            None,
        )
        results.append(
            SmokeQuery(
                chunk_id=point.chunk_id,
                query=_preview(point.text),
                rank=rank,
                score=hits[rank - 1].score if rank is not None else None,
            )
        )
    return results


def _spread[T](items: Sequence[T], count: int) -> list[T]:
    """从序列里**均匀**取 `count` 个。

    不是取前 N 个：一篇制度的前几块几乎必然来自同一节（标题路径相同），
    只测它们等于只验证了文档开头。采用均匀取样，覆盖到尾部那些
    "表格块""附则"等结构差异最大的部分——真正容易出问题的正是它们。
    """
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


_WHITESPACE = re.compile(r"\s+")


def _preview(text: str, limit: int = 60) -> str:
    """烟测查询的展示形态。报告要能被人读懂"当时查的是什么"。"""
    flat = _WHITESPACE.sub(" ", text).strip()
    return flat if len(flat) <= limit else f"{flat[:limit]}…"


# ------------------------------------------------------------------ 状态推进


def _resume_or_reject(
    existing: KnowledgeDocumentRecord | None,
    checksum: str,
    metadata: DocumentMetadata,
    *,
    force: bool,
) -> IngestionReport | None:
    """幂等命中 → 返回现有结果；同版本不同内容 → 报错；其余 → None（继续入库）。

    三种情形分开判，顺序不能换：
    1. 内容与版本都相同且**已发布** → 11.9 的"相同指纹重复上传直接返回现有结果"；
    2. 内容不同但版本相同且没给 `force` → 拒绝（见模块 docstring 决定 3）；
    3. 其余（上一次失败要重试、`--force` 重建）→ 继续走完整流程。
    """
    if existing is None:
        return None
    if existing.checksum == checksum:
        if existing.status is DocumentStatus.ACTIVE:
            return IngestionReport(
                document_id=metadata.document_id,
                record_id=existing.id,
                logical_key=metadata.logical_key,
                version=metadata.version,
                title=existing.title,
                source_file="",
                format="",
                checksum=checksum,
                status=existing.status,
                skipped=True,
                chunk_count=existing.chunk_count or 0,
                storage_path=existing.storage_path,
                embedding_version=existing.embedding_version or "",
                parser_version=existing.parser_version or PARSER_VERSION,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
        return None
    if force:
        return None
    raise AgentError(
        ErrorCode.INVALID_ARGUMENT,
        f"{metadata.logical_key}@{metadata.version} 已存在但内容不同"
        f"（库中 {existing.checksum[:12]}…，本次 {checksum[:12]}…）。"
        "改了内容就必须升版本号——同一个版本号承载两份内容，"
        "会让此前引用过该版 chunk_id 的证据指向另一段文字。"
        "确认要原地重建请加 `force=True`（`make ingest FORCE=1`）。",
        details={
            "logical_key": metadata.logical_key,
            "version": metadata.version,
            "stored_checksum": existing.checksum,
            "incoming_checksum": checksum,
        },
    )


def _new_record(
    existing: KnowledgeDocumentRecord | None,
    metadata: DocumentMetadata,
    *,
    checksum: str,
    storage_path: str,
    embedding_version: str,
    started: datetime,
) -> KnowledgeDocumentRecord:
    """构造推进到 PROCESSING 的那一行。

    `id` 与 `created_at` 沿用已有行的：重建同一版时，这一行仍然是**同一行**
    （见 `knowledge_repo` 模块 docstring）。`chunk_count` 与 `error_summary`
    显式清空——它们描述的是"上一次的结果"，留着会让一条失败记录
    看起来像带着上一次的块数成功了。
    """
    return KnowledgeDocumentRecord(
        id=existing.id if existing is not None else new_id(IdPrefix.DOCUMENT),
        logical_key=metadata.logical_key,
        version=metadata.version,
        title=metadata.title,
        document_type=metadata.document_type,
        department=metadata.department,
        storage_path=storage_path,
        checksum=checksum,
        effective_from=metadata.effective_from,
        effective_to=metadata.effective_to,
        classification=metadata.classification,
        source_kind=metadata.source_kind,
        allowed_roles=metadata.allowed_roles,
        status=DocumentStatus.PROCESSING,
        parser_version=PARSER_VERSION,
        embedding_version=embedding_version,
        chunk_count=None,
        error_summary=None,
        created_by=metadata.created_by,
        created_at=existing.created_at if existing is not None else started,
        updated_at=started,
    )


async def _archive_original(storage: ObjectStorage, path: Path, key: str, fmt: str) -> None:
    """原文归档。**先落存储再解析**——哪怕后面解析失败，"这份文件长什么样"
    也已经留下了，日后补上 OCR 或修好解析器时要从它重新出发（11.9）。

    读文件走 `to_thread`：上限 50MB，同步读会把事件循环按住
    （`make ingest` 是逐篇跑的批处理，但 `ruff` 的 ASYNC240 指的正是这类调用，
    将来从 API 侧调进来时它就是真的阻塞）。

    ⚠️ **归档的原文始终是"最近一次尝试"的那份文件**。路径由
    `(logical_key, version)` 唯一确定（11.9），所以 `--force` 重建失败时
    新文件已经覆盖了旧文件，而库里那行被还原成了旧版本——此时
    **行的 `checksum` 与归档原文对不上**。
    这不是可以忽略的小事：`reindex` 正是拿归档原文去重建 collection 的。
    判据在入库报告里（`checksum` 是本次尝试的、`previous_checksum` 是重建前的），
    重建索引之前先比对这三个值。
    """
    payload = await asyncio.to_thread(path.read_bytes)
    await storage.put(key, payload, content_type=_FORMAT_META[fmt][1])


async def _settle_failure(
    state: _RunState,
    documents: KnowledgeDocumentRepository,
    vector_store: VectorStore,
    existing: KnowledgeDocumentRecord | None,
    summary: str,
) -> None:
    """失败收尾。**分成"动过向量库"与"没动过"两条路，判据是 `state.upserted`。**

    这个判断不是讲究，是上一版真正的缺陷：把"删掉这一版的 point"当成无条件动作，
    `--force` 重建一篇已发布文档时，只要解析或 embedding 失败，
    就会把**上一次成功发布的那批 point 一起删掉**——而那时 `state.upserted`
    还是 False，本轮一个 point 都没写。库里那行随后被置成 FAILED，
    于是检索侧少了一版，且没有任何地方会报错。

    所以：

    - **本轮写过向量库** → 整批删掉（11.9 的"失败版本不得被在线查询命中"），
      行置 FAILED。状态位只挡得住**读者**，挡不住"下一版入库时
      chunk_id 撞上这些残留"，所以删除是必须的。
    - **本轮没碰过向量库** → 库里原封不动，把行**还原成入库前那一份**。
      否则会出现最难查的一种不一致：库里说 FAILED、向量库里那一版
      仍然是 ACTIVE 且查得到，"这份文档到底能不能用"有两个答案。
      首次入库失败时 `existing` 为 None，没有可还原的东西，直接置 FAILED。
    """
    state.error_summary = summary
    if state.upserted:
        await vector_store.delete_document(state.key)
        state.status = DocumentStatus.FAILED
        state.record = await documents.save(
            touch(state.record, DocumentStatus.FAILED, error_summary=summary)
        )
        return
    if existing is not None:
        state.record = await documents.save(existing)
        state.status = existing.status
        return
    state.status = DocumentStatus.FAILED
    state.record = await documents.save(
        touch(state.record, DocumentStatus.FAILED, error_summary=summary)
    )


async def _finalize(state: _RunState, storage: ObjectStorage) -> IngestionReport:
    report = _build_report(state)
    key = report_key(state.metadata.logical_key, state.metadata.version)
    await storage.put(
        key,
        report.model_dump_json(indent=2).encode("utf-8"),
        content_type="application/json",
    )
    return report.model_copy(update={"report_path": key})


def _build_report(state: _RunState) -> IngestionReport:
    return IngestionReport(
        document_id=state.key,
        record_id=state.record.id,
        logical_key=state.metadata.logical_key,
        version=state.metadata.version,
        title=state.metadata.title,
        source_file=str(state.path),
        format=state.fmt,
        checksum=state.checksum,
        status=state.status,
        unsupported=state.unsupported,
        chunk_count=len(state.chunk_ids),
        page_count=state.page_count,
        dropped=state.dropped,
        empty_sparse_chunks=state.empty_sparse_chunks,
        smoke=state.smoke,
        published_count=state.published_count,
        storage_path=state.storage_path,
        error_summary=state.error_summary,
        parser_version=state.record.parser_version or PARSER_VERSION,
        embedding_version=state.record.embedding_version or "",
        previous_checksum=state.previous_checksum,
        started_at=state.started,
        finished_at=datetime.now(UTC),
    )


__all__ = [
    "DocumentMetadata",
    "IngestionReport",
    "SmokeQuery",
    "ingest_document",
    "original_key",
    "report_key",
    "sha256_file",
    "validate_content",
    "validate_format",
    "validate_size",
]
