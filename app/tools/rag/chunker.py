"""分块（详细设计 11.3）。

输入是 `parser.py` 产出的结构化块，输出是待向量化的 chunk 列表。

## 这一层决定的其实是"检索能命中什么"

分块粒度是 RAG 里最难事后补救的参数：切大了，一个 chunk 里混着三节内容，
召回后模型要在一堆无关文字里找答案；切小了，单块信息不完整，
「华东的净销售额是多少」可能因为答案被切到隔壁块而召回不到。
11.3 给的是一组**目标值**（500–800 字、重叠 80–120），并明写
「上述默认值必须通过 RAG 评测集校准，而非视为永久常量」——
所以它们在 `rag` 配置段里，不在这个文件里当常量写死。

## 三条不变式

1. **一个自然段不因固定长度被切断**。只有超过目标块长的**单个段落**才按句子边界二次切分，
   且切点必须落在句读号之后——在半句话中间断开的 chunk，向量是"两半语义的平均"，
   两边的检索都变差。
2. **每个 chunk 都带标题路径**，且写进正文首行。挂在元数据里不够：
   元数据不参与向量化，而「销售政策 > 渠道折扣 > 华东区域」恰恰是
   「这个数字属于哪一节」的答案；不写进正文，检索就看不到它。
3. **表格行不是段落**。11.3 要求表格以「表头 + 当前行」序列化、每块保留表名——
   一行裸数据脱离表头就不可解读（`华东 | 1,050,000` 是什么的 105 万？），
   所以每行都是独立 chunk，**不参与"过短相邻块合并"**：
   把 33 行合并成一块，就等于把表头规则作废了。

## chunk_id 为什么是确定性派生

11.5 的入库幂等建立在「相同 chunk 必然得到相同 point id」上：
`point_id = BLAKE2b(chunk_id)`，所以 `chunk_id` 必须由
**文档键 + 块序号**确定性派生，重复入库天然覆盖同一条记录，不需要先查后写。
这也是 `app/core/ids.py::deterministic_id` 唯一的正当用法——
它平时警告"可预测的主键会被枚举"，而在这里，可预测正是需求本身。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.core.ids import IdPrefix, deterministic_id
from app.tools.rag.parser import Block, ParsedDocument, TableBlock

#: 句读号。中文的句子边界就是这几个，**不含逗号**——
#: 在逗号处切会把「净销售额 = 含税 − 折扣 − 退货，其中…」这种定义句拆散。
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])\s*")

#: 表格序列化时的列分隔符。用 ` | ` 而不是制表符：制表符在归一化那一步会被
#: 压成空格（`normalize` 的空白压缩），到分词时已经分不出列边界了。
_COLUMN_SEPARATOR = " | "


class Chunk(BaseModel):
    """一个待向量化的块。

    Attributes:
        chunk_id: `chk_` 前缀的 26 位 ID，由 (document_key, 序号) 确定性派生。
        text: 参与向量化的正文，**首行是标题路径**（见模块 docstring 不变式 2）。
        section_path: 标题路径，如 `("华东区域渠道折扣政策", "第二章 职责分工")`。
            与 `text` 的首行重复是刻意的：一份给人看（引用展示），一份给模型看。
        page_no: PDF 才有；其它格式与 `Block.page_no` 同样为 None。
        char_start / char_end: 在**文档规范化文本**中的区间（见 `_DocumentText`）。
        is_table: 是否表格行块。批量刷新、重排等操作要能把它挑出来。
        table_caption: 表格的表名（11.3「每块保留表名」）。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    section_path: tuple[str, ...] = ()
    page_no: int | None = None
    char_start: int = 0
    char_end: int = 0
    is_table: bool = False
    table_caption: str | None = None


class _DocumentText:
    """块的顺序拼接与偏移账本。

    `char_start` / `char_end` 是 11.3 要求的「原文件偏移」。
    偏移取自**规范化后的文本**而不是原文件字节：原文件里 PDF 的偏移是
    「第几页第几行」这种渲染器相关的东西，同一次入库重复跑都可能不同，
    拿它做定位只会得到一串对不上的数字。规范化文本的偏移是稳定的，
    也足够回答"这块内容在文档的哪个位置"。
    """

    def __init__(self, blocks: Sequence[Block]) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []
        cursor = 0
        for block in blocks:
            text = _block_text(block)
            self._starts.append(cursor)
            self._ends.append(cursor + len(text))
            cursor += len(text) + 1  # +1 是块之间的换行

    def start(self, index: int) -> int:
        return self._starts[index]

    def end(self, index: int) -> int:
        return self._ends[index]


def _block_text(block: Block) -> str:
    """块在文档文本里的形态。表格用序列化后的样子，正文用原文。"""
    if block.kind == "TABLE" and block.table is not None:
        return serialize_table(block.table)
    return block.text


def serialize_table(table: TableBlock) -> str:
    """表格 → 文本：表名 + 表头 + 数据行。

    与单行的 `serialize_table_row` 保持**同一种列序与分隔符**——
    两块拼在一起必须仍然是一张合法的表，否则"表头还原"就只是看着像。
    """
    lines: list[str] = []
    if table.caption:
        lines.append(f"表：{table.caption}")
    if table.header:
        lines.append(_COLUMN_SEPARATOR.join(table.header))
    lines.extend(_COLUMN_SEPARATOR.join(row) for row in table.rows)
    return "\n".join(lines)


def serialize_table_row(table: TableBlock, row: Sequence[str]) -> str:
    """一行 → 「表名 + 表头 + 当前行」（11.3 的表格序列化）。

    表头与当前行的**列数可能不一致**（PDF 抽取会丢空格子）。这里按表头列数补齐，
    不截断：少一列会让后面所有列错位，那比缺值是更糟的错——错位的表读起来
    仍然"像"一张对的表，没人会去数。
    """
    lines: list[str] = []
    if table.caption:
        lines.append(f"表：{table.caption}")
    if table.header:
        lines.append(_COLUMN_SEPARATOR.join(table.header))
    cells = list(row) + [""] * max(0, len(table.header) - len(row))
    lines.append(_COLUMN_SEPARATOR.join(cells[: len(table.header)] or cells))
    return "\n".join(lines)


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_key: str,
    settings: Settings,
    title: str | None = None,
) -> list[Chunk]:
    """把一篇解析好的文档切成 chunk。

    Args:
        parsed: `parse_document` 的结果。
        document_key: 文档的稳定身份（用 `logical_key@version`），
            **只参与 `chunk_id` 派生**——同一个 `chunk_id` 必须对应同一份内容，
            换文档键就是换一批 ID。
        settings: 分块参数取自 `settings.rag`（11.3 要求它们可被评测校准）。
        title: 文档标题，作为标题路径的**根**。入库时取自文档元数据（清单 / DB）。
            PDF 里那一行标题既没有样式也没有编号，解析层认不出它是标题，
            不显式给进来，路径就会从「一、编制说明」开始，
            而「这段话出自哪份文件」在检索侧正是靠路径回答的。
    """
    if not parsed.blocks:
        # 扫描件走到这里：没有块就没有 chunk。**返回空表而不是抛错**——
        # 11.2 允许"明确标记不支持"，由入库层记 FAILED 并写明原因。
        return []

    tuning = settings.rag
    offsets = _DocumentText(parsed.blocks)
    chunks: list[Chunk] = []
    section_path: list[str] = [title] if title else []
    root_depth = 1 if title else 0
    buffer: _ParagraphBuffer | None = None
    carry = ""

    def emit(
        text: str,
        first: int,
        last: int,
        page_no: int | None,
        *,
        is_table: bool = False,
        table_caption: str | None = None,
    ) -> None:
        path = tuple(section_path)
        path_line = " > ".join(path)
        chunks.append(
            Chunk(
                chunk_id=chunk_id_for(document_key, len(chunks)),
                # 标题路径写进正文首行：元数据不参与向量化，而"这个数字属于哪一节"
                # 恰恰要靠它（见模块 docstring 不变式 2）
                text=f"{path_line}\n{text}" if path_line else text,
                section_path=path,
                page_no=page_no,
                char_start=offsets.start(first),
                char_end=offsets.end(last),
                is_table=is_table,
                table_caption=table_caption,
            )
        )

    def flush() -> None:
        nonlocal buffer
        if buffer is not None and buffer.text:
            emit(buffer.text, buffer.first, buffer.last, buffer.page_no)
        buffer = None

    for index, block in enumerate(parsed.blocks):
        if index == 0 and title and block.text == title:
            # 文档自己的标题行**已经由路径承载**，不该再当正文进一次。
            # 不跳过的话首行就是「区域折扣授权额度表 > 区域折扣授权额度表」这类重复，
            # 而重复的标题会把这个 token 的权重拉高，让"问标题"淹没"问内容"。
            continue
        if block.kind == "HEADING":
            flush()
            section_path = _descend(section_path, block, root_depth)
            continue
        if block.kind == "TABLE" and block.table is not None:
            # 表格前后必然断开：把表格行并进正文块会让「表头 + 当前行」失效
            flush()
            for row in block.table.rows:
                emit(
                    serialize_table_row(block.table, row),
                    index,
                    index,
                    block.page_no,
                    is_table=True,
                    table_caption=block.table.caption,
                )
            continue

        if not block.text:
            continue
        for piece, is_continuation in _split_long_paragraph(block.text, tuning.chunk_target_chars):
            if buffer is None:
                buffer = _ParagraphBuffer(carry, index, index, block.page_no)
                carry = ""
            elif _would_overflow(buffer, piece, tuning.chunk_target_chars, tuning.chunk_min_chars):
                carry = _overlap_tail(buffer.text, tuning.chunk_overlap_chars)
                emit(buffer.text, buffer.first, buffer.last, buffer.page_no)
                buffer = _ParagraphBuffer(carry, index, index, block.page_no)
                carry = ""
            buffer.append(piece, index)
            # 二次切分出来的片段**自成一块**：它们本来就是按目标块长切的，
            # 再攒下一段就把块长又顶回去了
            if is_continuation:
                carry = _overlap_tail(buffer.text, tuning.chunk_overlap_chars)
                emit(buffer.text, buffer.first, buffer.last, buffer.page_no)
                buffer = None
    flush()
    return chunks


class _ParagraphBuffer:
    """正在累积的正文块。页码取**最后一个块**的：一个 chunk 跨了页要能说清它到哪结束。"""

    __slots__ = ("_parts", "first", "last", "page_no")

    def __init__(self, initial: str, first: int, last: int, page_no: int | None) -> None:
        self._parts: list[str] = [initial] if initial else []
        self.first = first
        self.last = last
        self.page_no = page_no

    @property
    def text(self) -> str:
        return "\n".join(self._parts)

    def append(self, piece: str, index: int) -> None:
        self._parts.append(piece)
        self.last = index


def _build_chunk(
    chunk_id: str,
    *,
    text: str,
    section_path: tuple[str, ...],
    page_no: int | None,
    char_start: int,
    char_end: int,
    is_table: bool = False,
    table_caption: str | None = None,
) -> Chunk:
    """组装 chunk：**标题路径写进正文首行**（见模块 docstring 不变式 2）。"""
    path_line = " > ".join(section_path)
    body = f"{path_line}\n{text}" if path_line else text
    return Chunk(
        chunk_id=chunk_id,
        text=body,
        section_path=section_path,
        page_no=page_no,
        char_start=char_start,
        char_end=char_end,
        is_table=is_table,
        table_caption=table_caption,
    )


def chunk_id_for(document_key: str, ordinal: int) -> str:
    """`(document_key, 序号)` → `chk_` + 22 位 Base32。

    序号是**块在文档内的下标**，不是内容的哈希：文档改一个字，
    其后所有 chunk 的 ID 都会变——这正是想要的。11.9 的幂等是
    「同一版本重复入库不产生新记录」，而版本变了本就该是另一批记录
    （`(logical_key, version)` 唯一，文档版本是显式的）。
    用内容哈希反而会让"改一个字"变成"多一条孤立记录"，
    旧记录不会被覆盖，只会在检索里和新记录打架。
    """
    return deterministic_id(IdPrefix.CHUNK, f"{document_key}#{ordinal}")


def _descend(section_path: list[str], heading: Block, root_depth: int) -> list[str]:
    """按标题层级维护路径：同级或更低级的标题**替换**掉它覆盖的那一段。

    直接 append 会让路径越挂越长（`一、` → `一、 > 二、` → `一、 > 二、 > 三、`），
    于是每个 chunk 都带着一串同级的兄弟标题，既没有层级含义也污染向量。

    `root_depth` 是**根的层数**（给了文档标题就是 1），层级要**加**在它上面：
    `depth = root_depth + (level - 1)`。

    这个加号不能省，也不能写成取 max。写成 `max(level - 1, root_depth)` 时，
    一级标题碰巧是对的，**二级标题会把它的一级父标题挤掉**——
    `二、经营业绩回顾 > （一）整体业绩` 变成只剩 `（一）整体业绩`。
    那看起来仍然像一个合理的路径，只是层级塌了一层，
    直到有人问「这个数字出自哪一节」才发现中间那层没了。
    """
    depth = root_depth + max(heading.level - 1, 0)
    return [*section_path[:depth], heading.text]


def _would_overflow(buffer: _ParagraphBuffer, piece: str, target: int, minimum: int) -> bool:
    """再放一段就超目标块长——且当前已经够长。

    **`minimum` 这个条件是必要的**：没有它，短段落会被一个个单独成块
    （第一段 200 字、第二段一来就超了），几十字一块，检索时上下文全丢了。
    语料里制度条文正是这种短段落。
    """
    current = len(buffer.text)
    return current + len(piece) + 1 > target and current >= minimum


def _split_long_paragraph(text: str, target: int) -> list[tuple[str, bool]]:
    """超长段落按句子边界二次切分（11.3）。

    返回 (片段, 是否为切分产物)。只有**单个段落就超长**时才切——
    "一个自然段不因固定长度被无意义切断"指的是正常段落，
    而一段两千字的制度如果整块入库，它自己就是一个大杂烩。
    """
    if len(text) <= target:
        return [(text, False)]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + len(sentence) > target:
            pieces.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return [(piece, True) for piece in pieces]


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """取上一块的结尾若干**整句**作为下一块的前缀（11.3 的重叠）。

    按整句取而不是按字符数硬截：硬截会造出一个从半句话开始的 chunk，
    它的向量是"半句 + 新内容"的平均，两边都代表不了。
    """
    if overlap_chars <= 0:
        return ""
    sentences = [s for s in _SENTENCE_END.split(text) if s]
    tail = ""
    for sentence in reversed(sentences):
        if len(tail) + len(sentence) > overlap_chars * 2:
            break
        tail = sentence + tail
        if len(tail) >= overlap_chars:
            break
    if not tail:
        return ""
    return tail.strip()
