"""文档解析：文件 → 结构化块（详细设计 11.1 的 Parse 与 Normalize 两步 / 11.2）。

## 这一层要解决的问题

11.2 列了四种格式，它们**能提供的信息量差别很大**，而 11.3 的分块要求
「表格以表头 + 当前行序列化」「页码写入 metadata」「标题路径保留」——
三件事都要求解析层先把结构吐出来，否则分块只能拿到一坨纯文本：

| 格式 | 标题 | 表格 | 页码 |
|---|---|---|---|
| PDF | 靠**版式约定**认（`一、` / `第X条`） | `pdfplumber` 能还原结构 | 有 |
| DOCX | 段落样式直接给 | 原生结构 | 无（DOCX 没有稳定页概念） |
| Markdown | `#` 层级 | 竖线表格 | 无 |
| TXT | setext 下划线 | **不支持**，按段落处理（见 `_parse_text`） | 无 |

四种格式各有一个解析器，归一化成同一组 `Block`，
让下游的分块只面对一种形状。

## PDF 为什么按表格外接框分带

`page.extract_text()` 与 `page.extract_tables()` 会**各自包含表格内容**，
两者都用就等于把表格正文在 chunk 里放两遍。反过来只取文本又拿不到表头。
本模块的做法是把页面按表格的外接框切成若干水平带：带内取文本、带就是表格，
再按纵坐标拼回阅读顺序。这样表格在文档里的**位置**也保住了——
位置决定它落在哪个标题路径下，而标题路径正是「这个数字属于哪一节」的依据。

## 清洗掉什么，以及为什么必须报出来

11.3 要求「去除重复页眉页脚」，16.11.2 又把「页眉页脚 / 修订历史 / 附件目录」
列为必须能被清洗的缺陷。本模块处理三类：

1. **页眉**：页首行，且（跨页重复）或（以 `[INTERNAL]`/`[CONFIDENTIAL]` 密级标记结尾）；
2. **页脚**：页尾行，且（跨页重复）或（含「第 N 页」）；
3. **修订历史 / 版本变更记录 / 附件目录**整节。

「跨页重复」要先把数字归一成 `#` 再比：页脚里的页码每页都不同，
按原文比会一页都匹配不上——这正是清洗规则最常见的失效方式，且**不报错**。

**所有被丢掉的文本都进 `ParsedDocument.dropped`**。清洗是唯一一类「正确时无声、
错误时也无声」的操作：多丢一行不会有任何症状，直到某天有人问「制度里明明写了」。
把丢弃项交出来，入库报告与 `make chunk` 才能把它显示给人看。

## 扫描件不是错误

图片型 PDF 没有文本层，`extract_text()` 返回空。**不抛异常**——
详设 11.2 允许「走 OCR 或明确标记不支持」，而 16.11.2 指定的期望行为是
`knowledge_document.status = FAILED` 且 `error_summary` 写明原因（切片内不做 OCR）。
解析层只如实报告 `has_text_layer=False`，由入库层决定怎么记这笔账。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from app.core.errors import AgentError, ErrorCode

#: 块类型。**只有三种**：分块只需要区分「这是标题（决定路径）」「这是正文」
#: 「这是表格（要按表头+行序列化）」，再多的类型下游也用不上。
BlockKind = Literal["HEADING", "PARAGRAPH", "TABLE"]

#: 解析 + 清洗 + 分块这条链的版本，落进 `knowledge_document.parser_version`（16.8）。
#:
#: **一个版本号盖三个环节，是有意的**：三者共同决定"同一份文件切出哪些 chunk"，
#: 而 16.8 只留了一列。分开记的话，改清洗规则而 parser_version 没动，
#: 库里会显示"还是那一版建的"，于是没人知道该不该重建——
#: 而那正是这一列存在的唯一理由（11.5：换规则必须全量重建）。
#: 因此**改动 parser.py 或 chunker.py 里任何影响输出的逻辑，都要升它**。
#:
#: 做成字符串而不是 int：它要落 `VARCHAR(32)`，且注释里已经说明了它的复合语义，
#: 写成 `rag-2` 比写成 `2` 更不容易被误当成"解析器的第 2 版"。
PARSER_VERSION = "rag-1"

#: 支持的扩展名 → 格式名。**这里是唯一的一份**：文件校验、CLI、测试都读它，
#: 免得出现「入库放行了 .md 但 CLI 说不认识」这类两处清单不同步的问题。
SUFFIX_TO_FORMAT: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}

#: 密级标记，出现在页眉末尾（`示例科技股份有限公司 XXX [INTERNAL]`）。
#: 页眉用它作判据之一，是因为**单页文档没有"跨页重复"可依**——
#: 语料里 3 份 PDF 只有一页，只靠重复判定会漏掉它们的页眉。
_CLASSIFICATION_MARK = re.compile(r"\[(?:INTERNAL|CONFIDENTIAL)\]$")

#: 页码特征，出现在页脚（`文件编号：AR-2025 版本：v1.0 第 2 页`）。
_PAGE_NUMBER = re.compile(r"第\s*\d+\s*页")

#: PDF / TXT 的标题版式约定。**PDF 没有样式信息**，标题只能按中文公文里
#: 通行的编号写法认：`一、` `第一章` 是一级，`（一）` `第一条` 是二级。
#: 这类约定是**可枚举的**，比"用字号猜标题"稳，也比"整篇当一个正文块"有用。
_HEADING_L1 = re.compile(r"^(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+章)")
_HEADING_L2 = re.compile(r"^(?:（[一二三四五六七八九十]+）|第[一二三四五六七八九十]+条)")

#: 表格标题行（`表：分区域经营情况`）。PDF 与 TXT 里它是独立一行文本，
#: 要挂到紧随其后的表格上；DOCX / Markdown 同理。
_TABLE_CAPTION = re.compile(r"^表[：:]\s*(.+)$")

#: 纯分隔线（`------` / `====` / `____`）。Markdown 的 `---`、TXT 里表格的
#: 对齐横线都长这样。**不是内容，不进 chunk**。
_RULE_LINE = re.compile(r"^[-=_~\s]{3,}$")

#: setext 下划线：**单段**横线。区分它与 TXT 表格的对齐横线，靠的是"有没有内部空格"——
#: 生成器写的表格分隔行是 `----  ----------`（多段），标题下划线是 `----------------`（一段）。
#: 这条区别不写下来的话，表格的表头行会被当成 setext 标题，整篇的标题路径就错位了。
_SETEXT_UNDERLINE = re.compile(r"^[-=]{4,}$")

#: 整行加粗（`**指标信息**`）。Markdown 里它兼作表名与小标题，
#: 靠"下一行是不是表格"来区分（见 `_parse_markdown`）。
_WHOLE_LINE_BOLD = re.compile(r"^\*\*(.+)\*\*$")

#: 要整节丢弃的小节标题（16.11.2 第 8 项）。
#:
#: **为什么这两类可以丢**：修订记录在语料里是「v1.0 首次发布」这类台账，
#: 讲不清任何口径；附件目录同理，指向的文件根本不存在。
#: 而它们会挤占稀疏维度、稀释召回——每个 chunk 里都躺着一段和问题无关的版本号。
#: 若哪天修订记录承载了口径变更的说明（如「v1.2 起扣除退货」），
#: **就不能再丢**：那是 DEFINITION 冲突的证据。判据是内容，不是标题。
_DROPPED_SECTIONS: tuple[str, ...] = ("修订历史", "版本变更记录", "修订记录", "附件目录")


class TableBlock(BaseModel):
    """一张表。`caption` 是「表：XXX」里的 XXX，可能没有。"""

    model_config = ConfigDict(frozen=True)

    caption: str | None = None
    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()


class Block(BaseModel):
    """解析后的结构化块。

    Attributes:
        kind: HEADING / PARAGRAPH / TABLE。
        text: 标题与正文的文本；表格块为空串（内容在 `table` 里）。
        level: 标题层级，从 1 起；非标题为 0。
        page_no: 该块所在的页码。**只有 PDF 有**——
            DOCX 没有稳定的页概念（分页由渲染器决定），Markdown / TXT 同理。
            给不出就不给，不用"默认第 1 页"糊弄过去：那会让页码过滤静默失效。
        table: 表格内容。
    """

    model_config = ConfigDict(frozen=True)

    kind: BlockKind
    text: str = ""
    level: int = 0
    page_no: int | None = None
    table: TableBlock | None = None


class ParsedDocument(BaseModel):
    """一篇文档的解析结果。

    Attributes:
        blocks: 按阅读顺序排列的块。
        page_count: 页数；非 PDF 为 0（表示"不适用"，不是"零页"）。
        has_text_layer: 是否抽到了文字。**图片型扫描件为 False**，
            入库层据此把文档标记为 FAILED（11.2 / 16.11.2）。
        dropped: 清洗掉的文本，供入库报告与排查展示。见模块 docstring。
    """

    model_config = ConfigDict(frozen=True)

    blocks: tuple[Block, ...] = ()
    page_count: int = 0
    has_text_layer: bool = True
    dropped: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """正文纯文本（**不含表格**）。

        表格的文本形态由分块层决定（`chunker.serialize_table`），
        这里再给一份只会多出第二种序列化写法，而两份写法迟早会不一致。
        本属性用于调试与"清洗后还剩多少正文"的快速核对。
        """
        return "\n".join(b.text for b in self.blocks if b.kind != "TABLE")


class TableLike(Protocol):
    """`pdfplumber` 的表格对象与本模块之间唯一的约定。

    用它而不是直接标注 `pdfplumber.table.Table`，是因为**类型从外部库进来后
    会一路渗到测试替身里**：测试要造一张假表就得构造一个真的 pdfplumber 对象。
    这里只依赖"能取到 bbox 与二维文本"这两件事，替身随便写。
    """

    bbox: tuple[float, float, float, float]

    def extract(self) -> list[list[str | None]]: ...


# --------------------------------------------------------------------- 入口


def detect_format(path: Path) -> str:
    """按扩展名判格式。不认得的扩展名**报错而不是猜**。"""
    fmt = SUFFIX_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"不支持的文件格式：{path.suffix or '（无扩展名）'}",
            details={"path": str(path), "supported": "、".join(sorted(SUFFIX_TO_FORMAT))},
        )
    return fmt


def parse_document(path: str | Path) -> ParsedDocument:
    """解析一篇文档。格式按扩展名判定（`detect_format`）。"""
    file = Path(path)
    fmt = detect_format(file)
    if not file.is_file():
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT, "待解析的文件不存在", details={"path": str(file)}
        )
    try:
        if fmt == "pdf":
            return _parse_pdf(file)
        if fmt == "docx":
            return _parse_docx(file)
        if fmt == "md":
            return _parse_markdown(file.read_text(encoding="utf-8"))
        return _parse_text(file.read_text(encoding="utf-8"))
    except AgentError:
        raise
    except Exception as exc:
        # 损坏的文件、编码不对的文本：都是"调用方给的这份文件用不了"，
        # 属于入参问题而不是本模块的 bug。details 带上原因便于定位，
        # **不带堆栈**（19.4 脱敏纪律；堆栈进日志由调用方决定）。
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            "文件解析失败，可能已损坏或编码不受支持",
            details={"path": str(file), "error": type(exc).__name__, "reason": str(exc)[:200]},
        ) from exc


# --------------------------------------------------------------------- 公共工具


def _clean_cell(value: Any) -> str:
    """单元格文本归一：`None` → 空串，内部空白压成单个空格。

    PDF 抽取出的单元格常带换行（`华东\\nR01`），不清掉的话换行会跟着进 chunk。
    等归一化那一步再处理就太晚了——表格序列化时已经按行拆错了。
    """
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _classify_line(line: str) -> tuple[BlockKind, int]:
    """按版式约定判断一行是标题还是正文，返回 (类型, 层级)。"""
    if _HEADING_L1.match(line):
        return "HEADING", 1
    if _HEADING_L2.match(line):
        return "HEADING", 2
    return "PARAGRAPH", 0


def _join_wrapped(lines: list[str]) -> str:
    """把被排版拆行的段落拼回一段。

    **中文之间不插空格**：`…退货冲减 4,479.62` + `万元。` 拼成 `…4,479.62万元。`
    才是原文的样子；插了空格反而切出 `4,479.62 万元` 这种原文里不存在的 token 形态。

    两侧都是 ASCII 字母数字时才插空格（英文被拆行的情况），
    否则 `net` + `sales` 会拼成 `netsales`——一个谁都匹配不到的词。
    """
    out = ""
    for line in lines:
        if out and _needs_space(out[-1], line[0]):
            out += " "
        out += line
    return out


def _needs_space(left: str, right: str) -> bool:
    return left.isascii() and left.isalnum() and right.isascii() and right.isalnum()


def _text_to_blocks(
    text: str, page_no: int | None, dropped: set[str] | None = None
) -> tuple[list[Block], str | None]:
    """一段文本 → 块序列。返回 (块, 末尾未消费的表名)。

    表名单独返回是因为它要挂到**紧随其后的表格**上，而表格不一定在同一段文本里
    （PDF 里表名在文本带内、表格是下一个带）。交给调用方跨带传递。

    页眉页脚在这里、**在拼段之前**剔除：`_join_wrapped` 会把多行合成一行，
    合完再按行比对就永远匹配不上——这类"顺序放反了所以规则失效"的写法不报错，
    只会让页脚安静地进 chunk。
    """
    blocks: list[Block] = []
    buffer: list[str] = []
    caption: str | None = None

    def flush() -> None:
        if buffer:
            blocks.append(Block(kind="PARAGRAPH", text=_join_wrapped(buffer), page_no=page_no))
            buffer.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or _RULE_LINE.match(line):
            continue
        if dropped and _normalize_furniture(line) in dropped:
            continue
        matched_caption = _TABLE_CAPTION.match(line)
        if matched_caption:
            flush()
            caption = matched_caption.group(1).strip()
            continue
        kind, level = _classify_line(line)
        if kind == "HEADING":
            flush()
            blocks.append(Block(kind="HEADING", text=line, level=level, page_no=page_no))
            continue
        buffer.append(line)
    flush()
    return blocks, caption


def _table_block(table: TableLike, caption: str | None, page_no: int | None = None) -> Block:
    """`pdfplumber` 的表格对象 → Block。

    **第一行当表头**：这是 11.3「表头 + 当前行」序列化的前提。
    没有表头的表格（罕见）退化成"首行也是数据"，序列化时靠空表头兜底。
    """
    rows = [_clean_row(row) for row in table.extract()]
    return Block(kind="TABLE", page_no=page_no, table=_build_table(rows, caption))


def _clean_row(row: list[Any]) -> list[str]:
    return [_clean_cell(cell) for cell in row]


def _build_table(rows: list[list[str]], caption: str | None) -> TableBlock:
    """二维文本 → TableBlock。第一行是表头，其余是数据行。"""
    kept = [row for row in rows if any(row)]
    if not kept:
        return TableBlock(caption=caption)
    return TableBlock(
        caption=caption,
        header=tuple(kept[0]),
        rows=tuple(tuple(row) for row in kept[1:]),
    )


# --------------------------------------------------------------------- 页眉页脚


def _normalize_furniture(line: str) -> str:
    """页眉页脚比对用的归一形态：数字一律变 `#`。

    页脚是 `文件编号：AR-2025 版本：v1.0 第 2 页`，每页只有页码不同。
    按原文比对会**一页都匹配不上**，跨页重复判定直接失效——
    而失效的表现是"页脚进了 chunk"，不报错、不告警。
    """
    return re.sub(r"\d+", "#", line.strip())


def _detect_page_furniture(page_texts: list[str]) -> tuple[set[str], list[str]]:
    """从各页文本里认出页眉页脚。

    返回 (归一化后的匹配集合, **原文**样例)。两者都要：
    匹配集合用于剔除，原文用于报告——报告里给出归一化后的
    `文件编号：AR-# 版本：v#.# 第 # 页` 没人看得懂，
    而"为什么这几行不见了"恰恰要靠它来回答。
    """
    first_lines: list[str] = []
    last_lines: list[str] = []
    for text in page_texts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])

    repeated_first = _repeated(first_lines)
    repeated_last = _repeated(last_lines)

    matched: set[str] = set()
    samples: dict[str, str] = {}
    for line in first_lines:
        # 单页文档没有"跨页重复"可依，密级标记是它唯一的结构特征
        if _normalize_furniture(line) in repeated_first or _CLASSIFICATION_MARK.search(line):
            matched.add(_normalize_furniture(line))
            samples.setdefault(_normalize_furniture(line), line)
    for line in last_lines:
        if _normalize_furniture(line) in repeated_last or _PAGE_NUMBER.search(line):
            matched.add(_normalize_furniture(line))
            samples.setdefault(_normalize_furniture(line), line)
    return matched, list(samples.values())


def _repeated(lines: list[str]) -> set[str]:
    """出现在 ≥2 页里的行（归一化后）。

    **阈值取 2 而不是"过半"**：两页的文档只要页眉一致就该认出来，
    按比例判定会让两页文档（语料里占多数）整体失效。
    """
    counts: dict[str, int] = {}
    for line in lines:
        key = _normalize_furniture(line)
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= 2}


# --------------------------------------------------------------------- 章节清洗


def _inherit_captions(blocks: list[Block]) -> list[Block]:
    """让跨页表格的续接部分继承表名（16.11.2 的「表头还原」）。

    一张表撑破一页时，后续页会**重复表头**（生成器用 `repeatRows=1` 埋的坑），
    `pdfplumber` 因此给出若干张"表头相同、表名缺失"的独立表格。
    不继承表名的话，第 4 页那些行就只剩一行裸数据，谁也说不清它属于哪张表——
    而「表头还原正确」这条门禁问的正是这个。

    判据是**紧邻且表头完全相同**。风险在于两张真正不同的表恰好表头相同、
    又恰好紧挨着——那种情况下续接表会挂上上一张的表名。
    两害相权：漏继承必然让续页数据不可追溯，误继承只是多一个可读的表名，
    且**每一行都带着自己的表头**，读者仍能判出它是什么表。
    """
    out: list[Block] = []
    previous: TableBlock | None = None
    for block in blocks:
        if block.kind != "TABLE" or block.table is None:
            previous = None
            out.append(block)
            continue
        table = block.table
        if table.caption is None and previous is not None and previous.header == table.header:
            table = table.model_copy(update={"caption": previous.caption})
        previous = table
        out.append(block.model_copy(update={"table": table}))
    return out


def _drop_boilerplate_sections(blocks: list[Block]) -> tuple[list[Block], list[str]]:
    """丢掉整节样板内容（修订记录 / 附件目录）。返回 (保留的块, 被丢的文本)。

    判据是**小节标题命中**，不是"看见版本号就丢"：只丢这些标题下的内容，
    正文里提到「v1.2 起扣除退货」这类口径变更不受影响——
    **那正是 DEFINITION 冲突要用的证据**。

    命中标题后一直丢到**下一个同级或更高级标题**为止，
    避免把标题下的子节漏在外面。
    """
    kept: list[Block] = []
    dropped: list[str] = []
    skip_level: int | None = None
    for block in blocks:
        if block.kind == "HEADING":
            if any(name in block.text for name in _DROPPED_SECTIONS):
                skip_level = block.level
                dropped.append(block.text)
                continue
            if skip_level is not None and block.level <= skip_level:
                skip_level = None
        if skip_level is not None:
            if block.text:
                dropped.append(block.text)
            elif block.table is not None:
                dropped.append(f"[表格] {block.table.caption or '未命名'}")
            continue
        kept.append(block)
    return kept, dropped


# --------------------------------------------------------------------- PDF


def _parse_pdf(path: Path) -> ParsedDocument:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        pages = list(pdf.pages)
        page_texts = [page.extract_text() or "" for page in pages]
        if not any(text.strip() for text in page_texts):
            # 图片型扫描件：没有文本层。**不是错误**，见模块 docstring。
            return ParsedDocument(blocks=(), page_count=len(pages), has_text_layer=False)

        matched, furniture_samples = _detect_page_furniture(page_texts)
        blocks: list[Block] = []
        for page_no, page in enumerate(pages, start=1):
            blocks.extend(_page_blocks(page, page_no, matched))
        blocks = _inherit_captions(blocks)

    kept, section_dropped = _drop_boilerplate_sections(blocks)
    return ParsedDocument(
        blocks=tuple(kept),
        page_count=len(pages),
        has_text_layer=True,
        dropped=tuple(furniture_samples) + tuple(section_dropped),
    )


def _page_blocks(page: Any, page_no: int, dropped: set[str]) -> list[Block]:
    """一页 → 块序列（按阅读顺序）。见模块 docstring 的「按表格外接框分带」。

    表格之间的文本带用 `page.crop` 取，保证表格前后的正文都还在原来的位置上——
    位置决定它属于哪个标题路径，而标题路径决定「这个数字出自哪一节」。
    """
    tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
    blocks: list[Block] = []
    caption: str | None = None
    cursor = 0.0
    for table in tables:
        _, top, _, bottom = table.bbox
        if top > cursor:
            band = page.crop((0, cursor, page.width, top))
            band_blocks, caption = _text_to_blocks(band.extract_text() or "", page_no, dropped)
            blocks.extend(band_blocks)
        blocks.append(_table_block(table, caption, page_no))
        caption = None
        cursor = max(cursor, bottom)
    if cursor < page.height:
        band = page.crop((0, cursor, page.width, page.height))
        band_blocks, _ = _text_to_blocks(band.extract_text() or "", page_no, dropped)
        blocks.extend(band_blocks)
    return blocks


# --------------------------------------------------------------------- DOCX


def _parse_docx(path: Path) -> ParsedDocument:
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    dropped = _docx_furniture(document)

    blocks: list[Block] = []
    caption: str | None = None
    pending: list[str] = []

    def flush() -> None:
        if pending:
            blocks.append(Block(kind="PARAGRAPH", text=_join_wrapped(pending)))
            pending.clear()

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            matched = _TABLE_CAPTION.match(text)
            if matched:
                flush()
                caption = matched.group(1).strip()
                continue
            level = _heading_level(paragraph)
            if level:
                flush()
                blocks.append(Block(kind="HEADING", text=text, level=level))
                continue
            pending.append(text)
        elif child.tag == qn("w:tbl"):
            flush()
            rows = [
                _clean_row([cell.text for cell in row.cells]) for row in Table(child, document).rows
            ]
            blocks.append(Block(kind="TABLE", table=_build_table(rows, caption)))
            caption = None
    flush()

    kept, section_dropped = _drop_boilerplate_sections(blocks)
    return ParsedDocument(
        blocks=tuple(kept),
        page_count=0,
        has_text_layer=bool(kept),
        dropped=tuple(dropped) + tuple(section_dropped),
    )


def _docx_furniture(document: Any) -> list[str]:
    """DOCX 的页眉页脚**不用猜**：它们在文档结构里是独立的部件。

    四种部件都要看：默认页眉/页脚，以及"首页不同""奇偶页不同"这两组
    （Word 里勾了相应选项才会写进去）。只看默认的那一对，
    会在部分文档上悄悄漏掉一半页眉——而症状同样是"页眉进了 chunk"。
    """
    dropped: list[str] = []
    for section in document.sections:
        parts = (
            section.header,
            section.footer,
            section.first_page_header,
            section.first_page_footer,
            section.even_page_header,
            section.even_page_footer,
        )
        for part in parts:
            dropped.extend(p.text.strip() for p in part.paragraphs if p.text.strip())
    return dropped


def _heading_level(paragraph: Any) -> int:
    """DOCX 标题层级来自**段落样式**（生成器用 `add_heading(level=N)`）。

    样式名形如 `Heading 1`（英文模板）或 `标题 1`（中文模板）。
    两种都要认：用哪个由 Word 的语言版本决定，只认一种会让文档悄悄少掉全部标题，
    而症状只是"分块粒度变粗"，完全看不出是标题没认出来。
    """
    name = str(getattr(paragraph.style, "name", "") or "")
    matched = re.search(r"(?:Heading|标题)\s*(\d+)", name)
    return int(matched.group(1)) if matched else 0


# --------------------------------------------------------------------- Markdown


def _parse_markdown(text: str) -> ParsedDocument:
    """Markdown：`#` 标题、`|` 表格、其余按段落。

    `page_no` 一律为 None（**不用行号冒充页码**：行号会被下游当成页码过滤条件，
    静默筛掉一批 chunk）。
    """
    blocks: list[Block] = []
    lines = text.splitlines()
    index = 0
    pending: list[str] = []
    caption: str | None = None

    def flush() -> None:
        if pending:
            blocks.append(Block(kind="PARAGRAPH", text=_join_wrapped(pending)))
            pending.clear()

    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or _RULE_LINE.match(line):
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush()
            blocks.append(
                Block(
                    kind="HEADING",
                    text=_strip_emphasis(heading.group(2)).strip(),
                    level=len(heading.group(1)),
                )
            )
            continue
        if line.startswith("|"):
            rows, index = _read_markdown_table(lines, index - 1)
            if rows:
                flush()
                blocks.append(Block(kind="TABLE", table=_build_table(rows, caption)))
                caption = None
                continue
        matched = _TABLE_CAPTION.match(line)
        if matched:
            flush()
            caption = matched.group(1).strip()
            continue
        if line.startswith("```"):
            # 代码块：内容保留成正文。**不丢**——丢内容比多几个标记更糟。
            index = _skip_fence(lines, index)
            continue
        bold = _WHOLE_LINE_BOLD.match(line)
        if bold and _next_line_starts_table(lines, index):
            # `**指标信息**` 紧接一张 `|` 表格：它是表名，不是正文。
            # 不认出来的话，每个指标文档都会多出一块 `净销售额指标口径说明 > 指标信息`
            # 的 13 字垃圾块——它不含任何信息，却会参与召回、挤占 Top-K。
            flush()
            caption = bold.group(1).strip()
            continue
        pending.append(_strip_emphasis(line))
    flush()

    kept, section_dropped = _drop_boilerplate_sections(blocks)
    return ParsedDocument(
        blocks=tuple(kept),
        page_count=0,
        has_text_layer=bool(kept),
        dropped=tuple(section_dropped),
    )


def _strip_emphasis(line: str) -> str:
    """去掉 Markdown 强调标记。保留 `**净销售额**` 里的文字，只剥星号。"""
    return re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", line)


def _next_line_starts_table(lines: list[str], index: int) -> bool:
    """下一行（跳过空行）是不是表格。

    只在**紧邻**时才把加粗行当表名：加粗也用来写小标题，
    隔了一段还去认会把它从正文里抹掉，那比多一块垃圾块更糟。
    """
    while index < len(lines):
        candidate = lines[index].strip()
        if candidate:
            return candidate.startswith("|")
        index += 1
    return False


def _read_markdown_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    """读一张 `|` 表格，返回 (行, 下一行下标)。

    第二行是 `|---|---|` 对齐行：**它必须被当成表头分隔而不是数据**，
    否则表头会变成 `---`，11.3 的「表头 + 当前行」直接错位。
    """
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", cell or "") for cell in cells):
            rows.append(cells)
        index += 1
    return rows, index


def _skip_fence(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip().startswith("```"):
        index += 1
    return index + 1


# --------------------------------------------------------------------- TXT


def _parse_text(text: str) -> ParsedDocument:
    """纯文本：按段落切分（11.2）。

    **表格不解析**：11.2 对 TXT 的要求只有「按段落和长度切分」，
    而固定宽度表格的列边界要靠字符对齐去猜——猜错的表现是单元格内容错位，
    比"表格降级成段落"糟得多。语料里 TXT 只占 2/88，不值得为它冒这个险。

    标题仍按 setext 下划线认（`标题` + `-----`）：这是纯文本里最通行的写法，
    认出来能让这类文档也有标题路径；不认的话它们整篇只有一个路径层级。
    """
    lines = text.splitlines()
    blocks: list[Block] = []
    pending: list[str] = []
    index = 0

    def flush() -> None:
        if pending:
            blocks.append(Block(kind="PARAGRAPH", text=_join_wrapped(pending)))
            pending.clear()

    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or _RULE_LINE.match(line):
            continue
        if index < len(lines) and _SETEXT_UNDERLINE.match(lines[index].strip()):
            flush()
            blocks.append(Block(kind="HEADING", text=line, level=1))
            index += 1
            continue
        pending.append(line)
    flush()

    kept, section_dropped = _drop_boilerplate_sections(blocks)
    return ParsedDocument(
        blocks=tuple(kept),
        page_count=0,
        has_text_layer=bool(kept),
        dropped=tuple(section_dropped),
    )


__all__ = [
    "PARSER_VERSION",
    "SUFFIX_TO_FORMAT",
    "Block",
    "BlockKind",
    "ParsedDocument",
    "TableBlock",
    "TableLike",
    "detect_format",
    "parse_document",
]
