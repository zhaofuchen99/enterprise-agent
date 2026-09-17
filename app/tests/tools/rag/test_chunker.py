"""分块（详细设计 11.3）。

分块是"参数错了要到检索评测才看得出来"的环节：那时你面对的是 Recall@8 掉了几个点，
完全不知道是切大了、切小了、标题路径没挂上，还是重叠没生效。
所以这里的用例大多是**拿一组小块参数把边界行为逼出来**，
而不是拿默认的 800/500/100 去跑一篇真文档——后者只能证明"它没崩"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.tools.rag.chunker import (
    Chunk,
    chunk_document,
    chunk_id_for,
    serialize_table,
    serialize_table_row,
)
from app.tools.rag.parser import Block, ParsedDocument, TableBlock, parse_document

#: 语料目录。**不存在时相关用例整体跳过**：`data/` 已 gitignore，
#: 新克隆的仓库里没有语料，而那不该让单测变红（生成语料要连业务库）。
_CORPUS = Path("data/corpus")


def _settings(
    settings: Settings, *, target: int = 40, minimum: int = 20, overlap: int = 10
) -> Settings:
    """把分块参数调到能在几行文本里触发边界的量级。"""
    rag = settings.rag.model_copy(
        update={
            "chunk_target_chars": target,
            "chunk_min_chars": minimum,
            "chunk_overlap_chars": overlap,
        }
    )
    return settings.model_copy(update={"rag": rag})


def _paragraph(text: str) -> Block:
    return Block(kind="PARAGRAPH", text=text)


def _heading(text: str, level: int = 1) -> Block:
    return Block(kind="HEADING", text=text, level=level)


def _chunk(blocks: list[Block], settings: Settings, *, title: str | None = None) -> list[Chunk]:
    return chunk_document(
        ParsedDocument(blocks=tuple(blocks)),
        document_key="doc@v1.0",
        settings=settings,
        title=title,
    )


# ------------------------------------------------------------------ 标题路径


def test_section_path_starts_with_document_title(settings: Settings) -> None:
    """标题是路径的根。

    PDF 里那一行标题既没有样式也没有编号，解析层认不出它是标题——
    不显式给进来，路径就从「一、编制说明」开始，
    而「这段话出自哪份文件」在检索侧正是靠路径回答的。
    """
    chunks = _chunk(
        [_heading("一、指标定义"), _paragraph("净销售额的定义。")],
        _settings(settings),
        title="净销售额指标口径说明",
    )

    assert chunks[0].section_path == ("净销售额指标口径说明", "一、指标定义")


def test_same_level_heading_replaces_instead_of_appending(settings: Settings) -> None:
    """同级标题**替换**而不是追加。

    直接 append 会让路径越挂越长（`一、 > 二、 > 三、`），
    于是每个 chunk 都带着一串同级的兄弟标题——既没有层级含义，也污染向量。
    """
    chunks = _chunk(
        [_heading("一、甲"), _paragraph("第一段。"), _heading("二、乙"), _paragraph("第二段。")],
        _settings(settings),
        title="文件",
    )

    assert chunks[0].section_path == ("文件", "一、甲")
    assert chunks[1].section_path == ("文件", "二、乙")


def test_second_level_heading_nests_under_the_first(settings: Settings) -> None:
    chunks = _chunk(
        [_heading("二、经营回顾"), _heading("（一）整体业绩", level=2), _paragraph("正文。")],
        _settings(settings),
        title="报告",
    )

    assert chunks[0].section_path == ("报告", "二、经营回顾", "（一）整体业绩")


def test_document_title_line_is_not_repeated_in_body(settings: Settings) -> None:
    """文档自己的标题行已经由路径承载，不该再当正文进一次。

    不跳过的话首行就是「区域折扣授权额度表 > 区域折扣授权额度表」，
    重复的标题会把那个 token 的权重拉高，让"问标题"淹没"问内容"。
    """
    chunks = _chunk(
        [_paragraph("区域折扣授权额度表"), _paragraph("本表按区域核定额度。")],
        _settings(settings),
        title="区域折扣授权额度表",
    )

    assert chunks[0].text == "区域折扣授权额度表\n本表按区域核定额度。"


# ------------------------------------------------------------------ 表格


_TABLE = TableBlock(
    caption="分区域经营情况",
    header=("区域", "净销售额（万元）"),
    rows=(("华东", "49,398.95"), ("华南", "51,771.55")),
)


def test_table_row_is_serialized_with_caption_and_header() -> None:
    """11.3 的「表头 + 当前行」，且每块保留表名。

    一行裸数据脱离表头就不可解读——`华东 | 49,398.95` 是什么的 49,398.95？
    """
    text = serialize_table_row(_TABLE, ("华东", "49,398.95"))

    assert text == "表：分区域经营情况\n区域 | 净销售额（万元）\n华东 | 49,398.95"


def test_table_row_pads_short_rows_instead_of_truncating() -> None:
    """缺列的**补齐**，不截断。

    截断会让后面所有列错位，而错位的表读起来仍然"像"一张对的表，没人会去数。
    """
    text = serialize_table_row(_TABLE, ("华东",))

    assert text.splitlines()[-1] == "华东 | "


def test_serialize_table_matches_row_serialization() -> None:
    """整表序列化与单行序列化必须是**同一种写法**。

    两块拼起来要仍然是一张合法的表，否则"表头还原"只是看着像。
    """
    whole = serialize_table(_TABLE)
    lines = whole.splitlines()

    assert lines[0] == "表：分区域经营情况"
    assert lines[1] == "区域 | 净销售额（万元）"
    assert lines[2] == serialize_table_row(_TABLE, ("华东", "49,398.95")).splitlines()[-1]


def test_each_table_row_becomes_its_own_chunk(settings: Settings) -> None:
    """表格行**不参与"过短相邻块合并"**。

    把 33 行合并成一块，就等于把「表头 + 当前行」这条规则作废了。
    """
    chunks = _chunk(
        [Block(kind="TABLE", table=_TABLE, page_no=3)],
        _settings(settings),
        title="报告",
    )

    assert len(chunks) == 2
    assert all(c.is_table for c in chunks)
    assert all(c.table_caption == "分区域经营情况" for c in chunks)
    assert all(c.page_no == 3 for c in chunks)
    assert [c.text.splitlines()[-1] for c in chunks] == ["华东 | 49,398.95", "华南 | 51,771.55"]


def test_table_row_without_header_keeps_the_row(settings: Settings) -> None:
    """没有表头的表格仍要出行，而不是因为"没表头"整张丢掉。"""
    chunks = _chunk(
        [Block(kind="TABLE", table=TableBlock(rows=(("甲", "乙"),)))],
        _settings(settings),
    )

    assert len(chunks) == 1
    assert chunks[0].text.endswith("甲 | 乙")


# ------------------------------------------------------------------ 正文分块


def test_short_paragraphs_merge_within_one_heading(settings: Settings) -> None:
    """同一标题下的短段落要合并。

    不合并的话几十字就是一块（语料里的制度条文正是这种短段落），
    检索时上下文全丢——这恰是 11.3「过短相邻块在同一标题内合并」要防的。
    """
    blocks = [_heading("一、定义")] + [_paragraph(f"第{i}句话。") for i in range(6)]
    chunks = _chunk(blocks, _settings(settings, target=40, minimum=20, overlap=0))

    assert len(chunks) < 6, "短段落没有被合并"
    assert all(len(c.text) <= 60 for c in chunks)


def test_heading_breaks_the_buffer(settings: Settings) -> None:
    """新标题必须断开——否则新一节的正文会挂在上一节的标题路径下。"""
    chunks = _chunk(
        [
            _heading("一、甲"),
            _paragraph("甲的内容。"),
            _heading("二、乙"),
            _paragraph("乙的内容。"),
        ],
        _settings(settings, target=1000, minimum=10, overlap=0),
        title="文件",
    )

    assert [c.section_path[-1] for c in chunks] == ["一、甲", "二、乙"]


def test_long_paragraph_is_split_at_sentence_boundaries(settings: Settings) -> None:
    """超长段落按句子边界二次切分，且**不在半句话中间断**。

    半句话的 chunk，向量是"两半语义的平均"，两边的检索都变差。
    """
    sentences = [f"这是第{i}个用于测试的句子，长度差不多。" for i in range(8)]
    chunks = _chunk([_paragraph("".join(sentences))], _settings(settings, target=40, minimum=10))

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.text.splitlines()[-1]
        assert body.endswith("。"), f"切点没有落在句读号之后：{body!r}"


def test_overlap_carries_the_previous_tail(settings: Settings) -> None:
    """重叠让相邻块共享一段上下文（11.3 的 80–120 字）。"""
    blocks = [_paragraph(f"第{i}句足够长的话，用来把块撑开。") for i in range(6)]
    chunks = _chunk(blocks, _settings(settings, target=40, minimum=20, overlap=12))

    assert len(chunks) >= 2
    tail = chunks[0].text.splitlines()[-1]
    assert tail[-10:] in chunks[1].text


def test_zero_overlap_means_no_shared_text(settings: Settings) -> None:
    blocks = [_paragraph(f"第{i}句足够长的话，用来把块撑开。") for i in range(6)]
    chunks = _chunk(blocks, _settings(settings, target=40, minimum=20, overlap=0))

    assert len(chunks) >= 2
    assert chunks[0].text.splitlines()[-1] not in chunks[1].text


# ------------------------------------------------------------------ 身份与偏移


def test_chunk_id_is_deterministic() -> None:
    """11.5 的幂等建立在"同一份 chunk 每次派生同一个 ID"上。

    不稳定的话，重复入库会造出第二条记录而不是覆盖第一条，
    向量库里于是出现两份一模一样的内容，都参与召回。
    """
    assert chunk_id_for("policy/a@v1.0", 3) == chunk_id_for("policy/a@v1.0", 3)


def test_chunk_id_changes_with_document_or_position() -> None:
    assert chunk_id_for("policy/a@v1.0", 3) != chunk_id_for("policy/a@v1.0", 4)
    assert chunk_id_for("policy/a@v1.0", 3) != chunk_id_for("policy/a@v2.0", 3)


def test_chunk_id_matches_project_id_shape() -> None:
    """与 `app/core/ids.py` 的 26 位约定一致：`chk_` + 22 位 Crockford Base32。"""
    chunk_id = chunk_id_for("policy/a@v1.0", 0)

    assert chunk_id.startswith("chk_")
    assert len(chunk_id) == 26
    assert set(chunk_id[4:]) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_char_offsets_point_into_the_document_text(settings: Settings) -> None:
    """偏移是 11.3 要求的「原文件偏移」，取自规范化文本。"""
    blocks = [_heading("一、定义"), _paragraph("甲" * 30), _paragraph("乙" * 30)]
    chunks = _chunk(blocks, _settings(settings, target=1000, minimum=10, overlap=0))

    assert chunks[0].char_start > 0
    assert chunks[0].char_end > chunks[0].char_start


def test_scanned_document_yields_no_chunks(settings: Settings) -> None:
    """没有块就没有 chunk。**返回空表而不是抛错**——
    入库层据此记 FAILED，异常会把"这份文件读不了"变成"解析器坏了"。
    """
    parsed = ParsedDocument(blocks=(), has_text_layer=False, page_count=3)

    assert chunk_document(parsed, document_key="x@v1", settings=settings) == []


def test_chunking_is_reproducible(settings: Settings) -> None:
    """同一份输入两次分块必须得到完全一样的结果。

    这是幂等的前提：分块一旦有随机性，`chunk_id` 随序号漂移，
    第二次入库就会覆盖掉一批本不该动的记录。
    """
    blocks = [
        _heading("一、甲"),
        _paragraph("正文。" * 20),
        _heading("二、乙"),
        _paragraph("更多。" * 20),
    ]
    first = _chunk(blocks, _settings(settings))
    second = _chunk(blocks, _settings(settings))

    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


# ------------------------------------------------------------------ 真实语料上的门禁


pytestmark_corpus = pytest.mark.skipif(
    not (_CORPUS / "corpus_report.json").exists(),
    reason="演示语料未生成（data/ 已 gitignore），先执行 make corpus",
)


def _corpus_documents() -> list[dict[str, Any]]:
    payload = json.loads((_CORPUS / "corpus_report.json").read_text(encoding="utf-8"))
    return list(payload["documents"])


@pytestmark_corpus
def test_corpus_headers_and_footers_never_reach_chunks(settings: Settings) -> None:
    """**16.11.2 第 8 项：页眉页脚清洗后不进入 chunk。**

    这条缺陷的性质是"注入了但没人消费"——语料里每页都印着页眉页脚，
    能不能挡住全看清洗逻辑；挡住了没有任何反馈，漏了也不报错。
    所以断言落在产物上：把 88 篇的 chunk 全拼起来找那几句样板文字。
    """
    leaked: list[str] = []
    for doc in _corpus_documents():
        parsed = parse_document(str(doc["path"]))
        chunks = chunk_document(
            parsed,
            document_key=f"{doc['logical_key']}@{doc['version']}",
            settings=settings,
            title=str(doc["title"]),
        )
        blob = "\n".join(c.text for c in chunks)
        for fragment in ("示例科技股份有限公司", f"文件编号：{doc['id']}"):
            if fragment in blob:
                leaked.append(f"{doc['id']}: {fragment}")

    assert not leaked, f"页眉页脚进了 chunk：{leaked[:5]}"


@pytestmark_corpus
def test_corpus_cross_page_tables_keep_their_header(settings: Settings) -> None:
    """**16.11.2 第 7 项：跨页表格的表头还原正确。**

    只对清单里标了 `CROSS_PAGE_TABLE` 的文档断言：这些文档的表格撑破了页，
    续页会重复表头。要求每一行都带着表头与表名——
    缺了表名的行没人说得清它属于哪张表。
    """
    checked = 0
    for doc in _corpus_documents():
        if "CROSS_PAGE_TABLE" not in (doc.get("defects") or []):
            continue
        parsed = parse_document(str(doc["path"]))
        chunks = chunk_document(
            parsed,
            document_key=f"{doc['logical_key']}@{doc['version']}",
            settings=settings,
            title=str(doc["title"]),
        )
        rows = [c for c in chunks if c.is_table]
        assert rows, f"{doc['id']} 标注了跨页表格，却一块表格行都没切出来"
        for chunk in rows:
            assert chunk.table_caption, f"{doc['id']} 有表格行缺表名：{chunk.text!r}"
            header = chunk.text.splitlines()[-2]
            assert " | " in header, f"{doc['id']} 表格行缺表头：{chunk.text!r}"
        # 跨页的判据：这些行分布在不同页上（否则它根本没跨页）
        pages = {c.page_no for c in rows}
        assert len(pages) > 1, f"{doc['id']} 的表格行都在同一页 {pages}，跨页缺陷没被覆盖"
        checked += 1

    assert checked >= 1, "清单里没有任何 CROSS_PAGE_TABLE 文档，这条门禁等于没测"
