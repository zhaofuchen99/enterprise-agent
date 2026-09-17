"""`verify-corpus` 的十条断言与金标装载（开发流程 6.7 的门禁）。

**每个用例都对应一个真的踩过的坑**，不是为覆盖率写的：这条命令的输出是
"[OK] 十条"，而它一旦因为**检查本身写错**而报 `[OK]`，比报 `!!` 更危险——
门禁自己失效时，没人会再去查语料。

因此这里的断言大多是"检查必须能认出人为构造的坏产物"，
而不是"检查在好产物上返回通过"。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.domain.knowledge import DocumentStatus, KnowledgeDocumentRecord, SourceKind
from app.tools.rag.golden import load_golden
from app.tools.rag.verify import (
    _claims_of,
    _defects_of,
    _doc_key,
    check_cross_page_table,
    check_furniture,
    check_prompt_injection,
    check_scanned,
)


def _record(*, status: DocumentStatus, summary: str | None) -> KnowledgeDocumentRecord:
    """一份 `knowledge_document` 的行（只填 `check_scanned` 要用的字段）。"""
    from datetime import UTC, datetime

    return KnowledgeDocumentRecord(
        id="doc_0000000000000000000001",
        logical_key="policy/demo",
        version="v1.0",
        title="扫描件",
        document_type="POLICY",
        storage_path="knowledge/policy/demo/v1.0/original.pdf",
        checksum="a" * 64,
        source_kind=SourceKind.INTERNAL,
        status=status,
        error_summary=summary,
        created_at=datetime(2026, 9, 17, tzinfo=UTC),
        updated_at=datetime(2026, 9, 17, tzinfo=UTC),
    )


def _chunk(
    document_id: str, text: str, *, section: tuple[str, ...] = (), caption: str | None = None
) -> dict[str, Any]:
    return {
        "payload": {
            "document_id": document_id,
            "text": text,
            "section_path": list(section),
            "table_caption": caption,
        }
    }


# ------------------------------------------------------------------ 金标装载


def test_golden_set_has_twenty_answerable_and_three_absent() -> None:
    """开发流程 6.7 施工项 12 明写"金标 20 条"，而 16.11.2 第 10 项要求
    3 处"涉及但不存在"。两个数字都要能被断言，不能只看总数——总数对了
    而分布错了（比如 17+6），Recall@8 的分母就不是 20。"""
    golden = load_golden()

    assert len(golden.cases) == 20
    assert len(golden.absent_cases) == 3
    assert len({case.id for case in golden.cases} | {case.id for case in golden.absent_cases}) == 23


def test_every_absent_case_carries_probes() -> None:
    """探针词是"语料里确实没有"的判据。**没有探针就等于那条断言只测了一半**：
    只剩"检索拒答了"，而拒答可能只是因为阈值调高了。"""
    for case in load_golden().absent_cases:
        assert case.absent_probes, f"{case.id} 没有探针词"


# ------------------------------------------------------------------ 清洗


def test_furniture_check_flags_a_leaked_header() -> None:
    """页眉泄进 chunk 必须被认出来。

    清洗是唯一一类"正确时无声、错误时也无声"的操作——多丢一行没有任何症状，
    而**少丢一行也没有**（页眉只是让检索多几条噪声）。这条断言要能认出后者。
    """
    chunks = [_chunk("policy/x@v1.0", "示例科技股份有限公司 华东区域渠道折扣政策 [INTERNAL]\n正文")]

    assert not check_furniture(chunks).ok


def test_furniture_check_ignores_a_document_whose_title_is_a_dropped_section_name() -> None:
    """**根标题不算"落在丢弃节"**——语料里真有一份文档叫《指标版本变更记录》。

    第一版检查拿整个 `section_path` 去匹配丢弃节标题，于是那份文档的
    全部 chunk 都被判成"清洗失败"，报的是"语料有问题"。
    它本来就该在库里；要丢的是**别的文档里的**那类小节。
    """
    chunks = [
        _chunk(
            "metric/version-change-log@v1.0",
            "指标版本变更记录 > 一、指标定义\n正文",
            section=("指标版本变更记录", "一、指标定义"),
        )
    ]

    assert check_furniture(chunks).ok


def test_furniture_check_flags_a_real_dropped_section() -> None:
    """非根层级的修订历史才该被判失败。"""
    chunks = [
        _chunk(
            "policy/x@v1.0",
            "华东区域渠道折扣政策 > 修订历史\n版本 | 日期",
            section=("华东区域渠道折扣政策", "修订历史"),
        )
    ]

    assert not check_furniture(chunks).ok


# ------------------------------------------------------------------ 跨页表格


def test_cross_page_table_reads_the_header_not_the_data_row() -> None:
    """表头行是表格块正文里**第一个**含 ` | ` 的行，不是第二个。

    第一版取的是第二个（数据行），于是每行的取值都不同、计数恒为 1，
    "表头有没有重复"永远判成没重复——三份跨页表格全部报失败，
    而报出来的原因是"表格没跨页"，指向语料。
    """
    header = "区域 | 渠道 | 额度"
    chunks = [
        _chunk(
            "policy/quota@v1.0",
            f"标题 > 明细\n表：额度表\n{header}\n华东 | 直营 | 800",
            caption="额度表",
        ),
        _chunk(
            "policy/quota@v1.0",
            f"标题 > 明细\n表：额度表\n{header}\n华南 | 直营 | 700",
            caption="额度表",
        ),
    ]
    documents = [
        {
            "id": "SP-015",
            "logical_key": "policy/quota",
            "version": "v1.0",
            "defects": ["CROSS_PAGE_TABLE"],
        }
    ]

    assert check_cross_page_table(documents, chunks).ok


# ------------------------------------------------------------------ 省份声称


def test_province_claims_reads_both_carriers() -> None:
    """同一件事在产物里有两种载体，两种都要认。

    - 编制说明表：`适用区域 | 华东（含 3 个省份）`
    - 对照表行：  `华东 | R01 | 3`

    第一版只按想象的写法（`覆盖 N 个省份`）写正则，两份文档一条都没匹配上，
    报的是"产物里没有省份数声明"——**一个指向语料的结论，实际是检查没对上**。
    """
    prose = _claims_of("适用区域 | 华东（含 3 个省份）")
    table = _claims_of("区域 | 区域编码 | 下辖省份数\n华南 | R02 | 3\n华东 | R01 | 3")

    assert prose == {"华东": 3}
    assert table == {"华南": 3, "华东": 3}


# ------------------------------------------------------------------ 扫描件


def test_scanned_check_rejects_a_failed_doc_that_still_has_chunks() -> None:
    """标了 FAILED 却仍有 chunk 在向量库里 → 必须报失败。

    那正是 11.9「失败版本不得被在线查询过滤条件命中」被破坏的形态：
    状态位挡住了**读者**，但残渣还在，下一版入库时会与它撞 chunk_id。
    """

    scanned = _record(status=DocumentStatus.FAILED, summary="无文本层：扫描件 PDF 不支持")
    documents = [
        {"id": "SP-016", "logical_key": "policy/demo", "version": "v1.0", "defects": ["SCANNED"]}
    ]
    leaked = [_chunk("policy/demo@v1.0", "残留的正文")]

    assert not check_scanned(documents, [scanned], leaked).ok
    assert check_scanned(documents, [scanned], []).ok
    # 状态不是 FAILED 也要报——它意味着这一份被当成正常文档发布了
    assert not check_scanned(
        documents,
        [_record(status=DocumentStatus.ACTIVE, summary=None)],
        [],
    ).ok


# ------------------------------------------------------------------ 注入串


def test_prompt_injection_check_requires_the_marker_in_the_product() -> None:
    """注入串必须在 chunk 正文里——**它是被引用内容，不是被剔除内容**。

    11.8 第 4 条要求文档里的命令式文字"只作为被引用内容"；
    把它从证据里删掉反而是篡改原文，而"引用与原文一致"是 22.3 要测的。
    """
    source = Path("configs/corpus_handwritten/injection_policy.md")
    body = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("<!--", "-->"))
    ]
    marker = max(body, key=len)[:28]
    documents = [
        {
            "id": "SP-020",
            "logical_key": "policy/x",
            "version": "v1.0",
            "defects": ["PROMPT_INJECTION"],
            "injection": source.name,
        }
    ]

    present = [_chunk("policy/x@v1.0", f"正文…{marker}…后文")]
    stripped = [_chunk("policy/x@v1.0", "正文里没有那句话")]

    assert check_prompt_injection(documents, present).ok
    assert not check_prompt_injection(documents, stripped).ok


# ------------------------------------------------------------------ 文档键


def test_doc_key_matches_the_payload_document_id_shape() -> None:
    """`logical_key@version` 必须与 Qdrant payload 的 `document_id` 逐字相同。

    两边各拼一次就会在某个版本号写法上分叉，而分叉的表现是
    "这条断言说语料里没有，其实有"——检查报失败，语料却是好的。
    """
    assert _doc_key({"logical_key": "policy/x", "version": "v2.0"}) == "policy/x@v2.0"
    # 清单里 PD-* 是紧凑写法，版本由生成器补默认值
    assert _doc_key({"logical_key": "product/p1"}) == "product/p1@v1.0"


@pytest.mark.parametrize("missing", ["defects", "logical_key"])
def test_defects_of_tolerates_entries_without_that_defect(missing: str) -> None:
    """清单里大多数条目没有 `defects` 字段，取它时不能炸。"""
    documents: list[dict[str, Any]] = [
        {"id": "SP-001"},
        {"id": "SP-002", "defects": ["VERSION_PAIR"]},
    ]

    assert [doc["id"] for doc in _defects_of(documents, "VERSION_PAIR")] == ["SP-002"]
