"""从检索候选生成证据（详细设计 13.1 / 11.7 第 9 步）。

**这是 RAG 侧「答案有证据」的全部产出。** 与 SQL 侧的 `tools/sql/evidence.py`
共用同一个 `Evidence` 模型、同一张 `agent_evidence` 表——到了 Analysis 与
Reviewer 那里，看到的只有 `Evidence`，它不关心这条来自一次 `SELECT`
还是某篇制度的第 3 节。这正是两条链路可以被并列比较、冲突检测能够
跨来源工作的前提。

## 一条候选一条证据

与 SQL 侧的「一行一条证据」同理由：Top 8 的每一条都是**彼此独立的陈述**，
把它们合并成一条会让下游无法单独引用其中某一条，
而 13.4 的冲突检测正是逐条比对的。

## `claim` 是原文，不是转述

分块的 `text` 已经自带标题路径（首行，见 `chunker.py` 的不变式），
所以它就是一条自解释、可引用的陈述。**不让模型润色**——理由同 SQL 侧：
模型转述一遍会丢精度，而后面所有一致性检查都要拿它去对账。

文档里的 `忽略以上规则`、`调用工具` 这类文本**照原样进 claim**：
11.8 第 4 条要求"只作为被引用内容，不进入系统指令"。剔除它们反而更糟——
那是**篡改证据原文**，而引用与原文不一致是 22.3 明写要测的东西。
防注入的责任在 prompt 组装侧（把文档文本放在有标签的数据字段里），不在这里。

## 两处拿不到的信息，如实留空而不是编

- **`metric_code` / `definition_version`**：分块不携带指标 code，而按
  `logical_key` 反推（`metric/order-count-definition`）是把一个自由文本约定
  当语义用。13.4 第 5 步的 DEFINITION 冲突因此暂时只覆盖 SQL 侧，
  文档侧要等 Phase 9 把文档与指标目录挂钩。**留 `None` 是不报错的**，
  所以这一条必须记在文档里，否则"冲突检测少了一半"不会有人发现。
- **`scope`**：分块里没有区域/渠道/产品线维度字段（16.11.2 的
  「区域范围不同」是文档**正文声称的**省份数与库不符，属于 Analysis
  从文本里读出来的结论，不是分块元数据）。硬塞 `{"department": ...}`
  会让 13.4 的维度比对多出一个永远对不上的键。

## `event_time` 只在生效区间**两端都有**时给出

`TimeRange` 是闭开区间，两端都不能为空（SQL 侧那条「Q3 = [7-01, 10-01)」
的论证就建立在这一点上）。语料里大量制度只有起始日、没有终止日
（长期有效），给它们编一个终止日就是**凭空造一个事实**，
而 13.4 会拿它去比 TIME 冲突。原始日期始终进 `locator`，
不丢信息，只是不进那个需要闭合区间的字段。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, time

from app.core.ids import IdPrefix, new_id
from app.domain.evidence import Evidence, Reliability, TimeRange
from app.tools.rag.metadata import ChunkMetadata
from app.tools.rag.schemas import RetrievedChunk

#: 文档类型 → 可靠性（详设 13.2 的五条优先级）。
#:
#: 13.2 不是"给每类来源打个分"，而是按**用途**判断，这里把它落成一张表：
#:
#: - 第 2 条「判断制度要求：目标日期已生效的正式制度优先」→
#:   `POLICY` / `METRIC` 是**规范性**文档，它说的就是"应该怎么做"，给 HIGH；
#: - 第 3 条「解释管理层报告：报告可作为当期叙述证据，但其数字要与数据库对账」→
#:   报告是**叙述性**的，给 MEDIUM（16.11.2 那 5 处「报告数字与 DB 差 1–2%」
#:   正是这一条在起作用）；
#: - 第 4 条「补充市场信号：权威外部来源可补充，不能改写内部指标事实」→
#:   外部材料给 LOW，而 FR-SEARCH-001「外部不得覆盖内部事实」靠的就是
#:   它与 internal 证据的等级差（Phase 9 的 SOURCE 冲突）。
_RELIABILITY: dict[str, Reliability] = {
    "POLICY": "HIGH",
    "METRIC": "HIGH",
    "REPORT": "MEDIUM",
    "PRODUCT": "MEDIUM",
    "OTHER": "MEDIUM",
}

#: 外部来源的可靠性。**不论文档类型**：13.2 第 4 条约束的是来源性质，
#: 一份外部机构写的"报告"不会因为它也叫 REPORT 就获得内部报告的地位。
_RELIABILITY_EXTERNAL: Reliability = "LOW"

#: `claim` 的长度上限。分块本身有上限（`chunk_target_chars`），
#: 这里是最后一道保险：真实语料里偶尔会有整段表格被塞进一个块的情况。
_MAX_CLAIM_CHARS = 1200


def build_document_evidence(
    chunks: Sequence[RetrievedChunk],
    *,
    question: str,
    retrieved_at: datetime | None = None,
) -> list[Evidence]:
    """Top K 条候选 → 证据列表（11.7 第 9 步）。

    **候选为空时返回空列表**，与 SQL 侧同：没有证据好过一条误导性的证据。
    而"检索为空"这件事本身在上游已经落成 `NO_RELEVANT_KNOWLEDGE`
    （见 `retriever.py` 的门禁），不靠这里再判一次。
    """
    moment = retrieved_at or datetime.now(UTC)
    return [_evidence_of(chunk, question=question, retrieved_at=moment) for chunk in chunks]


def _evidence_of(chunk: RetrievedChunk, *, question: str, retrieved_at: datetime) -> Evidence:
    meta = chunk.metadata
    return Evidence(
        id=new_id(IdPrefix.EVIDENCE),
        source_type="DOCUMENT",
        title=_title(question, chunk),
        claim=_clip(chunk.text),
        locator={
            # 定位信息取 11.7 第 9 步与 13.1 的交集。**`logical_key` 与
            # `document_version` 分列而不是给一个 `document_id`**：
            # `document_id` 是 `logical_key@version` 的拼接形态（见 `metadata.py`），
            # 而引用侧要按 `logical_key` 分组看"这份制度的各个版本"，
            # 反解一个复合键正是那条注释里警告过的做法。
            "chunk_id": chunk.chunk_id,
            "logical_key": meta.logical_key,
            "document_version": meta.document_version,
            "document_id": meta.document_id,
            "section_path": list(meta.section_path),
            "page_no": meta.page_no,
            "char_start": meta.char_start,
            "char_end": meta.char_end,
            "is_table": meta.is_table,
            "table_caption": meta.table_caption,
            # 表格行块在整张表里的位置 `[第几行, 共几行]`（11.7 第 ⑧ 步的前一半）。
            # **非表格块为 None**：那表示"这一条不是表格的一部分"，
            # 不是"表有 0 行"。下游据它判断证据是否只覆盖了表的一部分——
            # 少了它，"拿 8 行求和当区域合计"这件事在产物上看不出来。
            "table_row": list(chunk.table_row) if chunk.table_row else None,
            # 原始生效日期进 locator：`event_time` 需要闭合区间，
            # 而只有起始日的制度给不出终止日（见模块 docstring）
            "effective_from": meta.effective_from.isoformat() if meta.effective_from else None,
            "effective_to": meta.effective_to.isoformat() if meta.effective_to else None,
            "published_at": meta.published_at.isoformat() if meta.published_at else None,
            "checksum": meta.checksum,
            # 检索侧的分数**进 locator**：它解释"为什么这条被选中"，
            # 而 13.4 的比对只看 claim 与 scope，多这两个键不影响分组
            # 第 ⑧ 步补回来的行块没有检索分——**留 `None` 不填 0**，
            # 同下面 `dense_score` 的理由；`expanded` 说明它为什么没有
            "fusion_score": None if chunk.fusion_score is None else round(chunk.fusion_score, 6),
            "expanded": chunk.expanded,
            "dense_score": None if chunk.dense_score is None else round(chunk.dense_score, 6),
            "source_kind": meta.source_kind.value,
        },
        event_time=_effective_range(meta),
        retrieved_at=retrieved_at,
        # 见模块 docstring：分块不带指标 code，留空而不是按 logical_key 反推
        metric_code=None,
        definition_version=None,
        # 同上：分块没有维度字段，硬塞会让 13.4 的维度比对多出对不上的键
        scope={},
        reliability=_reliability(meta),
        # 对**正文**取哈希：同一段文字出现在两个版本里时，它们说的就是同一件事，
        # 13.4 第 1 步的"判同"要的正是这个粒度。版本差异由 locator 承担，
        # 混进哈希会让"两个版本说了同样的话"变成两条不同证据
        content_hash=hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        access_level=meta.classification,
    )


def _effective_range(meta: ChunkMetadata) -> TimeRange | None:
    """生效区间 → `TimeRange`，两端缺一即返回 None（见模块 docstring）。

    `effective_to` 按**当天结束**取 23:59:59.999999 而不是零点：
    制度写"有效期至 2025-06-30"，那一天本身仍然有效。
    这与 `vector_store._in_effective_range` 的闭区间语义是同一件事的两种表达——
    那边是日期比较（闭区间），这边是闭开区间（`TimeRange.contains` 用 `< end`），
    所以终点必须落在当天之内。两处不一致的话，同一条制度在
    "能不能被检索到"与"证据上的时间区间"上会给出不同答案。
    """
    if meta.effective_from is None or meta.effective_to is None:
        return None
    return TimeRange(
        start=datetime.combine(meta.effective_from, time.min, tzinfo=UTC),
        end=datetime.combine(meta.effective_to, time.max, tzinfo=UTC),
    )


def _reliability(meta: ChunkMetadata) -> Reliability:
    if meta.source_kind.value == "EXTERNAL":
        return _RELIABILITY_EXTERNAL
    return _RELIABILITY.get(meta.document_type, "MEDIUM")


def _title(question: str, chunk: RetrievedChunk) -> str:
    """标题 = 制度名 + 末级章节。

    **不带 `question`**（SQL 侧带了）：SQL 的标题是「问题（结果第 N 行）」，
    因为一次查询的行本身没有名字；而文档分块自带 `title` 与 `section_path`，
    「华东区域渠道折扣政策 > 第二章 折扣标准」才是它在引用里该有的样子。
    把问题拼进去会让同一条制度在不同问题下的证据标题各不相同，
    而 13.4 的 `group_key()` 在无 metric_code 时正是拿 `title` 分组的。
    """
    section = chunk.metadata.section_path[-1] if chunk.metadata.section_path else ""
    return f"{chunk.metadata.title} > {section}" if section else chunk.metadata.title


def _clip(text: str) -> str:
    if len(text) <= _MAX_CLAIM_CHARS:
        return text
    return f"{text[:_MAX_CLAIM_CHARS]}…（已截断，完整内容见 locator 指向的分块）"


__all__ = ["build_document_evidence"]
