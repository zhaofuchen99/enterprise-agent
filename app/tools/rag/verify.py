"""`verify-corpus`：11 类缺陷注入的逐条检出（开发流程 6.7 的门禁）。

**11 而不是 16.11.2 的 10**：那 10 项之外多了一条
`DEFINITION_CONFLICT`（口径版本与指标目录不一致）——它是第 2 项
「口径不一致」的**另一个维度**（VALUE 是同一版口径下数字对不上，
它是两版口径），机制与判据都不同，所以独立成一条。

## 为什么断言的是**产物**而不是清单

`configs/corpus_manifest.yaml` 里写 `defects: [CROSS_PAGE_TABLE]` 只是一句声明。
第一版语料就是这样：三份「跨页表格」的标注齐全、**一张都没跨页**——
`force_split` 只是把表挪到新页开头，实测本版式下需 ≥35 行才撑破一页
（CLAUDE.md 约定 11）。清单说什么不是证据，产物里真的有什么才是。

所以这里**只从产物取值**：Qdrant 里的 chunk（`make ingest` 的产物）、
`knowledge_document` 的行、以及业务库的维度表。清单只用来提供"本该有多少条"。

## 两种标注，判据是**比对的两端在不在这里**

- `[OK]`：这条断言**在这里就是它的检出点**——比对的双方都能在离线拿到。
  例：SCOPE（文档声称的省份数 vs 业务库 `dim_region`）、
  **DEFINITION**（文档 payload 的口径版本 vs 指标目录的 `version`）。
- `[注入]`：语料侧的注入**已确认在产物里**，但真正的结论要**跨源比对**才出得来，
  而那个动作发生在一次任务里（检索到文档 + 查到库）。**不标 [OK]**——
  标了就是把"语料里有"说成"系统检得出"，而那是两件事。
  例：VALUE（报告的含税口径 vs 库里的净销售额）、
  TIME（文档自称的期间 vs 它自己的截止日——权威的另一侧 MD-010 并没有
  真的被检索进来比对）。

⚠️ **`[注入]` 不等于"系统检不出"**：13.4 的三类检测器都已经实现
（`nodes/conflict.py`），这两类在真实任务里是**报得出来的**。标注说的是
**"这条断言本身证不了那件事"**，不是"能力不存在"。

## 与 `make eval-rag` 的分工

`eval-rag` 测**检索质量**（Recall@8），本命令测**语料质量**。
两者缺一不可：语料里没有缺陷时，Recall@8 可以很漂亮而整条 RAG 链路
什么冲突都发现不了（开发流程风险 R-17 说的正是这个）。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.domain.knowledge import DocumentStatus, KnowledgeDocumentRecord
from app.domain.period import period_bounds
from app.domain.user import PermissionScope, UserRole
from app.tools.rag.golden import AbsentCase
from app.tools.rag.retriever import Retriever
from app.tools.rag.schemas import RagQueryArgs, RetrievalOutcome

MANIFEST_PATH = Path("configs/corpus_manifest.yaml")
HANDWRITTEN_DIR = Path("configs/corpus_handwritten")

#: 值冲突的机制标记（`scripts/gen_corpus.REPORT_BASIS_NOTE`）。
#: **从产物里找这句话**，而不是查清单标注：报告的销售额按含税口径列示，
#: 而 MD-001 定义的净销售额要扣退货——两者的差额就是 VALUE 冲突的由来。
_REPORT_BASIS_NOTE = "本月销售额按含税口径列示，未扣除退货冲减"

#: 报告里标注的实际统计截止日。与 MD-010 规定的"每月最后一日 24:00"不符的
#: 那几篇，就是 TIME 冲突的注入点。
_CUTOFF = re.compile(r"数据统计截止至\s*(\d{4}-\d{2}-\d{2})")

#: 区域政策里声称的省份数。**同一件事在产物里有两种载体，两种都要认**：
#:
#: - SP-015 的编制说明表：`适用区域 | 华东（含 3 个省份）`
#: - MD-008 的对照表行：  `华东 | R01 | 3`（列为 `区域 | 区域编码 | 下辖省份数`）
#:
#: 第一版只写了 `覆盖 N 个省份`（凭想象写的），两份文档一条都没匹配上，
#: 报出来的是"产物里没有省份数声明"——**一个指向语料的结论，实际是我的正则没对上**。
#: 只认 `（含 N 个省份）` 也会漏掉 MD-008，所以表格那一形也要认。
_PROVINCE_CLAIM = re.compile(r"([一-龥]{2,6})（含\s*(\d+)\s*个省份）")
_PROVINCE_TABLE_ROW = re.compile(r"^\s*(\S+?)\s*\|\s*R\d+\s*\|\s*(\d+)\s*$", re.MULTILINE)

#: 页眉页脚与修订历史的特征串。**清洗掉的内容不该出现在任何 chunk 里**——
#: 清洗是唯一一类"正确时无声、错误时也无声"的操作（见 `parser.py` 的说明）。
_FURNITURE_MARKERS = (
    "示例科技股份有限公司",
    "文件编号：",
    "[INTERNAL]",
    "[CONFIDENTIAL]",
)

#: 整节丢弃的标题（`parser._drop_boilerplate_sections`）。依据是 16.11.2 第 8 项。
_DROPPED_SECTIONS = ("修订历史", "版本变更记录", "版本历史")


@dataclass(frozen=True)
class CheckResult:
    """一条检查的结果。

    `injected_only=True` 表示这条**只**验证了语料侧（见模块 docstring 的两种标注）。
    """

    name: str
    expected: str
    detail: str
    ok: bool
    injected_only: bool = False

    @property
    def mark(self) -> str:
        if not self.ok:
            return "!!"
        return "[注入]" if self.injected_only else "[OK]"


def _load_manifest() -> list[dict[str, Any]]:
    """读语料清单，**返回生成器规范化之后的条目**。

    不能直接 `yaml.safe_load`：清单里 `PD-*` 是紧凑写法（只有 `product_code`），
    `logical_key` / `version` / `department` 由 `gen_corpus.load_manifest`
    按默认值补出来。自己再补一遍就是**第二条"这份文档的 logical_key 是什么"
    的规则**，而它与生成器分叉的表现是"断言说语料里没有，其实有"。

    导入 `scripts.` 在本项目有先例（`test_tokenizer.py` 用 `scripts.gen_dict`
    的自检式断言），而这里的方向是**单向**的：`gen_corpus` 只依赖
    `app.core.config`，不会反向 import 回来。
    """
    from scripts.gen_corpus import load_manifest

    _, specs = load_manifest(MANIFEST_PATH)
    return [
        {
            **spec.raw,
            "logical_key": spec.logical_key,
            "version": spec.version,
            "source_kind": spec.source_kind,
            "department": spec.department,
        }
        for spec in specs
    ]


def _defects_of(documents: Sequence[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [doc for doc in documents if name in (doc.get("defects") or [])]


def _doc_key(spec: dict[str, Any]) -> str:
    """清单条目 → `logical_key@version`。

    **只有一个地方拼它**：这个键要与 Qdrant payload 的 `document_id`
    （`metadata.document_key`）逐字相同，两边各拼一次就会在某个版本号写法上分叉，
    而分叉的表现是"这条断言说语料里没有，其实有"。
    """
    return f"{spec['logical_key']}@{spec.get('version') or 'v1.0'}"


def _text_of(chunks: Sequence[dict[str, Any]]) -> dict[str, str]:
    """`document_id` → 该文档全部 chunk 正文的拼接。"""
    merged: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        payload = chunk["payload"]
        merged[payload["document_id"]].append(payload.get("text") or "")
    return {key: "\n".join(parts) for key, parts in merged.items()}


def check_furniture(chunks: Sequence[dict[str, Any]]) -> CheckResult:
    """页眉页脚与修订历史必须**一个 chunk 都不进**（16.11.2 第 8 项）。"""
    offenders: list[str] = []
    for chunk in chunks:
        payload = chunk["payload"]
        text = payload.get("text") or ""
        sections = list(payload.get("section_path") or [])
        path = " > ".join(sections)
        for marker in _FURNITURE_MARKERS:
            if marker in text:
                offenders.append(f"{payload['document_id']} 含页眉页脚「{marker}」")
        # **只看根标题之外的层级**：`section_path[0]` 是文档标题，
        # 而语料里真有一份文档就叫《指标版本变更记录》（MD-011，METRIC 类）。
        # 把它也算成"丢弃节"的话，那份文档的全部 chunk 都会被判成清洗失败——
        # 而它本来就该在库里，是**别的文档里的**修订历史整节要被丢掉。
        if any(name in section for name in _DROPPED_SECTIONS for section in sections[1:]):
            offenders.append(f"{payload['document_id']} 的 chunk 落在丢弃节「{path}」")
    return CheckResult(
        name="页眉页脚 / 修订历史",
        expected="全部清洗",
        detail="；".join(offenders[:5]) if offenders else f"{len(chunks)} 个 chunk 均无残留",
        ok=not offenders,
    )


def check_cross_page_table(
    documents: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]]
) -> CheckResult:
    """跨页表格的表头必须**在产物里真的重复出现**（11.3 的表头还原）。

    判据是"同一张表名下的 chunk 里，表头行出现 ≥2 次"——只数一次说明表格
    没跨页，而清单里那条标注就白写了（第一版语料正是如此）。
    """
    by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        payload = chunk["payload"]
        caption = payload.get("table_caption")
        if not caption:
            continue
        # 表格块的正文形如「标题路径 / 表：表名 / 表头 / 当前行」，
        # 含 ` | ` 的两行里**第一行是表头**（第二行是当前数据行）。
        # 取错成第二行的话，每行的取值都不相同，计数恒为 1，
        # 于是"表头有没有重复"这个判据永远判成没重复。
        lines = [line for line in (payload.get("text") or "").splitlines() if " | " in line]
        if lines:
            by_doc[payload["document_id"]][lines[0]] += 1

    problems: list[str] = []
    for spec in _defects_of(documents, "CROSS_PAGE_TABLE"):
        key = _doc_key(spec)
        repeated = [header for header, count in by_doc.get(key, Counter()).items() if count >= 2]
        if not repeated:
            problems.append(f"{spec['id']} 的表头未重复出现（表格没跨页）")
    expected = len(_defects_of(documents, "CROSS_PAGE_TABLE"))
    return CheckResult(
        name="跨页表格",
        expected=f"{expected} 份",
        detail="；".join(problems) if problems else f"{expected} 份的表头均已还原",
        ok=not problems,
    )


def check_scanned(
    documents: Sequence[dict[str, Any]],
    records: Sequence[KnowledgeDocumentRecord],
    chunks: Sequence[dict[str, Any]],
) -> CheckResult:
    """扫描件必须走"明确标记不支持"（11.2），而不是解析出 0 块之后静默成功。"""
    wanted = {_doc_key(spec) for spec in _defects_of(documents, "SCANNED")}
    ingested = {chunk["payload"]["document_id"] for chunk in chunks}
    by_key = {f"{r.logical_key}@{r.version}": r for r in records}

    problems: list[str] = []
    for key in sorted(wanted):
        record = by_key.get(key)
        if record is None:
            problems.append(f"{key} 没有入库记录")
            continue
        if record.status is not DocumentStatus.FAILED:
            problems.append(f"{key} 的状态是 {record.status.value}，应为 FAILED")
        elif "文本层" not in (record.error_summary or ""):
            problems.append(f"{key} 标了 FAILED 但没说清原因")
        if key in ingested:
            problems.append(f"{key} 被标 FAILED 却仍有 chunk 在向量库里")
    return CheckResult(
        name="扫描件 PDF",
        expected=f"{len(wanted)} 份（标记不支持）",
        detail="；".join(problems) if problems else f"{len(wanted)} 份均已标 FAILED 且无向量",
        ok=not problems,
    )


def check_prompt_injection(
    documents: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]]
) -> CheckResult:
    """注入串必须**在 chunk 正文里**（它是被引用内容，不是被剔除内容）。

    特征串的取法与生成器的自检**逐字一致**（注入文件里最长的一行前 28 字）：
    两处取法不一致时，一条本该通过的注入会被判成没注入，
    而排查方向会跑到解析器上去。
    """
    texts = _text_of(chunks)
    problems: list[str] = []
    specs = _defects_of(documents, "PROMPT_INJECTION")
    for spec in specs:
        source = HANDWRITTEN_DIR / str(spec.get("injection"))
        if not source.exists():
            problems.append(f"{spec['id']} 的手工样本缺失：{source.name}")
            continue
        body = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith(("<!--", "-->"))
        ]
        marker = max(body, key=len)[:28]
        key = _doc_key(spec)
        if marker not in texts.get(key, ""):
            problems.append(f"{spec['id']} 的注入串不在产物里")
    return CheckResult(
        name="文档内 Prompt Injection",
        expected=f"{len(specs)} 份",
        detail="；".join(problems) if problems else f"{len(specs)} 份的注入串均在 chunk 正文里",
        ok=not problems,
    )


def check_value_conflict(
    documents: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]]
) -> CheckResult:
    """报告的口径与 MD-001 的净销售额口径**确实不同**（VALUE 冲突的由来）。

    这是 `[注入]`：产物里能证明"报告的销售额按含税口径列示、未扣退货"，
    而 MD-001 定义的净销售额要扣退货——两者的差额就是冲突。
    但**把它检出来**要在一次真实任务里：得同时检索到那份报告、又查到库里的数，
    `conflict._document_conflicts` 才比得上。所以这里是 `[注入]`。
    """
    texts = _text_of(chunks)
    specs = _defects_of(documents, "VALUE_CONFLICT")
    with_note = [
        spec["id"] for spec in specs if _REPORT_BASIS_NOTE in texts.get(_doc_key(spec), "")
    ]
    missing = [spec["id"] for spec in specs if spec["id"] not in with_note]
    return CheckResult(
        name="报告与 DB 数字差异",
        expected=f"{len(specs)} 处",
        detail=(
            f"{len(with_note)} 篇的口径声明已在产物里；缺：{'、'.join(missing)}"
            if missing
            else f"{len(with_note)} 篇均带「{_REPORT_BASIS_NOTE}」，与 MD-001 口径不同；"
            "跨源比对在任务里发生（`conflict._document_conflicts`）"
        ),
        ok=not missing,
        injected_only=True,
    )


def check_time_conflict(
    documents: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]]
) -> CheckResult:
    r"""报告的统计截止日与它**自称的统计期间**对不上（TIME 冲突的由来）。

    判据是从产物的正文里**解析出实际截止日**，与它自称的那一期的最后一天比——
    不看清单的 `cutoff` 字段，因为那字段说的是"生成时想怎么截"，
    而产物里写了什么才是读者看到的。期间取清单的 `report.period`。

    ⚠️ **期间必须来自 `report.period`，不能从 `logical_key` 推**：
    这一版之前用 `(\d{4})-(\d{2})$` 从 `report/quarterly-2025-Q3` 里抠月末，
    而那个正则**匹配不上**（结尾是 `Q3` 不是 `-09`），于是直接 `continue`——
    3 处 TIME 注入里实际只检了月报那一处，另外两处**静默通过**。
    换成 `period_bounds` 之后四种期间记号（年/半年/季/月）全都覆盖。

    `[注入]` 的理由同 `check_value_conflict`：这里比的是"文档自称的期间 vs
    文档自称的截止日"，权威的另一侧（MD-010 的关账制度）并没有真的被检索比对，
    而运行时的 TIME 判据（`conflict._claim_truncated`）走的正是同一条规则。
    """
    texts = _text_of(chunks)
    problems: list[str] = []
    specs = _defects_of(documents, "TIME_CONFLICT")
    for spec in specs:
        key = _doc_key(spec)
        found = _CUTOFF.search(texts.get(key, ""))
        if found is None:
            problems.append(f"{spec['id']} 的产物里没有截止日标注")
            continue
        stated = date.fromisoformat(found.group(1))
        period = str((spec.get("report") or {}).get("period") or "")
        bounds = period_bounds(period)
        if bounds is None:
            problems.append(
                f"{spec['id']} 的 report.period「{period}」不是可识别的期间记号，"
                "判不出它自称的是哪一期"
            )
            continue
        last_day = bounds[1] - timedelta(days=1)
        if stated >= last_day:
            problems.append(
                f"{spec['id']} 的截止日 {stated} 不早于期间 {period} 的末日 {last_day}，"
                "没构成 TIME 冲突"
            )
    return CheckResult(
        name="统计截止日与自称期间不符",
        expected=f"{len(specs)} 处",
        detail=(
            "；".join(problems)
            if problems
            else f"{len(specs)} 处的截止日均早于其自称期间的末日（运行时判据同规则）"
        ),
        ok=not problems,
        injected_only=True,
    )


def check_definition_conflict(
    documents: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    versions: Mapping[str, str],
) -> CheckResult:
    """口径说明声明的口径版本与指标目录**真的有出入**（DEFINITION 冲突的由来）。

    ⚠️ **这条是 `[OK]` 而不是 `[注入]`**，理由与 VALUE/TIME 不同：那两条比的
    是"产物内部的两处声明"（或与一个硬编码惯例），而这条比的是**两个权威文件**
    ——文档侧的口径版本（清单 → payload）与目录里的 `metric.version`。
    两端都在这里，不需要检索也不需要运行时，**这就是它的检出点**。

    断言三件事（缺一条这个注入都可能以"已生效"的名义静默失效）：

    1. 带 `metric_code` 的口径说明，**它的 `metric_code` 真的进了 payload**
       —— 这一条是那个"新字段没写进 payload"的经典失败的防线：
       少了它，文档侧证据的 `metric_code` 恒为 None，DEFINITION 永远检不出来。
    2. 标了 `DEFINITION_CONFLICT` 的那篇，payload 里的版本**确实**与目录不等。
    3. 其余口径说明的版本**确实**与目录相等——清单那句"版本号必须与目录一致"
       的自动化。
    """
    payloads: dict[str, set[str | None]] = defaultdict(set)
    for chunk in chunks:
        payload = chunk["payload"]
        if payload.get("metric_code"):
            payloads[str(payload.get("document_id"))].add(payload.get("definition_version"))

    problems: list[str] = []
    flagged = 0
    for spec in documents:
        code = spec.get("metric_code")
        if not code:
            continue
        injected = "DEFINITION_CONFLICT" in (spec.get("defects") or ())
        flagged += int(injected)
        seen = payloads.get(_doc_key(spec))
        if not seen:
            problems.append(f"{spec['id']}（{code}）的 metric_code 没进 payload")
            continue
        expected = versions.get(str(code))
        if expected is None:
            problems.append(f"{spec['id']} 的 metric_code={code} 在指标目录里不存在")
            continue
        for value in sorted(seen, key=str):
            if injected and value == expected:
                problems.append(
                    f"{spec['id']} 标了 DEFINITION_CONFLICT，但 payload 里的口径版本"
                    f"（{value}）与目录（{expected}）相同，没构成不一致"
                )
            if not injected and value != expected:
                problems.append(
                    f"{spec['id']} 的口径版本 {value} 与目录 {expected} 不一致，"
                    "但清单里没标 DEFINITION_CONFLICT——一处没被登记的注入"
                )
    return CheckResult(
        name="口径版本与指标目录不一致",
        expected=f"{flagged} 处",
        detail=(
            "；".join(problems)
            if problems
            else f"{flagged} 处的口径版本与目录不符（其余口径说明均与目录一致）"
        ),
        ok=not problems and flagged > 0,
        #: **`[OK]`**：比对两端都在产物与目录里，见 docstring
        injected_only=False,
    )


def check_scope_conflict(
    documents: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    provinces: dict[str, int],
) -> CheckResult:
    """区域政策声称的省份数与业务库**实际不符**（SCOPE 冲突的由来）。

    这一条是**在这里就能检出的**：文档正文声称一个数，业务库
    `dim_region.province_count` 是另一个数，两个来源就同一件事给出不同事实。
    与 VALUE/TIME 不同，它不需要口径计算——把两个数摆在一起就够了。
    """
    texts = _text_of(chunks)
    problems: list[str] = []
    specs = _defects_of(documents, "SCOPE_CONFLICT")
    for spec in specs:
        key = _doc_key(spec)
        claims = _claims_of(texts.get(key, ""))
        if not claims:
            problems.append(f"{spec['id']} 的产物里没有省份数声明")
            continue
        # **判据是"这份文档里至少有一处与库不符"，而不是"某个指定区域不符"**：
        # 清单里 SP-015 有 `region: 华东`、MD-008 没有那个字段，
        # 按区域取会漏掉后者，而报出来的是"产物里没有省份数声明"——
        # 又一个指向语料的错误结论。逐区域比对不依赖清单里有没有那个提示字段。
        mismatched = {r: c for r, c in claims.items() if r in provinces and provinces[r] != c}
        if not mismatched:
            problems.append(f"{spec['id']} 声称的省份数与库里全部一致，没构成冲突")
    return CheckResult(
        name="区域范围不同",
        expected=f"{len(specs)} 处",
        detail="；".join(problems) if problems else f"{len(specs)} 处的声称值与 dim_region 不符",
        ok=not problems,
    )


def _claims_of(text: str) -> dict[str, int]:
    """从产物正文里取出**所有**被声称的省份数，`{区域: 声称数}`。

    两种载体都认（见 `_PROVINCE_CLAIM`），两种**都能拿到区域名**——
    散文那一种的区域名在同一个表格行里（`适用区域 | 华东（含 3 个省份）`），
    所以正则把它一起捕获，不需要另设占位键。
    """
    claims: dict[str, int] = {
        match.group(1): int(match.group(2)) for match in _PROVINCE_TABLE_ROW.finditer(text)
    }
    for match in _PROVINCE_CLAIM.finditer(text):
        claims.setdefault(match.group(1), int(match.group(2)))
    return claims


def check_source_pair(
    documents: Sequence[dict[str, Any]],
    records: Sequence[KnowledgeDocumentRecord],
    chunks: Sequence[dict[str, Any]],
) -> CheckResult:
    """外部材料与内部报告**成对存在且都已入库**（SOURCE 冲突的前提）。

    13.2 第 4 条「外部不得覆盖内部事实」（FR-SEARCH-001）在检索侧靠的是
    两者**同时进证据**、由 Conflict 模块判优先级——所以这里要验的是
    "两边都在库里"，而不是"外部被过滤掉了"。过滤掉的话，冲突永远发现不了，
    而检索看起来一切正常。
    """
    by_key = {f"{r.logical_key}@{r.version}": r for r in records}
    ingested = {chunk["payload"]["document_id"] for chunk in chunks}
    problems: list[str] = []
    pairs = [
        spec
        for spec in documents
        if spec.get("source_kind") == "EXTERNAL" and spec.get("pair_with")
    ]
    for spec in pairs:
        key = _doc_key(spec)
        if key not in ingested:
            problems.append(f"{spec['id']}（外部）未入库")
        record = by_key.get(key)
        if record is not None and record.source_kind.value != "EXTERNAL":
            problems.append(f"{spec['id']} 的 source_kind 不是 EXTERNAL")
        partner = next((d for d in documents if d.get("id") == spec.get("pair_with")), None)
        if partner is None:
            continue
        partner_key = _doc_key(partner)
        if partner_key not in ingested:
            problems.append(f"{spec['pair_with']}（内部）未入库")
        partner_record = by_key.get(partner_key)
        if partner_record is not None and partner_record.source_kind.value != "INTERNAL":
            problems.append(f"{spec['pair_with']} 的 source_kind 不是 INTERNAL")
    return CheckResult(
        name="外部与内部相悖",
        expected=f"{len(pairs)} 组",
        detail=(
            "；".join(problems)
            if problems
            else f"{len(pairs)} 组的两侧均已入库且 source_kind 正确；"
            "SOURCE 冲突的检出**尚未实现**（判据是「方向相反」，"
            "而外部材料现在是自由文本，取数要知道那是哪个指标）"
        ),
        ok=not problems,
        #: **`[注入]` 而不是 `[OK]`**：这里验的是"两支材料都在库里、
        #: `source_kind` 标对了"，而《5.5 业务规则》要的是"外部信号与内部
        #: 事实方向相反时不许覆盖内部事实"——那件事没有检测器
        #: （`conflict.py` 的清单里 SOURCE 仍在"不做"那一栏）。
        #: 只验了"材料齐备"却标 `[OK]`，就是把"语料里有"说成"系统检得出"。
        injected_only=True,
    )


async def check_version_pair(
    documents: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    retriever: Retriever,
) -> CheckResult:
    """同名制度的失效版本**不被召回**（21.8 验收 2 的"有效期过滤正确"）。

    这条要真的跑检索：只断言"两版都在库里"是不够的，两版都在正是
    VERSION_PAIR 的定义，而它要测的是**过滤生效**——失效那一版入库成功
    却查不到，才是正确行为。
    """
    pairs = _defects_of(documents, "VERSION_PAIR")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in pairs:
        grouped[spec["logical_key"]].append(spec)

    problems: list[str] = []
    checked = 0
    for logical_key, specs in grouped.items():
        ordered = sorted(specs, key=lambda s: str(s.get("version") or ""))
        older, newer = ordered[0], ordered[-1]
        if not older.get("effective_to"):
            problems.append(f"{logical_key} 的旧版没有终止日，构不成版本对")
            continue
        # 取一个**新旧版本区间之外**的时点：新的生效日起之后
        moment = date.fromisoformat(str(newer.get("effective_from")))
        outcome = await retriever.retrieve(
            RagQueryArgs(question=str(newer["title"]), as_of=moment),
            scope=PermissionScope(role=UserRole.ADMIN),
        )
        versions = {chunk.metadata.document_version for chunk in outcome.candidates}
        checked += 1
        if str(older.get("version")) in versions:
            problems.append(f"{logical_key} 在 {moment} 仍召回了失效版本 {older.get('version')}")
        if str(newer.get("version")) not in versions:
            problems.append(f"{logical_key} 在 {moment} 没有召回生效版本 {newer.get('version')}")
    return CheckResult(
        name="同名制度跨版本",
        expected=f"{len(grouped)} 组",
        detail=(
            "；".join(problems)
            if problems
            else f"{checked} 组的版本过滤均生效（失效版不召回、生效版召回）"
        ),
        ok=not problems and bool(grouped),
    )


def _judged_by(outcome: RetrievalOutcome) -> str:
    """拒答判据的可读描述。

    **两条路径报的不是同一个量**：重排生效时给的是重排分（它是这次真正的判据），
    没生效时才是稠密余弦。混着报的话，"为什么没拒答"会指向一个这次根本没
    参与判定的数。
    """
    if outcome.rerank_applied:
        if outcome.best_rerank_score is None:
            return "无候选可排"
        return f"重排分最高 {outcome.best_rerank_score:.4f}（阈值 {outcome.rerank_threshold}）"
    return f"最高余弦 {outcome.best_dense_score:.4f}"


async def check_absent(
    cases: Sequence[AbsentCase],
    corpus_text: str,
    retriever: Retriever,
) -> CheckResult:
    """「涉及但不存在」的制度：**语料里真的没有**，且问题**真的被拒答**。

    两侧都要验，缺一不可：
    - 只验"语料里没有" → 检索照常返回 8 条无关片段，用户照样拿到编造的答案；
    - 只验"被拒答" → 可能是阈值调高导致的，而语料里其实有相关内容。
    """
    problems: list[str] = []
    for case in cases:
        # 用**手写的探针词**，不用整串也不用片段：
        # - 整串匹配会因为一个虚词就对不上（「跨境出海业务管理办法」vs 语料里的写法）；
        # - 按 3 字切片会把「业务管理」「管理办法」这类通用组合算成命中，
        #   实测第一版就是这么报出假警报的。探针词逐条手写、注明实测次数，
        #   在 `eval_rag_golden.yaml` 里可复核。
        hits = [probe for probe in case.absent_probes if probe in corpus_text]
        if hits:
            problems.append(f"{case.id}：语料里出现了探针词「{'、'.join(hits)}」")
        outcome = await retriever.retrieve(
            RagQueryArgs(question=case.question), scope=PermissionScope(role=UserRole.ADMIN)
        )
        if not outcome.no_relevant_knowledge:
            problems.append(f"{case.id}：检索没有拒答（{_judged_by(outcome)}）")
    return CheckResult(
        name="无答案提问",
        expected=f"{len(cases)} 处",
        detail="；".join(problems) if problems else f"{len(cases)} 处均未命中且被拒答",
        ok=not problems,
    )


__all__ = [
    "HANDWRITTEN_DIR",
    "MANIFEST_PATH",
    "CheckResult",
    "check_absent",
    "check_cross_page_table",
    "check_furniture",
    "check_prompt_injection",
    "check_scanned",
    "check_scope_conflict",
    "check_source_pair",
    "check_time_conflict",
    "check_value_conflict",
    "check_version_pair",
]
