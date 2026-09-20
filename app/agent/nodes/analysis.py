"""`analysis` 节点：把证据组织成结论（详细设计 13.4 / 13.5）。

## 两件事，一件靠模型、一件不靠

- **结论的措辞靠模型**：把一堆证据组织成"人话"，这是语言模型该干的事；
- **证据的编号靠代码**：模型看到的是 `[E1]`、`[E2]`，填回来的也是编号。
  编号↔`evd_` id 的映射在这里做（见 `prompts/analysis.py` 的说明）——
  让模型逐字抄 26 位 Base32 字符串，出错率高得没有道理，
  而抄错一个字符的后果是引用指向不存在的证据。

## 模型失败时降级成"把证据列出来"，不编

降级产物是 `direct_answer` 说明"未能生成综合分析" + `limitations` 写明原因
+ `claims` 留空。**不把证据拼成一段像结论的话**：那正是"看起来正常"
的失败——用户看到的是一段有引用、有结论的答案，只是那句话不是模型写的、
也不是任何证据支持的。

## 冲突检测还没有

详设 6.1 的 `conflict_detect` 是独立节点（Phase 9）。这一版 `conflicts` 恒为空，
**不是"没检出冲突"，是"还没做检测"**——两者在最终答案里说法完全不同，
所以 `final` 在渲染时会说明这一点（见 `nodes/final.py`）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.nodes.conflict import render as render_conflicts
from app.agent.prompts.analysis import ANALYSIS_PROMPT
from app.agent.schemas.analysis import AnalysisResult, InvestigationStep, SupportedClaim
from app.agent.schemas.plan import StepResult
from app.agent.state import AgentState
from app.core.errors import AgentError
from app.domain.evidence import Evidence
from app.infrastructure.model_gateway import ModelGateway

#: 「表格证据不完整」最多列几张表。语料里的表格块占比很高（SP-015 一篇就 87 块），
#: 一次检索命中十来张表是可能的，而**限制清单是给人读的**——
#: 十几行同名句式会把真正要看的那一条淹掉。超出部分汇总成一行说明。
_MAX_TABLE_LIMITS = 3

#: 单条证据最多渲染多少字。**证据正文可能是上千字的文档分块**，
#: 全塞进 prompt 会让 token 花在"用户已经知道出处的那部分"上；
#: 而 `final` 渲染引用时用的是完整正文，不受这里影响。
_EVIDENCE_CHARS = 600

#: prompt 里最多放多少条证据。与详设 7.1 的 `evidence_max` 无关，
#: 这是**发送上限**：1871 个分块的语料里，一次问题最多也就召回十来条。
_MAX_EVIDENCE = 30


def build_analysis_node(
    gateway: ModelGateway,
) -> Callable[[AgentState], Any]:
    """构造 `analysis` 节点。"""

    async def analysis(state: AgentState) -> dict[str, Any]:
        evidence = list(state.get("evidence") or [])
        if not evidence:
            # **没有证据时不调模型**：把空列表发给它，得到的大概率是一段
            # 用常识补出来的答案——而 11.8 要防的正是这件事。
            return {
                "analysis_result": AnalysisResult(
                    direct_answer="没有检索到可支撑该问题的内容。",
                    # **这一条由代码判定，不经模型**：一条证据都没有，
                    # 就是"证据里没有问题所问的那件事"，不需要问模型。
                    refused=True,
                    conflicts=tuple(item.description for item in (state.get("conflicts") or [])),
                    limitations=tuple(_no_evidence_limitations(state)),
                )
            }

        numbered = _number(evidence)
        try:
            result = await gateway.invoke_structured(
                ANALYSIS_PROMPT,
                AnalysisResult,
                question=state.get("user_query") or "",
                evidence=_render(numbered),
                # **冲突在分析之前就算好了**（详设 6.1 的
                # `evidence_aggregate → conflict_detect → analysis`）：
                # 让模型在写结论时就知道哪里对不上，而不是写完再补一段。
                conflicts=render_conflicts(state.get("conflicts") or []),
            )
        except AgentError as exc:
            return {
                "analysis_result": _degraded(evidence, exc),
                "errors": [exc],
            }

        return {
            "analysis_result": _with_resolved_ids(result.value, numbered, state),
        }

    return analysis


def _number(evidence: list[Evidence]) -> dict[str, Evidence]:
    """证据 → `{E1: Evidence, E2: ...}`（发送顺序即编号顺序）。

    **顺序必须稳定**：`state["evidence"]` 的顺序由 reducer 的合并顺序决定，
    而模型填回来的编号要映射回同一条证据。中间再排一次序（比如按 id）
    会让编号与内容错位——那是最难发现的一类错，因为每条引用的格式都对。
    """
    return {f"E{index}": item for index, item in enumerate(evidence[:_MAX_EVIDENCE], start=1)}


def _render(numbered: dict[str, Evidence]) -> str:
    lines: list[str] = []
    for label, item in numbered.items():
        source = "业务数据库" if item.source_type == "SQL" else "企业知识库"
        lines.append(
            f"[{label}] （来源：{source}｜{item.title}{_table_note(item)}）\n"
            f"{item.claim[:_EVIDENCE_CHARS]}"
        )
    return "\n\n".join(lines)


def _table_note(item: Evidence) -> str:
    """表格行证据带上它在整张表里的位置，其余证据为空串。

    **这是给模型看的**，而模型的输入里原本没有任何东西能表明"这张表还有别的行"：
    表格按行分块，召回 8 行与召回整张表在证据列表上长得一模一样。
    实测踩到过——模型拿 8 行求和当成区域合计，真值是它的近两倍。

    写在**证据行上**而不是靠 prompt 里的一句通则：通则要求模型自己想起
    "这一条可能是残缺的"，而位置是这一条证据自带的属性，没有推理余地。
    """
    row = item.locator.get("table_row")
    if not isinstance(row, list) or len(row) != 2:
        return ""
    return f"｜该表第 {row[0]} 行，共 {row[1]} 行"


def _with_resolved_ids(
    result: AnalysisResult, numbered: dict[str, Evidence], state: AgentState
) -> AnalysisResult:
    """把模型给的 `E1` 编号换回真实证据 id。

    **未知编号整条丢掉并记账**：模型偶尔会引用一个不存在的编号
    （`[E9]` 而只给了 5 条），把它原样写进 `evidence_ids` 会让
    最终答案渲染出一个指向空处的引用——而引用列表看起来完全正常。
    丢掉之后那条 claim 仍在（它的文字是模型写的），只是没有引用，
    这一点由 `final` 的"无引用结论"检测兜住。
    """
    claims: list[SupportedClaim] = []
    for claim in result.claims:
        resolved = tuple(numbered[label].id for label in claim.evidence_ids if label in numbered)
        claims.append(claim.model_copy(update={"evidence_ids": resolved}))
    return result.model_copy(
        update={
            "claims": tuple(claims),
            # **冲突由代码写进 `conflicts`，不采用模型在正文里提没提**
            # （详设 13.3 的 `Conflict` 是结构化字段）。模型只负责在
            # `direct_answer` 里把它说清楚，那是措辞的事。
            "conflicts": tuple(item.description for item in (state.get("conflicts") or [])),
            "limitations": (*result.limitations, *_pipeline_limitations(state)),
            "investigation_chain": _chain(state),
        }
    )


def _chain(state: AgentState) -> tuple[InvestigationStep, ...]:
    """`findings` → 推理链（详设 13.5：**由代码组装，不由模型生成**）。

    这一版没有 `plan_deltas`（演进记录），所以 `triggered_step_id` 恒为
    `None`——链上只有"查了什么"，还没有"因为发现了什么所以又查了什么"。
    接 `plan_extend` 时补这一列，形状不用改。
    """
    return tuple(
        InvestigationStep(
            order=index,
            finding_id=finding.id,
            finding_statement=finding.statement,
            evidence_ids=finding.evidence_ids,
        )
        for index, finding in enumerate(state.get("findings") or [], start=1)
    )


def _pipeline_limitations(state: AgentState) -> tuple[str, ...]:
    """流程本身的限制，**必须与模型写的那些并列**。

    这些是"系统知道、模型不知道"的事实：哪一步跑了但没结果、
    哪一步失败了。只让模型写限制的话，它会漏掉自己没看到的那些
    （空结果虽然进了 prompt 的证据列表，但"少了哪一路"它推不出来）。
    """
    limits: list[str] = []
    for result in (state.get("step_results") or {}).values():
        # **没有错误码时不要写「（None）」**：SQL 的空结果走的是
        # `SUCCEEDED` + `payload.is_empty`（9.4 明写「SQL 空集不算失败」），
        # 它压根没有错误码。印出「（None）」会让读者以为哪里漏填了字段，
        # 而这只是两条"跑了但空"的路径载体不同。
        reason = f"（{result.error_code}）" if result.error_code else ""
        if result.empty:
            limits.append(f"{result.step_id}{reason}按当前条件未取得结果，相应结论缺少该来源的支撑")
            # **「看不到」不能与「查不到」共用一句话**（与约定 36 同一条理由）：
            # 两者在执行结果上完全同形——都只是"零行"——而处置相反。
            # 只说"没查到"，用户会去怀疑数据；而真相可能是他自己的授权范围。
            if result.data_scope:
                limits.append(_scope_limitation(result))
        elif result.status.value == "FAILED":
            limits.append(f"{result.step_id}{reason}执行失败，该来源未纳入分析")
    # **文档来源没有数据权限这一层**，而受限用户读到的文档是全量正文。
    # 这是取舍不是遗漏：语料是公司级报告、本身跨区域，按区域过滤会让 RAG
    # 大面积失效。但代价必须说出来——不说的话，一个只被授权看华东的用户
    # 会从文档里读到华南的数字，而"两条路给出的是同一个数"恰恰是他判断
    # "我到底能不能看"的唯一线索。范围要点名：「部分数据」等于没说。
    limits.extend(_incomplete_tables(state))
    scope = state.get("permission_scope")
    if scope is not None and not scope.unrestricted and _cites_documents(state):
        areas = "、".join(scope.region_ids)
        limits.append(
            f"本结论引用了知识库文档证据：数据权限（{areas}）只作用于数据库查询，"
            "文档内容未经过滤，可能包含授权范围之外区域的数据"
        )
    for question in state.get("open_questions") or []:
        limits.append(f"未解决的问题：{question}")
    for error in state.get("errors") or []:
        limits.append(f"执行期间的错误：{error.code.value} - {error.message}")
    return tuple(limits)


def _cites_documents(state: AgentState) -> bool:
    """本次结论是否用到了文档来源的证据。"""
    return any(item.source_type == "DOCUMENT" for item in state.get("evidence") or ())


def _incomplete_tables(state: AgentState) -> list[str]:
    """只引用了一部分行的表，逐张列出来（11.7 第 ⑧ 步的缺口）。

    **它是"表格按行分块"这个取舍的补丁**：一张 16 行的表变成 16 个块之后，
    召回 8 行与召回整张表在证据列表上完全同形。实测踩到过——模型拿 8 行
    求和当成区域合计（7,146.22 万，真值 13,249.31 万），而 Reviewer 那六条
    确定性检查没有一条看得出证据残缺。**"数没数全"是产出的属性，不是推理的结论**，
    所以它由代码判、不由模型判。

    表名相同但文档或章节不同的表**分别列出**：两张同名的表本来就该分开说。
    """
    cited: dict[tuple[str, ...], set[int]] = {}
    totals: dict[tuple[str, ...], int] = {}
    for item in state.get("evidence") or ():
        row = item.locator.get("table_row")
        if not isinstance(row, list) or len(row) != 2:
            continue
        key = (
            str(item.locator.get("document_id")),
            str(item.locator.get("section_path")),
            str(item.locator.get("table_caption")),
        )
        cited.setdefault(key, set()).add(int(row[0]))
        totals[key] = int(row[1])

    incomplete = [
        (key[2], totals.get(key, 0), len(rows))
        for key, rows in cited.items()
        if len(rows) < totals.get(key, 0)
    ]
    listed = incomplete[:_MAX_TABLE_LIMITS]
    limits = [
        f"表格「{caption}」共 {total} 行，本次证据只覆盖其中 {cited_rows} 行："
        "不得用这些行求和当作整表合计，该表的汇总值需查数据库或查阅原文"
        for caption, total, cited_rows in listed
    ]
    if len(incomplete) > len(listed):
        # **截断时如实说还差几条**：静默截断会让读者以为列出来的就是全部，
        # 而那正是这条限制要防的错觉。
        limits.append(f"另有 {len(incomplete) - len(listed)} 张表的证据同样不完整，未逐条列出")
    return limits


def _scope_limitation(result: StepResult) -> str:
    """数据权限导致的空结果，**单出一条**限制。

    写"可能源于授权范围"而不是"没有这个数据"：从一次执行的结果上，
    这两种原因本来就分不出来——把其中一种写成结论，等于替用户做了一个
    我们并没有依据的判断。而"需要与数据负责人确认"是此刻唯一可行动的下一步。
    """
    scope = "、".join(result.data_scope or ())
    return (
        f"{result.step_id}已按数据权限限定在 {scope}：未取得结果可能源于"
        "授权范围而非数据不存在，两者从本次执行无法区分，需确认权限后再下结论"
    )


def _no_evidence_limitations(state: AgentState) -> list[str]:
    """没有证据时，把"为什么没有"说清楚——它是**排查方向**。"""
    limits = ["本次检索没有取得任何证据"]
    limits.extend(_pipeline_limitations(state))
    if not (state.get("task_list") or []):
        limits.append("问题未被识别为需要查询数据源（可能是澄清或超出范围）")
    return limits


def _degraded(evidence: list[Evidence], exc: AgentError) -> AnalysisResult:
    """模型不可用时的降级产物（见模块 docstring）。"""
    return AnalysisResult(
        direct_answer=(f"已取得以下证据，但综合分析的生成失败，未能给出结论。（{exc.code.value}）"),
        # **降级不是拒答**：这里没有作答是因为模型/服务不可用，
        # 不是因为"语料里没有这件事"。混成 `True` 会让一次故障
        # 在统计上长得像一次正确的拒答——**而故障恰恰是最需要被看见的**。
        refused=False,
        # **claims 留空而不是把证据拼成结论**：见模块 docstring
        claims=(),
        limitations=(
            f"分析生成失败：{exc.message}",
            f"已取得 {len(evidence)} 条证据，但未经综合分析，不能作为结论使用",
        ),
    )


__all__ = ["build_analysis_node"]
