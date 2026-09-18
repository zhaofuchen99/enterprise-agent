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

from app.agent.prompts.analysis import ANALYSIS_PROMPT
from app.agent.schemas.analysis import AnalysisResult, InvestigationStep, SupportedClaim
from app.agent.state import AgentState
from app.core.errors import AgentError
from app.domain.evidence import Evidence
from app.infrastructure.model_gateway import ModelGateway

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
        lines.append(f"[{label}] （来源：{source}｜{item.title}）\n{item.claim[:_EVIDENCE_CHARS]}")
    return "\n\n".join(lines)


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
        if result.empty:
            limits.append(
                f"{result.step_id}（{result.error_code}）按当前条件未取得结果，"
                "相应结论缺少该来源的支撑"
            )
        elif result.status.value == "FAILED":
            limits.append(f"{result.step_id} 执行失败（{result.error_code}），该来源未纳入分析")
    for question in state.get("open_questions") or []:
        limits.append(f"未解决的问题：{question}")
    for error in state.get("errors") or []:
        limits.append(f"执行期间的错误：{error.code.value} - {error.message}")
    return tuple(limits)


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
        # **claims 留空而不是把证据拼成结论**：见模块 docstring
        claims=(),
        limitations=(
            f"分析生成失败：{exc.message}",
            f"已取得 {len(evidence)} 条证据，但未经综合分析，不能作为结论使用",
        ),
    )


__all__ = ["build_analysis_node"]
