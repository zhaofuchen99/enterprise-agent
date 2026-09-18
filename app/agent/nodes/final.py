"""`final` 节点：渲染最终答案（详细设计 6.2 的 `final_answer`）。

## 它**不写**新结论，只组装

所有的判断在 `analysis` 里做完了。`final` 只做四件事：按固定版式拼 Markdown、
把 `evidence_ids` 展开成引用、把限制列出来、把失败路径也渲染成人能读的东西。

**为什么是代码渲染而不是再让模型润色一遍**：润色会引入新的措辞，
而这条答案要能逐句回溯到证据（Reviewer 在 Phase 8 会按引用核对措辞，
详见 14.3）。模型每润色一次，那句"根据制度，净销售额应扣退货"就可能
变成"净销售额需要扣除退货"——**看起来一样，但它现在是一句没有引用来源的话**。

## 引用渲染成什么

`[E1]` 这类编号在这里**被换回可读的出处**（制度名 + 章节，或 SQL 的查询指纹），
而不是 26 位的 `evd_` id：答案是给人看的，`evd_xxx` 对人没有意义；
需要 id 的场合（Reviewer 核对、前端跳转）从 `answer_payload.evidence` 取。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.schemas.analysis import AnalysisResult
from app.agent.state import AgentState
from app.domain.evidence import Evidence


def build_final_node() -> Callable[[AgentState], Any]:
    """构造 `final` 节点。**纯函数式，无外部依赖。**"""

    def final(state: AgentState) -> dict[str, Any]:
        analysis = state.get("analysis_result")
        evidence = list(state.get("evidence") or [])

        # `analysis is None` 说明 supervisor 之前就失败了（模型不可用 / 计划非法）：
        # **不要编一个答案**，把已经发生的错误如实说出去。
        answer = _failure_answer(state) if analysis is None else _render(analysis, evidence, state)

        return {
            "final_answer": answer,
            "answer_payload": _payload(analysis, evidence, state),
        }

    return final


def _render(analysis: AnalysisResult, evidence: list[Evidence], state: AgentState) -> str:
    by_id = {item.id: item for item in evidence}
    lines: list[str] = [analysis.direct_answer.strip(), ""]

    if analysis.claims:
        lines.append("## 结论与依据")
        for index, claim in enumerate(analysis.claims, start=1):
            lines.append(f"{index}. {claim.text}")
            if claim.evidence_ids:
                refs = "；".join(
                    _cite(by_id[item_id]) for item_id in claim.evidence_ids if item_id in by_id
                )
                lines.append(f"   - 依据：{refs}")
            else:
                # **没有引用的结论要标出来**：13.5 要求 FACT 必须由直接证据支持，
                # 而"忘了填引用"与"这条其实是推断"在产物上长得一样。
                # 标出来让人去核对，比默认它是对的强。
                lines.append(f"   - 依据：（无引用，性质为{claim.kind}，请人工核对）")
        lines.append("")

    # **未解决的问题要并进限制，但不能重复**：`analysis` 节点已经把
    # `open_questions` 拼过一次（`_pipeline_limitations`），而两处的措辞
    # 不同（一处带"未解决的问题："前缀）——按整句去重会留下两行说同一件事。
    # 按**步骤号**去重：那是这两个来源共同的、稳定的标识。
    questions = state.get("open_questions") or []
    limits = list(analysis.limitations)
    mentioned = " ".join(limits)
    limits.extend(
        f"未解决的问题：{question}"
        for question in questions
        if question.split("：", 1)[0] not in mentioned
    )
    if limits:
        lines.append("## 限制与未覆盖")
        lines.extend(f"- {item}" for item in limits)
        lines.append("")

    if not analysis.conflicts:
        # **把"没检出冲突"与"还没做检测"分开说**：冲突检测属 Phase 9，
        # 现在恒为空。不写这一句的话，读答案的人会以为
        # "没有冲突"是被验证过的结论，而它其实只是个空列表。
        lines.append("> 本次未执行多源冲突检测（该能力属 Phase 9）。")
        lines.append("")

    if analysis.follow_up_questions:
        lines.append("## 可能还要问")
        lines.extend(f"- {item}" for item in analysis.follow_up_questions)

    return "\n".join(lines).strip()


def _cite(evidence: Evidence) -> str:
    """一条证据 → 人能读的出处。

    两条链路各有各的定位方式（详设 13.1 的 `locator`）：
    文档给"制度名 > 章节"，SQL 给查询指纹与结果切片。
    **不在这里读 `locator` 里没有的键**：那会让一条缺字段的证据
    渲染成"依据：（某条证据）"——看起来像格式问题，实际是上游漏填。
    """
    locator = evidence.locator
    if evidence.source_type == "DOCUMENT":
        # `locator` 的值类型是 `object`（详设 13.1 的说明：键按来源类型变化，
        # 收成一个联合类型只会让每种来源都要处理另外几种的字段）。
        # 这里显式判一下再拼，而不是 `str(part) for part in ...`——
        # 后者在 `section_path` 是字符串时会把它逐字符拆开。
        path = locator.get("section_path")
        section = " > ".join(str(part) for part in path) if isinstance(path, list) else ""
        page = f"，第 {locator['page_no']} 页" if locator.get("page_no") else ""
        return f"《{evidence.title}》{f'（{section}）' if section else ''}{page}"
    if evidence.source_type == "SQL":
        fingerprint = str(locator.get("sql_fingerprint", ""))[:12]
        return f"业务数据库查询 {fingerprint}"
    return evidence.title


def _failure_answer(state: AgentState) -> str:
    errors = state.get("errors") or []
    lines = ["本次任务未能完成。", ""]
    if errors:
        lines.append("## 原因")
        lines.extend(f"- {error.code.value}：{error.message}" for error in errors)
    else:  # pragma: no cover - 走到 final 却既无分析也无错误，属实现缺陷
        lines.append("## 原因\n- 未产生分析结果，且没有记录到具体错误")
    return "\n".join(lines)


def _payload(
    analysis: AnalysisResult | None, evidence: list[Evidence], state: AgentState
) -> dict[str, Any]:
    """结构化答案（详设 7.1 的 `answer_payload`，供 API 返回）。

    **带完整证据**：Markdown 里的引用是人读的，前端要跳转需要 id 与 locator。
    在这里给全，比让前端去正则解析 Markdown 可靠。
    """
    return {
        "direct_answer": analysis.direct_answer if analysis else None,
        "claims": [
            claim.model_dump(mode="json") for claim in (analysis.claims if analysis else ())
        ],
        "limitations": list(analysis.limitations) if analysis else [],
        "open_questions": list(state.get("open_questions") or []),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "step_results": {
            step_id: result.model_dump(mode="json")
            for step_id, result in (state.get("step_results") or {}).items()
        },
        "plan_revision": state.get("plan_revision", 0),
    }


__all__ = ["build_final_node"]
