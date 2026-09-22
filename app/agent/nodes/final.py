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

from collections.abc import Callable, Sequence
from typing import Any

from app.agent.schemas.analysis import AnalysisResult
from app.agent.state import AgentState
from app.domain.evidence import Evidence


def build_final_node() -> Callable[[AgentState], Any]:
    """构造 `final` 节点。**纯函数式，无外部依赖。**"""

    def final(state: AgentState) -> dict[str, Any]:
        analysis = state.get("analysis_result")
        evidence = list(state.get("evidence") or [])
        review = state.get("review_result")

        if analysis is None:
            # supervisor 之前就失败了（模型不可用 / 计划非法）：
            # **不要编一个答案**，把已经发生的错误如实说出去。
            answer = _failure_answer(state)
        elif review is not None and review.status == "FAIL":
            # 14.3 的一票否决：**结论没有依据时不把答案放出去**。
            # 这类答案看起来和别的答案一样，只是那句结论是空口说的，
            # 而人会拿它去做决定——拦住它是 Reviewer 存在的全部意义。
            answer = _blocked_answer(analysis, review)
        else:
            answer = _render(analysis, evidence, state, review)

        return {
            "final_answer": answer,
            "answer_payload": _payload(analysis, evidence, state),
        }

    return final


def _blocked_answer(analysis: AnalysisResult, review: Any) -> str:
    """审查未通过时的答案。

    **把"没通过"和"为什么"一起说出去**，而不是回一个通用文案：
    用户看到的应该是一条可行动的说明（哪条结论没依据），
    而不是"系统出错了"。
    """
    lines = ["本条回答未通过发布前的落地检查，因此不作为结论提供。", ""]
    lines.append("## 未通过的原因")
    lines.extend(f"- {item.message}" for item in review.blocking)
    lines.append("")
    lines.append(f"> 审查：{review.status}（{review.reason_code}）")
    if analysis.limitations:
        lines.append("")
        lines.append("## 补充说明")
        lines.extend(f"- {item}" for item in analysis.limitations)
    return "\n".join(lines)


def _render(
    analysis: AnalysisResult,
    evidence: list[Evidence],
    state: AgentState,
    review: Any = None,
) -> str:
    by_id = {item.id: item for item in evidence}
    lines: list[str] = [analysis.direct_answer.strip(), ""]

    if analysis.claims:
        lines.append("## 结论与依据")
        for index, claim in enumerate(analysis.claims, start=1):
            lines.append(f"{index}. {claim.text}")
            if claim.evidence_ids:
                refs = _cite_all(
                    [by_id[item_id] for item_id in claim.evidence_ids if item_id in by_id]
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

    if analysis.conflicts:
        lines.append("## 数据不一致（需人工核对）")
        # **冲突必须显式列出来，而且要说清"没有谁对谁错的结论"**：
        # 检测器只发现不一致，判谁对要靠 13.2 的证据优先级（Phase 9）。
        # 只说"发现冲突"而不说"未判定"，读者会默认系统已经选好了。
        lines.extend(f"- {item}" for item in analysis.conflicts)
        lines.append("")
        lines.append("> 以上不一致已检出但**未判定哪一方为准**（证据优先级判定属 Phase 9）。")
        lines.append("")
    else:
        # **把"没检出"与"没检测"分开说**：切片内只做 VALUE 一类（见
        # `nodes/conflict.py` 的清单）。写"没有冲突"会让读答案的人以为
        # 五类都比过了，而实际只比了一类。
        lines.append("> 本次只对「同口径数值」做了比对，未覆盖口径/时点/范围/来源四类冲突。")
        lines.append("")

    if review is not None and review.warnings:
        # **审查的警告也进"限制"**：它是"这条答案有个已知的弱点"，
        # 与证据层面的限制对读者是同一种信息。分成两节只会让人只看其中一节。
        lines.append("## 发布前检查的提醒")
        lines.extend(f"- {item.message}" for item in review.warnings)
        lines.append("")

    if analysis.follow_up_questions:
        lines.append("## 可能还要问")
        lines.extend(f"- {item}" for item in analysis.follow_up_questions)

    return "\n".join(lines).strip()


def _cite_all(items: Sequence[Evidence]) -> str:
    """一条结论引用的全部证据 → 出处列表，**同源的合并成一条**。

    ## 为什么必须合并

    原先逐条按 `evidence_id` 渲染，于是同一章节同一页的 8 个表格行块
    渲染出 **8 句一模一样的话**（2026-09-20 复跑实测）。读的人第一反应是
    "格式坏了"——而它其实是"这条结论引用了同一张表的 8 行"，
    一件**正常且必须能被看出来**的事。

    ## 合并之后行号要留着

    `（该表第 1–3、5 行）` 正是"引用能不能表达表的一部分"那个问题的答案：
    **能，而且必须能**。否则合并会把"引用了整张表"与"引用了其中 8 行"
    渲染成同一句话——而这两件事在"证据够不够"上完全不同，
    约定 78 那个缺口（8 行当整表合计）就住在它们的差别里。

    SQL 侧同理：一次查询的多行原先也渲染成同一句（只有指纹没有行号）。
    """
    groups: dict[str, list[Evidence]] = {}
    for item in items:
        # 分组键就是**渲染出来的那句出处**：合并的条件正是"读起来是同一个地方"。
        # 另立一套键（比如按 `locator` 的某几个字段）会与渲染逻辑各说各话，
        # 而漂移的症状是"合并了但看起来没合并"——反过来也一样。
        groups.setdefault(_cite(item), []).append(item)
    return "；".join(f"{base}{_merged_suffix(members)}" for base, members in groups.items())


def _merged_suffix(members: Sequence[Evidence]) -> str:
    """合并掉的那些证据**多出来的信息**：哪些行 / 一共几条。

    只列行号与条数，不列 id：引用是给人读的，id 在结构化证据里。
    """
    rows = [
        int(row[0])
        for item in members
        if _table_row(item) is not None and (row := _table_row(item))
    ]
    if rows:
        caption = members[0].locator.get("table_caption")
        name = f"表「{caption}」" if isinstance(caption, str) and caption else "该表"
        return f"（{name}第 {_ranges(rows)} 行）"
    # `result_slice` 是 **0 基**的（它是给程序切数组用的），而人读的行号从 1 起——
    # 与 `tools/sql/evidence.py` 的标题（`结果第 {index + 1} 行`）保持同一个口径。
    # 不加这个 1，引用里会出现「结果第 0 行」，而那一行不存在。
    slices = [int(slice_[0]) + 1 for item in members if (slice_ := _result_slice(item)) is not None]
    if slices:
        return f"（结果第 {_ranges(slices)} 行）"
    # 既不是表格行也不是 SQL 行，却仍然撞在同一句出处上——说明它们确实是
    # 同一处的多个块。**如实说有几条**，不要让合并把"2 条"说得像"1 条"。
    return f"（{len(members)} 条）" if len(members) > 1 else ""


def _table_row(item: Evidence) -> tuple[int, int] | None:
    row = item.locator.get("table_row")
    if isinstance(row, list) and len(row) == 2 and all(isinstance(v, int) for v in row):
        return int(row[0]), int(row[1])
    return None


def _result_slice(item: Evidence) -> tuple[int, int] | None:
    value = item.locator.get("result_slice")
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return int(value[0]), int(value[1])
    return None


def _ranges(values: Sequence[int]) -> str:
    """`[1, 2, 3, 5]` → `1–3、5`。

    连续区间压成一段：8 个行号逐个列出来没人看，而"第 1–3、5 行"
    一眼就知道是"这张表的一头一尾加一行"。
    """
    ordered = sorted(set(values))
    if not ordered:  # 调用方都判过非空，这里是防"以后多一个调用方"
        return ""
    parts: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(f"{start}–{previous}" if start != previous else str(start))
        start = previous = value
    parts.append(f"{start}–{previous}" if start != previous else str(start))
    return "、".join(parts)


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
        # **标题里已经含了这段章节时不要再拼一遍**：文档证据的 `title` 是
        # 「文档名 > 末级章节」（见 `tools/rag/evidence._title`），而
        # `section_path` 是完整路径——两者常常拼成「《A > B》（A > B）」，
        # 读起来像渲染坏了，而它只是同一个地方被说了两遍。
        suffix = f"（{section}）" if section and section not in evidence.title else ""
        page = f"，第 {locator['page_no']} 页" if locator.get("page_no") else ""
        return f"《{evidence.title}》{suffix}{page}"
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
        # **`refused` 必须在这里**：它已经渲染进 Markdown（人读），
        # 该由程序读的那一份就在这里。漏了它，`make demo` 的拒答判据
        # 只能回去猜词——而那正是它被加进来的原因（见 `AnalysisResult.refused`）。
        # 没有分析产物时给 `None` 而不是 `False`：那是"没走到这一步"，
        # 与"走到了且没拒答"是两件事。
        "refused": analysis.refused if analysis else None,
        # **`conflicts` 必须在这里**：它已经渲染进 Markdown（人读），
        # 也落进了 `agent_conflict` 表（程序读），而 API 的任务详情读的
        # **正是这个 payload**。漏了它，前端与 `make demo` 的判据都看到空列表——
        # 而答案里明明写着「数据不一致」。
        # 实测踩到：`make demo` 因此把一条**检出了冲突**的任务报成"没有检出冲突"。
        "conflicts": list(analysis.conflicts) if analysis else [],
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
        # 审查结论进 payload：API 与前端要能看到"这条答案经过检查、结论如何"
        "review": (
            state["review_result"].model_dump(mode="json")
            if state.get("review_result") is not None
            else None
        ),
    }


__all__ = ["build_final_node"]
