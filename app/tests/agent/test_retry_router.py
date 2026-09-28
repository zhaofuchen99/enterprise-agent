"""审查驱动的重试回路（详细设计 14.3 / 14.4 / 6.6.1）。

**这一组用例要钉住的东西只有两件**，而它们都不是"重试能不能发生"：

1. **预算不许借用**（FR-REV-002 业务规则 1）。四类预算各自存在的意义就是
   它们**分别**可观测；一旦某条路径能从别的预算里"借"，"某一类用完了"
   这件事在数据上就再也看不出来——而它正是排查"为什么这次没补证"的
   第一个问题。
2. **回路会停**。加回边最贵的失败不是"重试没发生"，是"重试停不下来"，
   而那时唯一的护栏是 LangGraph 的 `recursion_limit`，它触发时留下的是
   一次 `INTERNAL_ERROR` + 空轨迹——**一个查不出原因的收尾**。
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from app.agent.nodes.retry_router import build_retry_router_node, choose_retry
from app.agent.nodes.reviewer import review
from app.agent.schemas.analysis import AnalysisResult
from app.agent.schemas.plan import StepResult, StepStatus, TaskStep
from app.agent.schemas.review import ReviewResult
from app.agent.state import AgentState, Route


def _review(**overrides: Any) -> ReviewResult:
    base: dict[str, Any] = {"status": "RETRY", "retry_target": "rag", "reason_code": "R"}
    return ReviewResult.model_validate({**base, **overrides})


# ------------------------------------------------------------------ 路由表


def test_a_non_retry_verdict_routes_nowhere() -> None:
    """`PASS` / `FAIL` 都不重试——路由表第一行的判据是 `status != "RETRY"`。"""
    for status in ("PASS", "FAIL", "CLARIFY"):
        assert (
            choose_retry(
                _review(status=status), review_retries_left=1, expansions_left=1, replans_left=1
            )
            is None
        )


@pytest.mark.parametrize(
    ("target", "expected"),
    [("sql", Route.SQL), ("rag", Route.RAG), ("analysis", Route.ANALYSIS)],
)
def test_evidence_targets_route_to_their_node(target: str, expected: Route) -> None:
    assert (
        choose_retry(
            _review(retry_target=target), review_retries_left=1, expansions_left=0, replans_left=0
        )
        is expected
    )


def test_replan_is_tried_before_anything_else() -> None:
    """三个 `if` 的顺序不能换：**先 replan、再 expand、最后才是补一路**。

    顺序换了的症状是：一个本该作废整份计划重来的任务，会先去补一条 SQL。
    两种处置的代价差一个数量级，而结果看起来都"做了点什么"。
    """
    review = _review(retry_target="replan")

    assert (
        choose_retry(review, review_retries_left=1, expansions_left=1, replans_left=1)
        is Route.REPLAN
    )
    # replan 预算为 0 而补证预算还有 → **不借**，直接不路由
    assert choose_retry(review, review_retries_left=1, expansions_left=1, replans_left=0) is None


class TestBudgetIsolation:
    """22.10.4 的四条预算隔离判据。

    ⚠️ 这里**只验"不借"**这一件事。另外三条（演进预算耗尽不影响 SQL 自修复、
    SQL 修复不消耗演进预算、expand 与执行阶段共用 `expansions_left`）
    分别由 `test_sql_self_repair_*`（`tools/sql`）与 `reflect` 的用例覆盖——
    它们约束的是**别的**回边，混在这一组里测会让人以为它们是同一个机制。
    """

    def test_a_zero_budget_is_never_borrowed_from_another(self) -> None:
        """补证预算为 0 → 不路由，**哪怕别的预算还剩着**。

        "借用"是这个模块最容易犯的错（写起来就是少一个 `and`），
        而它的症状是**四类预算的计数不再各自可信**——排查时看到
        "补证用了 0 次"，而实际上它靠借来的预算跑了两次。
        """
        review = _review(retry_target="rag")

        assert (
            choose_retry(review, review_retries_left=0, expansions_left=99, replans_left=99) is None
        )

    def test_expand_needs_the_expansion_budget(self) -> None:
        """`expand` 要的是 `expansions_left`（14.3 明写它与执行阶段共用一份）。

        **它不许借 `review_retries_left`**：那个还满着，而预算为 0 时结果
        仍是"不重试"。借了的话四类预算的计数就不再各自可信——
        排查时看到"补证用了 0 次"，而它其实靠借来的预算跑过。
        """
        review = _review(retry_target="expand")

        assert (
            choose_retry(review, review_retries_left=9, expansions_left=0, replans_left=9) is None
        )
        # 预算够 → 交给 `plan_extend`（14.3：由它决定具体步骤）
        assert (
            choose_retry(review, review_retries_left=9, expansions_left=1, replans_left=9)
            is Route.EXPAND
        )


# ------------------------------------------------------------------ 节点


def _state(**overrides: Any) -> AgentState:
    base: dict[str, Any] = {
        "task_list": [TaskStep(id="step_01", objective="查数", tool="sql_query")],
        "step_results": {"step_01": StepResult(step_id="step_01", status=StepStatus.SUCCEEDED)},
        "review_result": _review(),
        "review_retries_left": 1,
        "expansions_left": 1,
        "replans_left": 1,
    }
    return cast(AgentState, {**base, **overrides})


def test_the_router_appends_a_new_step_instead_of_reopening_the_old_one() -> None:
    """重试**追加新步骤**，不把旧步骤重置成 PENDING。

    重置更省事，但它会同时打破两条被写下来的前提：`pending_steps` 的
    "SUCCEEDED / FAILED / SKIPPED 都不会被重复执行"，与 `merge_step_results` 的
    "后退（SUCCEEDED → PENDING）在本实现里不会发生"。

    追加还保住了一件可观测的事：**两次尝试各占一行**。覆盖的话
    `step_results` 里只剩最后一次，"这一步跑过两次、第一次是空的"事后无从分辨。
    """
    update = build_retry_router_node()(_state())

    assert update["retry_route"] is Route.RAG
    assert update["review_retries_left"] == 0
    added = [step for step in update["task_list"] if step.id not in {"step_01"}]
    assert len(added) == 1
    assert added[0].tool == "rag_retrieve"
    assert added[0].id != "step_01", "id 不能撞——撞了就是覆盖旧结果"


def test_the_budget_is_only_spent_when_the_retry_actually_happens() -> None:
    """**路由不出去就不花预算**。

    顺序反了的症状：一次没能发生的重试先扣掉一次预算，而那次预算再也要不回来
    ——下一次真正需要补证时，预算显示为 0，而"为什么是 0"没有任何痕迹。
    """
    exhausted = _state(review_retries_left=0, review_result=_review(retry_target="rag"))

    update = build_retry_router_node()(exhausted)

    assert update["retry_route"] is None
    assert "review_retries_left" not in update, "没路由出去就不能扣"


def test_an_unroutable_retry_is_downgraded_and_recorded() -> None:
    """预算耗尽 → 降级，**并且记一条**（14.4 点名要 `retry_budget_exhausted`）。

    不记的话，"审查要求补证但没补"与"审查压根没要求"在产物上长得一样，
    而两者的处置完全不同：前者要查为什么预算打满，后者说明判据该收紧。
    """
    update = build_retry_router_node()(
        _state(review_retries_left=0, review_result=_review(retry_target="rag"))
    )

    degraded = update["review_result"]
    assert degraded.status == "FAIL", "没有澄清问题 → FAIL（14.4 的降级规则）"
    assert "retry_budget_exhausted" in {item.code for item in degraded.issues}


def test_an_unroutable_retry_with_a_question_becomes_clarify() -> None:
    """有澄清问题时优先 CLARIFY：**能问清就还有出路**，而 FAIL 是"不发出去了"。"""
    update = build_retry_router_node()(
        _state(
            review_retries_left=0,
            review_result=_review(retry_target="rag", clarification_question="你说的口径是哪个？"),
        )
    )

    assert update["review_result"].status == "CLARIFY"


def test_expand_only_routes_and_touches_no_budget() -> None:
    """`expand` 只写路由，**两样预算一个都不动**。

    - 不追加步骤：那是 `plan_extend` 的事（14.3 "Reviewer 不直接指定 SQL"）；
    - 不扣 `review_retries_left`：它花的是 `expansions_left` 那一份；
    - 不扣 `expansions_left`：由 `plan_extend` 扣——而且**它是入口扣**，
      理由见 `nodes/plan_extend.py`（被拒的演进不追加步骤，若这里扣、
      那里也扣，或哪都不扣，`retry_router → plan_extend → …` 那条闭合回路
      就绕不完，而 `_guard` 数的是 `step_results`，它永远不涨）。
    """
    update = build_retry_router_node()(
        _state(review_result=_review(retry_target="expand", missing_evidence=("渠道维度的明细",)))
    )

    assert update["retry_route"] is Route.EXPAND
    assert "task_list" not in update, "步骤由 plan_extend 生成"
    assert "review_retries_left" not in update
    assert "expansions_left" not in update
    # 它也不能被降级：路由是成功的，"预算耗尽"才是降级那条路
    assert "review_result" not in update


def test_a_passing_review_changes_nothing() -> None:
    """PASS 时路由节点是个空操作——它每次任务都会跑，不能有副作用。"""
    update = build_retry_router_node()(_state(review_result=_review(status="PASS")))

    assert update == {"retry_route": None}


# ------------------------------------------------------------------ 审查侧


def _analysis(*kinds: str) -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "direct_answer": "答案",
            "refused": False,
            "claims": [
                {"text": f"结论{i}", "kind": kind, "evidence_ids": []}
                for i, kind in enumerate(kinds, start=1)
            ],
        }
    )


def _consulted(*tools: str) -> AgentState:
    return {
        "task_list": [
            TaskStep(id=f"step_{i:02d}", objective="查", tool=tool)
            for i, tool in enumerate(tools, start=1)
        ],
        "step_results": {
            f"step_{i:02d}": StepResult(step_id=f"step_{i:02d}", status=StepStatus.SUCCEEDED)
            for i, _ in enumerate(tools, start=1)
        },
    }


def test_an_uncited_fact_triggers_a_retry_to_the_source_not_yet_consulted() -> None:
    """**FACT 没有引用、而还有一路没查过** → RETRY，指向那一路。

    这是本版唯一一类的重试触发，而它被选中的理由是**重试能改变结果**：
    证据补回来之后，`analysis` 要么给这条结论找到依据，要么证明它确实无据
    ——两种结果都比"直接把整条答案拦下"好。
    """
    state = {
        "analysis_result": _analysis("FACT"),
        **_consulted("sql_query"),
        "review_retries_left": 1,
    }

    result = review(state["analysis_result"], state)  # type: ignore[arg-type]

    assert result.status == "RETRY"
    assert result.retry_target == "rag", "查过 sql 没查过 rag → 补 rag"
    assert result.reason_code == "CLAIM_WITHOUT_EVIDENCE_RETRY"


def test_an_uncited_fact_after_both_sources_fails_instead() -> None:
    """两路都查过了还缺引用 → **FAIL 而不是 RETRY**。

    这是 "重试必须能改变结果" 那条判据的落点：再补也只是把同一个动作
    重做一遍，而 14.3 明写"不得以'文风不够好'为由触发昂贵 Tool 重试"。
    """
    state = cast(
        AgentState,
        {
            "analysis_result": _analysis("FACT"),
            **_consulted("sql_query", "rag_retrieve"),
            "review_retries_left": 1,
        },
    )

    result = review(_analysis("FACT"), state)

    assert result.status == "FAIL"
    assert result.retry_target is None


def test_the_retry_needs_budget_as_well_as_an_unconsulted_source() -> None:
    """判据是**两件事同时成立**：还有一路没查 **且** 还有预算。

    少了后半句，预算就成了摆设；少了前半句，就变成"任何 FAIL 都先重试一次"。
    """
    state = cast(
        AgentState,
        {"analysis_result": _analysis("FACT"), **_consulted("sql_query"), "review_retries_left": 0},
    )

    result = review(_analysis("FACT"), state)

    assert result.status == "FAIL"


def test_a_required_step_that_never_ran_is_not_a_retry() -> None:
    """`REQUIRED_STEP_NOT_RUN` 同样 BLOCKING，但**不重试**。

    它是流程出了问题（计划里的必需步骤没跑），补一路取证解决不了——
    14.3 把"关键数据源不可用"归在 FAIL 那一行。这条用例钉的是
    "不是所有 BLOCKING 都该重试"。
    """
    state = cast(
        AgentState,
        {
            "analysis_result": _analysis("FACT"),
            "task_list": [TaskStep(id="step_01", objective="查", tool="sql_query", required=True)],
            "step_results": {},
            "review_retries_left": 1,
        },
    )

    result = review(_analysis("FACT"), state)

    assert result.status == "FAIL"
    assert "REQUIRED_STEP_NOT_RUN" in {item.code for item in result.issues}


# ------------------------------------------------------------------ 最后一道护栏


class TestStepBudgetGuard:
    """`max_total_steps` —— **回边写错时唯一会拦住它的东西**。

    在这之前，跑飞了的唯一后果是 LangGraph 的 `recursion_limit` 抛
    `GraphRecursionError`，而那一路走到底的产物是 `INTERNAL_ERROR` +
    `trace_incomplete` + 五张产出表全空——**一次"什么东西坏了"的收尾，
    而不是一个说得清的结论**。这条护栏把"用尽了"导回正常出口。
    """

    def test_the_guard_stops_the_loop_once_the_step_budget_is_spent(self) -> None:
        from app.agent.graph import _NODE_CONFLICT, _guard

        def would_loop_forever(_state: Any) -> str:
            return "sql"  # 一条永远想接着跑的判据

        settings = _settings(max_total_steps=3)
        guarded = _guard(would_loop_forever, settings)

        def step(index: int) -> StepResult:
            return StepResult(step_id=f"step_{index:02d}", status=StepStatus.SUCCEEDED)

        spent = cast(AgentState, {"step_results": {step(i).step_id: step(i) for i in range(3)}})

        assert guarded(spent) == _NODE_CONFLICT, "超限要收敛，不是继续绕"
        # 未超限时护栏**不插手**——它只兜底，不改正常路由
        assert guarded({"step_results": {}}) == "sql"

    def test_the_budget_counts_steps_not_rounds(self) -> None:
        """判据是"已经跑了多少步"而不是"绕了多少圈"。

        "圈数"在 State 里没有载体：`plan_revision` 只数计划演进，
        补证与 replan 都不加它。而每次重试都**追加新步骤**，
        所以 `step_results` 的键数数得准。
        """
        from app.agent.graph import _over_step_budget

        assert _over_step_budget(cast(AgentState, {"step_results": {}}), 1) is False
        one = StepResult(step_id="step_01", status=StepStatus.SUCCEEDED)
        assert _over_step_budget(cast(AgentState, {"step_results": {"step_01": one}}), 1) is True


def _settings(**loop_overrides: Any) -> Any:
    from app.core.config import Settings

    return Settings(loop={**Settings().loop.model_dump(), **loop_overrides})
