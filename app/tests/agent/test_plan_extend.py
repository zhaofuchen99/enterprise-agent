"""`plan_extend` 节点：计划演进（详细设计 6.6.3 / 8.5）。

## 这一组用例要钉住的三件事

1. **它是唯一的写者**——`task_list` / `plan_revision` / `expansions_left`
   三样要么一起变、要么都不变。只变其中两样的症状是
   「`plan_deltas` 说有一步、`task_list` 里没有」，而两边都看不出来。
2. **校验失败不报错**，但**必须留痕**（6.6.3 要求记 `plan_extend_rejected`）。
   不报错是对的——演进失败不该让已经拿到的证据白费；完全静默不行——
   那时"模型提了没通过的步骤"与"模型压根没提"长得一样。
3. **预算在入口扣**。它是一个被真问题逼出来的选择，见
   `nodes/plan_extend.py` 的模块说明：不在入口扣的话，
   `retry_router → plan_extend → conflict → analysis → reviewer → retry_router`
   这条闭合回路**绕不完**，而 `_guard` 数的是 `step_results`，它永远不涨。
"""

from __future__ import annotations

from typing import Any, cast

from app.agent.nodes.plan_extend import build_plan_extend_node
from app.agent.schemas.plan import (
    Finding,
    PlanExtension,
    ProgressAssessment,
    ProposedStep,
    StepResult,
    StepStatus,
    TaskStep,
)
from app.agent.schemas.review import ReviewResult
from app.agent.state import AgentState, Route
from app.core.config import LoopSettings, Settings
from app.tests.fakes import FakeFailure, FakeModelGateway


def _settings(**overrides: Any) -> Settings:
    """一份能改循环上限的配置（其余取默认值，用例只调它关心的那一个）。"""
    return Settings().model_copy(update={"loop": LoopSettings(**overrides)})


def _state(**overrides: Any) -> AgentState:
    base: dict[str, Any] = {
        "task_id": "tsk_TEST",
        "user_query": "2025年Q3华东净销售额是多少",
        "task_list": [TaskStep(id="step_01", objective="查数", tool="sql_query")],
        "step_results": {"step_01": StepResult(step_id="step_01", status=StepStatus.SUCCEEDED)},
        "findings": [],
        "plan_revision": 0,
        "plan_deltas": [],
        "expansions_left": 1,
        "review_retries_left": 1,
        "replans_left": 1,
    }
    return cast(AgentState, {**base, **overrides})


def _expand(**overrides: Any) -> ProgressAssessment:
    """`reflect` 判 EXPAND 时的判定——**步骤是它算好的，不需要模型**。"""
    base: dict[str, Any] = {
        "decision": "EXPAND",
        "reason": "sql_query 未取得有用结果（EMPTY_RESULT），补一路 rag_retrieve",
        "proposed_steps": (
            TaskStep(id="step_02", objective="从制度与报告库补查", tool="rag_retrieve"),
        ),
    }
    return ProgressAssessment.model_validate({**base, **overrides})


async def _run(
    state: AgentState,
    *,
    settings: Settings | None = None,
    reply: Any = None,
    failure: Any = None,
    responses: list[Any] | None = None,
) -> tuple[dict[str, Any], FakeModelGateway]:
    gateway = FakeModelGateway(
        responses=[reply] if reply is not None else (responses or []),
        failure=failure,
    )
    node = build_plan_extend_node(settings or _settings(), gateway)
    return await node(state), gateway


# ------------------------------------------------------------------ 确定性入口


async def test_the_reflect_path_merges_the_step_without_calling_the_model() -> None:
    """`reflect` 送来的步骤直接用，**一次模型调用都不发生**。

    这条是约定 34 在演进上的延续：判定要保证"循环会不会收敛"不依赖
    云模型连通性，而演进跟着它一起确定性——否则整条执行回边都会
    因为一次网络抖动而时灵时不灵。
    """
    update, gateway = await _run(_state(progress_assessment=_expand()))

    assert gateway.calls == [], "这条路上不该调模型"
    assert [step.id for step in update["task_list"]] == ["step_01", "step_02"]
    assert update["plan_revision"] == 1
    assert update["expansions_left"] == 0
    assert list(update["plan_deltas"][0].added_step_ids) == ["step_02"]


async def test_the_new_step_id_is_reassigned_by_code() -> None:
    """传入的 id 一律不采信，由 `plan_extend` 重发一个。

    撞号的后果是**新步骤的结果覆盖旧步骤的**（`merge_step_results` 就是
    "新值覆盖"），而它不报错——事后只看到"这一步跑过一次"。
    """
    state = _state(
        progress_assessment=_expand(
            proposed_steps=(TaskStep(id="step_01", objective="换个说法再查", tool="sql_query"),)
        )
    )

    update, _ = await _run(state)

    assert update["task_list"][-1].id == "step_02", "撞上已有 id 时要让开"


# ------------------------------------------------------------------ 模型入口


async def test_the_retry_path_asks_the_model_for_steps() -> None:
    """`retry_router` 那条路上步骤由模型生成（14.3：Reviewer 不指定 SQL）。"""
    state = _state(
        retry_route=Route.EXPAND,
        review_result=ReviewResult.model_validate(
            {
                "status": "RETRY",
                "retry_target": "expand",
                "reason_code": "INSUFFICIENT_EVIDENCE",
                "missing_evidence": ("渠道维度的明细",),
            }
        ),
    )
    reply = PlanExtension.model_validate(
        {
            "reason": "补一条按渠道拆解的 SQL",
            "steps": [{"objective": "按渠道拆解净销售额", "tool": "sql_query"}],
        }
    )

    update, gateway = await _run(state, reply=reply)

    assert [call.schema for call in gateway.calls] == ["PlanExtension"]
    # 审查说的"缺什么"要真的进了 prompt——否则模型是在盲猜方向
    assert "渠道维度的明细" in gateway.calls[0].variables["reason"]
    assert update["task_list"][-1].objective == "按渠道拆解净销售额"
    assert update["plan_deltas"][0].reason == "补一条按渠道拆解的 SQL"


async def test_a_model_with_nothing_to_add_does_not_burn_a_place_in_the_plan() -> None:
    """模型回空列表 → 计划不变，但**预算照样扣**（它在入口扣）。

    这是入口扣预算的代价，如实钉住：被拒的演进也算一次尝试。
    换取的是"那条闭合回路绕不完"被堵死——见模块 docstring。
    """
    state = _state(retry_route=Route.EXPAND, review_result=None)

    update, _ = await _run(state, reply=PlanExtension.model_validate({"reason": "无", "steps": []}))

    assert "task_list" not in update
    assert "plan_revision" not in update
    assert update["expansions_left"] == 0
    assert update["plan_extend_rejected"], "被拒也要留痕，否则与'压根没提'分不开"


async def test_a_model_failure_degrades_to_no_expansion_not_to_failure() -> None:
    """模型不可用 → **按不演进收敛**，不判 FAIL。

    演进是增强不是必经路径：一次模型侧的抖动不该让一条证据已经拿齐的
    答案作废。但也不能静默——`errors` 里留一条，拒绝理由里写明是模型没跑成，
    好把"模型坏了"与"模型提的步骤不合格"分开。
    """
    state = _state(retry_route=Route.EXPAND, review_result=None)

    update, _ = await _run(state, failure=FakeFailure.UNAVAILABLE)

    assert "analysis_result" not in update
    assert update["errors"], "降级必须留痕"
    assert any("模型未能给出演进步骤" in item for item in update["plan_extend_rejected"])
    assert "task_list" not in update


# ------------------------------------------------------------------ 校验


async def test_a_duplicate_direction_is_dropped() -> None:
    """去重键 `(工具, 归一化目标)` 命中已有步骤 → 丢弃那一步（6.6.3 第 6 条）。

    **归一化到"只留字"**：标点与空白的差别不算两个方向，
    而模型换个标点重提一遍是常见形态。
    """
    state = _state(
        progress_assessment=_expand(
            proposed_steps=(
                TaskStep(id="x", objective="查数。", tool="sql_query"),
                TaskStep(id="y", objective="按渠道拆解", tool="sql_query"),
            )
        )
    )

    update, _ = await _run(state)

    assert [step.objective for step in update["task_list"][1:]] == ["按渠道拆解"]
    assert any("方向重复" in item for item in update["plan_extend_rejected"])


async def test_the_step_budget_truncates_the_batch() -> None:
    """单次演进最多 `max_steps_per_expansion` 步，多的丢掉（6.6.3 第 2 条）。"""
    state = _state(
        progress_assessment=_expand(
            proposed_steps=tuple(
                TaskStep(id=f"s{i}", objective=f"方向{i}", tool="sql_query") for i in range(5)
            )
        )
    )

    update, _ = await _run(state, settings=_settings(max_steps_per_expansion=2))

    assert len(update["task_list"]) == 3, "1 步已有的 + 2 步上限"
    assert any("单次演进最多 2 步" in item for item in update["plan_extend_rejected"])


async def test_the_total_step_cap_wins_over_the_per_round_cap() -> None:
    """总步数上限是硬的：它比单轮上限更靠后，所以谁先到就按谁来截。"""
    state = _state(
        progress_assessment=_expand(
            proposed_steps=tuple(
                TaskStep(id=f"s{i}", objective=f"方向{i}", tool="sql_query") for i in range(3)
            )
        )
    )

    update, _ = await _run(state, settings=_settings(max_total_steps=2, max_steps_per_expansion=6))

    assert len(update["task_list"]) == 2, "上限就是 2 步，没得商量"
    assert any("计划总步数上限" in item for item in update["plan_extend_rejected"])


async def test_without_budget_the_model_is_not_called_at_all() -> None:
    """预算为 0 → 连模型都不调。

    结论必然是"不演进"，而调一次模型是真花钱的；这条路的判据只看
    State 里的一个整数，不需要问任何人。
    """
    update, gateway = await _run(
        _state(
            expansions_left=0,
            progress_assessment=_expand(),
            retry_route=Route.EXPAND,
        )
    )

    assert gateway.calls == []
    assert "task_list" not in update
    assert any("预算已用尽" in item for item in update["plan_extend_rejected"])


# ------------------------------------------------------------------ 触发源


async def test_the_trigger_finding_points_at_the_empty_step() -> None:
    """`trigger_finding_id` 指向**查空了的那一步**的发现。

    判据是"哪个步骤的结果是空的"：那正是 `reflect` 判 EXPAND 的理由
    （`_assess` 只在有 `empty` 结果时才提这一路）。没有它的话，
    22.10.3 的「每个 `origin=EXTENDED` 的步骤都能反查到存在的 `finding_id`」
    就写不出来。
    """
    finding = Finding(id="fnd_EMPTY", statement="查数：按当前条件未取得结果", step_id="step_01")
    state = _state(
        progress_assessment=_expand(),
        findings=[finding],
        step_results={
            "step_01": StepResult(
                step_id="step_01",
                status=StepStatus.SUCCEEDED,
                empty=True,
                error_code="EMPTY_RESULT",
            )
        },
    )

    update, _ = await _run(state)

    assert update["plan_deltas"][0].trigger_finding_id == "fnd_EMPTY"


async def test_a_finding_for_a_non_empty_step_is_not_claimed_as_the_trigger() -> None:
    """有 finding 但**那一步不是空的** → 不拿它顶替，留空。

    编一个"最近的发现"当触发源，会让推理链上凭空多出一条
    "因为发现了 X 所以查了 Y"的因果——而它并不存在。
    同 `rerank_score` 那条：没算过就没有值。
    """
    finding = Finding(id="fnd_OK", statement="查数（3 条证据）", step_id="step_01")
    state = _state(progress_assessment=_expand(), findings=[finding])

    update, _ = await _run(state)

    assert update["plan_deltas"][0].trigger_finding_id is None


# ------------------------------------------------------------------ Delta 的形状


async def test_the_delta_id_is_the_deduplication_key() -> None:
    """`PlanDelta.id` = `"{task_id}:{revision_no}"`——详设 7.2 要的去重键。

    用它当 `id`，`plan_deltas` 就能直接走 `merge_by_id`；用随机 id 的话，
    同一版本会被记两次，而两条一致的记录看起来完全正常。
    """
    update, _ = await _run(_state(progress_assessment=_expand()))

    delta = update["plan_deltas"][0]
    assert delta.id == "tsk_TEST:1"
    assert delta.revision_no == 1
    assert delta.trigger == "EXTEND"
    assert delta.skipped_step_ids == ()


async def test_the_proposed_step_type_never_carries_an_id() -> None:
    """`ProposedStep` 没有 id 字段——id 由代码分配，这是它的定义。

    这条钉的是**形状**而不是行为：一旦有人给它加上 id，
    模型就会开始填它，而填错的表现是"新步骤覆盖旧步骤的结果"。
    """
    assert "id" not in ProposedStep.model_fields
