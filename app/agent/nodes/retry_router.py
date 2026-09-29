"""`retry_router` 节点：审查不通过之后往哪走（详细设计 14.4）。

## 它解决的是什么

在这之前，`reviewer` 只有两种处置：整体 PASS 放行，或整体 FAIL 拦下
（`final` 输出"审查未通过"）。**中间那一档不存在**——而 14.3 的判定表里
它占了两行：

> | 存在可通过一次 Tool 或 Analysis 修复的问题，且有预算 | RETRY |
> | 证据缺口需要一个新的下钻方向才能补齐（retry_target=expand） | RETRY |

少了这一档，"结论缺一条证据"的处置就只剩"整条答案不放出去"，
而正确做法是**先去把那一条证据补回来**，补不到再拦。

## 四类预算各自约束一条回边，不许借用

FR-REV-002 业务规则 1：**相互独立且都必须非负**。三者在这条路径上相遇：

| 预算 | 约束的回边 | 本节点的动作 |
|---|---|---|
| `review_retries_left` | `retry_router → sql/rag/analysis` | 回 Tool / analysis 时 -1 |
| `expansions_left` | 演进回边（`reflect` 与 `expand` 共用） | **本节点不扣**，`plan_extend` 扣 |
| `replans_left` | 作废整份计划回 `supervisor` | 回 supervisor 时 -1 |

`expansions_left` 那一行是 14.3 的原文要求："同一份 `expansions_left` 预算被
`reflect` 和 `retry_router` 共用……避免'执行阶段激进 + 审查阶段再来一轮'
导致总耗时失控。"

**预算为 0 时降级，不借**（14.4 原文）：能澄清就 CLARIFY，否则 FAIL，
并记一条 `retry_budget_exhausted`。借用会让"某一类预算用完了"这件事
在数据上完全看不出来——四类预算各自存在的意义就是它们**分别**可观测。

## 为什么这个节点要写一个 `retry_route` 字段

`reflect` 立的规矩是"节点不写路由，由条件边从更新后的 State 推出来"，
那条规矩在这里不成立：**"没有待执行步骤"同时意味着两件相反的事**——
`retry_target=analysis`（要重跑分析）与"不重试了，去 final"。
两者在 State 上长得一模一样（都没有 PENDING 步骤），条件边推不出来。
所以这一处必须显式写下来，见 `state.AgentState.retry_route`。

## 谁会被真正产出

`choose_retry` 是 14.4 的逐字实现、五个目标都能路由，而能走到这里的
`retry_target` 有三个来源：

| 来源 | 目标 |
|---|---|
| 第一阶段的确定性判据（"FACT 结论没有引用"且还有一路没查过） | `sql` / `rag` |
| 第二阶段的模型审查（14.3 的"缺一整类信息"） | `expand`（偶尔也有 `sql`/`rag`） |
| 14.4 的降级 | 不路由，`CLARIFY` / `FAIL` |

另外两个的缺前提：

- `replan`：要"计划本身不可执行"这个判断——那是语义，而模型审查的四个
  布尔里没有哪一个问的是它，所以**目前没有产出者**；
- `search`：Tool 本身还没接（`app/tools/search/` 是空的），没有落点。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.schemas.plan import TaskStep
from app.agent.schemas.review import ReviewIssue, ReviewResult
from app.agent.state import (
    AgentState,
    Route,
    next_step_id,
    objective_for_route,
    tool_for_route,
)

#: `retry_target`（详设 14.4 的取值）-> 路由值。**只有 sql / rag 有落点**，
#: 其余三个见模块 docstring。工具名与目标说明**从 `state.COMPLEMENTS` 取**，
#: 那张表是「哪两路互补」的唯一出处。
_TOOL_TARGETS: dict[str, Route] = {"sql": Route.SQL, "rag": Route.RAG}


def choose_retry(
    review: ReviewResult,
    *,
    review_retries_left: int,
    expansions_left: int,
    replans_left: int,
) -> Route | None:
    """14.4 的路由表。**返回 `None` 表示不重试**（去 `final`）。

    逐字对应详设的 `choose_retry`，把 `RetryBudget` 拆成了三个整数——
    那个模型在本版没有第二个使用者，而 State 里存的就是这三个数。

    三个 `if` 的顺序不能换：先 replan（作废整份计划）、再 expand（加步骤）、
    最后才是"回某个 Tool/analysis"。顺序换了的话，一个本该重新规划的任务
    会先去补一条 SQL——**两种处置的代价差一个数量级**。
    """
    if review.status != "RETRY":
        return None
    if review.retry_target == "replan" and replans_left > 0:
        return Route.REPLAN
    if review.retry_target == "expand" and expansions_left > 0:
        # 交给 `plan_extend` 生成具体步骤（14.3：Reviewer 不直接指定 SQL）。
        # **本节点只路由，不加步骤也不扣预算**——预算是 `plan_extend` 在
        # 步骤真的被合入时才扣的。
        return Route.EXPAND
    if review.retry_target in _TOOL_TARGETS and review_retries_left > 0:
        return _TOOL_TARGETS[review.retry_target]
    if review.retry_target == "analysis" and review_retries_left > 0:
        return Route.ANALYSIS
    return None


def _exhausted_issue(review: ReviewResult) -> ReviewIssue:
    """预算耗尽时补记的那一条。

    **必须记**：否则"审查要求补证但没补"与"审查压根没要求"在产物上
    长得一样，而两者的处置完全不同（前者要调预算或查为什么打满，
    后者说明判据该收紧）。14.4 点名要记 `retry_budget_exhausted`。
    """
    return ReviewIssue(
        code="retry_budget_exhausted",
        severity="WARNING",
        message=(
            f"审查要求补证（retry_target={review.retry_target}），但对应预算已为 0，按受限回答处理"
        ),
    )


def build_retry_router_node() -> Callable[[AgentState], dict[str, Any]]:
    """构造 `retry_router` 节点。**纯确定性，不调模型**（同 `reflect`）。

    它做三件事，顺序有讲究：**先算路由、再消耗预算、最后才追加步骤**。
    反过来的话，一个路由不出去的重试会先花掉一次预算，而那次预算
    再也要不回来——预算的消耗必须是"这次真的重试了"的结果。
    """

    def retry_router(state: AgentState) -> dict[str, Any]:
        review: ReviewResult | None = state.get("review_result")
        if review is None:
            return {"retry_route": None}

        route = choose_retry(
            review,
            review_retries_left=state.get("review_retries_left", 0),
            expansions_left=state.get("expansions_left", 0),
            replans_left=state.get("replans_left", 0),
        )
        if route is None:
            update: dict[str, Any] = {"retry_route": None}
            if review.status == "RETRY":
                # 判了 RETRY 却没路由出去 = 预算不够或没有落点。
                # 补记一条，并按 14.4 降级：能澄清就澄清，否则拦下。
                update["review_result"] = _degrade(review)
            return update

        # **在这里统一 +1，不在下面几个分支里各写一遍**：REPLAN 与 EXPAND 都
        # 会提前 return（14.4 要求它们不扣 `review_retries_left`），写在下面
        # 会漏掉那两条路——而漏掉的表现是"某类重试的 attempt 永远不增"。
        update = {
            "retry_route": route,
            # 18.2 的 `task.retrying.attempt`：**本任务第几次合法重试**。
            # 它与四类预算正交——预算答"还能不能重试"，它答"这是第几次"。
            # **不从预算反推**：`expansions_left` 同时被 `reflect` 的
            # 计划演进正常路径消耗（那不是重试），减出来的会把那些算进来
            "retry_attempt": state.get("retry_attempt", 0) + 1,
        }
        if route is Route.REPLAN:
            # **作废整份计划**：留空列表让 supervisor 重新规划。
            # 已经跑过的 `step_results` 不清——它们是"已经知道的事实"，
            # 而 14.4 要求"重试时保留已验证证据"。
            update["task_list"] = []
            update["replans_left"] = max(0, state.get("replans_left", 0) - 1)
            return update

        if route is Route.EXPAND:
            # **提前返回**：`expand` 既不追加步骤（那是 `plan_extend` 的事），
            # 也不扣 `review_retries_left`——它花的是 `expansions_left`
            # 那一份，而那一份由 `plan_extend` 在步骤真被合入时才扣。
            # 落到下面那行统一扣 `review_retries_left` 的话，一次计划演进
            # 会**同时**吃掉一次补证预算，而四类预算各自可观测正是它们
            # 分开存在的意义（FR-REV-002 业务规则 1）。
            return update

        if route in (Route.SQL, Route.RAG):
            update["task_list"] = [*(state.get("task_list") or []), _retry_step(state, route)]
        # analysis 目标不追加步骤：`Route.ANALYSIS` 的落点就是重跑
        # `conflict → analysis`（证据没变时重跑会得到同样的结论，
        # 所以它只在**证据已经被补过**之后才有意义——这正是
        # `reviewer` 不把 analysis 作为首选目标的原因）。
        update["review_retries_left"] = max(0, state.get("review_retries_left", 0) - 1)
        return update

    return retry_router


def _degrade(review: ReviewResult) -> ReviewResult:
    """14.4 的降级：预算耗尽 → 有澄清问题就 CLARIFY，否则 FAIL。

    **CLARIFY 优先于 FAIL**：能问清就还有出路，而 FAIL 是"这条答案不发出去了"。
    两者都不是"重试"，所以降级**不会**消耗任何预算——它没有产生新的取证动作。
    """
    if review.clarification_question:
        return review.model_copy(
            update={
                "status": "CLARIFY",
                "issues": (*review.issues, _exhausted_issue(review)),
            }
        )
    return review.model_copy(
        update={
            "status": "FAIL",
            "issues": (*review.issues, _exhausted_issue(review)),
        }
    )


def _retry_step(state: AgentState, route: Route) -> TaskStep:
    """为这次重试**追加一个新步骤**。id 沿用 `reflect._next_step_id` 的规则。

    ## 为什么不"把旧步骤重置成 PENDING"

    那样更省事，但 `pending_steps` 的 docstring 明写"SUCCEEDED / FAILED /
    SKIPPED 都不会被重复执行——这条是详设 6.3 明写的，也是'演进后重跑旧步骤'
    这类错误的挡板"，而 `merge_step_results` 的注释说"后退（SUCCEEDED →
    PENDING）在本实现里不会发生"。重置状态位会同时打破这两条前提，
    而收益只是少存一行。

    追加新步骤还顺带保住了另一件事：**两次尝试的结果各占一行**。
    覆盖的话，`step_results` 里只剩最后一次，"这一步跑过两次、第一次是空的"
    这种事后无从分辨（同 `reflect._next_step_id` 的理由）。
    """
    return TaskStep(
        id=_next_step_id(state),
        objective=objective_for_route(route),
        tool=tool_for_route(route),
    )


def _next_step_id(state: AgentState) -> str:
    """序号接着计划往下排，**不重用已有 id**。

    **实现搬去了 `state.next_step_id`**（`reflect` / `plan_extend` 问的是
    同一个问题，三份各写一份就会漂移）。
    """
    return next_step_id(state)


__all__ = ["build_retry_router_node", "choose_retry"]
