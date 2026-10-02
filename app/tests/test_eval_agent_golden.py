"""Agent 端到端评测集**自己**的形状约束（`configs/eval_agent_golden.yaml`）。

**为什么金标集本身要测**：它是一份人工维护的 YAML，而它失效的方式是
"少了五条用例"或"某条用例其实什么都没断言"——两种都不会报错，
只会让通过率看起来还不错。这与 `test_verify.py` 的立意相同：
**门禁自己失效时，没人会再去查它**。

跑真链路的那部分由 `make eval-agent` 覆盖（需 `make run`）；这里只测
"这份清单是不是一份能使上劲的清单"，所以离线可跑。
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.eval_agent import CATEGORIES, GOLDEN_PATH, load_cases

#: 每一类的条数。**2026-09-28 从"每类 5 条"改成这张表**。
#:
#: 原因是 `任务循环` 按详设 22.10.5（"至少 15 条"）补齐了——它是**唯一**
#: 一类条数不同的，而原来那条 `总数 = 类数 × 5` 的断言在它补到 15 之后
#: 就成了一个说不上话的恒等式。
#:
#: 写死在测试里而不是从 YAML 数出来——**从被测对象推期望值等于没测**。
#: 分布也要能被断言：总数对了而分布错了（比如 12+5+2+1）某一类的通过率
#: 就没有意义了。
_EXPECTED_COUNTS: dict[str, int] = {
    "tool_selection": 5,
    # 需求规格 11.2 第 5 项原文要求「10 条多源冲突问题」，2026-10-02 补齐
    # （原先 5 条）。三类判据各占一块：VALUE 02、DEFINITION 06/07、
    # TIME 08，另有 04/05/09/10 四条零冲突用例守反方向。
    "conflict": 10,
    "reviewer": 5,
    "clarification": 5,
    # 详设 22.10.5 明写"循环类问题至少 15 条"。**其余十类不跟着涨**
    # （冲刺方案 §6 定的是每类 5 条），所以这里是一张表而不是一个数字。
    "loop": 15,
}

#: 一条用例至少要有一个 `expect_*` 键，否则它跑完不判任何东西、
#: 恒报通过。这不是理论风险：写清单时漏掉一行 `expect_sources`
#: 与写对了，在报告上长得一模一样。
_ASSERTION_PREFIX = "expect_"


def _cases() -> list[dict[str, Any]]:
    return load_cases(GOLDEN_PATH)


def test_the_set_has_the_planned_number_of_cases() -> None:
    """总量 = 各类条数之和（5 + **10** + 5 + 5 + 15 = 40）。

    数字要能被断言，不能只看总数——总数对了而分布错了（比如 12+5+2+1），
    某一类的通过率就没有意义了。
    """
    assert len(_cases()) == sum(_EXPECTED_COUNTS.values())


def test_every_category_has_the_planned_number_of_cases() -> None:
    counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    for case in _cases():
        counts[str(case["category"])] += 1

    assert counts == _EXPECTED_COUNTS


def test_every_case_asserts_something() -> None:
    """没有 `expect_*` 的用例是**恒过的用例**——它占着一个名额，什么都不验。"""
    for case in _cases():
        assertions = [key for key in case if key.startswith(_ASSERTION_PREFIX)]
        assert assertions, f"{case['id']} 没有写任何 expect_*，它跑完不判任何东西"


def test_case_ids_are_unique_and_namespaced() -> None:
    ids = [str(case["id"]) for case in _cases()]

    assert len(set(ids)) == len(ids), "id 重复会让 ONLY=... 一次跑两条"
    # 前缀让 id 在报告里能自解释（`agent-conflict-03` 一眼看出属于哪一类）。
    # **不按 category 逐个拼前缀**：那样 category 改名就要连带改 id，
    # 而 id 是唯一被 `--only` 和提交信息引用的东西。
    assert all(item.startswith("agent-") for item in ids)


def test_conflict_category_asserts_both_directions() -> None:
    """冲突这一类必须**两个方向都有**：检出真冲突，以及不误报。

    只钉"检出"的话，一个"见着两个数就报冲突"的实现会全绿——
    而误报同样是缺陷，只是它的症状（读者会去质疑那条冲突）比漏报明显。
    """
    conflict_cases = [case for case in _cases() if case["category"] == "conflict"]

    detects = [case for case in conflict_cases if case.get("expect_conflicts") is True]
    clean = [case for case in conflict_cases if case.get("expect_conflict_count") == 0]

    assert detects, "没有任何一条用例要求检出冲突"
    assert clean, "没有任何一条用例要求**不**报冲突（防误报的方向没有覆盖）"


def test_loop_category_asserts_both_directions() -> None:
    """任务循环这一类必须**两个方向都有**：该下钻时下钻了，以及不该下钻时没动。

    与冲突那一类同一条理由（`test_conflict_category_asserts_both_directions`）：
    只钉"演进了"的话，一个"见着空结果就无脑补一路"的实现会全绿——
    而**过度下钻同样是缺陷**，只是它的症状是白花时间，看起来完全正常。

    ⚠️ 这里同时拦一类**假断言**：`expect_plan_revision_min: 0` 看着像
    "钉了下限"，其实恒真（版本号不会是负数）。真要说"不该演进"，
    只有 `expect_plan_revision: 0`（精确）或 `expect_plan_delta_count: 0`
    算数——**一个恒真的断言比没有断言更糟**，因为它看起来是有保障的。
    """
    loop_cases = [case for case in _cases() if case["category"] == "loop"]

    expands = [case for case in loop_cases if case.get("expect_plan_revision_min")]
    idle = [
        case
        for case in loop_cases
        if case.get("expect_plan_revision") == 0 or case.get("expect_plan_delta_count") == 0
    ]

    assert expands, "没有任何一条用例要求发生演进"
    assert idle, "没有任何一条用例要求**不**演进（防过度下钻的方向没有覆盖）"


def test_a_loop_case_that_expects_an_expansion_does_not_only_use_a_floor() -> None:
    """钉下限的用例必须**同时钉住上界或来源**，不能只说"至少一次"。

    只写 `expect_plan_revision_min: 1` 的话，一个"每次都把两路都调一遍"
    的实现照样全绿——而那种实现根本不需要循环，它把 §8.4 第③条
    要验的东西绕过去了。上界（预算）与下钻来源（`origin=EXTENDED`）
    才是"这一步是补出来的"的证据。
    """
    for case in _cases():
        if not case.get("expect_plan_revision_min"):
            continue
        assert case.get("expect_plan_revision_max") or case.get("expect_extended_tools_contain"), (
            f"{case['id']} 只钉了演进次数的下限——"
            f"再加一条上界或 `expect_extended_tools_contain`，否则它验不出循环"
        )


def test_rejection_and_clarification_are_both_covered() -> None:
    """澄清与拒答是两条不同的路径（一个不跑步骤，一个跑完说"没有"），
    只覆盖一条时另一条退化了不会被发现。"""
    cases = _cases()

    assert [case for case in cases if case.get("expect_rejected") is True], "没有拒答用例"
    assert [case for case in cases if case.get("expect_clarification") is True], "没有澄清用例"


def test_a_restricted_account_is_exercised() -> None:
    """权限边界只有**换账号**才测得到：同一个问题，受限账号的限制清单
    必须多出几条（约定 79/80）。全用 admin 跑的话这条链路是空的。"""
    assert [case for case in _cases() if case.get("account")]


def test_model_dependent_cases_stay_a_small_minority() -> None:
    """观察用例（`stability: model-dependent`）**不进门禁**，所以它有一条
    极容易被滥用的捷径：**把一条老是失败的用例标成观察用例**，
    报告立刻变绿，而它藏起来的是一次真实的回归（约定 77 的同类警告）。

    约束的是数量而不是禁用：这类用例有必要（拒答行为确实取决于模型判定），
    但不可以变成多数——**多数用例都不受门禁保护时，这门禁就不存在了**。
    """
    cases = _cases()
    volatile = [case for case in cases if case.get("stability") == "model-dependent"]

    assert len(volatile) <= 2, f"观察用例过多（{len(volatile)}/{len(cases)}），门禁已经名存实亡"


def test_stability_values_are_from_the_known_vocabulary() -> None:
    """取值与 `configs/demo_questions.yaml` **同一套**。

    多出第三个取值（比如 `flaky`）会让两个入口的语义分叉——
    而"这个用例到底算不算数"正是最不该有两套说法的事。
    """
    for case in _cases():
        assert case.get("stability", "stable") in {"stable", "model-dependent"}, case["id"]


def test_number_assertions_are_numeric() -> None:
    """`expect_numbers` 写错时判据会**抛**（见 `missing_amounts`），
    而那会让一整条用例以"异常"的形式失败，指不到清单上。

    这里在装载期就拦一道，把问题指向**这一行**。
    """
    for case in _cases():
        for raw in case.get("expect_numbers") or []:
            float(str(raw).replace(",", ""))


def test_unknown_category_is_rejected(tmp_path: object) -> None:
    """表外的类别**报错而不是归到"其他"**：归错类的用例会让分类通过率
    失去意义，而它看起来只是一条普通的用例。"""
    from pathlib import Path

    path = Path(str(tmp_path)) / "golden.yaml"
    path.write_text(
        "version: 'test'\ncases:\n"
        "  - id: agent-x-01\n    category: 没这一类\n    question: 问\n    proves: 证明\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="不在表内"):
        load_cases(str(path))
