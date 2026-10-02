"""Agent 端到端评测集（开发流程 6.15 第 2 项 / 7.4 的构成）。

```bash
make eval-agent                    # 跑全部 20 条
make eval-agent ONLY=agent-conflict-01
make eval-agent-ask Q="还没固化的问题"   # 先跑一遍再固化（本文件的纪律）
```

## 它测的是组件层测不到的那一层

本项目已有三道组件层门禁，各自钉住一件事：

| 命令 | 测什么 | 判据 |
|---|---|---|
| `make eval-sql` | 生成的 SQL 与金标**结果集等价** | 10 题 |
| `make eval-rag` | 检索的 **Recall@8 / 定位一致** | 20 题 |
| `make verify-corpus` | 10 类缺陷**真的注入了产物** | 10 类 |
| 校验器用例 | SQL 安全与越权**100% 阻断** | 28 条 |

**三者全绿，端到端仍然可能是错的**。2026-09-20 实测到一次：
工具选对了、SQL 也对，而答案里的数字是错的（80 行进 8 行，差 46%），
Reviewer 给了 100 分。原因不在任何一个组件里，在**它们串起来之后**：
`reflect` 该不该补一路、冲突该不该报、审查该不该拦——
这些都是"图的行为"，只有把整条链路跑起来才看得见。

所以这一集**走 HTTP 而不是直接调图**（与 `scripts/demo.py` 同一条理由）：
HTTP → 限流 → 入库 → 队列 → Worker 领取 → 图 → 写回 → 查详情。
代价是它要求 `make run` 已经在跑。

## 分类统计，而不是一个笼统的通过率

`proves` 与 `category` 都不是装饰。**"18/20"本身不告诉你错的是哪一类能力**，
而"冲突这一类全错"才是可行动的结论。这与 `eval_sql.py` 按 `proves`
打印每题验证点是同一个理由（详设 16.11.3 第 4 条）。

类别取自开发流程 7.4 的构成表，只取**已实现**的五类：

| category | 中文 | 7.4 的条数 | 本集 |
|---|---|---:|---:|
| `tool_selection` | 工具选择与 Supervisor 路由 | 15 | 5 |
| `conflict` | 多源冲突 | 10 | 5 |
| `reviewer` | 审查（14.1 两个阶段） | 5 | 5 |
| `clarification` | 澄清、拒答与边界 | 10 | 5 |
| `loop` | 任务循环与自适应下钻 | 15 | 5 |

**未收进来的一类**：`异常恢复与重试`——它要的是**模型层面**的缺陷答案
（"这条结论缺一整类信息"），而那要靠审查第二阶段的判定，且那个判定
本来就会飘（约定 105 实测过）。写进来只能是一条恒绿的观察用例，
而一条恒绿的用例会让分类通过率失去意义。登记在案。

**`loop` 这一类现在能写了**（6.6.3 的 `plan_extend` 落地之后）：
它断言的是 `plan_revision` / `plan_deltas` / `steps[].origin` /
`trigger_finding_id`，而这几样在**确定性演进那条路上**（SQL 空 → 补 RAG）
是代码判定的，不依赖模型，所以这一类的断言是硬的。
⚠️ 7.4 要求 15 条，本集交 5 条（与其余四类同规模），其余按 115 口径登记。

## 判定与故障分开报，观察用例不进门禁

判据在 `scripts/agent_harness.py`（与 `make demo` **共用同一份**）。
这里多两层：

- **未判定**（`Verdict.unjudged`：任务失败、分析生成失败）与"行为不符"分开计数。
  **它仍然计入分母，这是刻意的**：排除在外的话，系统开始大面积失败任务时
  通过率反而会**上升**——那是最危险的一种静默，而它看起来完全正常
  （"通过的都通过了"）。
- **观察用例**（`stability: model-dependent`）**不进门禁**。它与 `make demo`
  是同一套取值、同一条理由：结果取决于模型的意图判定或检索排序，
  同一个问题两次跑可能不一样，混进通过率会让人把"模型这次判歪了"
  当成"系统坏了"。

⚠️ **这个区分不是"把门禁改宽"**（约定 77 警告过那条）：它不改变任何一条
用例的判定，只是把两类失败分开报。**观察用例的失败照样会打出来**——
它只是不算作回归。反过来，**新写的用例默认是 `stable`**：
把一条只因模型抖动而失败的用例标成观察用例，等于把回归藏起来，
而那正是这个标记最容易被滥用的方向。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.agent_harness import (
    DEMO_USERNAME,
    Verdict,
    amounts,
    ask,
    case_turns,
    evaluate,
    login,
    run_case,
)

#: 评测集路径。与 `eval_sql_golden.yaml` / `eval_rag_golden.yaml` 同属配置类文件。
GOLDEN_PATH = "configs/eval_agent_golden.yaml"

#: 类别 → 中文名。**取值封闭**：YAML 里出现表外的类别直接报错，
#: 不静默归到"其他"——一条归错类的用例会让分类通过率失去意义，
#: 而它看起来只是一条普通的用例。
CATEGORIES: dict[str, str] = {
    "tool_selection": "工具选择与路由",
    "conflict": "多源冲突",
    "reviewer": "审查（两阶段）",
    "clarification": "澄清、拒答与边界",
    "loop": "任务循环与自适应下钻",
}

#: 阶段基线。**先跑出真实数字，再定基线**——
#: 2026-10-02 冲突类补到 10 条之后（共 40 条）跑了两次全量，**结论不同**：
#:
#: | 次 | 稳定用例 | 失败 |
#: |---|---|---|
#: | 1 | 35/38 = 92% | `agent-conflict-03`、`agent-review-04`、`agent-clarify-05` |
#: | 2 | **38/38 = 100%** | 无 |
#:
#: ⚠️ **两次都记下来，而不是只记好看的那一次**：单次通过不是证据（约定 91 的
#: 原话），第二轮的满分有相当一部分来自模型这次没抖。
#: 第 1 轮的三条里，`agent-conflict-03` 是**真问题**（判据被检索运气绑住——
#: 同一季度有两份报告，问句不点名就会召回任一份，已改成钉「机制 + 行」）；
#: 另两条是**模型抖动**（审查第二阶段这次判了、下次不判），与约定 105 同类。
#: 分母 38 而不是 40：两条观察用例不进门禁。
#:
#: 定在 85% 而不是 92% 或 100%：基线贴着实测值会让门禁变成随机红——
#: 而一个会随机变红的门禁很快就会被忽略，那比没有门禁更糟
#: （与 `eval_sql.py` 的 `_DEFAULT_BASELINE` 同一条理由）。
#: 85% 折算成条数是 32.3/38，也就是**允许 5 条**因模型抖动失败，第 6 条报警。
#:
#: ⚠️ **分母涨到 38 之后，这条全局通过率拦不住"某一类（5 条）整体失效"了**，
#: 如实记下：丢掉 5 条只剩 33/38 = 86.8%，而判据是 `rate < 0.85`。
#: 涨到 40 条时这条灵敏度就是这么变的——**单类的退化请看报告里那张
#: 分类通过率表**（每一类单独报，本来就是为这个加的），
#: 而不是指望一个全局数字同时干两件事。要把它变成门禁（按类设阈值）是独立的一件事，
#: 已登记。
#:
#: ⚠️ **未判定计入分母**（见模块 docstring），所以系统大面积失败任务时
#: 这个数会掉下来——那是刻意的，不是误报。
_DEFAULT_BASELINE = 0.85


@dataclass(frozen=True)
class CaseOutcome:
    case: dict[str, Any]
    verdict: Verdict
    status: str
    sources: tuple[str, ...]
    refused: bool | None
    conflicts: tuple[str, ...]
    review: dict[str, Any] | None
    #: 走这一趟有没有演进（`plan_revision`）。**报告里单列一栏**：
    #: `loop` 那一类的判据全靠它，而它不在 `sources` / `conflicts` 里，
    #: 不带上就只能靠 `verdict.note` 读文字。
    plan_revision: int = 0
    #: 下钻出来的步骤（`origin=EXTENDED`）。同样是给报告读的。
    extended: tuple[str, ...] = ()

    @property
    def category(self) -> str:
        return str(self.case["category"])

    @property
    def stability(self) -> str:
        """`stable`（默认）或 `model-dependent`。

        与 `configs/demo_questions.yaml` **同一套取值、同一条理由**：
        后者的结果取决于模型的意图判定或检索排序，同一个问题两次跑可能
        不一样。混进通过率会让人把"模型这次判歪了"当成"系统坏了"——
        而两者的处置完全不同（一个去查它的判断依据，一个去查回归）。
        """
        return str(self.case.get("stability", "stable"))


def load_cases(path: str) -> list[dict[str, Any]]:
    """读评测集并**校验结构**。

    校验放在这里而不是靠"跑起来自然会发现"：一份写错类别的 YAML 跑起来
    完全正常（用例照跑、结果照打），只有分类统计悄悄少一块。
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise SystemExit(f"评测集 {path} 结构不对：顶层需要 version 与 cases 两个键")

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw["cases"]:
        for key in ("id", "category", "question", "proves"):
            if not item.get(key):
                raise SystemExit(f"评测集 {path} 里有一条用例缺少 {key}：{item.get('id') or item}")
        if item["id"] in seen:
            raise SystemExit(f"评测集 {path} 里 id 重复：{item['id']}")
        if item["category"] not in CATEGORIES:
            raise SystemExit(
                f"评测集 {path} 里 {item['id']} 的类别 {item['category']!r} 不在表内，"
                f"只有：{sorted(CATEGORIES)}"
            )
        seen.add(item["id"])
        cases.append(dict(item))
    return cases


# ------------------------------------------------------------------ 输出
def render(outcome: CaseOutcome) -> str:
    mark = "OK  " if outcome.verdict.ok else "FAIL"
    if outcome.verdict.unjudged:
        mark = "?   "
    elif not outcome.verdict.ok and outcome.stability != "stable":
        # 与 `make demo` 同一套记号：○ 是"这次没符合预期，而它是观察用例"。
        # 打成 FAIL 会让一次模型抖动看起来像一次回归。
        mark = "OBS "
    lines = [f"[{mark}] {outcome.case['id']}  {outcome.case['question']}"]
    lines.append(f"       验证：{outcome.case['proves']}")
    if outcome.plan_revision:
        # **只在真的演进过时打**：这一栏每次出现都会占一行，而绝大多数
        # 用例（单源查询、澄清）不演进——恒打一栏「演进 0 次」是噪声。
        lines.append(
            f"       演进：第 {outcome.plan_revision} 版"
            + (f"，下钻出 {sorted(set(outcome.extended))}" if outcome.extended else "")
        )
    if not outcome.verdict.ok:
        lines.append(f"       结论：{outcome.verdict.note}")
    return "\n".join(lines)


def _summary(outcomes: list[CaseOutcome]) -> tuple[int, int, str]:
    """算出分类通过率，返回 `(稳定用例通过数, 稳定用例总数, 供打印的总结)`。

    **门禁只看稳定用例**（`(通过, 总数)` 这对值就是给调用方算 `rate` 用的）。
    观察用例单独报——它们的失败是"模型这次判歪了"，与"系统坏了"是两件事，
    处置完全不同（见 `demo_questions.yaml` 的同一条说明）。
    """
    stable = [item for item in outcomes if item.stability == "stable"]
    volatile = [item for item in outcomes if item.stability != "stable"]

    by_category: dict[str, list[CaseOutcome]] = {key: [] for key in CATEGORIES}
    for outcome in outcomes:
        by_category[outcome.category].append(outcome)

    lines = ["", "分类通过率（每一类单独看，笼统的通过率说不出哪一类退化了）"]
    for key, label in CATEGORIES.items():
        items = by_category[key]
        if not items:
            # **空类别要报出来**：一条该类用例都没跑，与"该类全过"
            # 在通过率上长得一模一样。
            lines.append(f"  {label:<22} —— 本次没有这一类用例")
            continue
        passed = sum(1 for item in items if item.verdict.ok)
        lines.append(f"  {label:<22} {passed}/{len(items)}")

    stable_passed = sum(1 for item in stable if item.verdict.ok)
    unjudged = [item for item in outcomes if item.verdict.unjudged]
    failed = [item for item in stable if not item.verdict.ok and not item.verdict.unjudged]

    rate = stable_passed / len(stable) if stable else 0.0
    lines.append("")
    if stable:
        lines.append(f"稳定用例：{stable_passed}/{len(stable)} = {rate:.0%}")
    else:
        # **空分母不是 0%**。`ONLY=<观察用例>` 时若打「0/0 = 0%」，
        # 读的人会以为全挂了，而判据那边还会拿 0.0 去比基线 → 退出码 1：
        # 一次全绿的运行报成失败，而它看起来只是"这次没跑好"。
        lines.append("稳定用例：本次没有（只跑了观察用例）")
    if volatile:
        lines.append(
            f"观察用例：{sum(1 for item in volatile if item.verdict.ok)}/{len(volatile)}"
            "（结果取决于模型判定，**不进门禁**，可复现性见 YAML 的 `stability`）"
        )
    if unjudged:
        # **未判定单独报**：它计入分母（见模块 docstring），但不该与
        # "行为不符"混成一条结论——一个是系统故障，一个是能力退化。
        lines.append(f"  其中未判定 {len(unjudged)} 条（任务失败或分析生成失败，**已计入分母**）：")
        lines.extend(f"    - {item.case['id']}" for item in unjudged)
    if failed:
        lines.append(f"未通过：{', '.join(item.case['id'] for item in failed)}")
    return stable_passed, len(stable), "\n".join(lines)


# ------------------------------------------------------------------ 探针
def _probe(base: str, token: str, question: str) -> int:
    """跑一个**还没固化的**问题，把写 `expect_*` 需要的事实全部打出来。

    这是本项目的一条纪律的入口：**评测与演示问题都必须先用真链路跑一遍
    再固化**，否则固化的是"我以为会怎样"——`demo_questions.yaml` 里那条
    「2024年1月」就是这么踩出来的（那一年恰好有数据，根本不触发演进）。

    它比 `make demo-ask` 打得多：`review` / `progress_decision` /
    证据的 locator 都打出来，因为评测集的断言比演示集多，
    而"看不到的值"只能靠猜——那正是这个入口要消灭的东西。
    """
    detail = ask(base, token, question)
    steps = [step for step in detail.get("steps") or [] if step.get("status") != "PENDING"]

    print(f"任务 {detail['status']}｜id {detail.get('task_id')}｜意图 {detail.get('intent')}")
    print(f"走了：{sorted({step['tool'] for step in steps})}")
    print(f"refused：{detail.get('refused')}")
    print(f"演进判定：{detail.get('progress_decision')}｜理由：{detail.get('progress_reason')}")
    print(f"plan_revision：{detail.get('plan_revision')}")
    # **演进记录要打全**：`loop` 那一类的断言就写在这上面（版本号、加了哪几步、
    # 触发源），而它们不在答案里、也不在证据里——看不到就只能猜。
    for delta in detail.get("plan_deltas") or []:
        print(
            f"  delta v{delta.get('revision_no')} [{delta.get('trigger')}] "
            f"加了 {delta.get('added_step_ids')}｜触发发现 {delta.get('trigger_finding_id')}"
        )
        print(f"      理由：{delta.get('reason')}")
    print("步骤（含来源）：")
    for step in steps:
        print(
            f"  - {step.get('id')} [{step.get('status')}] {step.get('tool')} "
            f"origin={step.get('origin')} rev={step.get('revision_no')}｜{step.get('objective')}"
        )
    print(f"推理链：{detail.get('investigation_chain')}")

    print(f"冲突 {len(detail.get('conflicts') or [])} 条：")
    for item in detail.get("conflicts") or []:
        print(f"  - {item}")

    print("限制与未覆盖：")
    for item in detail.get("limitations") or []:
        print(f"  - {item}")

    print(f"审查：{detail.get('review')}")

    print("证据（title / metric_code / locator）：")
    for item in detail.get("evidence") or []:
        print(f"  - [{item.get('source_type')}] {item.get('title')}")
        print(f"      metric={item.get('metric_code')} scope={item.get('scope')}")
        print(f"      locator={item.get('locator')}")

    # **照**基准单位**填 `expect_numbers`**（判据会按单位还原答案里的数）。
    # 列表里混着年份与百分比等噪声，抄之前先认出哪个是要断言的那个值。
    print("答案里出现的数（含年份/百分比等噪声）：")
    for value in sorted(amounts(detail.get("final_answer_md") or "")):
        print(f"  - {value}")

    print("─" * 78)
    print(detail.get("final_answer_md") or "")
    return 0 if detail["status"] == "SUCCEEDED" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 端到端评测（需 make run 已在跑）")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API 地址")
    parser.add_argument("--golden", default=GOLDEN_PATH, help=f"评测集路径（默认 {GOLDEN_PATH}）")
    parser.add_argument("--only", default="", help="只跑指定 id，逗号分隔")
    parser.add_argument(
        "--baseline",
        type=float,
        default=_DEFAULT_BASELINE,
        help=f"通过率低于它则退出码非零（默认 {_DEFAULT_BASELINE:.0%}，见常量处的说明）",
    )
    parser.add_argument(
        "--ask", default="", help="跑一个还没固化的问题，打印写 expect_* 所需的事实"
    )
    parser.add_argument("--as", dest="ask_as", default="", help="以哪个账号跑 --ask")
    args = parser.parse_args(argv)

    try:
        login(args.base, args.ask_as or DEMO_USERNAME)
    except (httpx.HTTPError, OSError) as exc:
        print(f"连不上 API（{args.base}）：{exc}", file=sys.stderr)
        print("先在另一个终端跑 `make run`（只跑 make api 的话任务会永远停在 QUEUED）")
        return 2

    if args.ask:
        return _probe(args.base, login(args.base, args.ask_as or DEMO_USERNAME), args.ask)

    cases = load_cases(args.golden)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
        if not cases:
            print(f"没有匹配的用例：{args.only}", file=sys.stderr)
            return 1

    # **逐条刷新**：一轮全量要跑 15–25 分钟，而 `uv run` 不接终端时 stdout 是
    # 块缓冲的——不 flush 的话整个进程一行都不打，看起来像卡死了。
    # 一个"跑了 20 分钟没有任何输出"的命令，会让人以为它坏了然后 Ctrl-C。
    print(f"评测集：{args.golden}（{len(cases)} 题）")
    print(f"API：{args.base}\n", flush=True)

    outcomes: list[CaseOutcome] = []
    for case in cases:
        try:
            token = login(args.base, case.get("account") or DEMO_USERNAME)
            # **逐轮跑、逐轮判**（多轮用例共用同一个会话，见 `run_case`）。
            # 一个 `CaseOutcome` 只装一轮：多轮用例的每一轮各是一条可判定的
            # 行为，合成一条会丢掉"是哪一轮不成立"——而那正是排查的入口。
            turns = case_turns(case)
            details = run_case(args.base, token, case)
        except (httpx.HTTPError, TimeoutError) as exc:
            # 连不上或超时**不是这条用例的结论**，但也不能静默跳过——
            # 跳过的话"跑不动"会以"没这一类用例"的面目出现在报告里。
            outcomes.append(
                CaseOutcome(
                    case=case,
                    verdict=Verdict(False, f"**跑不动**：{exc}", unjudged=True),
                    status="ERROR",
                    sources=(),
                    refused=None,
                    conflicts=(),
                    review=None,
                )
            )
            print(render(outcomes[-1]), flush=True)
            continue

        for turn, detail in zip(turns, details, strict=True):
            outcome = CaseOutcome(
                case=turn,
                verdict=evaluate(turn, detail),
                status=str(detail.get("status")),
                sources=tuple(
                    step["tool"]
                    for step in (detail.get("steps") or [])
                    if step["status"] != "PENDING"
                ),
                refused=detail.get("refused"),
                conflicts=tuple(detail.get("conflicts") or []),
                review=detail.get("review"),
                plan_revision=int(detail.get("plan_revision") or 0),
                extended=tuple(
                    step["tool"]
                    for step in (detail.get("steps") or [])
                    if step.get("origin") == "EXTENDED"
                ),
            )
            outcomes.append(outcome)
            print(render(outcome), flush=True)

    passed, total, summary = _summary(outcomes)
    print(summary, flush=True)
    # **分母为空时不判基线**：`ONLY=<观察用例>` 的 rate 是 0.0 而它不代表
    # 任何失败——拿它去比基线会让一次全绿的运行返回非零。
    rate = passed / total if total else 0.0
    if args.baseline and total and rate < args.baseline:
        print(f"\n低于阶段基线 {args.baseline:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
