"""端到端演示（冲刺方案 §8.3 第 8 项）。

```bash
make run          # 另开一个终端：api + worker
make demo         # 跑固化下来的九条问题
make demo-ask Q="某条还没固化的问题"  # 先跑一遍再固化（本文件自己的纪律）
make demo ONLY=demo-cross
```

## 走 API 而不是直接跑图

`app.cli` 里已经有 `make sql` / `make retrieve`，它们直接调 Tool——
那证明的是"工具能跑"。这个脚本要证明的是**整条链路能跑**：
HTTP → 限流 → 入库 → 队列 → Worker 领取 → 图 → 写回 → 查详情。
两者的失败模式完全不同（队列没投递成功、Worker 没起来、权限没装载，
在 Tool 层全都看不到）。

代价是它**要求 `make run` 已经在跑**。脚本会在连不上时明确说出来，
而不是抛一个 `ConnectionRefusedError` 让人以为是代码坏了。

## 判定用 `configs/demo_questions.yaml` 的 `expect_*`，不靠人眼看

把答案打出来让人自己看，在演示时看起来一样，但它证明不了任何事——
而"这次演示过了"如果没有判据，它就不是一次验收。

**判据本身在 `scripts/agent_harness.py`**：它与回归评测（`make eval-agent`）
共用同一份，理由见那个模块的 docstring。本文件只管**怎么把它讲给人听**。
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

from scripts.agent_harness import DEMO_USERNAME, amounts, ask, evaluate, login

DEMO_PATH = Path("configs/demo_questions.yaml")


@dataclass(frozen=True)
class Outcome:
    case_id: str
    question: str
    status: str
    #: 这条用例是不是"结果取决于模型判定"。见 `configs/demo_questions.yaml`
    #: 的说明——**它不进通过率**，因为同一个问题两次跑可能不一样，
    #: 混进去会让人把"模型这次判歪了"当成"系统坏了"。
    stability: str
    sources: tuple[str, ...]
    answer: str
    conflicts: tuple[str, ...]
    review: dict[str, Any] | None
    ok: bool
    note: str


def _load_cases(only: str) -> list[dict[str, Any]]:
    raw = yaml.safe_load(DEMO_PATH.read_text(encoding="utf-8"))
    cases = list(raw.get("questions") or [])
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    return cases


def _ask(base: str, token: str, question: str) -> int:
    """跑一个**还没固化的**问题，把写 `expect_*` 需要的事实打出来。

    存在的理由是本项目自己的一条纪律：**演示问题必须先用真链路跑一遍再固化**，
    否则固化的是"我以为会怎样"——语料里那条「2024年1月」就是这么踩出来的
    （那一年恰好有数据，根本不触发演进）。没有这个入口时，写一条新用例
    要先手改 YAML 再跑 `--only`，而"先跑一遍"就变成了"先猜一遍"。

    ⚠️ **要断言的字段比这里打出来的多**（review / progress_decision 等），
    所以调试**评测集**时用 `make eval-agent-ask`——它打的是同一份事实的
    超集，并且用的是同一份解析。
    """
    detail = ask(base, token, question)
    steps = [step for step in detail.get("steps") or [] if step.get("status") != "PENDING"]
    print(f"任务 {detail['status']}｜id {detail.get('task_id')}")
    print(f"走了：{sorted({step['tool'] for step in steps})}")
    print(f"refused：{detail.get('refused')}｜冲突 {len(detail.get('conflicts') or [])} 条")
    print("限制与未覆盖：")
    for item in detail.get("limitations") or []:
        print(f"  - {item}")
    # **照**基准单位**填 `expect_numbers`**（判据会按单位还原答案里的数）。
    # 列表里混着年份与百分比等噪声，抄之前先认出哪个是要断言的那个值。
    print("答案里出现的数（含年份/百分比等噪声）：")
    for value in sorted(amounts(detail.get("final_answer_md") or "")):
        print(f"  - {value}")
    print("─" * 78)
    print(detail.get("final_answer_md") or "")
    return 0 if detail["status"] == "SUCCEEDED" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端演示（需 make run 已在跑）")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API 地址")
    parser.add_argument("--only", default="", help="只跑指定 ID，逗号分隔")
    parser.add_argument(
        "--ask", default="", help="跑一个还没固化的问题，打印写 expect_* 所需的事实"
    )
    parser.add_argument("--as", dest="ask_as", default="", help="以哪个演示账号跑 --ask")
    args = parser.parse_args(argv)

    try:
        login(args.base, args.ask_as or DEMO_USERNAME)
    except (httpx.HTTPError, OSError) as exc:
        # **说出该跑什么**，而不是抛一个连接异常。演示脚本的报错信息
        # 是"下一步做什么"，不是"哪里错了"。
        print(f"连不上 API（{args.base}）：{exc}", file=sys.stderr)
        print("先在另一个终端跑 `make run`（只跑 make api 的话任务会永远停在 QUEUED）")
        return 2

    if args.ask:
        return _ask(args.base, login(args.base, args.ask_as or DEMO_USERNAME), args.ask)

    cases = _load_cases(args.only)
    if not cases:
        print("没有匹配的演示问题，检查 configs/demo_questions.yaml 与 --only", file=sys.stderr)
        return 2

    outcomes: list[Outcome] = []
    for case in cases:
        print(f"\n{'=' * 78}\n▶ {case['id']}：{case['question']}")
        try:
            # **每条用例可以指定账号**（默认 admin）。权限相关的行为只有换一个
            # 受限账号才演示得出来——"同一个问题，两个账号看到的限制不同"
            # 本身就是一句话能讲清、且别处看不到的东西。
            token = login(args.base, case.get("account") or DEMO_USERNAME)
            detail = ask(args.base, token, case["question"])
        except (httpx.HTTPError, TimeoutError) as exc:
            outcomes.append(
                Outcome(
                    case["id"],
                    case["question"],
                    "ERROR",
                    case.get("stability", "stable"),
                    (),
                    "",
                    (),
                    None,
                    False,
                    str(exc),
                )
            )
            print(f"  ✗ {exc}")
            continue

        verdict = evaluate(case, detail)
        outcome = Outcome(
            case_id=case["id"],
            question=case["question"],
            status=str(detail.get("status")),
            stability=case.get("stability", "stable"),
            sources=tuple(
                step["tool"] for step in (detail.get("steps") or []) if step["status"] != "PENDING"
            ),
            answer=detail.get("final_answer_md") or "",
            conflicts=tuple(detail.get("conflicts") or []),
            review=detail.get("review"),
            ok=verdict.ok,
            note=verdict.note,
        )
        outcomes.append(outcome)
        mark = "✓" if outcome.ok else ("○" if outcome.stability != "stable" else "✗")
        print(f"  {mark} {verdict.note}")
        print(f"  证明：{case.get('proves', '')}")
        _print_answer(outcome)

    return _summary(outcomes)


def _print_answer(outcome: Outcome) -> None:
    answer = outcome.answer.strip()
    if not answer:
        print("  （没有答案）")
        return
    # 只打前 12 行：演示时要能一眼看到"直接回答 + 依据"，
    # 而完整答案在任务详情里。全量打印会把终端刷掉，
    # 下一个问题就看不见了——那正好是演示最需要连贯的时候。
    lines = answer.splitlines()
    print("  ┌" + "─" * 74)
    for line in lines[:12]:
        print(f"  │ {line[:72]}")
    if len(lines) > 12:
        print(f"  │ …（共 {len(lines)} 行，完整答案见任务详情）")
    print("  └" + "─" * 74)


def _summary(outcomes: list[Outcome]) -> int:
    print(f"\n{'=' * 78}\n汇总（判定依据 configs/demo_questions.yaml 的 expect_*）")
    for outcome in outcomes:
        mark = "✓" if outcome.ok else ("○" if outcome.stability != "stable" else "✗")
        print(f"  {mark} {outcome.case_id:14} {outcome.status:10} {outcome.note}")

    stable = [item for item in outcomes if item.stability == "stable"]
    volatile = [item for item in outcomes if item.stability != "stable"]
    failed = [item for item in stable if not item.ok]
    print()
    print(f"稳定用例 {len(stable) - len(failed)}/{len(stable)} 条符合预期")
    if volatile:
        # **不稳定的那几条单独报**：它们的结果取决于模型的意图判定或检索排序，
        # 同一个问题两次跑可能不一样。混进通过率会让人把"模型这次判歪了"
        # 当成"系统坏了"——而两者的处置完全不同。
        print(
            f"观察用例 {sum(1 for item in volatile if item.ok)}/{len(volatile)} 条符合预期"
            "（结果取决于模型判定，可复现性见 configs/demo_questions.yaml）"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
