"""端到端演示（冲刺方案 §8.3 第 8 项）。

```bash
make run          # 另开一个终端：api + worker
make demo         # 跑固化下来的六条问题
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
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.repositories.user_repo import DEMO_ACCOUNTS

DEMO_PATH = Path("configs/demo_questions.yaml")

#: 演示账号。**从 `DEMO_ACCOUNTS` 取而不是在这里再抄一遍口令**：
#: 抄一遍就有两处会漂移，而漂移的表现是"演示脚本登不上"——
#: 那时要先怀疑是不是口令改了，而它其实只是改了另一边。
#:
#: **用 admin**：它的数据范围不限（TBC-03），于是演示里不会因为
#: "华东之外的数据被权限挡掉"而出现看不懂的空结果——那不是系统坏了，
#: 但演示时解释它要花掉半分钟。
DEMO_USERNAME = "admin"
DEMO_PASSWORD = next(
    password for username, password, _, _ in DEMO_ACCOUNTS if username == DEMO_USERNAME
)

#: 单条任务最多等多久。图的正常耗时在 30–90 秒（含两次模型调用与一次检索），
#: 给 180 秒是留足余量又能在真的卡住时很快报出来。
_TASK_TIMEOUT_SECONDS = 180
_POLL_INTERVAL_SECONDS = 3


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


#: 用 `httpx`（项目已有的依赖）而不是 `urllib`：后者的错误类型零散
#: （`URLError` / `HTTPError` / `socket.error`），而这里只需要"连不上"与
#: "通了但报错"两类。演示脚本的异常处理越短越好——它出问题时
#: 正是在有人看着的时候。
_TIMEOUT = httpx.Timeout(30.0)


def _post(url: str, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def _get(url: str, *, token: str) -> dict[str, Any]:
    response = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


def _load_cases(only: str) -> list[dict[str, Any]]:
    raw = yaml.safe_load(DEMO_PATH.read_text(encoding="utf-8"))
    cases = list(raw.get("questions") or [])
    if only:
        wanted = {item.strip() for item in only.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    return cases


#: 数值断言里的数字。**带单位**——语料与答案里「万元」「亿元」是常态，
#: 不认单位会让一条正确的「11,196.70 万元」判成错的。
_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(亿元|亿|万元|万|千元|千)?")

#: 中文数词的单位倍率
_UNITS: dict[str, Decimal] = {
    "亿": Decimal(10**8),
    "亿元": Decimal(10**8),
    "万": Decimal(10**4),
    "万元": Decimal(10**4),
    "千": Decimal(10**3),
    "千元": Decimal(10**3),
}

#: 数值断言的**相对**容差。
#:
#: 取 0.1% 是因为**单位换算必然带来舍入**：真实值 111,967,031.73 元
#: 写成「11,196.70 万元」只差 0.0003%。而真正的算错差得远不止这个量级——
#: 实测那次把 8 行明细当成区域合计，差了 46%。
#: 两者之间有五个数量级的空档，阈值取在哪儿都不影响判定，取一个能容下
#: 舍入的值即可。
_NUMBER_TOLERANCE = Decimal("0.001")


def _amounts(text: str) -> list[Decimal]:
    """答案里出现的所有金额（**按单位还原成基准值**）。

    不还原单位的话，「1.12 亿」与「111,967,031.73」会被当成两个不相关的数，
    而它们说的是同一件事——**断言会因为写法不同而误报**，
    那种红叉比没有断言更糟（它会让人开始忽略断言）。
    """
    found: list[Decimal] = []
    for raw, unit in _NUMBER.findall(text):
        try:
            value = Decimal(raw.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - 正则已保证是数字
            continue
        found.append(value * _UNITS.get(unit, Decimal(1)))
    return found


def _close(found: Decimal, expected: Decimal) -> bool:
    if expected == 0:
        return found == 0
    return abs(found - expected) / abs(expected) <= _NUMBER_TOLERANCE


def _missing_amounts(answer: str, expected: list[str]) -> list[str]:
    """`expect_numbers` 里**没有出现**在答案中的那些。

    配置写错（不是数字）时直接抛——那是**这份清单自己的 bug**，
    静默跳过会让一条永远不会生效的断言看起来一直在通过。
    """
    found = _amounts(answer)
    missing: list[str] = []
    for raw in expected:
        try:
            want = Decimal(str(raw).replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError(f"demo_questions.yaml 的 expect_numbers 不是数字：{raw!r}") from exc
        if not any(_close(value, want) for value in found):
            missing.append(str(raw))
    return missing


def _evaluate(case: dict[str, Any], detail: dict[str, Any]) -> tuple[bool, str]:
    """按 `expect_*` 判定。**每条只给一个结论**——多个失败点会让报告读不出主因。

    **任务本身失败时先报那个**：`MODEL_OUTPUT_INVALID` 会让 `steps` 为空，
    而"期望 rag 实际 []"看起来像"工具选错了"——失败指错了层，
    排查方向会跑到意图判定上去，而真正的原因是一次模型输出抖动。
    这类"报告指向错误的一层"是本项目反复踩到的一类问题。
    """
    if detail.get("status") != "SUCCEEDED":
        code = detail.get("error_code")
        message = str(detail.get("error_message") or "")[:60]
        return False, f"**任务未成功**：{code} - {message}"

    steps = detail.get("steps") or []
    sources = {step["tool"] for step in steps if step["status"] != "PENDING"}
    conflicts = detail.get("conflicts") or []
    answer = detail.get("final_answer_md") or ""

    if case.get("expect_rejected"):
        # **判据读 `refused` 字段，不再认答案里的词**。
        #
        # 这里原先是一串字符串匹配（"没有找到"/"未能检索到"/…），它出过一次
        # 典型的漏判：同一条问题、同一份代码，模型说「没有找到」时判对，
        # 说「证据中没有…的条文」时判错——**四次里错一次**。判据在猜词，
        # 而演示现场出现一次红叉，代价远大于它省下的那点改动。
        # 现在 `refused` 由分析节点按 11.8 的判据直接给出（工具层拒答
        # 由代码置位），见 `AnalysisResult.refused`。
        if detail.get("refused") is True:
            return True, "按 11.8 拒答（refused=true）"
        # 分析生成失败时 `refused` 是 `False`（故障不是拒答），
        # 但此时报"没有拒答"会把一次故障说成一次编造——**失败指错了层**。
        # 这里认的是 `_degraded` 自己写下的那句，是**我们自己的字符串**，
        # 不是模型措辞，所以不含上面那种不确定性。
        if any("分析生成失败" in item for item in detail.get("limitations") or []):
            return False, "**未能判定**——综合分析生成失败，这条不是拒答行为的结果"
        return False, "**没有拒答**——refused=false，而语料里没有这条制度"

    if case.get("expect_clarification"):
        if not steps:
            return True, f"进入澄清：{answer.strip().splitlines()[0][:60]}"
        return False, f"**没有进入澄清**——直接跑了 {sorted(sources)}"

    expect = set(case.get("expect_sources") or [])
    if expect != sources:
        return False, f"**工具选择不符**：期望 {sorted(expect)}，实际 {sorted(sources)}"

    if case.get("expect_conflicts") and not conflicts:
        return False, "**没有检出冲突**——而这条问题的两个来源数字本就不一致"

    # **数值断言**：这是唯一能拦住「答案里的数字是错的」的判据。
    # 其余三处门禁都测不到它——`eval-sql` 测 SQL 与金标结果集等价、
    # `eval-rag` 测 Recall@8、`verify-corpus` 测缺陷注入，全在组件层；
    # 而 2026-09-20 实测到的那次，**四道门禁全绿、Reviewer 给 100 分，
    # 答案里的数是错的**（8 行明细当成区域合计，差 46%）。
    if missing := _missing_amounts(answer, case.get("expect_numbers") or []):
        return False, f"**答案里没有出现期望的数值**：{missing}"

    if missing_limits := [
        item
        for item in (case.get("expect_limitations_contain") or [])
        if item not in " ".join(detail.get("limitations") or [])
    ]:
        return False, f"**限制清单里缺少**：{missing_limits}"

    return True, f"走了 {sorted(sources)}" + (
        f"，检出 {len(conflicts)} 条冲突" if conflicts else ""
    )


def _wait(token: str, task_id: str, *, base: str) -> dict[str, Any]:
    deadline = time.monotonic() + _TASK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        detail: dict[str, Any] = _get(f"{base}/api/agent/tasks/{task_id}", token=token)["data"]
        if detail.get("status") in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return detail
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"任务 {task_id} 在 {_TASK_TIMEOUT_SECONDS} 秒内没有结束")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="端到端演示（需 make run 已在跑）")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API 地址")
    parser.add_argument("--only", default="", help="只跑指定 ID，逗号分隔")
    args = parser.parse_args(argv)

    try:
        token = _post(
            f"{args.base}/api/auth/login", {"username": DEMO_USERNAME, "password": DEMO_PASSWORD}
        )["data"]["access_token"]
    except (httpx.HTTPError, OSError) as exc:
        # **说出该跑什么**，而不是抛一个连接异常。演示脚本的报错信息
        # 是"下一步做什么"，不是"哪里错了"。
        print(f"连不上 API（{args.base}）：{exc}", file=sys.stderr)
        print("先在另一个终端跑 `make run`（只跑 make api 的话任务会永远停在 QUEUED）")
        return 2

    cases = _load_cases(args.only)
    if not cases:
        print("没有匹配的演示问题，检查 configs/demo_questions.yaml 与 --only", file=sys.stderr)
        return 2

    outcomes: list[Outcome] = []
    for case in cases:
        print(f"\n{'=' * 78}\n▶ {case['id']}：{case['question']}")
        try:
            created = _post(
                f"{args.base}/api/agent/chat", {"message": case["question"]}, token=token
            )["data"]
            detail = _wait(token, created["task_id"], base=args.base)
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

        ok, note = _evaluate(case, detail)
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
            ok=ok,
            note=note,
        )
        outcomes.append(outcome)
        mark = "✓" if ok else ("○" if outcome.stability != "stable" else "✗")
        print(f"  {mark} {note}")
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
