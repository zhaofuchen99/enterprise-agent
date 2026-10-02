"""性能基线压测（需求 8.1 的 NFR-P-01..08）。

## 目标值写死在这里，不跟实测走

`TARGETS` 抄的是需求规格 8.1 的**原文**。改成实测值就等于把门禁变成
"记录一下我们跑出来多少"——那是没有门禁。跑不达标就如实打 ✗ 并给根因
（NFR-P-08 就是这一种，见下）。

## ⚠️ 压测**绕开了产品配额**，这句话必须跟着数字一起说

单用户三道闸门都会挡住"20 并发"：同时运行上限 3、建任务 10/分钟、
状态查询 120/分钟。被限流时脚本**直接退出并提示怎么放开**——继续跑没有意义，
后面每一步都会因为"任务建不出来"而错位，而报告上看起来像延迟异常。

放开的代价如实记：它测的是**配额之外的容量**，不是"默认配置下的用户体验"。
配额本身另有单测与集成覆盖（`test_chat.py` 的 429、跨实例共享配额用例），
压测不重复覆盖它。

## ⚠️ NFR-P-08 与 NFR-P-06 在给定任务耗时下**数学互斥**

单任务耗时 S ≈ 5–15 秒、Worker 并发 W=4 时，第 k 个任务的排队等待
≈ `(⌈k/W⌉−1)×S`；第 20 个 ≈ `4S`，**远超 2 秒**。要让"20 个任务等待 P95 ≤ 2s"
成立，必须 W ≥ 20——而那正是 NFR-P-06 要的并发数。

这不是实现缺陷，是两条指标在给定 S 下的一致性冲突。需求原文已有
「性能目标为 MVP 基线，应在确定模型、硬件和数据规模后重新校准」，
所以脚本**不改目标、不改判据**，只在报告里给出根因与理论吞吐估算。

## 样本量的诚实边界

P95 是第 `⌈0.95n⌉` 个次序统计量——**n=20 时它就是第 19 大的值**，
换个随机种子可能差 30% 以上。亚秒级指标取 n≥100（代价低，没理由少取），
任务级默认 20 并在报告里注明"是估计值，不做跨版本精确比较"。
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from scripts.agent_harness import DEMO_USERNAME, get, login, post
from scripts.agent_harness import wait_for_task as _wait_for_task

#: 需求 8.1 的目标。**原文抄写，不跟实测走**。
TARGETS: dict[str, tuple[str, float]] = {
    "NFR-P-01": ("状态查询 P95", 0.300),
    "NFR-P-02": ("SSE 建连 P95", 1.000),
    "NFR-P-03": ("首个进度事件 P95", 1.000),
    "NFR-P-04": ("简单 SQL 任务 P95", 15.000),
    "NFR-P-05": ("多源分析任务 P95", 60.000),
    "NFR-P-08": ("排队等待 P95", 2.000),
}

#: 亚秒级探针的默认样本量。**没理由取小**——它便宜。
_PROBE_SAMPLES = 200
#: 任务级的默认样本量。每条要花十几秒到一分钟，取大了这个命令没人跑。
_TASK_SAMPLES = 20

_SQL_QUESTION = "2025年Q3华东地区的净销售额是多少？"
_MULTI_QUESTION = "2025年Q3华东地区净销售额同比下降，报告里是怎么解释的？"

_STATEMENT = "指标对照表（目标取自需求规格 8.1 原文，**未按实测调整**）"


@dataclass(frozen=True)
class Metric:
    """一条指标的实测结果。`p95 is None` 表示本次没测（场景没跑）。

    ⚠️ **`higher_is_better` 必须显式给**：需求 8.1 里大多数是"越小越好"（延迟），
    但 **NFR-P-06 是"并发数 ≥ 20"，越大越好**。方向写反的后果是它**恒判通过**
    ——实测踩到过：峰值只有 4 却打了 ✓，而那一行读起来像"并发达标了"。
    一个恒过的判据比没有判据更糟：它看起来是有保障的。
    """

    code: str
    label: str
    target: float
    unit: str = "s"
    p50: float | None = None
    p95: float | None = None
    worst: float | None = None
    n: int = 0
    note: str = ""
    higher_is_better: bool = False

    @property
    def ok(self) -> bool | None:
        if self.p95 is None:
            return None
        if self.higher_is_better:
            return self.p95 >= self.target
        return self.p95 <= self.target


@dataclass
class Observations:
    """一次跑批攒下的原始样本，按指标编号分组。"""

    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, code: str, seconds: float) -> None:
        self.values.setdefault(code, []).append(seconds)

    def summarise(self, code: str, *, note: str = "") -> Metric:
        return _summarise(code, self.values.get(code, []), note=note)


def _percentile(values: list[float], fraction: float) -> float:
    """次序统计量。

    `ceil(fraction * n) - 1` 就是 P95 的**定义**：n=20 时取第 19 大的值。
    换成插值会让小样本的 P95 看起来"平滑"，而那是估算不是实测。
    """
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, -(-int(fraction * len(ordered) * 1000) // 1000) - 1))
    return ordered[index]


def _summarise(code: str, values: list[float], *, note: str = "") -> Metric:
    label, target = TARGETS[code]
    if not values:
        return Metric(code=code, label=label, target=target, note=note or "本次没测")
    return Metric(
        code=code,
        label=label,
        target=target,
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        worst=max(values),
        n=len(values),
        note=note,
    )


async def _create_task(client: httpx.AsyncClient, base: str, token: str, question: str) -> str:
    """建任务。**被限流直接退出**——继续跑只会得到一串错位的数字。

    ⚠️ `post` / `get` / `wait_for_task` 是 `agent_harness` 里的**同步**函数
    （它们用 `httpx` 的阻塞 API，`demo.py` 与 `eval_agent.py` 共用同一份）。
    这里不 `await`，也因此**会短暂阻塞事件循环**——本脚本是按顺序一条条
    跑任务的，没有并发要调度，所以这个代价是零。**别在别处照抄这个写法**：
    一旦有并发，阻塞调用会把它们串行化。
    """
    response = post(f"{base}/api/agent/chat", {"message": question}, token=token)
    if response.get("code") == "RATE_LIMITED":
        raise SystemExit(
            "被限流了（429）。压测**必须**先放开三道配额、再重启 make run：\n"
            "  MAX_RUNNING_TASKS_PER_USER=100 RATE_LIMIT_CREATE_PER_MINUTE=1000\n"
            "  RATE_LIMIT_STATUS_PER_MINUTE=10000 RATE_LIMIT_STREAM_PER_MINUTE=1000\n"
            "⚠️ 这些是**配置上界**（`config.py` 的 `le=`），再往上写启动即失败——\n"
            "上界本身是对的，压测取的就是它允许的最大值。\n"
            "否则跑出来的是一串 429，而它看起来像'系统扛不住'。"
        )
    return str(response["data"]["task_id"])


async def _first_progress_seconds(
    client: httpx.AsyncClient, base: str, token: str, task_id: str
) -> float:
    """从**建连**到**第一个非 snapshot 事件**（NFR-P-03）。

    `snapshot` 是建连时立刻回的那一帧（来自任务对象、不消耗事件），
    所以它**不算**进度事件——算进去的话这个指标量的是"建连有多快"，
    与 NFR-P-02 变成同一件事。这是需求原文里"首个进度事件"的两种读法之一，
    报告里注明口径即可（另一种读法见 `make eval-agent` 的作者注）。
    """
    started = time.monotonic()
    async with client.stream(
        "GET",
        f"{base}/api/agent/tasks/{task_id}/stream",
        headers={"Authorization": f"Bearer {token}"},
    ) as response:
        async for line in response.aiter_lines():
            name = line.removeprefix("event: ") if line.startswith("event: ") else ""
            if name and name not in ("snapshot", "heartbeat"):
                return time.monotonic() - started
    raise TimeoutError("流在给出第一个进度事件之前就结束了")


async def _count_task_started(
    client: httpx.AsyncClient, base: str, token: str, task_id: str
) -> int:
    """重放事件流，数 `task.started` 的条数。**>1 就是重复执行。**

    **为什么用它而不是查库**（三个候选都不可靠）：
    - `agent_task.worker_id` **会被后一次执行覆盖**，重复执行看不出来；
    - 证据 / 工具调用落库是**整体替换**（`agent_repo.save` 先清后写），
      重跑也是同一批行数；
    - 助手消息用 `replace_for_task` 写（约定 113），**也不是 append-only**。

    `task.started` 由 `TaskRunner` 在 `claim` 成功后发一次，而 `claim` 是
    `QUEUED → RUNNING` 的条件更新——第二个领取者会失败、也就不发第二条。
    """
    count = 0
    async with client.stream(
        "GET",
        f"{base}/api/agent/tasks/{task_id}/stream",
        headers={"Authorization": f"Bearer {token}"},
    ) as response:
        async for line in response.aiter_lines():
            if line == "event: task.started":
                count += 1
            elif line == "event: done":
                break
    return count


def _seconds_between(start: Any, end: Any) -> float:
    return (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds()


async def _observe_one_task(
    client: httpx.AsyncClient,
    base: str,
    token: str,
    *,
    question: str,
    end_to_end_code: str,
) -> dict[str, float | str]:
    """跑一个任务，把这一条能贡献的样本都收回来。

    **一趟收三样**（P-03 / P-04 或 P-05 / P-08）：建一个任务要花十几秒到一分钟，
    为每条指标单独建任务会让这个命令跑不动，而三样本来就来自同一次执行。
    端到端用**服务端时钟**（started_at → finished_at），不掺客户端调度。
    """
    task_id = await _create_task(client, base, token, question)
    first_progress = await _first_progress_seconds(client, base, token, task_id)
    # ⚠️ **`wait_for_task` 返回的已经是 `data`**（它在内部取过一次
    # `["data"]`，见 `agent_harness.py`）。这里再取一层会 `KeyError: 'data'`
    # ——而它只在"任务跑完"这一条路径上炸，探针路径（只调不取）完全正常，
    # 所以症状是"probe 场景好着，任务场景一跑就崩"。
    data = _wait_for_task(token, task_id, base=base)
    return {
        "task_id": task_id,
        "first_progress": first_progress,
        end_to_end_code: _seconds_between(data["started_at"], data["finished_at"]),
    }


async def _probe_metrics(
    client: httpx.AsyncClient, base: str, token: str, *, samples: int
) -> tuple[list[Metric], list[str]]:
    """NFR-P-01 / P-02：**在一个终态任务上量**（空队列才不会量到排队）。"""
    task_id = await _create_task(client, base, token, _SQL_QUESTION)
    _wait_for_task(token, task_id, base=base)

    status_latencies: list[float] = []
    for _ in range(samples):
        started = time.monotonic()
        get(f"{base}/api/agent/tasks/{task_id}", token=token)
        status_latencies.append(time.monotonic() - started)

    connect_latencies: list[float] = []
    for _ in range(min(samples, 100)):
        started = time.monotonic()
        async with client.stream(
            "GET",
            f"{base}/api/agent/tasks/{task_id}/stream",
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            async for _line in response.aiter_lines():
                break
        connect_latencies.append(time.monotonic() - started)

    executions = await _count_task_started(client, base, token, task_id)
    return [
        _summarise("NFR-P-01", status_latencies),
        _summarise("NFR-P-02", connect_latencies),
    ], [
        f"重复执行检查：探针任务的 `task.started` 事件 {executions} 条"
        f"（{'正常' if executions == 1 else '**异常：被领取了多次**'}）。",
        "NFR-P-01 是**客户端侧**延迟（含本机回环），不能声称是服务端耗时。",
        "NFR-P-03 的口径是「建连到第一条非 snapshot 事件」；"
        "若把 snapshot 也算作进度事件，它会退化成 NFR-P-02。",
    ]


async def _capacity_metrics(
    client: httpx.AsyncClient, base: str, token: str, *, tasks: int
) -> tuple[list[Metric], list[str]]:
    """NFR-P-06：**一次性投 N 个任务**，看同时 RUNNING 的峰值。

    与 `sql`/`multi` 场景**必须分开跑**：那两个是"建一个、等一个"的串行，
    峰值恒为 1。这里要的是"同时有多少个在跑"，只能先把它们全投出去、
    再一边轮询一边看。

    判据两条，缺一不可：
    - **峰值 ≥ 20**（需求原文的并发数）；
    - **错误率不显著上升**——这里只报数字，不判"显著"：
      基线要与串行场景的错误率比，而那需要两次跑同题集；
      脚本把两个数字都打出来，判断留给人（这是"如实报告"不是"代替判断"）。

    ⚠️ **NFR-P-08（排队等待）只在这里量**：串行场景里任务是一个一个建的，
    排队等待恒为 0——量它等于量一个恒等式，而那个 0 会被读成"排队没问题"。
    真正会排队的就是这里：20 个任务一起投、Worker 只有 W 个槽位。
    """
    task_ids = []
    for _ in range(tasks):
        task_ids.append(await _create_task(client, base, token, _SQL_QUESTION))

    peak = 0
    while True:
        details = [get(f"{base}/api/agent/tasks/{tid}", token=token)["data"] for tid in task_ids]
        peak = max(peak, sum(1 for item in details if item["status"] == "RUNNING"))
        if all(item["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"} for item in details):
            break
        await asyncio.sleep(1.0)

    finished = sum(1 for item in details if item["status"] == "SUCCEEDED")
    failures = sum(1 for item in details if item["status"] == "FAILED")
    waits = _summarise(
        "NFR-P-08",
        [_seconds_between(item["queued_at"], item["started_at"]) for item in details],
    )
    duplicate_checks = 0
    for tid in task_ids:
        duplicate_checks += await _count_task_started(client, base, token, tid)
    return [
        Metric(
            code="NFR-P-06",
            label="并发运行任务数",
            target=float(tasks),
            unit="个",
            p50=float(peak),
            p95=float(peak),
            worst=float(peak),
            n=len(task_ids),
            note=f"需求要 ≥20；实测峰值 {peak}",
            # **越大越好**——漏掉它这一行会恒判 ✓（见 `Metric` 的说明）
            higher_is_better=True,
        ),
        waits,
    ], [
        f"容量场景：{tasks} 个任务并发投递，峰值同时 RUNNING **{peak}**，"
        f"成功 {finished}／失败 {failures}（错误率 {failures / tasks:.1%}）。",
        f"容量场景的 `task.started` 共 {duplicate_checks} 条"
        f"（{'每任务恰好一条，无重复执行' if duplicate_checks == tasks else '**异常**'}）。",
        "⚠️ 容量场景里 20 路并发模型调用可能撞服务商限流——错误率因此**部分取决于外部服务**，"
        "要在报告里与「系统扛不住」分开说。",
    ]


def _queue_wait_estimates(task_seconds: float, *, tasks: int = 20) -> str:
    """给定实测单任务耗时，估算不同 Worker 并发下的排队等待。

    **标注为估算**：它假设任务等长、到达即满、没有限流——真实系统三者都不成立。
    它唯一的用途是回答"把 W 提到多少才可能达标"。
    """
    return "／".join(
        f"W={workers}→{(max(1, -(-tasks // workers)) - 1) * task_seconds:.1f}s"
        for workers in (4, 8, 16, 20, 24)
    )


def _fmt(value: float | None, unit: str = "s") -> str:
    """**单位跟着指标走**：并发数是「个」而延迟是「秒」——统一打 `s`
    会让「峰值 4」印成 `4.000s`，而那一行读起来像一条延迟指标。"""
    if value is None:
        return "—"
    return f"{value:.0f}{unit}" if unit != "s" else f"{value:.3f}s"


def _print_report(metrics: list[Metric], *, notes: list[str], task_seconds: float) -> None:
    print()
    print(_STATEMENT)
    print(f"  {'编号':<10}{'指标':<16}{'目标':>9}{'P50':>10}{'P95':>10}{'最差':>10}{'n':>6}  判定")
    for item in metrics:
        verdict = "—" if item.ok is None else ("✓" if item.ok else "✗")
        # 目标值也跟单位走：并发数印 `20`，延迟印 `0.300s`
        target = f"{item.target:.0f}{item.unit}" if item.unit != "s" else f"{item.target:.3f}s"
        print(
            f"  {item.code:<10}{item.label:<16}{target:>9}"
            f"{_fmt(item.p50, item.unit):>10}{_fmt(item.p95, item.unit):>10}"
            f"{_fmt(item.worst, item.unit):>10}{item.n:>6}  {verdict}"
        )
    if task_seconds > 0:
        print(
            f"\n  理论排队等待（**估算**，按实测单任务 {task_seconds:.1f}s、20 个任务）："
            f"{_queue_wait_estimates(task_seconds)}"
        )
    for note in notes:
        print(f"  · {note}")


async def _run(args: argparse.Namespace) -> int:
    token = login(args.base, args.account)
    print(f"API：{args.base}｜场景：{args.scenario}｜账号：{args.account}", flush=True)
    print(
        "⚠️ 本命令**绕开产品配额**（单用户 3 个并发 / 建任务 10 每分钟）——"
        "它测的是配额之外的容量，不是默认配置下的用户体验。",
        flush=True,
    )

    metrics: list[Metric] = []
    notes: list[str] = [
        "P95 是第 ⌈0.95n⌉ 个次序统计量：n=20 时它就是第 19 大的值，"
        "换种子可能差 30% 以上——n 小的指标只看趋势，不做跨版本精确比较。",
    ]
    observed = Observations()
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        if args.scenario in ("all", "probe"):
            probe_metrics, probe_notes = await _probe_metrics(
                client, args.base, token, samples=args.samples or _PROBE_SAMPLES
            )
            metrics.extend(probe_metrics)
            notes.extend(probe_notes)

        scenarios = ["sql", "multi", "capacity"] if args.scenario == "all" else [args.scenario]
        if "capacity" in scenarios:
            capacity_metrics, capacity_notes = await _capacity_metrics(
                client,
                args.base,
                token,
                tasks=args.samples or _TASK_SAMPLES,
            )
            metrics.extend(capacity_metrics)
            notes.extend(capacity_notes)

        for scenario in (item for item in scenarios if item not in ("probe", "capacity")):
            count = args.samples or _TASK_SAMPLES
            question = _MULTI_QUESTION if scenario == "multi" else _SQL_QUESTION
            code = "NFR-P-05" if scenario == "multi" else "NFR-P-04"
            started_events = 0
            for index in range(count):
                row = await _observe_one_task(
                    client, args.base, token, question=question, end_to_end_code=code
                )
                observed.add("NFR-P-03", float(row["first_progress"]))
                observed.add(code, float(row[code]))
                started_events += await _count_task_started(
                    client, args.base, token, str(row["task_id"])
                )
                print(f"  …{scenario} {index + 1}/{count}", flush=True)
            notes.append(
                f"{scenario} 场景：{count} 个任务的 `task.started` 共 {started_events} 条"
                f"（{'每个任务恰好一条，无重复执行' if started_events == count else '**异常**'}）。"
            )

    metrics.extend(observed.summarise(code) for code in ("NFR-P-03", "NFR-P-04", "NFR-P-05"))
    ordered = sorted(metrics, key=lambda item: item.code)
    task_seconds = 0.0
    for code in ("NFR-P-04", "NFR-P-05"):
        samples = observed.values.get(code, [])
        if samples:
            task_seconds = sum(samples) / len(samples)
    _print_report(ordered, notes=notes, task_seconds=task_seconds)

    missed = [item for item in ordered if item.ok is False]
    if missed:
        print(
            f"\n未达标 {len(missed)} 项：{'、'.join(item.code for item in missed)}。"
            "**目标未修改**——根因见文件头（NFR-P-08 与 NFR-P-06 在给定任务耗时下互斥）。"
        )
    return 1 if missed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="性能基线压测（需求 8.1 的 NFR-P-01..08）")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API 地址")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", "probe", "sql", "multi", "capacity"],
        help="probe=亚秒级探针；sql/multi=串行任务级；capacity=并发投递；all=全跑",
    )
    parser.add_argument("--samples", type=int, default=0, help="0 = 按场景取默认样本量")
    parser.add_argument("--account", default=DEMO_USERNAME, help="用哪个账号跑")
    args = parser.parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except httpx.ConnectError:
        print(f"连不上 {args.base}。先在另一个终端跑 `make run`。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
