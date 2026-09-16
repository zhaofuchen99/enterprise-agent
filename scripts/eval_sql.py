"""SQL 金标评测（开发流程 6.6 的门禁：金标 SQL 正确率达到阶段基线）。

用法：
    uv run python scripts/eval_sql.py                 # 跑全部
    uv run python scripts/eval_sql.py --only sql-03   # 只跑一题（调 prompt 时用）
    uv run python scripts/eval_sql.py --baseline 0.8  # 低于基线则退出码非零

## 判定方式：结果集等价，不比对 SQL 字符串

详设 22.2 明文要求。同一条业务问题可以有多种正确写法，比对文本只会奖励
「抄得像」而不是「算得对」。

比对时把每个单元格规范成可比较的形式：数值按**两位小数**比（`Decimal`
不是 `float`——浮点误差会让两个相等的金额被判成不等，而这种失败最难查）；
其余按字符串比。`ordered: true` 的用例按顺序比（TOP-N 有业务含义），
其余按**多重集**比——SQL 不保证 GROUP BY 的输出顺序，按顺序比会把一个
正确的查询判成错的。

## 为什么金标 SQL 也要真跑一遍

金标写在 YAML 里，它自己也可能写错（日期边界、JOIN 写漏、口径混用）。
如果只拿模型结果去比一个**静态的期望值**，评测就退化成「比对两份人工产物」。
把金标也执行一次，两边都出自同一份冻结数据，比对的才是结果本身。

## 分类统计

除了总体正确率，脚本按 `proves` 字段打印每一题的验证点——因为
「正确率 8/10」本身不告诉你错的是哪一类能力，而「同比口径全错」
才是可行动的结论。这对应详设 16.11.3 第 4 条「每类缺陷标注对应注入项」。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.domain.user import PermissionScope, UserRole
from app.infrastructure.logging import SERVICE_CLI, setup_logging
from app.infrastructure.model_gateway import build_model_gateway
from app.tools.sql.schemas import SqlQueryArgs
from app.tools.sql.tool import build_cli_context, build_sql_query_tool

#: 默认金标集路径。与 `SQL_TOOL__CATALOG_PATH` 同属配置类文件。
GOLDEN_PATH = "configs/eval_sql_golden.yaml"

#: 数值判定的小数位。两位是金额与百分比在演示里的自然精度；
#: 再多一位就会把「口径正确但舍入方式不同」误判成错。
_QUANTUM = Decimal("0.01")

#: 阶段基线。**先跑出真实数字，再定基线**——这个值的来历是
#: 2026-09-16 的首次全量运行：内容正确率 10/10、列形状一致 9/10。
#:
#: 定在 80% 而不是 100%：题量只有 10，模型单次生成的波动就是 10 个点，
#: 基线贴着实测值会让门禁变成随机红。它拦的是**退步**，不是波动——
#: 调完 prompt 掉到 7/10 才是信号。
_DEFAULT_BASELINE = 0.8


@dataclass(frozen=True)
class GoldenCase:
    """一条金标用例。

    Attributes:
        compare: 结果集比对方式。

            - `exact`（默认）：逐格相等。**这是默认值**，因为「答案的形状也是答案的
              一部分」——下游 Analysis 与 Reviewer 要按列取数，多一列不会报错，
              但会让「哪个数是结论」变成由模型临场决定。
            - `subset`：金标每一行的取值序列必须是模型某一行的**子序列**。
              用于「模型多给中间量不算错」这一类问题：问「同比下降多少」时，
              模型返回「去年同期 / 本期 / 差值 / 百分比」四列，比只返回一个
              百分比**信息更全**，把它判成错会让评测分数衡量的是「形状是否
              与金标一致」而不是「算得对不对」。

              它并不宽松到失去意义：那个百分比必须真的出现在结果里，
              少算或算错的查询照样不通过。
    """

    id: str
    question: str
    golden_sql: str
    ordered: bool
    proves: str
    compare: str = "exact"


@dataclass
class CaseOutcome:
    case: GoldenCase
    passed: bool
    #: 结果形状也与金标完全一致（更严的那一档）。与 `passed` 一起报，
    #: 见 `compare_both` 的说明。
    shape_exact: bool = False
    reason: str = ""
    model_rows: list[list[Any]] | None = None
    golden_rows: list[list[Any]] | None = None
    model_sql: str = ""


def load_cases(path: str) -> list[GoldenCase]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise SystemExit(f"金标集 {path} 结构不对：顶层需要 version 与 cases 两个键")
    return [
        GoldenCase(
            id=str(item["id"]),
            question=str(item["question"]),
            golden_sql=str(item["golden_sql"]).strip(),
            ordered=bool(item.get("ordered", False)),
            proves=str(item.get("proves", "")),
            compare=str(item.get("compare", "exact")),
        )
        for item in raw["cases"]
    ]


# ------------------------------------------------------------------ 结果比对
def canonical(rows: Sequence[Sequence[Any]], *, ordered: bool) -> list[tuple[str, ...]]:
    """把结果集规范成可比较的形式。

    数值统一走 `Decimal` 并量化到两位小数：`float` 的二进制表示会让
    `111967031.73` 与 `111967031.7299999` 判成不等，而这类失败会以
    「模型算错了」的面目出现，排查方向完全跑偏。
    """
    normalized = [tuple(_cell(value) for value in row) for row in rows]
    return list(normalized) if ordered else sorted(normalized)


def _cell(value: Any) -> str:
    """单元格 → 可比较字符串。

    先试数值：能解析成 `Decimal` 的就按两位小数比。
    解析不了（日期、区域名）就原样比——但要把 `None` 显式写成 `NULL`，
    否则它会与字符串 `"None"` 混为一谈。
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return str(Decimal(str(value)).quantize(_QUANTUM))
    except (InvalidOperation, ValueError):
        return str(value)


def rows_equal(
    model_rows: Sequence[Sequence[Any]],
    golden_rows: Sequence[Sequence[Any]],
    *,
    ordered: bool,
    compare: str = "exact",
) -> bool:
    """两批结果是否等价。

    `compare="exact"`（默认）：逐格相等。
    `compare="subset"`：金标每一行的取值序列，必须是模型某一行的**子序列**
    （保持列序）。用途见 `GoldenCase.compare` 的说明。
    """
    produced = canonical(model_rows, ordered=ordered)
    expected = canonical(golden_rows, ordered=ordered)
    if compare == "exact":
        return produced == expected

    # 每个金标行占用一个**未被用掉**的模型行：不这样约束的话，
    # 一个模型行可以同时满足两条金标行，「两行都对了」就成了假象。
    remaining = list(produced)
    for row in expected:
        for index, candidate in enumerate(remaining):
            if _is_subsequence(row, candidate):
                remaining.pop(index)
                break
        else:
            return False
    return True


def compare_both(
    model_rows: Sequence[Sequence[Any]],
    golden_rows: Sequence[Sequence[Any]],
    *,
    ordered: bool,
) -> tuple[bool, bool]:
    """一次算出**两种判据**的结果：`(内容正确, 形状也一致)`。

    **为什么两个都要算**：逐题的 `compare` 由人来标，那就存在「把不好过的题
    标成 `subset` 来抬高分数」这条捷径。同时把严格判据的数字印出来之后，
    这条捷径就失效了——放宽了多少，在报告里一眼看得见。

    没有正例的严格判据会被无脑放宽，而只有严格判据会把「多给了一列主键」
    当成算错。两个数字一起看，才是对「模型算得对不对」的完整描述。
    """
    content_ok = rows_equal(model_rows, golden_rows, ordered=ordered, compare="subset")
    shape_ok = rows_equal(model_rows, golden_rows, ordered=ordered, compare="exact")
    return content_ok, shape_ok


def _is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    """`needle` 是否是 `haystack` 的子序列（顺序敏感、不要求连续）。"""
    cursor = iter(haystack)
    return all(any(item == wanted for item in cursor) for wanted in needle)


# ------------------------------------------------------------------ 单题评测
async def evaluate_case(tool: Any, ctx: Any, case: GoldenCase) -> CaseOutcome:
    """一题 = 两次真跑：模型路径一次、金标 SQL 一次。"""
    outcome = CaseOutcome(case=case, passed=False)

    produced = await tool.execute(SqlQueryArgs(question=case.question), ctx)
    if produced.status == "FAILED":
        error = produced.error
        outcome.reason = (
            f"模型路径失败：{error.code if error else '未知'} - {error.message if error else ''}"
        )
        # 失败时**更要**把模型写的那条 SQL 带出来：只说「校验没通过」，
        # 排查的人还得重跑一次才能看到它到底写了什么。
        attempts = (produced.payload or {}).get("attempts") or []
        if attempts:
            last = attempts[-1]
            outcome.model_sql = str(last.get("sql") or "")
            outcome.reason += f"｜{last.get('error_summary') or ''}"
        return outcome

    payload = produced.payload or {}
    outcome.model_sql = str(payload.get("normalized_sql", ""))
    outcome.model_rows = [list(row) for row in payload.get("rows") or []]

    golden = await tool.run_sql(case.golden_sql, ctx)
    if golden.status == "FAILED":
        # 金标自己跑不动是**评测集的问题**，不是模型的问题——必须与
        # 「模型算错」区分开，否则会去改 prompt 而真正要改的是这个 YAML。
        error = golden.error
        outcome.reason = (
            f"金标 SQL 无法执行（评测集自身的问题）：{error.safe_detail if error else '未知原因'}"
        )
        return outcome
    outcome.golden_rows = [list(row) for row in (golden.payload or {}).get("rows") or []]

    content_ok, shape_ok = compare_both(
        outcome.model_rows, outcome.golden_rows, ordered=case.ordered
    )
    outcome.shape_exact = shape_ok
    outcome.passed = content_ok if case.compare == "subset" else shape_ok
    if not outcome.passed:
        outcome.reason = "结果集与金标不等价"
    return outcome


# ------------------------------------------------------------------ 输出
def render(outcome: CaseOutcome) -> str:
    mark = "OK  " if outcome.passed else "FAIL"
    lines = [f"[{mark}] {outcome.case.id}  {outcome.case.question}"]
    if outcome.passed:
        lines.append(f"       证明：{outcome.case.proves}")
        return "\n".join(lines)
    lines.append(f"       证明：{outcome.case.proves}")
    lines.append(f"       原因：{outcome.reason}")
    if outcome.model_sql:
        lines.append(f"       模型 SQL：{outcome.model_sql[:200]}")
    if outcome.model_rows is not None:
        lines.append(f"       模型结果：{outcome.model_rows[:5]}")
    if outcome.golden_rows is not None:
        lines.append(f"       金标结果：{outcome.golden_rows[:5]}")
    return "\n".join(lines)


async def run(settings: Settings, cases: list[GoldenCase]) -> list[CaseOutcome]:
    gateway = build_model_gateway(settings)
    tool = build_sql_query_tool(settings, gateway)
    # 评测按**全量**口径跑：金标 SQL 不带数据范围，用受限范围会给两边
    # 各加一层过滤，把「口径写错了」与「权限过滤生效了」混在一起。
    ctx = build_cli_context(region_ids=(), timeout_seconds=settings.task_timeout_seconds)
    ctx = ctx.model_copy(
        update={"permission_scope": PermissionScope(role=UserRole.ADMIN, region_ids=())}
    )
    try:
        outcomes: list[CaseOutcome] = []
        for case in cases:
            try:
                outcomes.append(await evaluate_case(tool, ctx, case))
            except AgentError as exc:
                outcomes.append(
                    CaseOutcome(case=case, passed=False, reason=f"调用异常：{exc.code.value}")
                )
        return outcomes
    finally:
        await tool.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQL 金标评测")
    parser.add_argument("--golden", default=GOLDEN_PATH, help=f"金标集路径（默认 {GOLDEN_PATH}）")
    parser.add_argument("--only", default="", help="只跑指定 id，逗号分隔")
    parser.add_argument(
        "--baseline",
        type=float,
        default=_DEFAULT_BASELINE,
        help=f"正确率低于它则退出码非零（默认 {_DEFAULT_BASELINE:.0%}，见常量处的说明）",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(SERVICE_CLI, settings.log_level)

    cases = load_cases(args.golden)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case.id in wanted]
        if not cases:
            print(f"没有匹配的用例：{args.only}", file=sys.stderr)
            return 1

    print(f"金标集：{args.golden}（{len(cases)} 题）")
    print(f"模型：{settings.model_name}｜Schema 目录版本由每次调用记录\n")

    outcomes = asyncio.run(run(settings, cases))
    for outcome in outcomes:
        print(render(outcome))

    passed = sum(1 for outcome in outcomes if outcome.passed)
    shape = sum(1 for outcome in outcomes if outcome.shape_exact)
    rate = passed / len(outcomes) if outcomes else 0.0
    print(f"\n金标 SQL 正确率：{passed}/{len(outcomes)} = {rate:.0%}")
    # 两个数一起报：差值就是「模型多给了中间量/主键列」的那部分。
    # 只报前者会让人以为模型连列形状都写对了；只报后者会把正确回答判成错。
    print(f"  其中列形状与金标完全一致：{shape}/{len(outcomes)}")
    if 0 < rate < 1:
        failed = ", ".join(outcome.case.id for outcome in outcomes if not outcome.passed)
        print(f"未通过：{failed}")
    if args.baseline and rate < args.baseline:
        print(f"低于阶段基线 {args.baseline:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
