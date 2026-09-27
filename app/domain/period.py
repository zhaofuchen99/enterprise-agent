"""统计期间记号（`2025` / `2025-H1` / `2025-Q3` / `2025-08`）与它对应的区间。

## 为什么这需要单独一个地方

「这一版说的是哪一段时间」在三处出现，而它们的形态各不相同：

| 出现处 | 形态 |
|---|---|
| 语料清单的 `report.period` | **记号**（`2025-Q3`） |
| 分块 payload 的 `stat_period` | **记号**（同一份，逐字带下来） |
| SQL 证据的 `event_time` | **区间**（`>= '2025-07-01' AND < '2025-10-01'`） |

判「两边的数是不是同一期的数」必须把它们换成同一种表示。**记号与区间的
换算规则只写在这里**——在第二个地方再写一遍，漂移的症状是"某类冲突再也
检不出来"，而没有任何地方会报错（`verify._time_of` 现在仍自己解一遍
`logical_key`，见那边的说明）。

## 记号是**闭集**，区间是半开

四种记号、四种跨度，没有第五种：

    YYYY      全年      [Jan 1, 次年 Jan 1)
    YYYY-HN   半年      [Jan 1, Jul 1) 或 [Jul 1, 次年 Jan 1)
    YYYY-QN   季度      [Q 首月 1 日, +3 个月)
    YYYY-MM   月度      [当月 1 日, +1 个月)

**"区间归不到这四种"与"记号解不出来"是同一个结论：期间未知。**
未知不等于"不限期间"——判据里两者处置相反（见 `conflict._comparable`）。
"""

from __future__ import annotations

import re
from datetime import date

#: 半年 / 季度 / 月度三条正则。**顺序即优先级**：`2025-01` 不能落到"年份"那一条上
#: （`^\d{4}$` 要求整串只有四位，所以 `2025-01` 落不进去）。
#:
#: 大小写不敏感是为了 `2025-q3` 这类写法能归一，**不是**为了容忍
#: `2025Q3`（没有连字符）——连字符是记号词汇表的一部分，
#: 而放宽它会让 `202508` 这种"到底是年月还是别的"变得有歧义。
_TOKENS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(\d{4})-H([12])$", re.IGNORECASE),
    re.compile(r"^(\d{4})-Q([1-4])$", re.IGNORECASE),
    re.compile(r"^(\d{4})-(\d{2})$"),
)
_YEAR = re.compile(r"^(\d{4})$")


def _shift(day: date, months: int) -> date:
    """月份加减，**只按整月**（本项目所有跨度都是整月）。"""
    total = day.year * 12 + (day.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def period_bounds(token: str) -> tuple[date, date] | None:
    """期间记号 → 半开区间 `[start, end)`；解不出来返回 `None`。

    `None` 表示**这不是一个期间记号**，不是"不限期间"。
    """
    text = token.strip()
    if (matched := _YEAR.match(text)) is not None:
        start = date(int(matched.group(1)), 1, 1)
        return start, _shift(start, 12)
    for pattern in _TOKENS:
        matched = pattern.match(text)
        if matched is None:
            continue
        year, value = int(matched.group(1)), int(matched.group(2))
        if "-H" in text.upper():
            start = date(year, 1 if value == 1 else 7, 1)
            return start, _shift(start, 6)
        if "-Q" in text.upper():
            start = date(year, (value - 1) * 3 + 1, 1)
            return start, _shift(start, 3)
        # **月份要自己校验范围**：`2025-13` 会被 `\d{2}` 放进来，
        # 而 `date(2025, 13, 1)` 抛的是 `ValueError`。
        # 一个"看着像期间"的串在这里炸掉，比返回 `None` 糟得多：
        # 调用方在冲突检测的循环里，没有任何地方预备接住它。
        if not 1 <= value <= 12:
            return None
        start = date(year, value, 1)
        return start, _shift(start, 1)
    return None


def canonical_period(start: date, end: date) -> str | None:
    """半开区间 `[start, end)` → 期间记号；**归不到四种跨度就返回 `None`**。

    归不上是常态而不是异常：`order_date >= '2025-01-01' AND < '2025-10-01'`
    （九个月）、同比查询里 `min/max` 包出来的 15 个月、以及"压根没有时间条件"
    都会落到这里。**它们都是"期间未知"**——按未知处置，不要硬凑一个记号：
    凑出来的记号会让一条跨期的比对照常发生，而它看起来完全正常。
    """
    if start.day != 1:
        return None
    if _shift(start, 1) == end:
        return f"{start.year}-{start.month:02d}"
    if start.month in (1, 4, 7, 10) and _shift(start, 3) == end:
        return f"{start.year}-Q{(start.month - 1) // 3 + 1}"
    if start.month in (1, 7) and _shift(start, 6) == end:
        return f"{start.year}-H{1 if start.month == 1 else 2}"
    if start.month == 1 and _shift(start, 12) == end:
        return f"{start.year}"
    return None


def canonical_token(token: str) -> str | None:
    """记号 → **规范化**后的记号；解不出来返回 `None`。

    走一趟区间再回来，是为了让"同一个期间的不同写法"（`2025-q3` / `2025-Q3`）
    落到同一个字符串上。**不这么做的话，两边写法不一致时相等判断恒假**，
    症状是那个期间的对照永远不比——而它看起来只是"这次没检出冲突"。
    """
    bounds = period_bounds(token)
    return None if bounds is None else canonical_period(*bounds)


__all__ = ["canonical_period", "canonical_token", "period_bounds"]
