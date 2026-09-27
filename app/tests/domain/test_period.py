"""统计期间记号与区间的换算（`app/domain/period.py`）。

这一层的全部价值在于**"两边的数是不是同一期的数"有一个共同的答案**。
所以用例大多在测"归不上就说归不上"——硬凑一个记号比返回 `None` 危险得多：
凑出来的记号会让一条跨期的比对照常发生，而它看起来完全正常。
"""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.period import canonical_period, canonical_token, period_bounds


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("2025", (date(2025, 1, 1), date(2026, 1, 1))),
        ("2025-H1", (date(2025, 1, 1), date(2025, 7, 1))),
        ("2025-H2", (date(2025, 7, 1), date(2026, 1, 1))),
        ("2025-Q1", (date(2025, 1, 1), date(2025, 4, 1))),
        ("2025-Q3", (date(2025, 7, 1), date(2025, 10, 1))),
        ("2025-Q4", (date(2025, 10, 1), date(2026, 1, 1))),
        ("2025-08", (date(2025, 8, 1), date(2025, 9, 1))),
        ("2025-12", (date(2025, 12, 1), date(2026, 1, 1))),
    ],
)
def test_period_bounds_covers_the_four_shapes(token: str, expected: tuple[date, date]) -> None:
    """四种记号各自的跨度。**跨年是常态**：Q4 与 12 月都落在次年 1 月 1 日结束。"""
    assert period_bounds(token) == expected


@pytest.mark.parametrize("token", ["2025-Q5", "2025-H3", "2025-13", "2025年Q3", "", "总", "2025-8"])
def test_unknown_tokens_are_none_rather_than_guesses(token: str) -> None:
    """解不出来就是 `None`——**不猜**。

    `2025-8`（月份没补零）与 `2025年Q3`（中文写法）都在这里挡掉：
    它们看着像期间，而按它们算出来的区间会在"相等判断"里**恒假**，
    症状是那个期间的对照永远不比——而它看起来只是"这次没检出冲突"。
    """
    assert period_bounds(token) is None


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (date(2025, 1, 1), date(2026, 1, 1), "2025"),
        (date(2025, 7, 1), date(2026, 1, 1), "2025-H2"),
        (date(2025, 7, 1), date(2025, 10, 1), "2025-Q3"),
        (date(2025, 8, 1), date(2025, 9, 1), "2025-08"),
        (date(2025, 12, 1), date(2026, 1, 1), "2025-12"),
    ],
)
def test_canonical_period_names_regular_spans(start: date, end: date, expected: str) -> None:
    assert canonical_period(start, end) == expected


def test_canonical_period_rejects_spans_that_are_not_a_period() -> None:
    """**归不上是常态，不是异常**，而且它必须返回 `None`。

    这三条都是真实会出现的：

    - 九个月（题目给了个跨期的区间）；
    - 同比查询那 15 个月——`_extract_time_range` 取的是 WHERE 里全部时间谓词的
      min/max 包络，两个 `CASE WHEN` 各一个季度，包起来正好是它；
    - 不是从 1 号开始的区间。

    给它们硬凑一个记号，等于说"这段时间就是某一期"，而下一句就是拿它去比。
    """
    assert canonical_period(date(2025, 1, 1), date(2025, 10, 1)) is None  # 九个月
    assert canonical_period(date(2024, 7, 1), date(2025, 10, 1)) is None  # 同比的包络
    assert canonical_period(date(2025, 8, 15), date(2025, 9, 15)) is None  # 不是整月


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("2025-Q3", "2025-Q3"),
        ("2025", "2025"),
        ("2025-08", "2025-08"),
        # 走一趟区间再回来：**同一个期间的另一种写法要落到同一个字符串上**，
        # 否则两边写法不一致时相等判断恒假，而症状只是"这次没检出冲突"
        ("2025-q3", "2025-Q3"),
        ("2025-h1", "2025-H1"),
        (" 2025-Q3 ", "2025-Q3"),
    ],
)
def test_canonical_token_normalises_equivalent_spellings(token: str, expected: str) -> None:
    assert canonical_token(token) == expected


def test_canonical_token_of_a_non_period_is_none() -> None:
    assert canonical_token("2025年第三季度") is None
