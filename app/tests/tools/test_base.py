"""工具层的公共件：**错误码 → 错误类别**这张表（详设 9.4）。

这张表决定的是**下一步动作**（重试 / 定向修复 / 终止步骤），所以"一个码归错类"
不是文档问题，是行为问题。它以出错的方式出过一次：RAG 工具把
`UPSTREAM_UNAVAILABLE` 贴成 `VALIDATION`，而 `VALIDATION` 的默认动作是
"让模型重试解析"——对一次服务故障建议了一个错的动作。
"""

from __future__ import annotations

from app.core.errors import ErrorCode
from app.tools.base import _CODE_ERROR_CLASS, error_class_of


def test_upstream_outage_is_transient() -> None:
    """上游不可用 → `TRANSIENT`（重试一次），不是 `VALIDATION`（让模型重试解析）。

    这条钉的是 RAG 工具那处贴错标签的回归：服务挂了与"输出不合规"是两件事，
    后者重试解析能好，前者不能。
    """
    assert error_class_of(ErrorCode.UPSTREAM_UNAVAILABLE.value) == "TRANSIENT"
    assert error_class_of(ErrorCode.MODEL_RATE_LIMITED.value) == "TRANSIENT"


def test_an_unknown_code_falls_back_to_internal() -> None:
    """码表外的码落 `INTERNAL`——**不猜**。

    猜的代价是：一个我们其实没想清楚的失败被归进某个类别，而类别会被
    下游当成本次该怎么办的依据。落 `INTERNAL` 至少会失败并记录堆栈。
    """
    assert error_class_of("NO_SUCH_CODE") == "INTERNAL"


def test_task_timeout_is_deliberately_absent() -> None:
    """`TASK_TIMEOUT` **故意不在这张表里**——这条把理由钉成断言。

    19.1 给它的是**任务总预算**语义（504 / 不重试 /「返回已有证据和限制」），
    而工具层遇到的超时是**单次调用**的时间预算（`SQL_TOOL__TIMEOUT_SECONDS`
    默认 10 秒）——两者差一个量级，用 504 会让客户端误判"整个任务终结了"。
    单次超时按 `UPSTREAM_UNAVAILABLE`（503 / 可重试）报，而"是超时还是连接被拒"
    由 `error_class` 表达（`SqlExecutionError` 自己带 `TIMEOUT`）。

    ⚠️ 这条曾经是一处**死映射**（表里有、全仓没有生产者），危害是读代码的人
    会以为有个产生点、去找它、找不到、然后怀疑是不是自己漏了什么。
    """
    assert "TASK_TIMEOUT" not in _CODE_ERROR_CLASS
    assert error_class_of(ErrorCode.TASK_TIMEOUT.value) == "INTERNAL"


def test_every_mapped_code_exists_in_the_error_table() -> None:
    """表里的键必须都是 19.1 那张封闭表里的码——防拼错与自造码。

    自造码的下场是它永远不会被命中（真正的码走兜底成了 `INTERNAL`），
    而这一行映射看起来像是有保障的。
    """
    known = {item.value for item in ErrorCode}
    assert set(_CODE_ERROR_CLASS) <= known
