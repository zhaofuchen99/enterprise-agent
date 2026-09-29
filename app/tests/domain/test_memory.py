"""多轮上下文的两个纯函数（`app/domain/memory.py`）。

这一层最便宜，所以把"归纳规则"的边界都钉在这里——上面接的图与仓储
只要保证"消息与口径按顺序传进来"就够了。

⚠️ 四条用例里两条是**顺序断言而不是长度断言**（`test_..._oldest_first`、
`test_..._keeps_the_entities_and_the_latest_turn`）。长度对了而顺序反了
（丢的是最近一轮、留的是最旧一轮）会让代词解析落到**错误的落点**上，
而上下文看起来"有内容"。
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.conversation import Message, MessageRole
from app.domain.memory import (
    TurnResolution,
    build_conversation_context,
    group_turns,
    resolution_from_payload,
    resolution_to_payload,
    squash,
)

_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _message(
    index: int,
    role: MessageRole,
    content: str,
    *,
    task_id: str | None = None,
) -> Message:
    return Message(
        id=f"msg_{index:022d}",
        conversation_id="cnv_1",
        task_id=task_id or (f"tsk_{index}" if role is not MessageRole.SYSTEM_NOTICE else None),
        role=role,
        content=content,
        created_at=_NOW.replace(second=index % 60),
    )


def _turns(count: int) -> list[Message]:
    messages: list[Message] = []
    for index in range(count):
        messages.append(
            _message(2 * index, MessageRole.USER, f"第{index}轮问题", task_id=f"tsk_{index}")
        )
        messages.append(
            _message(
                2 * index + 1, MessageRole.ASSISTANT, f"第{index}轮答复", task_id=f"tsk_{index}"
            )
        )
    return messages


# ------------------------------------------------------------------ 窗口
def test_no_messages_means_an_empty_context_not_none() -> None:
    """空输入 → **各字段为空，而不是 `None`**。

    与 `load_conversation_context` 返回 `None` 是两件事：那个 `None` 表示
    "这一轮不必带上下文"，而这里返回的对象是"历史确实是空的"。
    `render_context` 对两者的处置相同（都渲染成"无历史"），
    但把它统一成 `None` 会让"消息读回来了、只是 0 条"这条路径消失。
    """
    context = build_conversation_context(group_turns([], {}, max_turns=10))

    assert context.recent_messages == ()
    assert context.confirmed_entities == {}
    assert context.previous_task_summary is None
    assert context.is_empty is True


def test_window_keeps_the_last_turns_and_drops_the_oldest_first() -> None:
    """11 轮 + `max_turns=10` → 丢掉**第 0 轮**，留下第 1..10 轮。

    断言的是**留下的是谁**，不是条数：丢错一头（丢最近、留最旧）时
    条数一样，而代词会去指一个十轮之前的话题。
    """
    messages = _turns(11)
    context = build_conversation_context(group_turns(messages, {}, max_turns=10))

    questions = [item.content for item in context.recent_messages if item.role is MessageRole.USER]
    assert questions == [f"第{index}轮问题" for index in range(1, 11)]


def test_a_turn_without_an_answer_does_not_swallow_the_next_question() -> None:
    """上一轮没答（失败 / 还在跑）时，问与答仍各归各的轮。

    第一版按"USER 后面跟 ASSISTANT"配对，遇到没有答复那一轮会把**下一条
    用户消息**当成它的答复吃掉——于是"我问了什么、它没答上来"这件事
    在上下文里消失，而下一轮的代词指到了一个错的地方。
    """
    messages = [
        _message(0, MessageRole.USER, "第一问", task_id="tsk_a"),
        _message(1, MessageRole.USER, "第二问", task_id="tsk_b"),
        _message(2, MessageRole.ASSISTANT, "第二问的答复", task_id="tsk_b"),
    ]

    turns = group_turns(messages, {}, max_turns=10)

    assert [turn.question for turn in turns] == ["第一问", "第二问"]
    assert turns[0].answer is None
    assert turns[1].answer == "第二问的答复"


# ------------------------------------------------------------------ 归纳
def test_entities_come_from_the_latest_turn_that_has_them() -> None:
    """口径取**最近一轮有口径的**，不跨轮合并。

    最后一轮是澄清（没有口径）时，再上一轮的口径仍然是解析代词的正确落点；
    而把两轮并起来会造出一份**从来没被问过的口径**（第 1 轮华东 + 第 3 轮华南）。
    """
    messages = _turns(3)
    resolutions = {
        "tsk_0": TurnResolution(entities={"region": "华东", "period": "2025-Q3"}),
        "tsk_1": TurnResolution(entities={"region": "华南", "period": "2025-Q4"}),
        "tsk_2": TurnResolution(unresolved_fields=("区域",)),
    }

    context = build_conversation_context(group_turns(messages, resolutions, max_turns=10))

    assert context.confirmed_entities == {"region": "华南", "period": "2025-Q4"}
    assert context.unresolved_fields == ("区域",)


def test_a_long_answer_is_excerpted() -> None:
    """助手回答进上下文时被截断——它是**会被回灌进下一次 prompt 的模型生成物**。"""
    messages = [
        _message(0, MessageRole.USER, "问", task_id="tsk_a"),
        _message(1, MessageRole.ASSISTANT, "答" * 500, task_id="tsk_a"),
    ]

    context = build_conversation_context(group_turns(messages, {}, max_turns=10))

    answer = context.recent_messages[1].content
    assert len(answer) < 500
    assert answer.endswith("…")


def test_squash_flattens_a_question_that_tries_to_break_the_table() -> None:
    """问题里的换行与引号会被压掉。

    这一段是拼进一张按行读的口径表里的，而问题是用户可控的：
    换行能凭空造出一行（看起来像表里的另一个字段）。
    """
    assert squash('华东\n口径：全公司 "特别"') == "华东 口径：全公司 特别"


# ------------------------------------------------------------------ 往返
def test_resolution_round_trips_through_the_payload() -> None:
    resolution = TurnResolution(
        entities={"metric": "净销售额", "period": "2025-Q3"}, unresolved_fields=("年度",)
    )

    payload = resolution_to_payload(resolution)

    assert resolution_from_payload(payload) == resolution


def test_unknown_payload_keys_are_dropped_on_read() -> None:
    """payload 是历史数据：**多出来的键不能进口径**。

    渲染侧按 `ENTITY_KEYS` 映射中文标签，一个不认识的口径键会以它自己的
    英文名渲染出来（`province 华南`）——模型读得懂，但读的人会以为漏了映射。
    """
    restored = resolution_from_payload(
        {"resolved_entities": {"region": "华东", "province": "江苏", "empty": ""}}
    )

    assert restored.entities == {"region": "华东"}
