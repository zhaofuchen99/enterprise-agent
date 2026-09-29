"""会话内短期记忆（详细设计 15.1 的 `ConversationContext`，FR-CHAT-003）。

## 它是什么，不是什么

**它是给模型的裁剪视图，不是会话的存储。** 会话里存了什么由
`app/domain/conversation.py` 的 `Message` 与 `Conversation` 说，
这里说的是「这一轮要拿多少历史、怎么归纳、超限丢谁」。

两者分开的直接理由是**来源不同**：对话文本来自 `agent_message`，
而「上一轮把问题解析成了什么口径」来自 `agent_task.result_json`。
合成一个对象会让"消息表空了"与"任务没解析出口径"变成同一个症状。

## 15.1 的五字段，本实现逐条落地

| 字段 | 本实现的来源 | 说明 |
|---|---|---|
| `recent_messages` | `agent_message` 最近 N 轮 | 助手回答**截断**（见 `_ANSWER_EXCERPT_CHARS`） |
| `confirmed_entities` | **最近一轮**有口径的任务 | **不跨轮合并**，理由见该字段的 docstring |
| `metric_definitions` | v1 恒空 | 需指标目录，跨层注入待论证（已登记） |
| `unresolved_fields` | 最近一轮的 `missing_fields` | |
| `previous_task_summary` | 最近一次成功任务的答案摘要 | 它是**不随转录被裁掉**的那一段 |

## 「轮」的边界

一轮 = 一条用户消息 + 它对应的助手消息，实现上取 `2 × max_turns` 条消息
（FR-CHAT-003 业务规则：默认保留**最近 10 轮**）。不真去配对的好处是
**上一轮没答完（失败 / 还在跑）时不会把它整轮丢掉**——那正是最需要看见的一轮。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.conversation import Message, MessageRole

#: 助手回答进上下文时截断到多少字。
#:
#: **不截断是两件事的风险**：一是 token——10 轮答案各带一张 Markdown 表格，
#: 上下文会比问题本身大一个数量级；二是**助手回答是模型生成物，会被回灌进
#: 下一次 prompt**，那是标准的注入面。截断不能消除注入，但它把「塞一整段
#: 指令」压成「塞一句」——而真正消除它的是「口径走结构化字段」这条设计
#: （见 `render_context`：数字与口径不来自这段文本）。
_ANSWER_EXCERPT_CHARS = 200

#: 口径键的**闭集**。渲染时映射成中文标签（`app/agent/prompts/supervisor.py`），
#: 而断言（`expect_resolved_contain`）用这里的英文键——
#: 用中文当键的话，改一次渲染措辞就会让所有历史用例失去意义。
ENTITY_KEYS: tuple[str, ...] = (
    "metric",
    "region",
    "channel",
    "product_line",
    "period",
    "comparison",
)


class TurnResolution(BaseModel):
    """一个任务解析出的口径（落 `agent_task.result_json`，无需迁移）。

    写成模型而不是直接塞 `dict`：它要**走出**进程（进 payload 再读回来），
    形状漂移时应当在校验处报错，而不是在渲染处变成一句奇怪的中文。
    """

    model_config = ConfigDict(frozen=True)

    entities: dict[str, str] = Field(default_factory=dict)
    unresolved_fields: tuple[str, ...] = ()


class ConversationTurn(BaseModel):
    """一轮对话：问了什么、答了什么、那一轮把问题解析成了什么口径。"""

    model_config = ConfigDict(frozen=True)

    task_id: str | None = None
    question: str
    answer: str | None = None
    entities: dict[str, str] = Field(default_factory=dict)
    unresolved_fields: tuple[str, ...] = ()


class ContextMessage(BaseModel):
    """进上下文的一条消息。**与 `Message` 不是同一个东西**：
    后者是存下来的原文（审计用），这一条是**裁过、压成单行**的视图。
    """

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str


class ConversationContext(BaseModel):
    """详细设计 15.1 的 `ConversationContext`。"""

    model_config = ConfigDict(frozen=True)

    recent_messages: tuple[ContextMessage, ...] = ()
    #: **语义是「上一轮系统解析出的口径」，不是「用户已确认」。**
    #:
    #: 详设 15.1 的原话是「模型推测不得写成"用户已确认"」，而这个名字
    #: （`confirmed_entities`）读起来正好像是前者。所以这一条要写死在这里：
    #: 填进来的值来自**上一轮任务真的解析出来的 `IntentResult`**（确定性读出，
    #: 不是模型此刻的猜测），而渲染进 prompt 时**禁止使用「已确认」措辞**
    #: ——`test_memory_context.py` 有一条断言钉着这件事（用户读到的措辞
    #: 与字段名不一致时，会按字段名理解为"用户确认过"，而那是不成立的）。
    #:
    #: **只取最近一轮，不跨轮合并**：把第 1 轮的「华东」与第 3 轮的「华南」
    #: 并起来会造出一份**从来没有被问过的口径**，而模型会拿它去解析代词
    #: （第 4 轮问「那 Q2 呢」，它可能挑中那个已经不成立的区域）。
    #: 会话里真正"累积"的是消息本身，这里只是给代词一个最近的落点。
    confirmed_entities: dict[str, str] = Field(default_factory=dict)
    #: v1 恒空：指标口径目录活在 `app/tools/sql`，为这一个字段把 tools 层
    #: 拖进 API 进程的装配链不合算（已登记）。空列表的语义是
    #: **"本轮没有注入口径定义"**，不是"没有口径"。
    metric_definitions: tuple[str, ...] = ()
    unresolved_fields: tuple[str, ...] = ()
    previous_task_summary: str | None = None

    @property
    def is_empty(self) -> bool:
        """有没有值得告诉模型的历史。

        渲染侧靠它决定「无历史」那句话——**不能靠 `recent_messages` 是否为空**
        单独判：转录被预算裁光、而实体的恒留段还在时，历史仍然是有的。
        """
        return not (
            self.recent_messages
            or self.confirmed_entities
            or self.previous_task_summary
            or self.unresolved_fields
        )


def squash(text: str) -> str:
    """压成单行、去掉引号。

    **不是美化**：这一段会被拼进一张按行读的口径表里，而问题是用户可控的。
    换行能凭空造出一行（看起来像表里的另一个字段），引号能让下一行的
    分隔符看起来属于这句话。压平之后这两个都做不到。
    """
    flat = " ".join(text.split())
    for char in ('"', "'", "「", "」", "『", "』", "`"):
        flat = flat.replace(char, "")
    return flat


def _excerpt(text: str | None) -> str | None:
    if not text:
        return None
    flat = squash(text)
    if len(flat) <= _ANSWER_EXCERPT_CHARS:
        return flat
    return flat[:_ANSWER_EXCERPT_CHARS] + "…"


def group_turns(
    messages: list[Message],
    resolutions: dict[str, TurnResolution],
    *,
    max_turns: int,
) -> tuple[ConversationTurn, ...]:
    """消息 + 各任务解析出的口径 → 按时间正序的轮次。

    **窗口在这里生效**（取最近 `2 × max_turns` 条），不在读取侧：
    读取侧只认 `limit` 这一个数字，把"10 轮 = 20 条"的换算放在一处，
    免得两处各写一遍而其中一处写成 10 条（症状是上下文只剩 5 轮，
    且没有任何地方报错）。
    """
    if max_turns <= 0 or not messages:
        return ()

    window = messages[-(2 * max_turns) :]

    turns: list[ConversationTurn] = []
    pending: Message | None = None
    for message in window:
        if message.role is MessageRole.USER:
            # 上一条用户消息还没等到答复就来了下一条（上一轮失败 / 被取消）：
            # 把它当成一轮**没有答复**的对话收下，而不是覆盖掉。
            # 覆盖的话，"我问了什么、它没答上来"这件事在上下文里彻底消失。
            if pending is not None:
                turns.append(_close(pending, None, resolutions))
            pending = message
            continue
        if message.role is MessageRole.ASSISTANT and pending is not None:
            turns.append(_close(pending, message, resolutions))
            pending = None

    if pending is not None:
        turns.append(_close(pending, None, resolutions))
    return tuple(turns)


def _close(
    question: Message, answer: Message | None, resolutions: dict[str, TurnResolution]
) -> ConversationTurn:
    resolution = resolutions.get(question.task_id or "") if question.task_id else None
    return ConversationTurn(
        task_id=question.task_id,
        question=squash(question.content),
        answer=_excerpt(answer.content if answer else None),
        entities=dict(resolution.entities) if resolution else {},
        unresolved_fields=tuple(resolution.unresolved_fields) if resolution else (),
    )


def build_conversation_context(
    turns: tuple[ConversationTurn, ...] | list[ConversationTurn],
) -> ConversationContext:
    """轮次 → `ConversationContext`。

    **字符预算不在这里**：预算约束的是「渲染出来的那段文本有多长」，
    而这里看不到渲染。把预算放在这里就得在这里重新实现一遍渲染的计量，
    两处一旦不一致，症状是"预算调小了却什么都没省"（或反过来）。
    它归 `app/agent/prompts/supervisor.py` 的 `render_context` 管。
    """
    ordered = list(turns)
    # 「最近一轮有口径的」而不是「最后一轮」：最后一轮很可能是失败或澄清，
    # 那一轮没有口径，而再上一轮的口径仍然是解析代词的正确落点。
    latest_with_entities = next(
        (turn for turn in reversed(ordered) if turn.entities),
        None,
    )
    latest_with_unresolved = next(
        (turn for turn in reversed(ordered) if turn.unresolved_fields),
        None,
    )
    latest_answer = next(
        (turn.answer for turn in reversed(ordered) if turn.answer),
        None,
    )

    messages: list[ContextMessage] = []
    for turn in ordered:
        messages.append(ContextMessage(role=MessageRole.USER, content=turn.question))
        if turn.answer is not None:
            messages.append(ContextMessage(role=MessageRole.ASSISTANT, content=turn.answer))

    return ConversationContext(
        recent_messages=tuple(messages),
        confirmed_entities=dict(latest_with_entities.entities) if latest_with_entities else {},
        unresolved_fields=(
            tuple(latest_with_unresolved.unresolved_fields) if latest_with_unresolved else ()
        ),
        previous_task_summary=latest_answer,
    )


def resolution_to_payload(resolution: TurnResolution) -> dict[str, Any]:
    """`TurnResolution` → 落进 `agent_task.result_json` 的形状。

    键名与 `result_json` 里的其他字段一样是**契约**——
    `app/api/schemas.py` 的 `TaskDetailData` 与演示脚本都按它读。
    """
    return {
        "resolved_entities": dict(resolution.entities),
        "unresolved_fields": list(resolution.unresolved_fields),
    }


def resolution_from_payload(payload: dict[str, Any] | None) -> TurnResolution:
    """读回来。**缺字段一律当空**：payload 是历史数据，
    没有这两个键的是本功能上线前的任务，那不是错误。
    """
    data = payload or {}
    entities = {
        str(key): str(value)
        for key, value in (data.get("resolved_entities") or {}).items()
        if key in ENTITY_KEYS and value is not None and str(value).strip()
    }
    unresolved = tuple(str(item) for item in (data.get("unresolved_fields") or []) if str(item))
    return TurnResolution(entities=entities, unresolved_fields=unresolved)


__all__ = [
    "ENTITY_KEYS",
    "ContextMessage",
    "ConversationContext",
    "ConversationTurn",
    "TurnResolution",
    "build_conversation_context",
    "group_turns",
    "resolution_from_payload",
    "resolution_to_payload",
    "squash",
]
