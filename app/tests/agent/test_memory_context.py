"""多轮上下文进图之后：喂给模型、以及**传给工具**。

这个文件里最承重的是最后两条（`test_the_sql_node_...`）：**上下文解析得再对，
下游读的仍是 `user_query` 的话整条链路等于没接**。「那 Q2 呢？」会被原样送进
SQL 生成器的 `业务问题：{question}`，指标与区域一个都带不过去——
而生成的 SQL 看起来完全正常（只是少了个区域）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.agent.nodes.supervisor import build_supervisor_node
from app.agent.nodes.tool_nodes import build_sql_node
from app.agent.prompts.supervisor import SUPERVISOR_PROMPT
from app.agent.schemas.plan import IntentResult, TaskStep
from app.core.config import Settings, get_settings
from app.domain.conversation import MessageRole
from app.domain.memory import ContextMessage, ConversationContext
from app.domain.user import PermissionScope, UserRole
from app.tests.fakes import FakeModelGateway
from app.tools.base import ToolResult

_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)


def _context(**overrides: Any) -> ConversationContext:
    base: dict[str, Any] = {
        "recent_messages": (
            ContextMessage(role=MessageRole.USER, content="2025年华东Q3销售额多少"),
            ContextMessage(
                role=MessageRole.ASSISTANT, content="2025年Q3华东净销售额为 11,196.70 万元"
            ),
            ContextMessage(role=MessageRole.USER, content="那Q2呢"),
        ),
        "confirmed_entities": {"metric": "净销售额", "region": "华东", "period": "2025-Q3"},
        "previous_task_summary": "2025年Q3华东净销售额为 11,196.70 万元",
    }
    base.update(overrides)
    return ConversationContext.model_validate(base)


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "user_query": "那Q2呢？",
        "sanitized_query": "那Q2呢？",
        "user_id": "usr_0000000000000000000001",
        "task_id": "tsk_0000000000000000000001",
        "trace_id": "trc_0000000000000000000001",
        "conversation_id": "cnv_0000000000000000000001",
        "permission_scope": PermissionScope(role=UserRole.ANALYST),
        "deadline_at": _NOW,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ 喂给模型
async def test_the_prompt_carries_the_previous_entities(settings: Settings) -> None:
    """上下文到了模型手里，且**用词是「上一轮系统解析」而不是「已确认」**。

    详设 15.1 的原话是「模型推测不得写成"用户已确认"」，而字段名叫
    `confirmed_entities`——模型看到的是**渲染出来的文本**，不是字段名。
    措辞错了不会报错，只会让模型把一份系统推测当成用户明确指定的口径，
    然后理直气壮地拿它去补全代词。
    """
    gateway = FakeModelGateway(
        responses=[IntentResult(intent="QUERY", required_sources=("sql",), confidence=0.9)]
    )

    await build_supervisor_node(settings, gateway)(
        _state(context_summary=_context())  # type: ignore[arg-type]
    )

    rendered = gateway.calls[0].variables["context"]
    assert "华东" in rendered
    assert "2025-Q3" in rendered
    assert "上一轮系统解析" in rendered
    assert "已确认" not in rendered


async def test_no_history_renders_an_explicit_line(settings: Settings) -> None:
    """没有历史时给一句**明确的话**，而不是空串——而且**不抛**。

    `PromptTemplate.render` 对**少传变量**是报错的。模板加了 `{context}`
    而节点忘了传，这一条会当场炸；这也是最容易漏的一处（两处
    `invoke_structured` 只要漏一处，就只在"模型第一次输出非法"时才会走到）。
    """
    gateway = FakeModelGateway(
        responses=[IntentResult(intent="QUERY", required_sources=("sql",), confidence=0.9)]
    )

    await build_supervisor_node(settings, gateway)(_state())  # type: ignore[arg-type]

    assert gateway.calls[0].variables["context"] == "（无历史：这是本会话的第一轮提问）"
    assert len(gateway.calls) == 1


def test_the_prompt_version_was_bumped() -> None:
    """模板改了就必须升版本——版本号会随结果进 span（`llm.prompt_version`），
    是回答"昨天还能用今天为什么变了"的唯一线索。"""
    assert SUPERVISOR_PROMPT.version == "1.1.0"


# -------------------------------------------------------- 传给工具（承重）
class _RecordingTool:
    """只记录参数、不回结果——本文件关心的是"问题有没有被换掉"。"""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def execute(self, args: Any, ctx: Any) -> ToolResult:
        self.questions.append(args.question)
        return ToolResult(
            call_id="tcl_0000000000000000000001",
            tool="sql_query",
            status="SUCCEEDED",
            started_at=_NOW,
            finished_at=_NOW,
            summary="命中 1 条",
            payload={},
            evidence=[],
        )


def _step() -> TaskStep:
    return TaskStep(
        id="step_01",
        objective="从业务数据库查出问题涉及指标的数值",
        tool="sql_query",
    )


@pytest.fixture
def settings() -> Settings:
    return get_settings()


async def test_the_sql_node_sends_the_resolved_question(settings: Settings) -> None:
    """**解析后的问题**被送去生成 SQL，而不是用户原话。

    这条是这一整片功能的落点：supervisor 把「那 Q2 呢」解析成
    「2025年Q2华东净销售额是多少」，而 SQL 生成器只拿得到问题文本
    （指标与区域在 `IntentResult` 的字段里，模板看不到）——
    不换的话，生成的是一个**没有区域、没有指标**的问题。
    """
    tool = _RecordingTool()
    state = _state(
        sanitized_query="2025年Q2华东净销售额是多少",
        task_list=[_step()],
    )

    await build_sql_node(settings, tool)(state)  # type: ignore[arg-type]

    assert tool.questions == ["2025年Q2华东净销售额是多少"]


async def test_without_a_rewrite_the_original_question_is_used(settings: Settings) -> None:
    """没有改写时退回原话——**不是空串**。

    空串送进 SQL 生成器会得到一条没有任何过滤条件的 SQL，
    而它照样跑得出来、照样有数字。
    """
    tool = _RecordingTool()
    state = _state(task_list=[_step()])

    await build_sql_node(settings, tool)(state)  # type: ignore[arg-type]

    assert tool.questions == ["那Q2呢？"]
