"""SQL 生成与修复的 prompt（开发流程 6.6 施工项 2、详设 10.8）。

这一层不做安全判断，因此测的是「送给模型的东西对不对」：
模型看到的表与不表、口径完不完整、规则里的 LIMIT 上限是不是真的在执行的那个。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agent.prompts.base import PromptVariableError
from app.core.config import Settings
from app.tests.fakes import FakeModelGateway
from app.tools.sql.generator import SqlGenerator
from app.tools.sql.prompts import SAFETY_RULES, SQL_GENERATION_PROMPT, SQL_REPAIR_PROMPT
from app.tools.sql.schema_provider import SchemaProvider
from app.tools.sql.schemas import SqlCandidate


def test_template_declares_all_variables() -> None:
    """`PromptTemplate.render` 对缺变量与多变量都会抛错——声明必须与正文一致。

    这里再断言一次是因为模板是**跨模块**的：调用方（`SqlGenerator`）传什么
    完全由这份声明决定，而声明写错了要到运行时才炸。
    """
    assert SQL_GENERATION_PROMPT.variables() == {"schema", "rules", "question", "objective"}
    assert SQL_REPAIR_PROMPT.variables() == {
        "schema",
        "rules",
        "question",
        "objective",
        "original_sql",
        "error",
    }


def test_template_versions_declared() -> None:
    """详设 22.9：改 prompt 必须升版本，否则历史记录里的 `prompt_version`
    会指向两段不同的文本，回归归因就断了。"""
    assert SQL_GENERATION_PROMPT.version
    assert SQL_REPAIR_PROMPT.version
    assert SQL_GENERATION_PROMPT.name != SQL_REPAIR_PROMPT.name


def test_rules_limit_comes_from_settings(settings: Settings, gateway: FakeModelGateway) -> None:
    """写死的 1000 与配置里的实际上限不一致时，模型会按一个不存在的上限写 SQL。"""
    generator = SqlGenerator(gateway, max_rows=250)

    assert "250" in generator._rules
    assert "1000" not in generator._rules


def test_rules_template_free_of_braces() -> None:
    """`SAFETY_RULES` 由 `str.format` 注入行数，正文里除了 `{max_rows}` 不能有别的花括号。

    多一个花括号会让 `SqlGenerator` 在**构造时**抛 KeyError——那还算好；
    更糟的是某个 `{` 恰好拼成了合法字段名，于是规则文本被静默替换成别的东西。
    """
    rendered = SAFETY_RULES.format(max_rows=1000)
    assert "{" not in rendered and "}" not in rendered


async def test_generate_prompt_carries_schema_and_rules(
    settings: Settings, gateway: FakeModelGateway
) -> None:
    provider = SchemaProvider.from_settings(settings)
    context = provider.select("2025 年华东 Q3 净销售额")
    generator = SqlGenerator(gateway, max_rows=settings.sql_tool.max_rows)
    gateway.responses = [SqlCandidate(sql="SELECT 1")]

    await generator.generate(question="净销售额", objective="", context=context)

    call = gateway.calls[0]
    assert call.prompt_name == SQL_GENERATION_PROMPT.name
    assert call.schema == "SqlCandidate"
    # 口径表达式必须进 prompt——这一步没做到，模型就会自己发明算法
    assert "SUM(fact_sales_order_item.net_amount)" in call.rendered
    # 分析目标缺省时用问题兜底，而不是留空
    assert "净销售额" in call.rendered


async def test_generated_output_validated_by_pydantic(
    settings: Settings, gateway: FakeModelGateway
) -> None:
    """开发流程 5.3：模型输出进 State 之前必须先经校验。

    少一个必填字段就该在这一层拒掉，而不是让一个形状不全的 dict 一路走到
    「生成 SQL 时报错」——那时错误信息看起来像模型写错了 SQL。
    """
    provider = SchemaProvider.from_settings(settings)
    context = provider.select("净销售额")
    generator = SqlGenerator(gateway, max_rows=settings.sql_tool.max_rows)
    gateway.responses = [{"selected_tables": []}]  # 缺 sql

    with pytest.raises(ValidationError):
        await generator.generate(question="净销售额", objective="", context=context)


async def test_repair_prompt_carries_sql_and_error(
    settings: Settings, gateway: FakeModelGateway
) -> None:
    provider = SchemaProvider.from_settings(settings)
    context = provider.select("净销售额")
    generator = SqlGenerator(gateway, max_rows=settings.sql_tool.max_rows)
    gateway.responses = [SqlCandidate(sql="SELECT 1")]

    await generator.repair(
        question="净销售额",
        objective="",
        context=context,
        original_sql="SELECT * FROM dim_region",
        error="第 7 步校验未通过：不允许 SELECT *",
    )

    rendered = gateway.calls[0].rendered
    assert "SELECT * FROM dim_region" in rendered
    assert "第 7 步校验未通过" in rendered


async def test_generate_passes_declared_variables(
    settings: Settings, gateway: FakeModelGateway
) -> None:
    """模板加一个变量而调用方没传（或反之）必须立刻炸，而不是发出一段残缺 prompt。

    残缺 prompt 发给模型的表现是「答非所问」，排查成本比一个启动期异常高一个数量级。
    `PromptTemplate.render` 对缺变量与多变量都会抛 `PromptVariableError`，
    这条断言把「声明」与「实际传的」钉在一起——它是唯一的漂移检测点。
    """
    provider = SchemaProvider.from_settings(settings)
    context = provider.select("净销售额")
    generator = SqlGenerator(gateway, max_rows=settings.sql_tool.max_rows)
    gateway.responses = [SqlCandidate(sql="SELECT 1")]

    await generator.generate(question="净销售额", objective="", context=context)

    assert set(gateway.calls[0].variables) == SQL_GENERATION_PROMPT.variables()

    gateway.responses = [SqlCandidate(sql="SELECT 1")]
    await generator.repair(
        question="净销售额", objective="", context=context, original_sql="SELECT 1", error="错"
    )
    assert set(gateway.calls[1].variables) == SQL_REPAIR_PROMPT.variables()


def test_render_raises_on_missing_variable() -> None:
    """`PromptVariableError` 是上面那条漂移检测的机制本身，单独确认它真的会抛。"""
    with pytest.raises(PromptVariableError):
        SQL_GENERATION_PROMPT.render(question="x")
