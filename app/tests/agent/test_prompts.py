"""Prompt 模板的版本纪律与变量校验（开发流程 6.5 施工项 3、7.2 回归触发矩阵）。"""

from __future__ import annotations

import pytest

from app.agent.prompts import ALL_PROMPTS, PromptTemplate, PromptVariableError
from app.agent.prompts.smoke import SMOKE_PROMPT, SMOKE_PROMPT_VERSION


def test_render_substitutes_variables() -> None:
    template = PromptTemplate(name="t", version="v1", template="问：{question}")
    assert template.render(question="华东区") == "问：华东区"


def test_render_rejects_missing_variable() -> None:
    """缺变量必须报错，绝不能把带着 `{question}` 字面量的 prompt 发给模型。"""
    template = PromptTemplate(name="t", version="v1", template="问：{question}")
    with pytest.raises(PromptVariableError, match="缺少变量"):
        template.render()


def test_render_rejects_unexpected_variable() -> None:
    """多传变量同样要报错。

    `str.format` 对多余的键是**静默忽略**的，所以只校验缺失的话，
    变量名拼错（`question` 写成 `questions`）会一路带到模型面前，
    表现为「答非所问」——比一个异常难查得多。
    """
    template = PromptTemplate(name="t", version="v1", template="问：{question}")
    with pytest.raises(PromptVariableError, match="未声明的变量"):
        template.render(question="a", questions="b")


def test_variables_uses_format_syntax_not_regex() -> None:
    """占位符解析走 `string.Formatter`，与 `str.format` 的真实语法一致。"""
    template = PromptTemplate(name="t", version="v1", template="{a} 与 {b[0]} 与 {{转义}}")
    assert template.variables() == frozenset({"a", "b"})


def test_all_prompts_is_not_empty() -> None:
    """清单为空说明模板写了却没登记——回归触发矩阵（开发流程 7.2）会漏掉它。"""
    assert ALL_PROMPTS


def test_all_prompts_have_unique_name_and_version() -> None:
    """`(name, version)` 必须唯一。

    重复会让 Trace 里的 `prompt_version` 指向两段不同的文本，
    而「改完 prompt 无法归因」正是引入版本号要解决的问题本身。
    """
    keys = [(prompt.name, prompt.version) for prompt in ALL_PROMPTS]
    assert len(keys) == len(set(keys))


def test_smoke_template_version_constant_matches_object() -> None:
    """模块导出的版本号常量与模板对象里的必须一致。

    两者漂移时，`make model-smoke` 打印的版本号与 Trace 里记的会对不上，
    而诊断脚本的全部意义就是让这两者能互相印证。
    """
    assert SMOKE_PROMPT.version == SMOKE_PROMPT_VERSION


def test_smoke_template_renders() -> None:
    assert "华东" in SMOKE_PROMPT.render(question="华东区销售额是多少？")
