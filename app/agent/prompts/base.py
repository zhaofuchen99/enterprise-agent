"""Prompt 模板与版本管理（开发流程 6.5 施工项 3、详细设计 4.2 的 `prompts/`）。

**为什么模板必须带版本号，并且必须随调用记录下来**：详细设计 22.9 规定
「每次修改 prompt、模型、Schema……必须运行对应回归集，并记录版本」。
没有 `prompt_version`，一次「这次检索质量怎么变差了」的排查就无法回答
「是模型变了、还是 prompt 被改了」——版本号与模型名必须落在同一条调用记录里，
才谈得上归因。开发流程 7.2 的回归触发矩阵也把「任意 prompt 模板」单列一行。

**这里只放模板本身，不放 JSON 输出契约。** 让模型「只输出 json」并给出格式示例
是**服务方的协议要求**（DeepSeek 的 JSON 模式要求 prompt 内出现字面量 "json"
并附格式示例），不是业务模板的组成部分。它由网关统一追加——换一个原生支持
structured output 的服务时，要改的是网关那一个文件，而不是这里的每一个模板。
见 `app/infrastructure/model_gateway.py` 的 `build_json_contract`。
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Final


class PromptVariableError(ValueError):
    """模板变量与调用方传入的对不上。

    继承内建 `ValueError` 而不是自造基类：这是**参数非法**，不是运行期故障，
    调用方（写错的开发者）应当看到它立刻炸掉，而不是被降级路径吞掉。
    """


@dataclass(frozen=True)
class PromptTemplate:
    """一个带版本的 prompt 模板。

    结构化子类型：网关侧的 `PromptSource` Protocol（`model_gateway.py`）
    定义了它需要满足的形状，本类**不显式继承**它。这是本项目的既有约定
    （见 `app/infrastructure/storage.py` 的 `LocalObjectStorage`）——
    让 `agent` 侧不必 import `infrastructure`，依赖方向保持单向。

    Attributes:
        name: 模板标识，用于日志与回归集映射。
        version: 版本号常量。**改动 template 必须同时升版本**——
            否则历史记录里的 `prompt_version` 会指向两段不同的文本。
            刻意做成人工维护的常量而不是内容哈希：哈希会因为改一个标点就变，
            而版本号只应在「这次改动可能影响输出」时才变，这个判断哈希做不了。
        template: 含 `{placeholder}` 的模板正文。
    """

    name: str
    version: str
    template: str

    def variables(self) -> frozenset[str]:
        """模板声明的**根**占位符名集合。

        用 `string.Formatter` 解析而不是正则：它认的是 `str.format` 的真实语法，
        正则只能匹配最朴素的 `{name}`，两者一旦不一致，就会出现
        「校验说没问题、format 却炸了」。

        解析结果要**截到根名**：`{b[0]}` 与 `{b.name}` 的字段名分别被解析成
        `b[0]` 与 `b.name`，而调用方传的是 `b=...`（取值由 format 自己做）。
        不截根名的话，校验会要求一个名叫 `b[0]` 的变量并拒绝真正的 `b`，
        把一个本来正确的调用挡在门外。
        """
        roots = (
            field_name.split(".", 1)[0].split("[", 1)[0]
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name
        )
        return frozenset(roots)

    def render(self, **variables: Any) -> str:
        """渲染模板。

        **缺变量和多变量都报错**，不是只报缺的：多传通常意味着变量名拼错了
        （`question` 写成 `questions`），而 `str.format` 对多余的键是静默忽略的，
        只校验缺失的话这个错会一路带到模型面前，表现为「答案答非所问」——
        比一个启动期异常难查得多。
        """
        required = self.variables()
        provided = set(variables)
        if missing := sorted(required - provided):
            raise PromptVariableError(
                f"prompt 模板 {self.name!r}({self.version}) 缺少变量：{missing}"
            )
        if unexpected := sorted(provided - required):
            raise PromptVariableError(
                f"prompt 模板 {self.name!r}({self.version}) 收到未声明的变量：{unexpected}；"
                f"模板声明的是 {sorted(required)}"
            )
        return self.template.format(**variables)


__all__: Final[tuple[str, ...]] = ("PromptTemplate", "PromptVariableError")
