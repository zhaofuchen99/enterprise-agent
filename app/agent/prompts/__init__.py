"""Prompt 模板目录（开发流程 6.5 施工项 3）。

每个模板一个模块，模块内导出该模板的版本号常量与模板对象；
本文件把它们汇总成 `ALL_PROMPTS`，供回归触发矩阵（开发流程 7.2）
与测试统一枚举。

**新增模板模块时必须登记到 `ALL_PROMPTS`**——`app/tests/agent/test_prompts.py`
会断言清单非空且 `(name, version)` 无重复，漏登记不会静默通过。
"""

from __future__ import annotations

from typing import Final

from app.agent.prompts.base import PromptTemplate, PromptVariableError
from app.agent.prompts.smoke import SMOKE_PROMPT

#: 全部模板。开发流程 7.2：「任意 prompt 模板」变更 → 跑对应评测集。
ALL_PROMPTS: Final[tuple[PromptTemplate, ...]] = (SMOKE_PROMPT,)

__all__ = ["ALL_PROMPTS", "SMOKE_PROMPT", "PromptTemplate", "PromptVariableError"]
