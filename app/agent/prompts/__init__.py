"""Prompt 模板目录（开发流程 6.5 施工项 3）。

每个模板一个模块，模块内导出该模板的版本号常量与模板对象；
本文件把它们汇总成 `ALL_PROMPTS`，供回归触发矩阵（开发流程 7.2）
与测试统一枚举。

**新增模板模块时必须登记到 `ALL_PROMPTS`**——`app/tests/agent/test_prompts.py`
会断言清单非空且 `(name, version)` 无重复，漏登记不会静默通过。
"""

from __future__ import annotations

from typing import Final

from app.agent.prompts.analysis import ANALYSIS_PROMPT
from app.agent.prompts.base import PromptTemplate, PromptVariableError
from app.agent.prompts.plan_extend import PLAN_EXTEND_PROMPT
from app.agent.prompts.review import REVIEW_PROMPT
from app.agent.prompts.smoke import SMOKE_PROMPT
from app.agent.prompts.supervisor import SUPERVISOR_PROMPT

#: 全部模板。开发流程 7.2：「任意 prompt 模板」变更 → 跑对应评测集。
#:
#: ⚠️ **这份清单曾经只有 `SMOKE_PROMPT`**（`analysis` 与 `supervisor` 那两个
#: 从写下起就没登记过），而用例只断言"非空且不重复"——于是"漏登记"这件事
#: 恰好是它测不出来的。本模块的 docstring 一直写着"新增模板必须登记"，
#: 但**清单不全时这条纪律无法被枚举执行**：照 `ALL_PROMPTS` 去跑回归集，
#: 会漏掉两个真正在用的模板。补齐它才是让那条纪律成立的前提。
ALL_PROMPTS: Final[tuple[PromptTemplate, ...]] = (
    SMOKE_PROMPT,
    SUPERVISOR_PROMPT,
    ANALYSIS_PROMPT,
    REVIEW_PROMPT,
    PLAN_EXTEND_PROMPT,
)

__all__ = [
    "ALL_PROMPTS",
    "ANALYSIS_PROMPT",
    "PLAN_EXTEND_PROMPT",
    "REVIEW_PROMPT",
    "SMOKE_PROMPT",
    "SUPERVISOR_PROMPT",
    "PromptTemplate",
    "PromptVariableError",
]
