"""链路自检模板（`make model-smoke`）。

**这个模板不参与任何业务分析**，它的唯一用途是回答「模型现在通不通」：
`make model-smoke` 拿它打一次真实调用，把延迟、token 用量、以及
「返回的是不是合法 JSON」打印出来。

之所以需要一个专门的诊断模板而不是复用业务模板：业务模板要到
Phase 4（SQL 生成）、Phase 6（Supervisor）才落地，而**密钥与连通性必须在
写业务代码之前就能验证**——否则第一次接真模型时会同时面对
「prompt 写得对不对」和「key 配得对不对」两个未知数。
"""

from __future__ import annotations

from typing import Final

from app.agent.prompts.base import PromptTemplate

#: 改动 `SMOKE_TEMPLATE` 时必须同步升版本（见 `PromptTemplate.version` 的说明）。
#: 本模板不进业务链路，但同样受这条纪律约束——诊断脚本的输出要与版本对得上，
#: 否则「昨天还通、今天不通」时无法判断是不是模板改了。
SMOKE_PROMPT_VERSION: Final[str] = "v1"

SMOKE_TEMPLATE: Final[str] = (
    "这是一次链路连通性自检，不涉及任何业务分析。\n"
    "请针对下面的问题给出你的判断与把握程度。\n\n"
    "问题：{question}\n"
)

SMOKE_PROMPT: Final[PromptTemplate] = PromptTemplate(
    name="smoke",
    version=SMOKE_PROMPT_VERSION,
    template=SMOKE_TEMPLATE,
)
