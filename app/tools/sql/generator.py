"""候选 SQL 生成与定向修复（详细设计 10.1 的 Prompt Construction / Candidate SQL
Generation，10.8 的自修复）。

**这个模块只负责「问模型要一条 SQL」，不做任何安全判断。** 校验是
`validator.py` 的事，执行是 `executor.py` 的事。分开的直接好处是：
「模型生成的 SQL 不安全」与「校验器漏放了一条」是两类完全不同的缺陷，
合在一个模块里时它们都表现为「危险的 SQL 被执行了」。

模型调用一律经 `ModelGateway`（详设 4.1「LLM 适配」），因此单元测试里
换成 `FakeModelGateway` 就能把这条链路完整跑一遍而不打网络。
"""

from __future__ import annotations

from typing import Final

from app.infrastructure.model_gateway import ModelGateway, StructuredResult
from app.tools.sql.prompts import SAFETY_RULES, SQL_GENERATION_PROMPT, SQL_REPAIR_PROMPT
from app.tools.sql.schemas import SchemaContext, SqlCandidate


class SqlGenerator:
    """把业务问题（或一次失败）转成 `SqlCandidate`。"""

    def __init__(self, gateway: ModelGateway, *, max_rows: int) -> None:
        self._gateway = gateway
        #: 进 prompt 的 LIMIT 上限。取配置值而不是写死 1000：
        #: 规则文本与实际执行上限不一致时，模型会按一个不存在的上限写 SQL。
        self._rules: Final[str] = SAFETY_RULES.format(max_rows=max_rows)

    async def generate(
        self,
        *,
        question: str,
        objective: str,
        context: SchemaContext,
    ) -> StructuredResult[SqlCandidate]:
        return await self._gateway.invoke_structured(
            SQL_GENERATION_PROMPT,
            SqlCandidate,
            schema=context.render(),
            rules=self._rules,
            question=question,
            objective=objective or question,
        )

    async def repair(
        self,
        *,
        question: str,
        objective: str,
        context: SchemaContext,
        original_sql: str,
        error: str,
    ) -> StructuredResult[SqlCandidate]:
        """带着错因重生成一次（详设 10.8）。

        `error` 必须**已脱敏**——它是数据库或校验器给出的原因摘要，
        调用方（`tool.py`）负责在传进来之前把原始 SQL 与取值摘掉。
        """
        return await self._gateway.invoke_structured(
            SQL_REPAIR_PROMPT,
            SqlCandidate,
            schema=context.render(),
            rules=self._rules,
            original_sql=original_sql,
            error=error,
            question=question,
            objective=objective or question,
        )


__all__ = ["SqlGenerator"]
