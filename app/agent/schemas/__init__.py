"""进入 LangGraph State 的 Pydantic 边界模型（开发流程 5.3、详设 7.4）。

**所有进入 State 的 LLM 输出与 Tool 输出必须先经 Pydantic 校验**，
禁止把任意 `dict` 直接写入 State。本目录是那些模型的归属地。

当前只有自检用的 `SmokeAnswer`。业务模型随各节点落地：
Phase 4 的 SQL 生成结果、Phase 6 的 `IntentResult` / `ProgressAssessment`（详设 7.4）。
"""

from __future__ import annotations

from app.agent.schemas.smoke import SmokeAnswer

__all__ = ["SmokeAnswer"]
