"""链路自检用的判据模型（`make model-smoke`）。

**不参与业务**，与 `app/agent/prompts/smoke.py` 的模板配套。

字段刻意**不加取值范围约束**：自检要回答的是「模型通不通、输出能不能被解析」，
而不是「模型答得准不准」。加一个 `ge` / `le` 会让「链路正常但模型给的值
越界」也报成失败，把连通性问题和一个纯粹的取值问题混在一起，
而后者该由 Phase 4 起的业务 Schema 去管。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SmokeAnswer(BaseModel):
    """一次自检的返回：一句判断 + 一个把握程度。"""

    answer: str = Field(description="针对自检问题的简短回答")
    confidence: int = Field(description="对上述回答的把握程度，0-100")
