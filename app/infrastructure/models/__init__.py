"""ORM 模型汇总（详细设计 16.3–16.9）。

**这个 `__init__` 必须导入全部模型**：Alembic 的 autogenerate 靠
`Base.metadata` 与数据库现状做差，未被导入的模型不会出现在 `metadata` 里，
表现为「迁移脚本里悄悄少了一张表」——而且不报错。
新增模型时同时在这里登记，是防止漏迁移的唯一手段。
"""

from __future__ import annotations

from app.infrastructure.models.catalog import AgentConfig, SchemaCatalog
from app.infrastructure.models.evidence import (
    AgentConflict,
    AgentEvidence,
    AgentReview,
    AgentTraceEvent,
)
from app.infrastructure.models.identity import (
    AgentConversation,
    AgentMessage,
    AppUser,
)
from app.infrastructure.models.knowledge import KnowledgeDocument, RagVocab
from app.infrastructure.models.task import (
    AgentFinding,
    AgentPlanRevision,
    AgentTask,
    AgentTaskStep,
    AgentToolCall,
)

__all__ = [
    "AgentConfig",
    "AgentConflict",
    "AgentConversation",
    "AgentEvidence",
    "AgentFinding",
    "AgentMessage",
    "AgentPlanRevision",
    "AgentReview",
    "AgentTask",
    "AgentTaskStep",
    "AgentToolCall",
    "AgentTraceEvent",
    "AppUser",
    "KnowledgeDocument",
    "RagVocab",
    "SchemaCatalog",
]
