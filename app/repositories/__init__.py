"""仓储层。

`Repositories` 是仓储依赖的**单一入口**，由 `app/main.py` 的 `wire_dependencies`
接收：生产装配塞 MySQL 实现，单元测试注入内存实现。

做成一个整体而不是三个独立参数，是因为三者必须**同源**——
「用户来自 MySQL、任务来自内存」这种组合在测试里不会有任何报错，
却会让「登录后查自己的任务」这类链路出现无法解释的空结果。
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.repositories.agent_repo import AgentArtifactRepository, SqlAgentArtifactRepository
from app.repositories.conversation_repo import (
    ConversationRepository,
    SqlConversationRepository,
)
from app.repositories.knowledge_repo import (
    KnowledgeDocumentRepository,
    SqlKnowledgeDocumentRepository,
)
from app.repositories.task_repo import SqlTaskRepository, TaskRepository
from app.repositories.user_repo import SqlUserRepository, UserRepository
from app.repositories.vocab_repo import SqlVocabRepository, VocabRepository


@dataclass(frozen=True, slots=True)
class Repositories:
    users: UserRepository
    conversations: ConversationRepository
    tasks: TaskRepository
    #: 稀疏检索词表（11.6.3）。它与其他三个的形状不同——**只增不改、没有删除**，
    #: 且读路径通常走快照而不是这张表（见 `tools/rag/vocabulary.py`）。
    #: 仍然放在这里，是因为"仓储依赖只有一个入口"这条纪律比形状整齐更重要：
    #: 另开一条装配路径，就会出现「谁忘了给 worker 装配词表仓储」这类问题。
    vocab: VocabRepository
    #: 知识文档版本（11.9）。与 `vocab` 同理放进同一入口：入库要同时用到两者，
    #: 分开装配就会出现"词表来自 MySQL、文档来自内存"这种在测试里毫无症状的组合。
    documents: KnowledgeDocumentRepository
    #: 执行产出（16.6 / 16.7 的五张表 + 轨迹）。放进来不是为了整齐，是为了
    #: **17.4 的轨迹接口能在单元测试里被验证**：那个接口直接读它，而单元测试
    #: 不连 MySQL——不从这里注入，接口用例就只能测"403 与 404"，
    #: "事件按序返回"这条主路径永远没有用例。
    artifacts: AgentArtifactRepository


def build_sql_repositories(sessions: async_sessionmaker[AsyncSession]) -> Repositories:
    """生产装配：全部走 MySQL（详细设计 16 章）。

    **放在这里而不是 `app/main.py`**：`app/worker.py` 也要用它，而 worker
    一旦 import `app.main` 就会把 FastAPI 加载进 Worker 进程——
    正是 L2「Worker 不得依赖 Web 框架」要防的事，但分层检查器只匹配
    `fastapi` / `starlette` 前缀，**查不出这种间接依赖**。
    放在仓储层这个「双方都向下依赖」的位置是唯一不会踩线的选择。
    """
    return Repositories(
        users=SqlUserRepository(sessions),
        conversations=SqlConversationRepository(sessions),
        tasks=SqlTaskRepository(sessions),
        vocab=SqlVocabRepository(sessions),
        documents=SqlKnowledgeDocumentRepository(sessions),
        artifacts=SqlAgentArtifactRepository(sessions),
    )
