"""用户、会话与消息（详细设计 16.3 / 16.4）。

`app_user` 之外的两张表承载 FR-CHAT-003「多轮上下文」：
会话存上下文摘要，消息存最近 N 轮的可见对话。
"""

from __future__ import annotations

from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base
from app.infrastructure.models.common import (
    json_obj_opt,
    ulid_pk,
    ulid_ref,
    ulid_ref_opt,
    utc_dt,
    utc_dt_opt,
)


class AppUser(Base):
    """最小用户与角色（16.3）。

    `password_hash` 可空是为将来接 SSO 预留——SSO 用户没有本地口令。
    """

    __tablename__ = "app_user"

    id: Mapped[ulid_pk]
    #: 唯一登录名。约束名由 db.py 的命名约定生成为 `uq_app_user_username`，
    #: 登录查询按它走，名字稳定可 grep。
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(128))
    #: ANALYST / ADMIN。用 VARCHAR 存字符串而不是 MySQL ENUM，见 common.py。
    role: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    #: 数据范围摘要。TBC-03 决议为 `{"region_ids": [...]}`。
    data_scope_json: Mapped[json_obj_opt]
    created_at: Mapped[utc_dt]
    updated_at: Mapped[utc_dt]

    __table_args__ = (Index("idx_user_status", "status"),)


class AgentConversation(Base):
    """会话与上下文摘要（16.4）。

    `deleted_at` 是逻辑删除标记：用户删除会话先逻辑删除，
    物理清理交给 `make cleanup`（16.12）。
    """

    __tablename__ = "agent_conversation"

    id: Mapped[ulid_pk]
    #: 不加单列索引：下面的 idx_conversation_user_time 以 user_id 为前导列，已覆盖。
    user_id: Mapped[ulid_ref]
    title: Mapped[str | None] = mapped_column(String(255))
    #: ConversationContext 的序列化结果（详细设计 15.1）
    context_summary_json: Mapped[json_obj_opt]
    last_message_at: Mapped[utc_dt_opt]
    created_at: Mapped[utc_dt]
    updated_at: Mapped[utc_dt]
    deleted_at: Mapped[utc_dt_opt]

    __table_args__ = (
        # 会话列表按「该用户的最近活跃」翻页，索引顺序必须与此一致
        Index("idx_conversation_user_time", "user_id", "last_message_at"),
    )


class AgentMessage(Base):
    """用户与助手可见消息（16.4）。

    **不得保存模型思维链**（FR-TRACE-001 业务规则）。
    `role` 为 USER / ASSISTANT / SYSTEM_NOTICE。
    """

    __tablename__ = "agent_message"

    id: Mapped[ulid_pk]
    #: 同 AgentConversation.user_id：单列索引由下面的复合索引前导列覆盖。
    conversation_id: Mapped[ulid_ref]
    #: 该消息由哪次任务产生；SYSTEM_NOTICE 可能没有对应任务，故可空。
    task_id: Mapped[ulid_ref_opt]
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    token_count: Mapped[int | None]
    created_at: Mapped[utc_dt]
    deleted_at: Mapped[utc_dt_opt]

    __table_args__ = (
        # 上下文构建按会话取最近 N 轮，与查询顺序一致
        Index("idx_message_conversation_time", "conversation_id", "created_at"),
    )
