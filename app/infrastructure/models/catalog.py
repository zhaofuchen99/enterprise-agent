"""Schema 白名单与运行配置（详细设计 16.9）。

两张表都**按版本发布、不原地修改**：`version` 变化即新行，
`schema_version` 会被写进任务结果，使「这条结论基于哪一版口径」可回溯。

**密钥不在这里**。`agent_config` 只存非密钥配置，
密钥只来自环境变量或密钥服务（19.5）。
"""

from __future__ import annotations

from sqlalchemy import Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db import Base
from app.infrastructure.models.common import (
    json_list_opt,
    json_obj_opt,
    ulid_pk,
    utc_dt,
    utc_dt_opt,
)


class SchemaCatalog(Base):
    """白名单 Schema 与指标目录（16.9）。

    SQL Tool 的 Schema 注入从这里取，**不从数据库反射生成**——
    白名单是安全边界，允许模型看见什么必须由配置显式决定（FR-ADM-001）。
    """

    __tablename__ = "schema_catalog"

    id: Mapped[ulid_pk]
    version: Mapped[str] = mapped_column(String(32))
    #: 数据源标识，对应 DATABASE_URL_BUSINESS_RO 指向的库
    datasource: Mapped[str] = mapped_column(String(64))
    #: 表与字段白名单
    tables_json: Mapped[json_obj_opt]
    #: 允许的 JOIN 路径。不声明 JOIN 的表不允许被连接，防笛卡尔积与越权关联。
    joins_json: Mapped[json_list_opt]
    #: 指标目录：指标名 → 口径、计算表达式、可用维度
    metrics_json: Mapped[json_obj_opt]
    #: DRAFT / PUBLISHED / ARCHIVED
    status: Mapped[str] = mapped_column(String(16))
    content_hash: Mapped[str] = mapped_column(String(64))
    created_by: Mapped[str | None] = mapped_column(String(26))
    created_at: Mapped[utc_dt]
    published_at: Mapped[utc_dt_opt]

    __table_args__ = (
        UniqueConstraint("version", name="uk_schema_catalog_version"),
        Index("idx_schema_catalog_status", "status"),
    )


class AgentConfig(Base):
    """非密钥运行配置（16.9）。

    与 `app/core/config.py` 的分工：环境变量决定**部署形态**（连哪个库、开哪些开关），
    这张表决定**运行期可调的策略**（阈值、提示词版本指向、预算）。
    """

    __tablename__ = "agent_config"

    id: Mapped[ulid_pk]
    key: Mapped[str] = mapped_column(String(128))
    #: 配置版本。改配置产生新行而非覆盖，便于回滚与对比。
    version: Mapped[int] = mapped_column()
    value_json: Mapped[json_obj_opt]
    #: ACTIVE / ARCHIVED
    status: Mapped[str] = mapped_column(String(16))
    updated_by: Mapped[str | None] = mapped_column(String(26))
    updated_at: Mapped[utc_dt]

    __table_args__ = (
        UniqueConstraint("key", "version", name="uk_agent_config_key_version"),
        Index("idx_agent_config_key_status", "key", "status"),
    )
