"""全局配置。

规范见开发流程 5.4：配置项必须有**默认值、上下限和环境覆盖规则**。
密钥只从环境变量加载，缺失即启动失败——禁止使用不安全的默认密钥。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoopSettings(BaseModel):
    """任务循环预算（详细设计 5.4）。

    四类预算相互独立，**不可借用**；修改后必须重跑循环类评测集。
    环境变量覆盖方式：`LOOP__MAX_TOTAL_STEPS=30`。
    """

    max_total_steps: int = Field(default=24, ge=1, le=100, description="单任务总步数上限")
    max_expansions: int = Field(default=2, ge=0, le=10, description="计划演进次数上限")
    max_steps_per_expansion: int = Field(default=6, ge=1, le=50, description="单次演进内步数上限")
    max_drilldown_depth: int = Field(default=3, ge=1, le=10, description="自适应下钻最大深度")
    min_step_seconds: float = Field(default=1.0, ge=0.0, description="步间最小间隔，防抖")
    duplicate_similarity_threshold: float = Field(
        default=0.92, ge=0.0, le=1.0, description="重复步骤判定阈值，需用真实数据校准"
    )
    #: 独立预算：SQL 修复 / 重新规划 / Reviewer 补证 各自的次数上限
    max_sql_repairs: int = Field(default=2, ge=0, le=5)
    max_replans: int = Field(default=1, ge=0, le=5)
    max_reviewer_evidence: int = Field(default=1, ge=0, le=5)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ 应用
    app_env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 - 容器内监听全网卡是预期行为，由编排层限制暴露面
    api_port: int = Field(default=8000, ge=1, le=65535)
    #: 任务超时必须与 loop.max_expansions 联动（开发流程 5.4），见下方校验
    task_timeout_seconds: int = Field(default=600, ge=30, le=7200)

    # -------------------------------------------------------- 任务配额与限流
    #: 单次问题最大长度。FR-CHAT-001 业务规则：默认 4,000 字，**可配置**。
    #: 改这里就够了，接口层不重复写死上限（见 app/api/schemas.py 的说明）。
    chat_message_max_length: int = Field(default=4000, ge=1, le=100_000)
    #: 同时运行任务数上限（详细设计 19.5 的 app.max_running_tasks_per_user）。
    #: 计数必须跨实例一致，Phase 1.5 起改由 Redis 承载。
    max_running_tasks_per_user: int = Field(default=3, ge=1, le=100)
    #: 固定窗口限流配额（详细设计 19.3）
    rate_limit_create_per_minute: int = Field(default=10, ge=1, le=1000)
    rate_limit_status_per_minute: int = Field(default=120, ge=1, le=10_000)
    rate_limit_upload_per_hour: int = Field(default=10, ge=1, le=1000)

    # ------------------------------------------------------ 模型（TBC-04 未决）
    model_provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_api_key: str = Field(min_length=1)
    model_base_url: str | None = None
    model_timeout_seconds: int = Field(default=120, ge=5, le=600)

    embedding_model: str = Field(min_length=1)
    embedding_api_key: str = Field(min_length=1)
    embedding_dim: int = Field(default=1024, ge=64, le=8192)

    reranker_model: str | None = None
    reranker_enabled: bool = False

    # ------------------------------------------------------------------ 数据库
    database_url_agent: str = Field(min_length=1)
    database_url_business_ro: str = Field(min_length=1)
    db_pool_size: int = Field(default=10, ge=1, le=100)

    # ------------------------------------------------------------------ Redis
    redis_url: str = Field(min_length=1)
    redis_max_connections: int = Field(default=50, ge=1, le=500)

    # ------------------------------------------------------------------ 向量库
    milvus_uri: str = Field(min_length=1)
    rag_score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    rag_top_k: int = Field(default=20, ge=1, le=200)
    rag_rerank_top_n: int = Field(default=8, ge=1, le=100)
    rag_user_dict: str = "configs/rag_user_dict.txt"
    rag_stopwords: str = "configs/rag_stopwords.txt"

    # ------------------------------------------------------------------ 存储
    storage_backend: Literal["s3", "local"] = "local"
    storage_local_root: str = "data/storage"
    minio_endpoint: str | None = None
    minio_access_key: str | None = None
    minio_secret_key: str | None = None
    minio_bucket: str = "agent-knowledge"

    # ------------------------------------------------------------------ 认证
    jwt_secret: str = Field(min_length=16)
    #: 限定为 HMAC 系列：`none` 会让任何人手写空签名通过校验，
    #: 用 Literal 把算法钉死，比在 decode 处逐个排除更不容易漏。
    jwt_algorithm: Literal["HS256", "HS384", "HS512"] = "HS256"
    jwt_expire_minutes: int = Field(default=60, ge=1, le=10080)

    # ------------------------------------------------------------------ 可观测
    otel_service_name: str = Field(min_length=1)
    otel_enabled: bool = False
    otlp_endpoint: str | None = None
    otel_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    sentry_dsn: str | None = None

    # ------------------------------------------------------------------ 外部搜索
    search_enabled: bool = False
    search_provider: str | None = None
    search_api_key: str | None = None

    # ------------------------------------------------------------------ 循环预算
    loop: LoopSettings = Field(default_factory=LoopSettings)

    # ------------------------------------------------------------------ 校验
    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Settings:
        if self.storage_backend == "s3":
            missing = [
                name
                for name, value in (
                    ("MINIO_ENDPOINT", self.minio_endpoint),
                    ("MINIO_ACCESS_KEY", self.minio_access_key),
                    ("MINIO_SECRET_KEY", self.minio_secret_key),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"STORAGE_BACKEND=s3 时必须提供：{', '.join(missing)}")

        if self.otel_enabled and not self.otlp_endpoint:
            raise ValueError("OTEL_ENABLED=true 时必须提供 OTLP_ENDPOINT")

        if self.search_enabled and not self.search_provider:
            raise ValueError("SEARCH_ENABLED=true 时必须提供 SEARCH_PROVIDER")
        return self

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试中修改环境变量后需调用 `get_settings.cache_clear()`。"""
    return Settings()
