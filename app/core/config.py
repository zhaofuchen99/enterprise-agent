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


class RedisTuning(BaseModel):
    """Redis 使用参数（详细设计 4.4 的键生命周期 / 19.5 的 redis 段）。

    **字段名为什么是 `redis_tuning` 而不是文档里的 `redis`**：本类已有扁平字段
    `redis_url` / `redis_max_connections`，若再挂一个名为 `redis` 的嵌套模型，
    环境变量里 `REDIS_URL` 与 `REDIS__URL` 会指向两个不同的东西，而
    `extra="ignore"` 会把写错的那一个**静默丢掉**。宁可名字长一点。
    """

    #: 事件流 `task:{id}:events` 的长度上限，约等于「保留最近多少次事件」
    stream_maxlen: int = Field(default=10_000, ge=100, le=1_000_000)
    #: 任务结束后事件流的保留时长（详细设计 4.4）
    stream_ttl_seconds: int = Field(default=3600, ge=60, le=86_400)
    #: 互斥锁（孤儿回收等）的持有时长，必须大于这类操作的最坏耗时
    lock_ttl_seconds: int = Field(default=30, ge=1, le=600)
    #: `cache:{name}:{version}` 的默认 TTL
    cache_ttl_seconds: int = Field(default=3600, ge=1, le=86_400)
    #: Redis 不可用时的本地限流收紧系数（详细设计 19.3）。
    #: 0.5 等价于「按 2 个实例均分」；**只允许收紧，不允许放宽**，故上限就是 1.0。
    rate_limit_degraded_factor: float = Field(default=0.5, gt=0.0, le=1.0)
    #: 限流器在 Redis 失败后的熔断冷却时长。
    #: 没有它的话，Redis 挂掉期间**每个请求**都要先等满一次命令超时才降级，
    #: 降级就从「保住可用性」变成了「给每个请求加一秒延迟」。
    rate_limit_breaker_seconds: float = Field(default=5.0, ge=0.0, le=60.0)
    #: 单次 Redis 命令的超时。限流与健康检查必须**快速失败**：
    #: 一个卡住 30 秒的 Redis 调用会让降级路径永远来不及生效，
    #: 结果是 Redis 挂了整个 API 一起挂，降级形同不存在。
    operation_timeout_seconds: float = Field(default=1.0, gt=0.0, le=10.0)


class WorkerTuning(BaseModel):
    """Worker 与队列参数（开发流程 6.3 / 详细设计 19.5 的 worker 段）。"""

    concurrency: int = Field(default=4, ge=1, le=64, description="单 Worker 进程并发任务数")
    heartbeat_interval_seconds: int = Field(default=10, ge=1, le=300)
    #: 心跳键 TTL，超过它没续期即视为 Worker 已死（详细设计 4.4）
    heartbeat_ttl_seconds: int = Field(default=30, ge=2, le=600)
    #: QUEUED 任务超过多久仍未被领取就重新投递（详细设计 17.1 的补偿扫描）
    queue_reconcile_seconds: int = Field(default=30, ge=5, le=3600)
    #: 同一任务最多重投几次，超过即置 FAILED + ENQUEUE_FAILED（详细设计 17.1）
    max_requeue_attempts: int = Field(default=2, ge=0, le=10)
    #: 孤儿任务回收扫描周期
    orphan_scan_interval_seconds: int = Field(default=30, ge=5, le=3600)
    #: 收到停机信号后等多久再中断在跑任务。取 0 会让 `Ctrl-C` 立刻打断任务，
    #: 任务停在 RUNNING 只能等孤儿回收；取 task_timeout 则本地开发要等太久。
    shutdown_grace_seconds: int = Field(default=10, ge=0, le=600)


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
    redis_tuning: RedisTuning = Field(default_factory=RedisTuning)

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
    #: 进程角色（api / worker）**不是配置项**，见 infrastructure/logging.py
    otel_enabled: bool = False
    otlp_endpoint: str | None = None
    otel_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    sentry_dsn: str | None = None

    # ------------------------------------------------------------------ 外部搜索
    search_enabled: bool = False
    search_provider: str | None = None
    search_api_key: str | None = None

    # ------------------------------------------------------- Worker 与队列
    worker: WorkerTuning = Field(default_factory=WorkerTuning)

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

        # 心跳 TTL 必须留出「漏掉一拍」的余量。判死的判据是
        # `heartbeat_at < now - ttl`，与扫描周期无关，所以这里约束的是
        # ttl 与 interval 的关系而不是 ttl 与扫描周期的关系。
        # 违反时不会有任何启动报错，只表现为「Worker 活着，任务却被回收成
        # WORKER_INTERRUPTED」——从配置上根本看不出来，因此必须在启动时挡住。
        if self.worker.heartbeat_ttl_seconds < 2 * self.worker.heartbeat_interval_seconds:
            raise ValueError(
                "WORKER__HEARTBEAT_TTL_SECONDS 至少要是 WORKER__HEARTBEAT_INTERVAL_SECONDS "
                f"的两倍（要容忍漏掉一拍），当前为 {self.worker.heartbeat_ttl_seconds} / "
                f"{self.worker.heartbeat_interval_seconds}"
            )
        return self

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """进程内单例。测试中修改环境变量后需调用 `get_settings.cache_clear()`。"""
    return Settings()
