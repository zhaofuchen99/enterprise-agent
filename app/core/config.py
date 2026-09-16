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


class ModelTuning(BaseModel):
    """模型调用参数（开发流程 6.5 / 详细设计 9.4 的错误分类表）。

    **字段名为什么是 `model_tuning` 而不是 `model`**：同 `RedisTuning` 的理由——
    本类已有扁平的 `model_name` / `model_api_key`，再挂一个名为 `model` 的嵌套模型，
    环境变量里 `MODEL_NAME` 与 `MODEL__NAME` 会指向两个不同的东西，
    而 `extra="ignore"` 会把写错的那一个**静默丢掉**。

    代价是环境变量前缀变长：`MODEL_TUNING__MAX_TOKENS=8192`，
    **不是** `MODEL__MAX_TOKENS`（后者不会报错，只会静默走默认值）。

    两项重试预算**相互独立、不可借用**（与 `loop` 的四类预算同一条纪律）：
    传输失败不消耗修复次数，反之亦然。详见 `model_gateway.py` 的说明。
    """

    #: TRANSIENT 类（429 / 5xx / 连接失败 / 超时）的退避重试次数
    max_transport_retries: int = Field(default=1, ge=0, le=3)
    #: VALIDATION 类（JSON 解析或 Schema 校验失败）的修复重试次数
    max_repair_retries: int = Field(default=1, ge=0, le=3)
    #: 指数退避的基数。第 n 次重试前等待 `base * 2**(n-1)` 秒
    backoff_base_seconds: float = Field(default=0.5, gt=0.0, le=10.0)
    #: 单次响应的输出上限。**不是可选的**——DeepSeek 的 JSON 模式明确要求设置它，
    #: 否则返回的 JSON 可能被中途截断，表现为「解析失败」而不是「输出太长」
    max_tokens: int = Field(default=4096, ge=256, le=32_768)
    #: 是否启用思考模式。**默认关闭**，理由见 `config.py` 模型段的注释
    thinking_enabled: bool = False


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

    # ------------------------------------- 模型（TBC-04 已决议，见 CLAUDE.md）
    #: 主聊天模型。当前为 `deepseek-flash`（`deepseek-chat` 与 `deepseek-v4-pro`
    #: 均已停用/被路由，实际只剩这一个可用模型名）。
    model_provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_api_key: str = Field(min_length=1)
    #: OpenAI 兼容端点。云模型与本地 Ollama 走**同一套协议**，
    #: 因此网关只有一个实现，换 provider 的成本是一个配置值而不是一段 `if`。
    #: **必填**：本项目的 provider 都不是 OpenAI 官方，没有可用的默认端点；
    #: 留空会让错误推迟到第一个任务跑起来之后，那时看到的是「任务失败」
    #: 而不是「配置没填」，排查成本差一个数量级。
    model_base_url: str = Field(min_length=1)
    #: 单次模型调用超时。详细设计 19.5 定的是 45s；Phase 3 前这里是 120s，
    #: 属代码与文档不一致，已按文档对齐
    model_timeout_seconds: int = Field(default=45, ge=5, le=600)

    #: Embedding 走本地 Ollama（bge-m3，1024 维），**base_url 与聊天模型不同**，
    #: 故必须单独配置。不配则回落到 `model_base_url`
    embedding_model: str = Field(min_length=1)
    embedding_api_key: str = Field(min_length=1)
    embedding_base_url: str | None = None
    embedding_dim: int = Field(default=1024, ge=64, le=8192)

    reranker_model: str | None = None
    reranker_enabled: bool = False

    #: **默认关闭思考**（`MODEL__THINKING_ENABLED=false`）。DeepSeek V4 的思考模式
    #: 默认开启且 effort=high，而本项目是**节点密集调用 + 全是结构化输出**：
    #: 思考态下 `temperature` 被忽略、思考 token 按输出计费，且带 `tools` 时
    #: 历史轮次的 `reasoning_content` 必须原样回传否则 400。
    #: 这些代价换不来结构化抽取的准确率，需要时再按节点开。
    model_tuning: ModelTuning = Field(default_factory=ModelTuning)

    # ------------------------------------------------------------------ 数据库
    database_url_agent: str = Field(min_length=1)
    database_url_business_ro: str = Field(min_length=1)
    #: 业务库的**写**连接。**只有 `scripts/business_seed.py` 会用它**——
    #: API / Worker 连业务库一律走 `DATABASE_URL_BUSINESS_RO`，这是
    #: 「SQL 安全不依赖模型」的底座（详细设计 19.2 第 5 条）。
    #:
    #: 可选且无默认值：种子脚本在 `make up` 之后按需运行，
    #: 而 `make test` / API 启动都不该因为它缺失而失败。缺失时脚本会
    #: 给出明确提示并退出（见 `_rw_url`），不会退化成静默跳过。
    database_url_business_rw: str | None = None
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

        if self.reranker_enabled and not self.reranker_model:
            raise ValueError("RERANKER_ENABLED=true 时必须提供 RERANKER_MODEL")

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
