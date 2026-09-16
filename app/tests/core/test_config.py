"""配置校验测试（开发流程 5.4）。

配置通过**环境变量**加载，因此这里也用环境变量构造，而不是直接传 init kwargs——
否则测的就不是真实的加载路径（嵌套配置 `LOOP__*` 只在环境变量下生效）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings

_MINIMAL: dict[str, str] = {
    "MODEL_PROVIDER": "p",
    "MODEL_NAME": "m",
    "MODEL_API_KEY": "k",
    "MODEL_BASE_URL": "https://example.invalid/v1",
    "EMBEDDING_MODEL": "e",
    "EMBEDDING_API_KEY": "k",
    "DATABASE_URL_AGENT": "mysql+asyncmy://a@127.0.0.1:3306/agent",
    "DATABASE_URL_BUSINESS_RO": "mysql+asyncmy://b@127.0.0.1:3307/business",
    "MILVUS_URI": "http://127.0.0.1:19530",
    "REDIS_URL": "redis://127.0.0.1:6379/0",
    "JWT_SECRET": "a-sufficiently-long-secret",
}


@pytest.fixture(autouse=True)
def _isolated(clean_env: None) -> None:
    """每个用例都从干净环境开始，避免用例之间通过 os.environ 串味。"""


def _load(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    for key, value in {**_MINIMAL, **overrides}.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_minimal_env_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load(monkeypatch)
    assert settings.app_env == "dev"
    assert settings.storage_backend == "local"
    assert settings.loop.max_total_steps == 24


@pytest.mark.parametrize("missing", sorted(_MINIMAL))
def test_missing_required_field_fails(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """缺失必需项必须启动失败，而不是用不安全的默认值顶上（开发流程 5.4）。"""
    for key, value in _MINIMAL.items():
        if key != missing:
            monkeypatch.setenv(key, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_short_jwt_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _load(monkeypatch, JWT_SECRET="short")


def test_s3_backend_requires_minio_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="MINIO_ENDPOINT"):
        _load(monkeypatch, STORAGE_BACKEND="s3")


def test_s3_backend_accepts_complete_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load(
        monkeypatch,
        STORAGE_BACKEND="s3",
        MINIO_ENDPOINT="http://127.0.0.1:9000",
        MINIO_ACCESS_KEY="ak",
        MINIO_SECRET_KEY="sk",
    )
    assert settings.storage_backend == "s3"


def test_otel_enabled_requires_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="OTLP_ENDPOINT"):
        _load(monkeypatch, OTEL_ENABLED="true")


def test_search_enabled_requires_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="SEARCH_PROVIDER"):
        _load(monkeypatch, SEARCH_ENABLED="true")


def test_reranker_enabled_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError, match="RERANKER_MODEL"):
        _load(monkeypatch, RERANKER_ENABLED="true")


def test_model_timeout_follows_design_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    """详细设计 19.5 定的是 45s；Phase 3 之前这里是 120s，属代码与文档不一致。"""
    assert _load(monkeypatch).model_timeout_seconds == 45


def test_thinking_mode_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeepSeek V4 服务端默认开启思考，必须由我们显式关掉。

    这是「默认值错了但不会报错」的典型：开着也能跑，只是更贵、更慢、
    且 `temperature` 静默失效。因此把默认值本身钉成断言。
    """
    assert _load(monkeypatch).model_tuning.thinking_enabled is False


def test_model_nested_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """前缀是 `MODEL_TUNING__` 而不是 `MODEL__`——嵌套字段名叫 `model_tuning`。

    写成 `MODEL__MAX_TOKENS` 不会报错，只会**静默走默认值**（`extra="ignore"`），
    是这套配置里最容易踩且最难发现的一种。因此把正确前缀钉成断言。
    """
    settings = _load(monkeypatch, MODEL_TUNING__MAX_REPAIR_RETRIES="2")
    assert settings.model_tuning.max_repair_retries == 2


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("MODEL_TUNING__MAX_TRANSPORT_RETRIES", "9"),
        ("MODEL_TUNING__MAX_REPAIR_RETRIES", "-1"),
        ("MODEL_TUNING__BACKOFF_BASE_SECONDS", "0"),
        ("MODEL_TUNING__MAX_TOKENS", "1"),
    ],
)
def test_model_tuning_bounds_enforced(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    """重试上限必须有界——无界重试会让一次失败的任务卡满整个 task_timeout。"""
    with pytest.raises(ValidationError):
        _load(monkeypatch, **{key: value})


def test_loop_nested_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """嵌套配置用双下划线覆盖，这是 `.env` 与 Compose 里的写法，必须验证。"""
    settings = _load(monkeypatch, LOOP__MAX_TOTAL_STEPS="30")
    assert settings.loop.max_total_steps == 30


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LOOP__MAX_EXPANSIONS", "99"),
        ("LOOP__DUPLICATE_SIMILARITY_THRESHOLD", "1.5"),
        ("LOOP__MIN_STEP_SECONDS", "-1"),
    ],
)
def test_loop_budget_bounds_enforced(monkeypatch: pytest.MonkeyPatch, key: str, value: str) -> None:
    """四类预算必须有上下限，配置写错不能让循环失去收敛保证。"""
    with pytest.raises(ValidationError):
        _load(monkeypatch, **{key: value})
