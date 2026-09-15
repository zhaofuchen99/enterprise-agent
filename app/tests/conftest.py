"""测试全局夹具。

原则：单元测试不依赖真实外部组件；需要真实 Redis / MySQL / Milvus 的用例
必须打 `@pytest.mark.integration` 标记（见 pyproject.toml 的 markers）。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

#: 测试用最小环境变量。
_TEST_ENV: dict[str, str] = {
    "APP_ENV": "test",
    "LOG_LEVEL": "WARNING",
    "MODEL_PROVIDER": "test-provider",
    "MODEL_NAME": "test-model",
    "MODEL_API_KEY": "test-key",
    "EMBEDDING_MODEL": "test-embedding",
    "EMBEDDING_API_KEY": "test-key",
    "DATABASE_URL_AGENT": "mysql+asyncmy://agent:agent@localhost:3306/agent_test",
    "DATABASE_URL_BUSINESS_RO": "mysql+asyncmy://ro:ro@localhost:3307/business_test",
    "MILVUS_URI": "http://localhost:19530",
    "REDIS_URL": "redis://localhost:6381/15",
    "JWT_SECRET": "test-secret-not-for-production",
    "STORAGE_BACKEND": "local",
    "OTEL_SERVICE_NAME": "api",
}

# 必须在任何测试模块导入 app.* 之前写入 os.environ：
# app/main.py 在**模块级**构造 FastAPI 实例，那一刻 Settings 的必需项校验就会执行，
# 而夹具要到用例运行时才生效。conftest 的导入早于测试模块的导入，这里是唯一的时机。
os.environ.update(_TEST_ENV)


def _clear_settings_cache() -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _settings_env() -> Iterator[None]:
    _clear_settings_cache()
    yield
    _clear_settings_cache()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """清空测试环境变量。

    用于否定用例（「缺失必需项必须启动失败」）——如果 os.environ 里留着值，
    缺失项会被兜住，断言永远是绿的。
    """
    for key in _TEST_ENV:
        monkeypatch.delenv(key, raising=False)
    _clear_settings_cache()
    yield
    _clear_settings_cache()


@pytest.fixture
def settings(_settings_env: None) -> object:
    from app.core.config import get_settings

    return get_settings()
