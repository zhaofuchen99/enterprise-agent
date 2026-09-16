"""测试全局夹具。

原则：单元测试不依赖真实外部组件；需要真实 Redis / MySQL / Milvus 的用例
必须打 `@pytest.mark.integration` 标记（见 pyproject.toml 的 markers）。
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Iterator

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

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
    # 长度 >= 32 字节：HS256 的密钥短于摘要长度会被 PyJWT 警告（RFC 7518 3.2）
    "JWT_SECRET": "test-secret-not-for-production-but-long-enough",
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


# --------------------------------------------------------------------- 应用装配
@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


def build_app(fake_redis: fakeredis.aioredis.FakeRedis) -> FastAPI:
    """绕开 lifespan 构造应用。

    lifespan 会去连真实 Redis，单元测试里不能跑；改为直接注入 fakeredis，
    并显式调用 `wire_dependencies`——**与生产走的是同一个装配函数**，
    这样测试覆盖到的依赖图不会和实际运行的那份分家。
    """
    from app.core.config import get_settings
    from app.main import create_app, wire_dependencies

    application = create_app()
    application.state.redis = fake_redis
    wire_dependencies(application, get_settings())
    return application


def build_client(app: FastAPI) -> AsyncClient:
    # raise_app_exceptions=False：让未捕获异常也走全局异常处理器，
    # 否则 httpx 会把异常直接抛进用例，500 分支根本测不到。
    return AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    )


@pytest.fixture
def app(fake_redis: fakeredis.aioredis.FakeRedis, _settings_env: None) -> FastAPI:
    return build_app(fake_redis)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with build_client(app) as ac:
        yield ac


@pytest.fixture
def make_app(
    fake_redis: fakeredis.aioredis.FakeRedis,
    _settings_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., FastAPI]]:
    """按环境变量覆盖后构造应用。

    用于验证配置驱动的行为（限流配额、并发上限、问题长度）。
    `get_settings` 是 lru_cache 单例，改完环境变量必须手动清缓存，
    否则拿到的是上一个用例的配置——这类串扰排查起来非常费时间。
    """
    from app.core.config import get_settings

    def _make(**env: str) -> FastAPI:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return build_app(fake_redis)

    yield _make
    get_settings.cache_clear()
