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

from app.infrastructure.queue import JobQueue

#: 测试用最小环境变量。
_TEST_ENV: dict[str, str] = {
    "APP_ENV": "test",
    "LOG_LEVEL": "WARNING",
    "MODEL_PROVIDER": "test-provider",
    "MODEL_NAME": "test-model",
    "MODEL_API_KEY": "test-key",
    # .invalid 是 RFC 2606 保留的顶级域名，**保证不可解析**：
    # 万一哪天单元测试真的发起了请求，这里会立刻失败而不是打到一个真实服务上
    "MODEL_BASE_URL": "https://model.invalid/v1",
    "EMBEDDING_MODEL": "test-embedding",
    "EMBEDDING_API_KEY": "test-key",
    "EMBEDDING_BASE_URL": "https://embedding.invalid/v1",
    # 指向**独立的测试库** agent_test，不是开发库 agent：
    # 契约测试会反复建表与清数据，指向开发库等于把本地数据当消耗品。
    # 凭据与 docker-compose.dev.yml 里的 agent-mysql 一致。
    "DATABASE_URL_AGENT": "mysql+asyncmy://agent:agent_pw@127.0.0.1:3308/agent_test",
    "DATABASE_URL_BUSINESS_RO": "mysql+asyncmy://readonly:readonly_pw@127.0.0.1:3307/business",
    "QDRANT_URL": "http://localhost:6333",
    "REDIS_URL": "redis://localhost:6381/15",
    # 长度 >= 32 字节：HS256 的密钥短于摘要长度会被 PyJWT 警告（RFC 7518 3.2）
    "JWT_SECRET": "test-secret-not-for-production-but-long-enough",
    "STORAGE_BACKEND": "local",
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


def build_app(fake_redis: fakeredis.aioredis.FakeRedis, queue: JobQueue | None = None) -> FastAPI:
    """绕开 lifespan 构造应用。

    lifespan 会去连真实 Redis、MySQL 与 arq，单元测试里不能跑；改为直接注入
    fakeredis、内存仓储与队列替身，并显式调用 `wire_dependencies`——
    **与生产走的是同一个装配函数**，这样测试覆盖到的依赖图不会和实际运行的那份分家。

    队列默认用 `FakeJobQueue`（arq 需要真实连接）。需要断言投递行为的用例
    可以自己传一个进来，之后从 `app.state.job_queue` 取回。

    **仓储注入内存实现，不连 MySQL**（Phase 2）：`make test` 的契约是
    「不依赖任何外部组件」。仓储的 MySQL 实现由 `make test-integration`
    下的契约测试覆盖——**同一批断言跑两个实现**，因此这里换成内存实现
    不会让 SQL 实现失去验证。
    """
    from app.core.config import get_settings
    from app.main import create_app, wire_dependencies
    from app.tests.fakes import (
        FakeJobQueue,
        FakeModelGateway,
        FakeSessionFactory,
        build_memory_repositories,
    )

    settings = get_settings()
    application = create_app()
    application.state.redis = fake_redis
    # 就绪探针会拿 sessions 执行 SELECT 1；注入替身而不是真实工厂，
    # 使探针的调用形状仍然被覆盖（503 分支见 integration 用例）。
    application.state.sessions = FakeSessionFactory()
    wire_dependencies(
        application,
        settings,
        queue=queue or FakeJobQueue(),
        # 模型网关注入替身：真实实现会建两个指向 `.invalid` 的 httpx 连接池，
        # 没人关它。替身的行为契约由 test_model_gateway_contract.py 的
        # 参数化用例与真实实现对齐。
        gateway=FakeModelGateway(),
        repositories=build_memory_repositories(settings),
    )
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
