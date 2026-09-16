"""ModelGateway 的真连集成用例（详设 22.1：集成测试覆盖 ModelGateway）。

`make test` 下的契约测试用 `httpx.MockTransport` 覆盖了协议解析与重试编排；
这里补的是**只有真连才能验的那部分**：真实服务返回的字段名、用量结构、
以及「不同文本确实拿到不同向量」这类抓不到但会疼的差异。

前置：本机 Ollama 在跑（向量化）；云模型另需 `.env` 里填好密钥。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

from app.agent.prompts import SMOKE_PROMPT
from app.agent.schemas import SmokeAnswer
from app.core.config import Settings, get_settings
from app.infrastructure.model_gateway import HttpModelGateway

pytestmark = pytest.mark.integration

#: `.env.example` 里的占位值。它们会被原样复制进 `.env`，
#: 因此「有没有配真模型」必须能识别出它们，否则用例会拿占位值去打真实服务。
_PLACEHOLDER_KEYS = frozenset({"dev-placeholder", "test-key", "test-provider", ""})
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]")

#: 需要从测试环境变量里摘掉、好让 `.env` 的真实值生效的键。
_TEST_ENV_MODEL_KEYS = (
    "MODEL_PROVIDER",
    "MODEL_NAME",
    "MODEL_API_KEY",
    "MODEL_BASE_URL",
    "MODEL_TIMEOUT_SECONDS",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_DIM",
)


@pytest.fixture
def real_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """读**真实** `.env` 的配置，绕开 conftest 注入的测试环境变量。

    conftest 在导入期用 `os.environ.update` 把模型配置钉成了占位值
    （`MODEL_BASE_URL` 指向 `.invalid`）——那是对的，它保证单元测试永远
    不会打到真实服务。但本文件的全部意义就是打真实服务，所以得先把那几个
    键从环境里摘掉，让 pydantic-settings 回落到 `.env` 文件。

    退出时清一次 settings 缓存：`get_settings` 是 `lru_cache` 单例，
    不清理的话后续用例会拿到这份「真实配置」，把开发者的密钥带进单元测试。
    """
    for key in _TEST_ENV_MODEL_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield Settings()
    get_settings.cache_clear()


@pytest.fixture
async def gateway(real_settings: Settings) -> AsyncIterator[HttpModelGateway]:
    instance = HttpModelGateway(real_settings)
    try:
        yield instance
    finally:
        await instance.aclose()


def _has_real_cloud_model(settings: Settings) -> bool:
    """是否配了真实云模型。

    判据是「密钥不像占位值」且「端点不是本机」——不引入专门的测试开关：
    开关本身也会被忘记设，而这两条判据直接来自配置的实际内容。
    """
    if settings.model_api_key in _PLACEHOLDER_KEYS:
        return False
    return not any(host in settings.model_base_url for host in _LOCAL_HOSTS)


async def test_live_structured_invoke(gateway: HttpModelGateway, real_settings: Settings) -> None:
    """真打一次云模型，走完整的「发请求 → 解析 → Pydantic 校验」链路。"""
    if not _has_real_cloud_model(real_settings):
        pytest.skip(
            "未配置真实云模型（MODEL_API_KEY 仍是占位值，或 MODEL_BASE_URL 指向本机）；"
            "在 .env 里填好密钥后本用例即生效"
        )

    result = await gateway.invoke_structured(
        SMOKE_PROMPT, SmokeAnswer, question="请判断：1 + 1 是否等于 2？"
    )

    assert isinstance(result.value, SmokeAnswer)
    assert result.value.answer
    assert result.model == real_settings.model_name
    assert result.prompt_version == SMOKE_PROMPT.version
    # 真实服务必须回传用量：它是 token 成本统计的唯一来源
    assert result.usage.total_tokens > 0


async def test_live_embedding_matches_configured_dimension(
    gateway: HttpModelGateway, real_settings: Settings
) -> None:
    """真打一次本地 Ollama，验证维度与配置一致、且不同文本拿到不同向量。

    第二条断言不是凑数的：若 `data` 的顺序没按 `index` 归位，
    两条文本可能拿到同一个向量，而**维度断言照样会通过**。
    """
    vectors = await gateway.embed(["华东地区 Q3 净销售额", "渠道折扣政策"])

    assert len(vectors) == 2
    assert len(vectors[0]) == real_settings.embedding_dim
    assert vectors[0] != vectors[1]
