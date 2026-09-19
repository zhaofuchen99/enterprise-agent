"""SSE 订阅令牌（详细设计 17.3.1）。

三条性质各自对应一个真实风险，逐条钉住：

| 性质 | 不成立时的症状 |
|---|---|
| **一次性** | 日志里翻到一条旧 URL 就能一直读别人的任务事件流 |
| **短时效** | 泄露的窗口从一分钟变成一个永久凭证 |
| **绑定 task_id** | 令牌变成"能读任意任务"的通行证 |

拒绝时**必须不区分原因**（过期 / 已用 / 签名错都是同一个 401）：
能区分的报错等于告诉探测者"这个令牌曾经有效"。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import jwt
import pytest

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.services.stream_token import StreamTokenService

_USER = "usr_0000000000000000000001"
_TASK = "tsk_0000000000000000000001"


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def settings() -> Settings:
    from app.core.config import get_settings

    return get_settings()


def _service(redis: fakeredis.aioredis.FakeRedis, settings: Settings) -> StreamTokenService:
    return StreamTokenService(redis=redis, settings=settings)


async def test_issued_token_can_be_consumed_once(
    redis: fakeredis.aioredis.FakeRedis, settings: Settings
) -> None:
    service = _service(redis, settings)

    issued = await service.issue(user_id=_USER, task_id=_TASK)
    claims = await service.consume(issued.token)

    assert claims.user_id == _USER
    assert claims.task_id == _TASK
    assert issued.expires_in == settings.sse_tuning.stream_token_ttl_seconds
    # 令牌只该出现在查询参数里，而 URL 由服务端拼（前端拼的话参数名会成为
    # 一个没有文档的约定，改一次客户端就静默连不上）
    assert issued.stream_url == f"/api/agent/tasks/{_TASK}/stream?token={issued.token}"


async def test_a_token_cannot_be_used_twice(
    redis: fakeredis.aioredis.FakeRedis, settings: Settings
) -> None:
    """**一次性**是这套设计的全部意义：泄露的 URL 多半已经被用掉了。"""
    service = _service(redis, settings)
    issued = await service.issue(user_id=_USER, task_id=_TASK)

    await service.consume(issued.token)

    with pytest.raises(AgentError) as failure:
        await service.consume(issued.token)
    assert failure.value.code is ErrorCode.AUTHENTICATION_REQUIRED


async def test_an_expired_token_is_rejected(
    redis: fakeredis.aioredis.FakeRedis, settings: Settings
) -> None:
    """短时效：过期之后连签名都验不过（`exp` 由 PyJWT 校验）。

    这里直接造一个**已经过期**的令牌，而不是 sleep 等它过期——
    等一秒的用例在 CI 上会变成偶发失败。
    """
    service = _service(redis, settings)
    past = datetime.now(UTC) - timedelta(seconds=settings.sse_tuning.stream_token_ttl_seconds + 1)
    expired = jwt.encode(
        {
            "sub": _USER,
            "task_id": _TASK,
            "jti": "stk_expired",
            "iat": int((past - timedelta(seconds=1)).timestamp()),
            "exp": int(past.timestamp()),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AgentError) as failure:
        await service.consume(expired)
    assert failure.value.code is ErrorCode.AUTHENTICATION_REQUIRED


async def test_a_token_without_a_marker_is_rejected(
    redis: fakeredis.aioredis.FakeRedis, settings: Settings
) -> None:
    """签名有效、`exp` 未到，但**消费标记不在了**（Redis 里没有 / 已过期）。

    这一条覆盖的是"令牌本身没过期、标记先过期"的那种错配——
    它与"已经用过"对外必须长得一模一样。
    """
    service = _service(redis, settings)
    token = jwt.encode(
        {
            "sub": _USER,
            "task_id": _TASK,
            "jti": "stk_no_marker",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(seconds=60)).timestamp()),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AgentError) as failure:
        await service.consume(token)
    assert failure.value.code is ErrorCode.AUTHENTICATION_REQUIRED


async def test_a_tampered_token_is_rejected(settings: Settings) -> None:
    """签名不对一律 401。**不区分**签名错与过期（见模块 docstring）。"""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    service = _service(redis, settings)
    issued = await service.issue(user_id=_USER, task_id=_TASK)

    with pytest.raises(AgentError) as failure:
        await service.consume(issued.token + "x")
    assert failure.value.code is ErrorCode.AUTHENTICATION_REQUIRED
    await redis.aclose()


async def test_the_token_carries_the_task_binding(
    redis: fakeredis.aioredis.FakeRedis, settings: Settings
) -> None:
    """令牌绑定的是**签发时那个任务**——调用方必须拿它跟路径里的比对。

    不比对的话，它就是一张"能读任意任务事件流"的通行证；而这张通行证
    是从"自己的任务"换来的，看不出任何异常。
    """
    service = _service(redis, settings)
    issued = await service.issue(user_id=_USER, task_id=_TASK)

    claims = await service.consume(issued.token)

    assert claims.task_id == _TASK
    assert claims.expires_at > datetime.now(UTC)
