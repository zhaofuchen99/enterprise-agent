"""带版本号的缓存 helper（开发流程 6.3 施工项 3）。

**放在 `infrastructure/` 而不是文档写的 `core/cache.py`**：缓存键格式
`cache:{name}:{version}` 写在详细设计 4.4「Redis 键命名与使用边界」里，
本质是一个 Redis 使用约定；而 `app/core/` 目前不含任何外部服务客户端
（config 是 pydantic、security 是 pyjwt，都是纯库）。把一个需要 Redis 的
helper 放进去，会让 core 从「横切底座」变成「又一个依赖服务的地方」。
已回写开发流程 6.3 的施工项清单。

**为什么缓存读出来还要过一次 Pydantic 校验**：版本号只覆盖
`schema_catalog.version` 这类**数据**版本，覆盖不了**代码**版本。
发布新版本改了模型字段、而数据版本号没变时，旧键会被新代码读到，
不校验的话解析出来的就是一个缺字段的对象，一路带到下游才炸。
校验失败按「未命中」处理并回源，让这类不一致自动消失。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

import redis.asyncio as aioredis
from pydantic import BaseModel, ValidationError

from app.infrastructure.redis import RedisKey, dump_json, load_json

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class _RedisLike(Protocol):
    """只用到这两个方法，便于注入 fakeredis 与测试替身。

    签名写成同步返回 `Any` 而不是 `async def`：redis-py 把命令的返回类型标成
    `Awaitable[T] | T`（同步与异步客户端共用一份签名），与 `Coroutine` 对不上，
    写成 `async def` 会让真实的客户端类型**无法**满足这个协议——
    那正是我们最想注入的那一个。
    """

    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: str, *, ex: int | None = ...) -> Any: ...


class VersionedCache:
    """`cache:{name}:{version}` 形态的只读缓存。

    **没有 `invalidate`**：这是刻意的。版本号进了键名，发布新版本时新键
    自然生效、旧键按 TTL 过期，不需要任何主动失效逻辑。留一个 `delete`
    接口只会诱使调用方回到「改完数据记得清缓存」那条路上——
    那不是设计，那是约定，约定迟早会被忘记。详见详细设计 4.4 纪律 1。
    """

    def __init__(self, redis: _RedisLike, *, default_ttl_seconds: int = 3600) -> None:
        self._redis = redis
        self._default_ttl = default_ttl_seconds

    async def get(self, name: str, version: str, *, model: type[TModel]) -> TModel | None:
        """读缓存。未命中、解析失败、Redis 不可用一律返回 None。"""
        key = RedisKey.cache(name, version)
        try:
            raw = await self._redis.get(key)
        except (aioredis.RedisError, OSError) as exc:
            # 缓存不可用不是故障，是「没缓存」。降级为回源，不让它中断请求。
            logger.warning("缓存读取失败，按未命中处理：%s", type(exc).__name__)
            return None

        payload = load_json(raw)
        if payload is None:
            return None
        try:
            return model.model_validate(payload)
        except ValidationError:
            # 代码版本与数据版本不一致（见模块 docstring）。按未命中处理。
            logger.warning("缓存内容与当前模型不兼容，按未命中处理：%s", key)
            return None

    async def set(
        self,
        name: str,
        version: str,
        value: BaseModel | dict[str, Any] | list[Any],
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """写缓存。失败只记日志——缓存写不进去不该让一次正常请求变成 500。"""
        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        key = RedisKey.cache(name, version)
        try:
            await self._redis.set(key, dump_json(payload), ex=ttl_seconds or self._default_ttl)
        except (aioredis.RedisError, OSError) as exc:
            logger.warning("缓存写入失败，已忽略：%s", type(exc).__name__)

    async def get_or_set(
        self,
        name: str,
        version: str,
        loader: Callable[[], Awaitable[TModel]],
        *,
        model: type[TModel],
        ttl_seconds: int | None = None,
    ) -> TModel:
        """未命中则调用 `loader` 回源并写回。`loader` 的异常**原样抛出**。

        不吞 `loader` 的异常：回源失败是真实故障（比如库连不上），
        把它伪装成缓存问题会让排查方向整个跑偏。
        """
        cached = await self.get(name, version, model=model)
        if cached is not None:
            return cached
        value = await loader()
        await self.set(name, version, value, ttl_seconds=ttl_seconds)
        return value
