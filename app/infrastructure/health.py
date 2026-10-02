"""依赖探针：`/health/ready` 的输入（详设 20.3、需求 8.2）。

## 为什么单独一个模块

`/health/ready` 住在 `app/main.py`，而它是 **L1 边界**——不许 import
`tools.*` / `agent.*`。要探的 Qdrant、业务库、对象存储都住在 `app/infrastructure/`
（L1 允许），所以**装配放这里，`main.py` 只消费**。探针由 lifespan 建一次、
挂到 `app.state`，不在每次请求里新建（ready 是会被高频打的）。

## 判 503 的唯一标准是「这个实例完全不能用」

今天只有 MySQL 满足：它是任务、会话、用户的**唯一权威**，连不上时登录、
建任务、查状态全部失败，且没有任何降级路径。

Qdrant / 业务库 / 对象存储都只影响**部分功能**——状态查询、已缓存的结果读取
仍然可用——所以它们失败一律 **200 + `degraded`**，绝不 503。报 not_ready 会让
编排层摘掉实例，而摘掉它恰恰是最不该做的事（同详设 19.3 的限流降级、
CLAUDE.md 约定 127 的队列积压：把"能力下降"报成"实例不可用"就会被摘掉）。

**队列积压阈值刻意不进这里**（约定 127 明文）：积压是"消费能力下降"，
而 Worker 全挂时才积压——那时正是要保住 API 实例的时候。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from sqlalchemy import text

from app.core.config import Settings
from app.infrastructure.db import create_engine
from app.infrastructure.storage import build_object_storage
from app.infrastructure.vector_store import build_vector_store

#: 探针结果的缓存时长。`/health/ready` 被编排层每隔几秒打一次，而依赖挂掉时
#: 每一次都要等满超时——不缓存的话「探测本身」变成一种负载，还让 ready 从
#: 毫秒级变成秒级。5 秒远小于编排层的摘除判定周期，不会让状态显得过时。
_PROBE_CACHE_SECONDS: Final[float] = 5.0

#: 对象存储探针 HEAD 的那个键。**故意用一个不存在的键**：我们要的是
#: "端点答不答得上来"，而不是"某个对象在不在"——用一个真实存在的键会让
#: 探针依赖"那份文件还在"，而那与可用性是两件事。
_STORAGE_PROBE_KEY: Final[str] = "health/probe"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """一条探针的结果。

    `error` 的形状与 `/health/ready` 原有的 `mysql(TypeError)` **保持一致**：
    读的人要能一眼看出**是哪一层、什么异常**，而不是一句"依赖不可用"。
    """

    name: str
    ok: bool
    latency_ms: int
    error: str | None = None


class DependencyProbe(Protocol):
    name: str

    async def probe(self) -> ProbeResult: ...

    async def aclose(self) -> None: ...


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


class _QdrantProbe:
    """Qdrant 连得上，且 collection 在。

    用 `VectorStore.count()` 而不是自己新建一个 Qdrant 客户端：公开接口里
    它是**唯一一次"真的问了一次库、且不以某条查询为前提"的调用**。它同时验了
    「连得上」与「collection 在」——只验前者的话，一个 collection 还没建的库
    也会被报成健康，而那正是"检索永远返回空"的现场。
    """

    name = "qdrant"

    def __init__(self, settings: Settings) -> None:
        self._store = build_vector_store(settings)

    async def probe(self) -> ProbeResult:
        started = time.monotonic()
        try:
            await self._store.count()
        except Exception as exc:
            return ProbeResult(self.name, False, _ms(started), f"{self.name}({type(exc).__name__})")
        return ProbeResult(self.name, True, _ms(started))

    async def aclose(self) -> None:
        await self._store.aclose()


class _BusinessDbProbe:
    """业务库连得上。**只 `SELECT 1`，不碰任何业务表。**

    引擎在这里建一次、复用——`create_engine` 是惰性的（不立即连接），
    而池化的连接在依赖恢复后能自己重建，所以没必要每次探测都换一个引擎。
    """

    name = "business_db"

    def __init__(self, settings: Settings) -> None:
        self._engine = create_engine(settings, url=settings.database_url_business_ro)

    async def probe(self) -> ProbeResult:
        started = time.monotonic()
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            return ProbeResult(self.name, False, _ms(started), f"{self.name}({type(exc).__name__})")
        return ProbeResult(self.name, True, _ms(started))

    async def aclose(self) -> None:
        await self._engine.dispose()


class _ObjectStorageProbe:
    """对象存储可达。两种后端**探的不是同一件事**，如实分开写：

    - `s3`：HEAD 一个不存在的小键。端点活着时 `head_object` 会以 404 回答
      （`S3ObjectStorage.exists` 只吞 404，其余照抛），所以"能返回就算可达"；
      端点是死的时候会抛连接错误。
    - `local`：根目录在不在。没有网络可打，这时探的是**配置有没有指到一个
      真的目录**——它是这一类问题里最容易在现场暴露的那一种。
    """

    name = "object_storage"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._storage: Any = None

    async def probe(self) -> ProbeResult:
        started = time.monotonic()
        try:
            if self._settings.storage_backend == "s3":
                self._storage = self._storage or build_object_storage(self._settings)
                await self._storage.exists(_STORAGE_PROBE_KEY)
            # ASYNC240 在这一行是**放行的**：一次本地 `stat` 是微秒级，
            # 套 `to_thread` 的调度开销比它本身还大；何况这条探针本来就有
            # 5 秒结果缓存，它不会成为每请求一次的系统调用。
            elif not Path(self._settings.storage_local_root).is_dir():  # noqa: ASYNC240
                return ProbeResult(
                    self.name,
                    False,
                    _ms(started),
                    f"{self.name}(root missing: {self._settings.storage_local_root})",
                )
        except Exception as exc:
            return ProbeResult(self.name, False, _ms(started), f"{self.name}({type(exc).__name__})")
        return ProbeResult(self.name, True, _ms(started))

    async def aclose(self) -> None:
        # 本地后端没有可关的东西；s3 的客户端是每次调用现开的（storage.py 的实现）
        return None


class _CachedProbe:
    """给一条探针加结果缓存（见 `_PROBE_CACHE_SECONDS`）。

    ⚠️ **缓存的只是"结果"，不是探针对象**：`aclose()` 必须**透传**给内部探针，
    否则连接池永远关不掉——症状是进程不退出、测试挂住，而排查方向会指向
    pytest 或 event loop，不是这里。
    """

    def __init__(
        self, inner: DependencyProbe, *, ttl_seconds: float = _PROBE_CACHE_SECONDS
    ) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        #: 名字取一次就固定：它标识的是"这条依赖是谁"，而 `_inner` 不会被替换。
        #: 写成 `@property` 转发的版本过不了 `DependencyProbe` 协议
        #: ——协议声明的是可写属性，而属性是只读的。
        self.name = inner.name
        self._cached: ProbeResult | None = None
        self._cached_at: float = 0.0

    async def probe(self) -> ProbeResult:
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < self._ttl:
            return self._cached
        result = await self._inner.probe()
        self._cached = result
        self._cached_at = now
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()


def build_dependency_probes(settings: Settings) -> list[DependencyProbe]:
    """装配 `/health/ready` 要探的那几条依赖。**这里是唯一的装配点。**

    与 `build_object_storage` / `build_sql_repositories` 同一条纪律：装配散在
    多处时，"API 探这个、Worker 探那个"这类不一致要到现场才会暴露。
    """
    return [
        _CachedProbe(_QdrantProbe(settings)),
        _CachedProbe(_BusinessDbProbe(settings)),
        _CachedProbe(_ObjectStorageProbe(settings)),
    ]


async def run_probes(
    probes: Sequence[DependencyProbe], *, timeout_seconds: float
) -> list[ProbeResult]:
    """并发跑一遍探针，**每条各自限时**。

    并发（而不是串行）的理由：这个值就是 ready 的最坏延迟上界，而不是各条
    之和——串行的话三条各 2 秒会让 ready 变成 6 秒。

    ⚠️ **超时与探针自己抛异常都算 `ok=False`，不让它抛出去**：健康检查的职责是
    **如实报告**，"这条依赖在规定时间内没答上来"本身就是一条可读的结论。
    抛出去会让整个 ready 变成 500，反而比 `degraded` 丢掉了更多信息。
    """
    timeout_ms = int(timeout_seconds * 1000)

    async def one(probe: DependencyProbe) -> ProbeResult:
        try:
            return await asyncio.wait_for(probe.probe(), timeout=timeout_seconds)
        except TimeoutError:
            return ProbeResult(probe.name, False, timeout_ms, f"{probe.name}(Timeout)")
        except Exception as exc:
            # 探针不挑异常——它就是来兜底的（连"探针自己写错了"也要变成一条
            # 可读的 degraded 记录，而不是让整个 ready 变成 500）
            return ProbeResult(probe.name, False, 0, f"{probe.name}({type(exc).__name__})")

    return list(await asyncio.gather(*(one(probe) for probe in probes)))


__all__ = [
    "DependencyProbe",
    "ProbeResult",
    "build_dependency_probes",
    "run_probes",
]
