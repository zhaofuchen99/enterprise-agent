"""任务队列——把 arq 关在这一个文件里（开发流程 6.3 施工项 6）。

**为什么要这一层**：CLAUDE.md 要求「infrastructure/ 层必须把外部组件隔离干净，
使换实现的成本控制在一个文件内」。队列是最可能被换掉的那个组件
（`arq` 与 `celery` 在详细设计 4.3 里本就是备选关系），而调用方
（`services/task_runner.py`）只该看见「投递一个任务」这一个动作。

**投递为什么会失败**：Redis 不可达时 `enqueue_job` 会抛异常。按详细设计 17.1，
此时任务已经写库成功（状态 QUEUED），正确做法是**不把这次请求判成失败**——
任务停在 QUEUED，由 Worker 侧的补偿扫描重新投递。
因此 `enqueue` 把失败收敛成一个 `False` 返回值而不是抛异常，
让调用方写完库之后的路径只有一种形状。
"""

from __future__ import annotations

import logging
from typing import Protocol

import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job, JobStatus

from app.core.config import Settings
from app.infrastructure.redis import RedisKey

logger = logging.getLogger(__name__)

#: 与 `app/worker.py` 里注册的协程函数同名。写错这个字符串不会有任何报错，
#: 只会表现为「任务入队了但永远没人执行」，所以两端共用这一个常量。
TASK_JOB_NAME = "run_agent_task"

#: 视为「还没被执行过」的作业状态。`complete` 刻意**不算**：
#: 一个已经跑完的作业而任务还停在 QUEUED（比如 Worker 领到时状态已不符，
#: 空跑一轮就结束了），若把它算作「还在」，补偿扫描会永远跳过它，
#: 任务就永久卡在 QUEUED 上，连失败都不会失败。
_PENDING_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.deferred, JobStatus.queued, JobStatus.in_progress}
)


class JobQueue(Protocol):
    """投递接口。返回值表示**本次是否真的入队**，不是任务是否成功。"""

    async def enqueue(
        self, *, task_id: str, trace_id: str, trace_context: dict[str, str]
    ) -> bool: ...

    async def is_pending(self, task_id: str) -> bool:
        """任务是否仍在队列里等待执行（排队中 / 执行中 / 延后）。

        补偿扫描靠它区分两种「QUEUED 很久了」：队列里还有 = 只是排队慢，
        队列里没有 = 真的丢了需要重投。没有这个区分，扫描只能靠猜，
        队列积压时会把一批正常排队的任务全部重投并计入失败次数。

        **名字故意不叫 `exists`**：实现里差点写出 `ArqRedis.exists(...)`，
        而那其实是 Redis 的通用 `EXISTS`（查 key 存不存在），
        在 arq 的 `job_id` 上永远返回 False。名字含糊会直接导致这种错误，
        而且它只在队列积压时才表现出后果，平时完全看不出来。
        """
        ...

    async def aclose(self) -> None: ...


class ArqJobQueue:
    """arq 实现。

    `_job_id` 固定取 `task_id`，这不是随手定的：它让 arq 自己完成去重——
    补偿扫描重新投递一个其实还在队列里的任务时，arq 会返回 None 而不是
    塞进第二个 job。没有这个约束，「重投」就会变成「同一个任务被两个 Worker
    同时领取」，而重复执行在 LLM 与 SQL 场景下都是要花钱的。
    """

    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool
        #: 投递与查询必须用同一个队列名，因此只在这里取一次
        self._queue_name = str(pool.default_queue_name)

    @classmethod
    async def create(cls, settings: Settings) -> ArqJobQueue:
        # RedisSettings.from_dsn 复用 REDIS_URL，不维护第二份 Redis 配置
        pool = await create_pool(
            RedisSettings.from_dsn(settings.redis_url),
            # 显式指定队列名：arq 的默认名是 `arq:queue`，与详细设计 4.4 写的
            # `q:agent` 不一致。不指定的话，按文档去 redis-cli 里找队列会找不到，
            # 而那种「文档说的键不存在」会让人怀疑整条链路都没跑起来。
            default_queue_name=RedisKey.queue(),
        )
        return cls(pool)

    async def enqueue(self, *, task_id: str, trace_id: str, trace_context: dict[str, str]) -> bool:
        try:
            job = await self._pool.enqueue_job(
                TASK_JOB_NAME,
                task_id,
                trace_id,
                trace_context,
                _job_id=task_id,
            )
        except (aioredis.RedisError, OSError) as exc:
            logger.warning(
                "任务投递失败，等待补偿扫描重投：%s", type(exc).__name__, extra={"task_id": task_id}
            )
            return False
        # job 为 None 说明同 id 的任务已在队列中，属于幂等命中，不是失败
        return job is not None

    async def is_pending(self, task_id: str) -> bool:
        # 两处都不能省：
        # 1. 不是 `self._pool.exists(...)`——那是 Redis 的 EXISTS，含义完全不同；
        # 2. `Job` **不会**继承连接池的 default_queue_name，必须显式传队列名，
        #    否则它去查 `arq:queue`，而任务其实躺在 `q:agent` 里，永远返回 not_found。
        job = Job(task_id, self._pool, _queue_name=self._queue_name)
        return await job.status() in _PENDING_STATUSES

    async def aclose(self) -> None:
        await self._pool.aclose()
