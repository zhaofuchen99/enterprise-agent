"""测试替身。放在 `app/tests/` 而不是生产模块里。

`FakeRateLimiter` 曾经写在 `services/rate_limit.py`，后来删掉了——
限流现在由 fakeredis 跑真实的 Lua 脚本，替身反而让测试覆盖不到真实路径。
**能跑真实实现时就别写替身**：替身只会验证「调用方按我设想的方式调用了」，
真实实现才验证「它真的做对了」。

队列是例外：arq 的 `ArqRedis` 需要真实连接才能 `enqueue_job`，
在单元测试里跑不起（integration 用例可以）。而投递行为本身需要被断言
（「创建任务后有没有投递」「投递失败时接口是否仍然 202」），因此保留替身。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.infrastructure.queue import JobQueue


@dataclass
class EnqueuedJob:
    task_id: str
    trace_id: str
    trace_context: dict[str, str]


@dataclass
class FakeJobQueue:
    """记录投递，并可按需模拟投递失败。

    `fail_with` 模拟 Redis 不可达：`ArqJobQueue.enqueue` 在那种情况下
    返回 False 而不是抛异常，替身必须复现这个契约，
    否则「投递失败时接口仍返回 202」的用例测的是替身的行为，不是代码的。
    """

    jobs: list[EnqueuedJob] = field(default_factory=list)
    #: 置 True 时模拟队列不可用
    fail_with: bool = False
    #: 已经「在队列里」的 job id。补偿扫描靠 `is_pending` 区分排队慢与真丢了，
    #: 用例通过直接改这个集合来构造两种场景。
    present: set[str] = field(default_factory=set)
    closed: bool = False

    async def enqueue(self, *, task_id: str, trace_id: str, trace_context: dict[str, str]) -> bool:
        if self.fail_with:
            return False
        if task_id in self.present:
            # arq 在 job_id 已存在时返回 None，即「没有真的入队」
            return False
        self.present.add(task_id)
        self.jobs.append(
            EnqueuedJob(task_id=task_id, trace_id=trace_id, trace_context=trace_context)
        )
        return True

    async def is_pending(self, task_id: str) -> bool:
        return task_id in self.present

    async def aclose(self) -> None:
        self.closed = True


def build_job_queue() -> JobQueue:
    """类型标注成协议，让用例里的替身与生产实现受同一份契约约束。"""
    return FakeJobQueue()
