"""任务仓储契约（开发流程 6.3 施工项 3 的配套）。

**同一份断言跑两个实现**：`InMemoryTaskRepository` 与 `RedisTaskRepository`
共用一个参数化的夹具。只测 Redis 实现的话，内存实现会悄悄烂掉；
只测内存实现的话，真正上线的那份没有被验证过。两者行为一致本身也是要求——
Phase 2 换成 MySQL 时，这个文件就是新实现的验收清单。

仓储里最容易出错的是 `update` 的**字段级语义**：Worker 心跳与用户取消
是两个进程对同一条记录的并发写。用「整份读出来、改、整份写回」的话，
心跳会把 `CANCEL_REQUESTED` 抹掉，表现为「点了取消但任务照跑」。
下面的 `test_heartbeat_does_not_clobber_status` 就是钉住这件事的。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import pytest

from app.core.config import Settings
from app.domain.task import Task, TaskStatus
from app.repositories.task_repo import (
    DuplicateTaskError,
    InMemoryTaskRepository,
    RedisTaskRepository,
    TaskPatch,
    TaskRepository,
)

_NOW = datetime(2026, 9, 16, 12, 0, 0, tzinfo=UTC)


def _settings() -> Settings:
    return Settings(
        model_provider="p",
        model_name="m",
        model_api_key="k",
        embedding_model="e",
        embedding_api_key="k",
        database_url_agent="mysql+asyncmy://a@localhost/a",
        database_url_business_ro="mysql+asyncmy://a@localhost/b",
        redis_url="redis://localhost:6379/0",
        milvus_uri="http://localhost:19530",
        jwt_secret="x" * 32,
    )


def _task(
    task_id: str = "tsk_0000000001AAAAAAAAAAAA",
    *,
    user_id: str = "usr_1",
    status: TaskStatus = TaskStatus.QUEUED,
    idempotency_key: str | None = None,
    queued_at: datetime = _NOW,
) -> Task:
    return Task(
        id=task_id,
        user_id=user_id,
        conversation_id="cnv_0000000001AAAAAAAAAAAA",
        trace_id="trc_0000000001AAAAAAAAAAAA",
        query_text="华东销售额为什么下降",
        status=status,
        idempotency_key=idempotency_key,
        queued_at=queued_at,
        created_at=queued_at,
        updated_at=queued_at,
    )


@pytest.fixture
async def redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture(params=["memory", "redis"])
def make_repo(
    request: pytest.FixtureRequest, redis: fakeredis.aioredis.FakeRedis
) -> Callable[[], TaskRepository]:
    """参数化的仓储构造器。两个实现跑完全相同的用例集。"""
    if request.param == "memory":
        return InMemoryTaskRepository
    return lambda: RedisTaskRepository(redis, _settings())


# ------------------------------------------------------------------ 建档
async def test_add_then_get_round_trips_every_field(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    task = _task(idempotency_key="key-1")
    await repo.add(task)

    loaded = await repo.get(task.id)

    assert loaded == task


async def test_get_returns_none_for_unknown_id(make_repo: Callable[[], TaskRepository]) -> None:
    assert await make_repo().get("tsk_不存在") is None


async def test_duplicate_task_id_is_rejected(make_repo: Callable[[], TaskRepository]) -> None:
    repo = make_repo()
    await repo.add(_task())

    with pytest.raises(DuplicateTaskError):
        await repo.add(_task())


async def test_duplicate_idempotency_key_is_rejected(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    await repo.add(_task("tsk_0000000001AAAAAAAAAAAA", idempotency_key="k"))

    with pytest.raises(DuplicateTaskError):
        await repo.add(_task("tsk_0000000002AAAAAAAAAAAA", idempotency_key="k"))


async def test_idempotency_key_is_scoped_to_user(
    make_repo: Callable[[], TaskRepository],
) -> None:
    """同一个键换个用户就是另一个键——否则一个用户能猜到别人的任务。"""
    repo = make_repo()
    await repo.add(_task("tsk_0000000001AAAAAAAAAAAA", user_id="usr_1", idempotency_key="k"))
    await repo.add(_task("tsk_0000000002AAAAAAAAAAAA", user_id="usr_2", idempotency_key="k"))

    found = await repo.find_by_idempotency_key("usr_2", "k")
    assert found is not None
    assert found.id == "tsk_0000000002AAAAAAAAAAAA"


async def test_find_by_idempotency_key_returns_none_when_absent(
    make_repo: Callable[[], TaskRepository],
) -> None:
    assert await make_repo().find_by_idempotency_key("usr_1", "没有这个键") is None


# ------------------------------------------------------------------ 并发配额
async def test_count_active_counts_non_terminal_statuses(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    for index, status in enumerate(
        [TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING_CLARIFICATION]
    ):
        await repo.add(_task(f"tsk_000000000{index}AAAAAAAAAAAA", status=status))

    assert await repo.count_active("usr_1") == 3


async def test_count_active_ignores_other_users(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    await repo.add(_task("tsk_0000000001AAAAAAAAAAAA", user_id="usr_1"))
    await repo.add(_task("tsk_0000000002AAAAAAAAAAAA", user_id="usr_2"))

    assert await repo.count_active("usr_1") == 1


@pytest.mark.parametrize(
    "terminal", [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
async def test_entering_terminal_releases_quota(
    make_repo: Callable[[], TaskRepository], terminal: TaskStatus
) -> None:
    """19.3：任务结束（任意终态）时必须递减。

    漏掉这一步，用户的并发配额会被泄漏的计数永久占住，
    表现为「这个用户再也创建不了任务」，而且重启也修不好。
    """
    repo = make_repo()
    task = _task()
    await repo.add(task)
    assert await repo.count_active("usr_1") == 1

    await repo.update(task.id, TaskPatch(status=terminal), at=_NOW)

    assert await repo.count_active("usr_1") == 0


# ------------------------------------------------------------------ 字段级更新
async def test_update_only_touches_the_given_fields(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    task = _task(idempotency_key="k")
    await repo.add(task)

    await repo.update(task.id, TaskPatch(intent="sales_drilldown"), at=_NOW)

    loaded = await repo.get(task.id)
    assert loaded is not None
    assert loaded.intent == "sales_drilldown"
    assert loaded.query_text == task.query_text
    assert loaded.idempotency_key == "k"


async def test_heartbeat_does_not_clobber_status(
    make_repo: Callable[[], TaskRepository],
) -> None:
    """心跳与取消是两个进程对同一条记录的并发写，绝不能互相覆盖。

    用「读出来、改、整份写回」的话，Worker 手里那份是几秒前读的，
    写回时会把用户刚写下的 CANCEL_REQUESTED 抹掉，任务照跑——
    这正是 `touch_heartbeat` 只改一个字段、不改整份记录的原因。
    """
    repo = make_repo()
    task = _task(status=TaskStatus.RUNNING)
    await repo.add(task)

    # 用户取消先落库
    await repo.update(task.id, TaskPatch(status=TaskStatus.CANCEL_REQUESTED), at=_NOW)
    # Worker 的下一拍心跳随后到达
    later = _NOW + timedelta(seconds=10)
    await repo.touch_heartbeat(task.id, later)

    loaded = await repo.get(task.id)
    assert loaded is not None
    assert loaded.status is TaskStatus.CANCEL_REQUESTED
    assert loaded.heartbeat_at == later


async def test_update_returns_none_when_task_is_missing(
    make_repo: Callable[[], TaskRepository],
) -> None:
    assert await make_repo().update("tsk_没有", TaskPatch(intent="x"), at=_NOW) is None


async def test_only_if_status_guards_the_transition(
    make_repo: Callable[[], TaskRepository],
) -> None:
    """领取互斥的实现手段：只有当前确实是 QUEUED 才写得进去。

    少了这层保护，两个 Worker 同时领到同一任务会双双开跑——
    同一个问题算两遍，两次 SQL、两次模型调用。
    """
    repo = make_repo()
    task = _task(status=TaskStatus.RUNNING)
    await repo.add(task)

    result = await repo.update(
        task.id, TaskPatch(status=TaskStatus.RUNNING), at=_NOW, only_if_status=TaskStatus.QUEUED
    )

    assert result is None


async def test_only_if_status_writes_when_it_matches(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    task = _task(status=TaskStatus.QUEUED)
    await repo.add(task)

    result = await repo.update(
        task.id, TaskPatch(status=TaskStatus.RUNNING), at=_NOW, only_if_status=TaskStatus.QUEUED
    )

    assert result is not None
    assert result.status is TaskStatus.RUNNING
    loaded = await repo.get(task.id)
    assert loaded is not None and loaded.status is TaskStatus.RUNNING


# ------------------------------------------------------------------ 扫描
async def test_list_stale_queued_filters_by_age_and_status(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    old = _NOW - timedelta(minutes=5)
    await repo.add(_task("tsk_0000000001AAAAAAAAAAAA", queued_at=old))
    await repo.add(_task("tsk_0000000002AAAAAAAAAAAA", queued_at=_NOW))
    await repo.add(_task("tsk_0000000003AAAAAAAAAAAA", queued_at=old, status=TaskStatus.RUNNING))

    stale = await repo.list_stale_queued(_NOW - timedelta(minutes=1), limit=10)

    assert [task.id for task in stale] == ["tsk_0000000001AAAAAAAAAAAA"]


async def test_list_stale_queued_respects_limit(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    old = _NOW - timedelta(minutes=5)
    for index in range(5):
        await repo.add(_task(f"tsk_000000000{index}AAAAAAAAAAAA", queued_at=old))

    stale = await repo.list_stale_queued(_NOW, limit=2)

    assert len(stale) == 2


async def test_list_stale_running_uses_heartbeat_age(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    await repo.add(_task("tsk_0000000001AAAAAAAAAAAA", status=TaskStatus.RUNNING))
    await repo.add(_task("tsk_0000000002AAAAAAAAAAAA", status=TaskStatus.RUNNING))
    # 第二个任务刚刚续过心跳：用未来的时间戳确保它一定不算过期
    await repo.touch_heartbeat("tsk_0000000002AAAAAAAAAAAA", _NOW + timedelta(minutes=1))

    stale = await repo.list_stale_running(_NOW + timedelta(seconds=30), limit=10)

    assert [task.id for task in stale] == ["tsk_0000000001AAAAAAAAAAAA"]


async def test_running_task_is_reclaimable_before_its_first_heartbeat(
    make_repo: Callable[[], TaskRepository],
) -> None:
    """Worker 在「领取之后、第一次心跳之前」挂掉，任务也必须能被扫到。

    只靠心跳来建索引的实现会在这里失守：任务进了 RUNNING 却不在索引里，
    孤儿回收永远发现不了它，任务一直挂着。
    """
    repo = make_repo()
    task = _task(status=TaskStatus.QUEUED)
    await repo.add(task)
    await repo.update(
        task.id, TaskPatch(status=TaskStatus.RUNNING), at=_NOW, only_if_status=TaskStatus.QUEUED
    )

    stale = await repo.list_stale_running(_NOW + timedelta(seconds=1), limit=10)

    assert [t.id for t in stale] == [task.id]


async def test_terminal_task_drops_out_of_both_scan_indexes(
    make_repo: Callable[[], TaskRepository],
) -> None:
    repo = make_repo()
    task = _task(queued_at=_NOW - timedelta(minutes=5))
    await repo.add(task)

    await repo.update(task.id, TaskPatch(status=TaskStatus.FAILED), at=_NOW)

    assert await repo.list_stale_queued(_NOW, limit=10) == []
    assert await repo.list_stale_running(_NOW + timedelta(days=1), limit=10) == []
