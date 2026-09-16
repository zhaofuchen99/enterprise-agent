"""用户仓储。

Phase 1 还没有数据库（app_user 表在 Phase 2 建），因此这里给出
`UserRepository` 协议 + 内存实现。**这是刻意留下的唯一替换点**：
Phase 2 只要补一个 SQLAlchemy 实现、改 `app/api/deps.py` 里的一行装配，
服务层与接口层完全不动。

演示账号只在非生产环境写入。这不只是为了防误用，更重要的是**让缺失可见**：
生产环境此仓储为空，登录必然返回 401，而不是悄悄用演示口令放人进来。
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.ids import IdPrefix, deterministic_id
from app.core.security import hash_password
from app.domain.user import User, UserRole
from app.infrastructure.db import session_scope
from app.infrastructure.models.identity import AppUser
from app.repositories._mapping import to_db_time, user_from_row, user_to_row_values

#: 演示账号：(用户名, 口令, 角色, 数据范围)。开发环境占位，不是密钥。
#: 公开（无下划线前缀）是为了让接口测试直接引用，避免测试里另抄一份口令——
#: 抄一份就等于埋了一个「改了种子忘了改测试」的哑掉断言。
DEMO_ACCOUNTS: tuple[tuple[str, str, UserRole, tuple[str, ...]], ...] = (
    ("analyst", "analyst-dev-pass", UserRole.ANALYST, ("华东",)),
    ("admin", "admin-dev-pass", UserRole.ADMIN, ()),
)


@lru_cache(maxsize=1)
def _demo_password_hashes() -> tuple[str, ...]:
    """scrypt 单次约几十毫秒，两个账号就是上百毫秒。

    应用实例在测试里会被反复构造，缓存到进程级可以让这笔开销只付一次。
    盐值不需要每次进程启动都不同——口令哈希的盐是防彩虹表，不是防重启。
    """
    return tuple(hash_password(password) for _, password, _, _ in DEMO_ACCOUNTS)


class UserRepository(Protocol):
    async def get_by_id(self, user_id: str) -> User | None: ...

    async def get_by_username(self, username: str) -> User | None: ...


class InMemoryUserRepository:
    """进程内实现。仅用于 Phase 1，重启即丢。"""

    def __init__(self, users: Iterable[User] = ()) -> None:
        self._by_id: dict[str, User] = {}
        self._by_username: dict[str, User] = {}
        for user in users:
            self.add(user)

    def add(self, user: User) -> None:
        if user.id in self._by_id or user.username in self._by_username:
            raise ValueError(f"用户重复：{user.username}")
        self._by_id[user.id] = user
        self._by_username[user.username] = user

    async def get_by_id(self, user_id: str) -> User | None:
        return self._by_id.get(user_id)

    async def get_by_username(self, username: str) -> User | None:
        return self._by_username.get(username)


def _audit_times() -> dict[str, datetime]:
    """`app_user` 的审计时间列（NOT NULL）。

    取「写入那一刻」而不是从外部传入：审计字段的语义就是**实际写入时间**，
    允许调用方指定等于允许它写一个假的。
    """
    now = to_db_time(datetime.now(UTC))
    return {"created_at": now, "updated_at": now}


class SqlUserRepository:
    """MySQL 实现（`app_user`）。Phase 2 起的生产实现。"""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_by_id(self, user_id: str) -> User | None:
        async with session_scope(self._sessions) as session:
            row = await session.get(AppUser, user_id)
            return user_from_row(row) if row is not None else None

    async def get_by_username(self, username: str) -> User | None:
        stmt = select(AppUser).where(AppUser.username == username).limit(1)
        async with session_scope(self._sessions) as session:
            row = (await session.execute(stmt)).scalar_one_or_none()
            return user_from_row(row) if row is not None else None

    async def upsert_demo_users(self, users: Iterable[User]) -> int:
        """幂等写入演示账号，返回新增条数。供 `make seed` 调用。

        **按 username 判存在而不是无脑 INSERT**：`make seed` 会被反复执行
        （换了机器、清了库、改了演示口令），每次都在唯一索引上炸掉
        会让它变成一个「只能用一次」的命令。
        """
        created = 0
        async with session_scope(self._sessions) as session:
            for user in users:
                exists = (
                    await session.execute(
                        select(AppUser.id).where(AppUser.username == user.username).limit(1)
                    )
                ).scalar_one_or_none()
                if exists is not None:
                    continue
                # 审计时间由仓储补，不从领域模型取：`User` 是「用户是什么」，
                # 而 created_at/updated_at 是「这行什么时候被写进来的」，
                # 属于存储的事实。放进领域模型会让每次构造 User 都要考虑它。
                session.add(AppUser(**user_to_row_values(user), **_audit_times()))
                created += 1
        return created


def seed_demo_users(settings: Settings) -> list[User]:
    """返回演示账号；`APP_ENV=prod` 时返回空列表。

    **ID 是确定性派生的**（见 `deterministic_id` 的说明）：两个 API 实例
    必须看到同一个 `usr_`，否则多实例的限流与并发配额各算各的，
    开发流程 6.3 的验收命令根本跑不起来。
    """
    if settings.is_prod:
        return []

    hashes = _demo_password_hashes()
    return [
        User(
            id=deterministic_id(IdPrefix.USER, username),
            username=username,
            display_name=username,
            role=role,
            region_ids=region_ids,
            password_hash=password_hash,
        )
        for (username, _, role, region_ids), password_hash in zip(
            DEMO_ACCOUNTS, hashes, strict=True
        )
    ]
