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
from functools import lru_cache
from typing import Protocol

from app.core.config import Settings
from app.core.ids import IdPrefix, new_id
from app.core.security import hash_password
from app.domain.user import User, UserRole

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


def seed_demo_users(settings: Settings) -> list[User]:
    """返回演示账号；`APP_ENV=prod` 时返回空列表。"""
    if settings.is_prod:
        return []

    hashes = _demo_password_hashes()
    return [
        User(
            id=new_id(IdPrefix.USER),
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
