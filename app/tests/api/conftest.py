"""接口测试夹具：登录并拿到可用的 Authorization 头。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.config import get_settings
from app.core.ids import IdPrefix, new_id
from app.core.security import create_access_token, hash_password
from app.domain.user import User, UserRole
from app.repositories.user_repo import DEMO_ACCOUNTS
from app.tests.conftest import build_client

#: 建临时用户时要算一次口令哈希，scrypt 约几十毫秒，放模块级只算一次
_OTHER_PASSWORD_HASH = hash_password("other-dev-pass")


def issue_headers(
    app: FastAPI, *, username: str = "other", role: UserRole = UserRole.ANALYST
) -> dict[str, str]:
    """新造一个用户并签发令牌。

    演示账号只有 analyst / admin 两个，而归属校验（FR-CHAT-002 的 403）
    需要「第三个身份」才能构造出来。
    """
    user = User(
        id=new_id(IdPrefix.USER),
        username=username,
        display_name=username,
        role=role,
        password_hash=_OTHER_PASSWORD_HASH,
    )
    app.state.user_repo.add(user)
    issued = create_access_token(user_id=user.id, role=role.value, settings=get_settings())
    return {"Authorization": f"Bearer {issued.token}"}


def demo_password(username: str) -> str:
    """从种子数据里取演示口令。

    不在这里另抄一份：抄一份等于埋了个「改了种子忘了改测试」的哑断言。
    """
    return next(password for name, password, _, _ in DEMO_ACCOUNTS if name == username)


async def login(client: AsyncClient, username: str) -> str:
    """登录并返回 Access Token。"""
    response = await client.post(
        "/api/auth/login", json={"username": username, "password": demo_password(username)}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["data"]["access_token"])


async def login_headers(client: AsyncClient, username: str = "analyst") -> dict[str, str]:
    return {"Authorization": f"Bearer {await login(client, username)}"}


@asynccontextmanager
async def app_client(
    app: FastAPI, username: str = "analyst"
) -> AsyncIterator[tuple[AsyncClient, dict[str, str]]]:
    """为**新构造的应用实例**建客户端并登录，返回 (client, headers)。

    演示账号的 ID 是确定性派生的，因此在**多个应用实例之间是同一个用户**，
    默认 app 上签发的令牌在新实例里同样有效——多实例限流与并发配额
    因此可以被端到端验收（开发流程 6.3 验收命令 1）。
    会话与任务仓储仍是进程内的，跨实例的可见性要等 Phase 2 接入 MySQL。
    """
    async with build_client(app) as client:
        yield client, await login_headers(client, username)


@pytest.fixture
async def analyst_token(client: AsyncClient) -> str:
    return await login(client, "analyst")


@pytest.fixture
async def admin_token(client: AsyncClient) -> str:
    return await login(client, "admin")


@pytest.fixture
async def analyst_headers(analyst_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {analyst_token}"}


@pytest.fixture
async def admin_headers(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}
