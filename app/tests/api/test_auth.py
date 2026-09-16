"""认证接口测试（TBC-08：自建账号 + 本地 JWT）。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.config import get_settings
from app.core.ids import IdPrefix, new_id
from app.core.security import create_access_token, hash_password
from app.domain.user import User, UserRole, UserStatus
from app.tests.api.conftest import demo_password

_PROTECTED = "/api/agent/tasks/tsk_0000000000000000000000"


async def test_login_returns_token_and_profile(client: AsyncClient) -> None:
    response = await client.post(
        "/api/auth/login", json={"username": "analyst", "password": demo_password("analyst")}
    )
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"code", "message", "data", "trace_id", "retryable"}
    assert body["code"] == "OK"
    data = body["data"]
    assert data["token_type"] == "Bearer"
    assert data["expires_in"] == get_settings().jwt_expire_minutes * 60
    assert data["user"]["role"] == "ANALYST"
    # 口令哈希绝不能出现在响应里
    assert "password" not in response.text.lower()


async def test_login_with_wrong_password_is_401(client: AsyncClient) -> None:
    response = await client.post(
        "/api/auth/login", json={"username": "analyst", "password": "definitely-wrong"}
    )
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "AUTHENTICATION_REQUIRED"
    assert body["retryable"] is False


async def test_unknown_user_and_wrong_password_are_indistinguishable(
    client: AsyncClient,
) -> None:
    """两种失败必须返回一模一样的响应。

    只要 code 或 message 有差别，攻击者就能拿它当用户名存在性的探针。
    """
    wrong_password = await client.post(
        "/api/auth/login", json={"username": "analyst", "password": "definitely-wrong"}
    )
    unknown_user = await client.post(
        "/api/auth/login", json={"username": "no-such-user", "password": "definitely-wrong"}
    )

    assert unknown_user.status_code == wrong_password.status_code == 401
    assert unknown_user.json()["code"] == wrong_password.json()["code"]
    assert unknown_user.json()["message"] == wrong_password.json()["message"]


async def test_disabled_account_is_403(client: AsyncClient, app: FastAPI) -> None:
    """账号被禁用是 403（认证过了但无权使用），与口令错误的 401 区分开。"""
    user = User(
        id=new_id(IdPrefix.USER),
        username="disabled",
        display_name="disabled",
        role=UserRole.ANALYST,
        password_hash=hash_password("disabled-dev-pass"),
        status=UserStatus.DISABLED,
    )
    app.state.user_repo.add(user)

    response = await client.post(
        "/api/auth/login", json={"username": "disabled", "password": "disabled-dev-pass"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "ACCESS_DENIED"


async def test_login_validation_error_does_not_echo_password(client: AsyncClient) -> None:
    """校验失败时**不得**把口令回显出来。

    pydantic 的报错默认带 input 字段，而登录接口的 input 就是明文口令；
    一旦回显，它会顺着错误提示、前端 toast 和访问日志一路扩散。
    """
    secret = "p@ssw0rd-should-never-be-echoed"
    response = await client.post("/api/auth/login", json={"username": "", "password": secret})
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ARGUMENT"
    assert secret not in response.text


async def test_expired_token_is_401(client: AsyncClient, analyst_token: str) -> None:
    settings = get_settings()
    payload = jwt.decode(analyst_token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    expired = {
        **payload,
        "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
        "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
    }
    token = jwt.encode(expired, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    response = await client.get(_PROTECTED, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


async def test_token_signed_with_other_secret_is_401(client: AsyncClient) -> None:
    """换了密钥签出来的令牌必须被拒——否则 JWT_SECRET 形同虚设。"""
    token = jwt.encode(
        {
            "sub": "usr_0000000000000000000000",
            "role": "ADMIN",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        },
        "a-completely-different-secret-value",
        algorithm="HS256",
    )
    response = await client.get(_PROTECTED, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


async def test_role_change_invalidates_existing_token(client: AsyncClient, app: FastAPI) -> None:
    """令牌签发后角色被改过，旧令牌必须立刻失效。

    这条是权限回收的唯一手段：令牌在有效期内无法撤销，
    因此每次请求都要以库里的角色为准，而不是信令牌里的声明。
    """
    user = User(
        id=new_id(IdPrefix.USER),
        username="role-shift",
        display_name="role-shift",
        role=UserRole.ANALYST,
        password_hash=hash_password("role-shift-dev-pass"),
    )
    app.state.user_repo.add(user)

    # 故意签发一张角色与库里不符的令牌
    issued = create_access_token(
        user_id=user.id, role=UserRole.ADMIN.value, settings=get_settings()
    )
    response = await client.get(_PROTECTED, headers={"Authorization": f"Bearer {issued.token}"})

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"
