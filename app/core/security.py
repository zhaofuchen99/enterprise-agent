"""口令哈希与 JWT 签发 / 校验（TBC-08 决议：自建账号 + 本地 JWT）。

本模块只依赖 `app.core.*`，**不 import `app.domain`**——core 是所有层的横切底座，
让它反向依赖领域模型会把依赖图绕成环。调用方负责把 `role` 字符串
转成 `UserRole` 再做校验。

口令哈希选 scrypt 而非 bcrypt/argon2：`hashlib.scrypt` 是标准库自带的，
在「每个中间件都要能论证必要性」的前提下，不为口令哈希再引一个三方依赖，
也避免 bcrypt 的 72 字节静默截断问题。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode

_SCHEME = "scrypt"
#: n=2^14 / r=8 / p=1 是 RFC 7914 的交互式登录推荐档，约 16MB 内存、几十毫秒
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16
_DERIVED_BYTES = 32


@dataclass(frozen=True, slots=True)
class IssuedToken:
    token: str
    expires_in: int


@dataclass(frozen=True, slots=True)
class TokenClaims:
    user_id: str
    role: str
    expires_at: datetime


def _b64encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _derive(password: str, salt: bytes, *, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen)


def hash_password(password: str) -> str:
    """返回 `scrypt$n$r$p$salt$hash`，参数随哈希一起存，日后调参不影响老数据。"""
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = _derive(password, salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_DERIVED_BYTES)
    return f"{_SCHEME}${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64encode(salt)}${_b64encode(derived)}"


def verify_password(password: str, encoded: str) -> bool:
    """恒定时间比对。任何解析失败都返回 False，不区分「格式坏」与「口令错」。"""
    try:
        scheme, raw_n, raw_r, raw_p, salt_b64, hash_b64 = encoded.split("$")
        if scheme != _SCHEME:
            return False
        salt = base64.b64decode(salt_b64, validate=True)
        expected = base64.b64decode(hash_b64, validate=True)
        derived = _derive(
            password,
            salt,
            n=int(raw_n),
            r=int(raw_r),
            p=int(raw_p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        # 编码串来自数据库而非用户输入，走到这里说明数据损坏，按「验证失败」处理即可
        return False
    return hmac.compare_digest(derived, expected)


def create_access_token(*, user_id: str, role: str, settings: Settings) -> IssuedToken:
    """签发 Access Token。演示版本不实现 Refresh Token（详细设计 17.7）。"""
    now = datetime.now(UTC)
    expires_in = settings.jwt_expire_minutes * 60
    payload: dict[str, Any] = {
        "sub": user_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return IssuedToken(token=token, expires_in=expires_in)


def decode_access_token(token: str, settings: Settings) -> TokenClaims:
    """校验并解析 Access Token。

    失败一律抛同一个 401，**不区分**「过期」「签名错」「字段缺失」——
    区分开等于给攻击者一个探测签名是否正确的方法（详细设计 19.2.1）。
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "iat", "sub", "role"]},
        )
    except jwt.InvalidTokenError as exc:
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "认证失败，请重新登录") from exc

    user_id = payload.get("sub")
    role = payload.get("role")
    if not isinstance(user_id, str) or not isinstance(role, str):
        raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "认证失败，请重新登录")

    return TokenClaims(
        user_id=user_id,
        role=role,
        expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=UTC),
    )
