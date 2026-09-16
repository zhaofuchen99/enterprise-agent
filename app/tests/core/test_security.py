"""口令哈希与 JWT。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import get_settings
from app.core.errors import AgentError, ErrorCode
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

_PASSWORD = "correct-horse-battery-staple"


def test_hash_then_verify() -> None:
    assert verify_password(_PASSWORD, hash_password(_PASSWORD))


def test_verify_rejects_wrong_password() -> None:
    assert not verify_password("wrong-password", hash_password(_PASSWORD))


def test_salt_makes_hashes_differ() -> None:
    """同样的口令两次哈希必须不同，否则彩虹表一次命中全部同口令账号。"""
    assert hash_password(_PASSWORD) != hash_password(_PASSWORD)


def test_verify_rejects_corrupted_hash() -> None:
    """哈希串损坏时返回 False，而不是把异常抛给调用方（会变成 500）。"""
    for corrupted in ("", "not-a-hash", "scrypt$only$two", "bcrypt$1$2$3$4$5", "scrypt$x$y$z$a$b"):
        assert not verify_password(_PASSWORD, corrupted)


def test_token_round_trip() -> None:
    settings = get_settings()
    issued = create_access_token(
        user_id="usr_0000000000000000000000", role="ADMIN", settings=settings
    )

    claims = decode_access_token(issued.token, settings)
    assert claims.user_id == "usr_0000000000000000000000"
    assert claims.role == "ADMIN"
    assert issued.expires_in == settings.jwt_expire_minutes * 60


def test_decode_rejects_expired_token() -> None:
    settings = get_settings()
    token = jwt.encode(
        {
            "sub": "usr_0000000000000000000000",
            "role": "ANALYST",
            "iat": int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
            "exp": int((datetime.now(UTC) - timedelta(hours=1)).timestamp()),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AgentError) as excinfo:
        decode_access_token(token, settings)
    assert excinfo.value.code is ErrorCode.AUTHENTICATION_REQUIRED


def test_decode_rejects_wrong_secret() -> None:
    settings = get_settings()
    token = jwt.encode(
        {
            "sub": "usr_0000000000000000000000",
            "role": "ANALYST",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        },
        "another-secret-entirely-but-long-enough",
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AgentError):
        decode_access_token(token, settings)


def test_decode_rejects_token_missing_required_claims() -> None:
    """少字段的令牌不该被当成合法凭证。"""
    settings = get_settings()
    token = jwt.encode(
        {
            "sub": "usr_0000000000000000000000",
            "iat": int(datetime.now(UTC).timestamp()),
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AgentError):
        decode_access_token(token, settings)
