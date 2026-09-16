"""登录与请求认证（TBC-08 决议：自建账号 + 本地 JWT）。

安全上的两个要点：

1. **用户名不存在时也要走一次哈希校验**。否则「用户不存在」会比
   「口令错误」快几十毫秒，攻击者据此就能枚举出哪些用户名是有效的。
2. **每次请求都以库里的角色为准**，不信任令牌里的 `role` 声明。
   否则用户角色被降级后，手里那张未过期的旧令牌仍然带着旧权限。

第 2 点之所以在 Phase 1 就要做：等 Phase 2 有了数据库再补，
已经发出去的令牌无法回收，属于只能靠改代码堵、不能靠配置补救的漏洞。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.domain.user import User
from app.repositories.user_repo import UserRepository


@dataclass(frozen=True, slots=True)
class LoginResult:
    user: User
    access_token: str
    expires_in: int


@lru_cache(maxsize=1)
def _timing_equalizer_hash() -> str:
    """抹平「用户不存在」与「口令错误」耗时差的哑哈希（见模块 docstring 第 1 点）。"""
    return hash_password("timing-equalizer-not-a-real-credential")


def _invalid_credentials() -> AgentError:
    # 不区分「用户名不存在」与「口令错误」，避免用户名枚举（详细设计 19.2）
    return AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "用户名或口令错误")


class AuthService:
    def __init__(self, users: UserRepository, settings: Settings) -> None:
        self._users = users
        self._settings = settings

    async def login(self, *, username: str, password: str) -> LoginResult:
        user = await self._users.get_by_username(username)
        if user is None:
            verify_password(password, _timing_equalizer_hash())
            raise _invalid_credentials()

        if not verify_password(password, user.password_hash):
            raise _invalid_credentials()

        if not user.is_active:
            # 禁用是「认证过了但无权使用」，与口令错误区别对待：
            # 后者不该透露任何细节（401），前者需要让用户知道找管理员（403）
            raise AgentError(ErrorCode.ACCESS_DENIED, "账号已被禁用，请联系管理员")

        issued = create_access_token(user_id=user.id, role=user.role.value, settings=self._settings)
        return LoginResult(user=user, access_token=issued.token, expires_in=issued.expires_in)

    async def authenticate(self, token: str) -> User:
        """把 Bearer Token 换成当前用户；任何一步不通过都返回同一个 401。"""
        claims = decode_access_token(token, self._settings)

        user = await self._users.get_by_id(claims.user_id)
        if user is None or not user.is_active:
            raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "认证失败，请重新登录")

        if user.role.value != claims.role:
            # 令牌签发后角色被改动过，旧令牌立即失效（见模块 docstring 第 2 点）
            raise AgentError(ErrorCode.AUTHENTICATION_REQUIRED, "认证失败，请重新登录")

        return user
