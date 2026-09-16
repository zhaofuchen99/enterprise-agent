"""用户领域模型（对应详细设计 16.3 的 app_user 表）。

MVP 只有 ANALYST / ADMIN 两种角色（详细设计 19.3）。
数据范围按 TBC-03 的决议简化为区域列表，不做行列级授权。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class UserRole(StrEnum):
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"


class UserStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class User(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    username: str
    display_name: str
    role: UserRole
    #: repr=False 防止误打印，exclude=True 防止 model_dump() 把它带进日志或响应体。
    #: 口令哈希泄露的后果等同于口令泄露，不值得依赖调用方自觉。
    password_hash: str = Field(repr=False, exclude=True)
    status: UserStatus = UserStatus.ACTIVE
    #: TBC-03 决议：`{"region_ids": [...]}`，ANALYST 限本区域，ADMIN 为全量
    region_ids: tuple[str, ...] = ()

    @property
    def is_active(self) -> bool:
        return self.status is UserStatus.ACTIVE
