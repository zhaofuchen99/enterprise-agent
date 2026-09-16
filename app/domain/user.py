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

    def permission_scope(self) -> PermissionScope:
        """转成 Tool 侧使用的权限范围。

        **只有这一个转换入口**：工具拿到的数据范围必须与登录用户实际拥有的
        完全一致，两处各自组装一次就等于两处可能不一致，而这类不一致的表现是
        「越权查询悄悄成功了」，不会有任何报错。
        """
        return PermissionScope(role=self.role, region_ids=self.region_ids)


class PermissionScope(BaseModel):
    """一次工具调用的数据权限范围（详细设计 9.1 的 `PermissionScope`）。

    TBC-03 把数据权限简化为一组区域（`data_scope_json` = `{"region_ids": [...]}`），
    不做行列级授权。落到 SQL Tool 就是详设 10.4 第 11 步的那个**服务端谓词**。

    Attributes:
        role: 角色。决定列的 `allowed_roles` 是否放行（详设 10.4 第 7 步）。
        region_ids: 可见区域集合。**空集合表示不限制**——演示库里的 `admin`
            账号正是空值（`app/repositories/user_repo.py` 的 `DEMO_ACCOUNTS`）。
            刻意不做成「ADMIN 角色即放开」：那是把权限判断挂在角色名上，
            加一个角色就要改一遍判断；挂在「范围是否为空」上则只有一种语义，
            也顺带让「给 ADMIN 限定某个区域」成为配置而不是改代码。
    """

    model_config = ConfigDict(frozen=True)

    role: UserRole = UserRole.ANALYST
    region_ids: tuple[str, ...] = ()

    @property
    def unrestricted(self) -> bool:
        return not self.region_ids
