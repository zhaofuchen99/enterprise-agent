"""本地账号登录接口（详细设计 17.7，TBC-08 决议：自建账号 + 本地 JWT）。

路径在 `/api/auth` 而不是 `/api/agent` 下：需求规格 7.1 的前缀约定是给业务接口的，
认证是所有业务接口的前置，混在里面反而不好找。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import AuthServiceDep, TraceIdDep
from app.api.schemas import (
    ApiResponse,
    LoginData,
    LoginRequest,
    SuccessCode,
    UserProfile,
    error_responses,
)
from app.core.errors import ErrorCode

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post(
    "/login",
    response_model=ApiResponse[LoginData],
    summary="本地账号登录",
    description=(
        "校验用户名与口令，返回 Bearer Access Token。"
        "演示版本不实现 Refresh Token，会话到期后重新登录。"
    ),
    responses=error_responses(
        ErrorCode.INVALID_ARGUMENT,
        ErrorCode.AUTHENTICATION_REQUIRED,
        ErrorCode.ACCESS_DENIED,
        ErrorCode.INTERNAL_ERROR,
    ),
)
async def login(
    payload: LoginRequest, auth: AuthServiceDep, trace_id: TraceIdDep
) -> ApiResponse[LoginData]:
    result = await auth.login(username=payload.username, password=payload.password)
    return ApiResponse(
        code=SuccessCode.OK,
        message="登录成功",
        data=LoginData(
            access_token=result.access_token,
            expires_in=result.expires_in,
            user=UserProfile.from_domain(result.user),
        ),
        trace_id=trace_id,
    )
