"""创建分析任务（详细设计 17.1）。

**Phase 1 是占位实现**：只校验入参、登记任务、返回 202，不启动 LangGraph。
真正的入队与执行在 Phase 1.5 接入（`services/task_runner.py`），
本文件的接口形状与响应体不会再变。

按需求规格 7.1，「调用方拿到 task_id 后用 SSE 订阅进展」，
所以这里返回的 `stream_url` 指向 Phase 10 才实现的端点——
先占住契约，避免前端在接口形状上返工。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.api.deps import (
    CurrentUser,
    SettingsDep,
    TaskServiceDep,
    TraceIdDep,
    require_create_task_rate_limit,
)
from app.api.schemas import ApiResponse, ChatRequest, SuccessCode, TaskCreatedData, error_responses
from app.core.errors import AgentError, ErrorCode
from app.infrastructure.logging import bind_context

router = APIRouter(prefix="/api/agent", tags=["任务"])


@router.post(
    "/chat",
    status_code=202,
    response_model=ApiResponse[TaskCreatedData],
    summary="创建分析任务",
    description=(
        "登记一个分析任务并返回 202。Phase 1 只落库不入队，任务会停在 QUEUED；"
        "入队与执行在 Phase 1.5 接入。\n\n"
        "带 `Idempotency-Key` 时，同一用户在该键有效期内的重复请求"
        "返回同一个任务；若该键已被用于**不同**的请求体，返回 409。"
    ),
    dependencies=[Depends(require_create_task_rate_limit)],
    responses=error_responses(
        ErrorCode.INVALID_ARGUMENT,
        ErrorCode.AUTHENTICATION_REQUIRED,
        ErrorCode.ACCESS_DENIED,
        ErrorCode.RATE_LIMITED,
        ErrorCode.TASK_CONFLICT,
        ErrorCode.INTERNAL_ERROR,
    ),
)
async def create_task(
    payload: ChatRequest,
    user: CurrentUser,
    service: TaskServiceDep,
    settings: SettingsDep,
    trace_id: TraceIdDep,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            max_length=128,
            description="幂等键，同用户范围内有效；长度上限对应 agent_task.idempotency_key",
        ),
    ] = None,
) -> ApiResponse[TaskCreatedData]:
    # 长度上限按配置实时校验（FR-CHAT-001：默认 4,000 字，**可配置**）。
    # 不在 ChatRequest 上写死 max_length，否则配置调大后校验反而更严。
    if len(payload.message) > settings.chat_message_max_length:
        raise AgentError(
            ErrorCode.INVALID_ARGUMENT,
            f"问题长度不能超过 {settings.chat_message_max_length} 字",
        )

    result = await service.create_task(
        user=user,
        message=payload.message,
        conversation_id=payload.conversation_id,
        idempotency_key=idempotency_key,
        # 任务的 trace_id 就是本次请求的 trace_id：API 与 Worker 的 span 必须同链
        # （详细设计 19.4.1），否则 Phase 11 的跨进程追踪验收过不了
        trace_id=trace_id,
    )
    task = result.task
    # 绑定后本次请求的后续日志自动带上任务标识，便于按 task_id 检索
    bind_context(task_id=task.id, conversation_id=task.conversation_id, status=task.status.value)

    return ApiResponse(
        code=SuccessCode.ACCEPTED,
        message="任务已创建" if result.created else "命中幂等键，返回已存在的任务",
        data=TaskCreatedData(
            task_id=task.id,
            conversation_id=task.conversation_id,
            trace_id=task.trace_id,
            status=task.status,
            stream_url=f"/api/agent/tasks/{task.id}/stream",
        ),
        trace_id=trace_id,
    )
