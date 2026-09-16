"""任务查询（详细设计 17.2）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import (
    CurrentUser,
    TaskServiceDep,
    TraceIdDep,
    require_task_status_rate_limit,
)
from app.api.schemas import ApiResponse, SuccessCode, TaskDetailData, error_responses
from app.core.errors import ErrorCode
from app.infrastructure.logging import bind_context

router = APIRouter(prefix="/api/agent", tags=["任务"])


@router.get(
    "/tasks/{task_id}",
    response_model=ApiResponse[TaskDetailData],
    summary="查询任务状态与结果",
    description=(
        "返回任务状态、意图、最终答案与错误摘要。"
        "任务不存在返回 404，跨用户访问返回 403（ADMIN 可跨用户查看）。\n\n"
        "步骤进度、证据、冲突与限制要等对应阶段产出后才有内容，"
        "届时按各自 Schema 增补字段。"
    ),
    dependencies=[Depends(require_task_status_rate_limit)],
    responses=error_responses(
        ErrorCode.AUTHENTICATION_REQUIRED,
        ErrorCode.ACCESS_DENIED,
        ErrorCode.TASK_NOT_FOUND,
        ErrorCode.RATE_LIMITED,
        ErrorCode.INTERNAL_ERROR,
    ),
)
async def get_task(
    task_id: str,
    user: CurrentUser,
    service: TaskServiceDep,
    trace_id: TraceIdDep,
) -> ApiResponse[TaskDetailData]:
    # 不对 task_id 做格式校验：需求规格 FR-CHAT-002 要求「任务不存在返回 404」，
    # 格式不对也属于「不存在」，不该变成 400。
    task = await service.get_task(user=user, task_id=task_id)
    bind_context(task_id=task.id, conversation_id=task.conversation_id, status=task.status.value)
    return ApiResponse(
        code=SuccessCode.OK,
        message="查询成功",
        data=TaskDetailData.from_domain(task),
        trace_id=trace_id,
    )


@router.post(
    "/tasks/{task_id}/cancel",
    status_code=202,
    response_model=ApiResponse[TaskDetailData],
    summary="请求取消任务",
    description=(
        "仅 QUEUED / RUNNING 的任务可取消，进入 `CANCEL_REQUESTED` 后由 Worker 确认。\n\n"
        "**这是请求不是命令**：Worker 可能在另一个实例上、也可能正阻塞在一次"
        "模型的调用里，因此底层不可中断时会等它返回后再转 `CANCELLED`。\n\n"
        "取消是幂等的：重复取消同一任务返回相同结果，不报错；"
        "已结束（SUCCEEDED / FAILED / CANCELLED）的任务返回 409。"
    ),
    dependencies=[Depends(require_task_status_rate_limit)],
    responses=error_responses(
        ErrorCode.AUTHENTICATION_REQUIRED,
        ErrorCode.ACCESS_DENIED,
        ErrorCode.TASK_NOT_FOUND,
        ErrorCode.TASK_CONFLICT,
        ErrorCode.RATE_LIMITED,
        ErrorCode.INTERNAL_ERROR,
    ),
)
async def cancel_task(
    task_id: str,
    user: CurrentUser,
    service: TaskServiceDep,
    trace_id: TraceIdDep,
) -> ApiResponse[TaskDetailData]:
    task = await service.cancel_task(user=user, task_id=task_id)
    bind_context(task_id=task.id, conversation_id=task.conversation_id, status=task.status.value)
    return ApiResponse(
        code=SuccessCode.ACCEPTED,
        message="取消请求已登记",
        data=TaskDetailData.from_domain(task),
        trace_id=trace_id,
    )
