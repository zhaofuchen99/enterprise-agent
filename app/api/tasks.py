"""任务查询（详细设计 17.2）。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.deps import (
    ArtifactsDep,
    CurrentUser,
    TaskServiceDep,
    TraceIdDep,
    require_task_status_rate_limit,
)
from app.api.schemas import (
    ApiResponse,
    SuccessCode,
    TaskDetailData,
    TaskTraceData,
    TraceEventData,
    error_responses,
)
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


@router.get(
    "/tasks/{task_id}/trace",
    response_model=ApiResponse[TaskTraceData],
    summary="查询任务的执行轨迹",
    description=(
        "按执行顺序返回每个节点的进入/离开事件。\n\n"
        "**这张表是事件流的权威重放来源**（详细设计 18.3）：Redis Stream 会被 "
        "MAXLEN 裁剪，客户端断线重连后要补历史必须回到这里。\n\n"
        "`after_sequence` 用于增量拉取——语义是「我已经有的最后一条」，"
        "因此是**严格大于**，用 `>=` 会让每次重连都重复拿到同一条。"
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
async def get_task_trace(
    task_id: str,
    user: CurrentUser,
    service: TaskServiceDep,
    artifacts: ArtifactsDep,
    trace_id: TraceIdDep,
    after_sequence: Annotated[int | None, Query(ge=0, description="只返回序号大于它的事件")] = None,
    limit: Annotated[int | None, Query(ge=1, le=500, description="最多返回多少条")] = None,
) -> ApiResponse[TaskTraceData]:
    # **先走一次 service 拿任务**：权限判定（跨用户 403）与任务存在性都在那里，
    # 直接用仓储读轨迹会绕过这两道检查——而轨迹里带着节点名与耗时，
    # 是**未授权用户不该看到**的执行细节。
    task = await service.get_task(user=user, task_id=task_id)
    events = await artifacts.list_trace_events(task.id, after_sequence=after_sequence, limit=limit)
    bind_context(task_id=task.id, conversation_id=task.conversation_id)
    return ApiResponse(
        code=SuccessCode.OK,
        message="查询成功",
        data=TaskTraceData(
            task_id=task.id,
            trace_id=task.trace_id,
            events=[
                TraceEventData(
                    sequence=item.sequence,
                    type=item.event_type,
                    node=item.node,
                    status=item.status,
                    duration_ms=item.duration_ms,
                    timestamp=item.created_at,
                )
                for item in events
            ],
            trace_incomplete=task.trace_incomplete,
        ),
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
