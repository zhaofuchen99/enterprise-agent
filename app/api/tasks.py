"""任务查询、轨迹与事件流（详细设计 17.2 / 17.3 / 17.4）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import (
    ArtifactsDep,
    CurrentUser,
    EventBusDep,
    SettingsDep,
    StreamPrincipalDep,
    StreamTokenDep,
    TaskServiceDep,
    TraceIdDep,
    require_task_status_rate_limit,
)
from app.api.schemas import (
    ApiResponse,
    StreamTokenData,
    SuccessCode,
    TaskDetailData,
    TaskTraceData,
    TraceEventData,
    error_responses,
)
from app.api.stream import SseFrame, format_frame, stream_frames
from app.core.errors import AgentError, ErrorCode
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


@router.post(
    "/tasks/{task_id}/stream-token",
    response_model=ApiResponse[StreamTokenData],
    summary="签发 SSE 订阅令牌",
    description=(
        "用请求头里的 Access Token 换一个**短时效、一次性、只对该任务有效**的令牌。\n\n"
        "为什么需要它：浏览器的 `EventSource` 设不了请求头，事件流只能靠查询参数认证；"
        "而把长期有效的 Access Token 放进 URL 会泄露给访问日志与浏览器历史"
        "（详细设计 17.3.1）。"
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
async def create_stream_token(
    task_id: str,
    user: CurrentUser,
    service: TaskServiceDep,
    tokens: StreamTokenDep,
    trace_id: TraceIdDep,
) -> ApiResponse[StreamTokenData]:
    # **先判归属再签发**：令牌等于一条"只读这条流"的通行证，
    # 发给非所有者就等于把别人的执行细节交出去
    task = await service.get_task(user=user, task_id=task_id)
    issued = await tokens.issue(user_id=user.id, task_id=task.id)
    bind_context(task_id=task.id, conversation_id=task.conversation_id)
    return ApiResponse(
        code=SuccessCode.OK,
        message="签发成功",
        data=StreamTokenData(
            stream_token=issued.token,
            expires_in=issued.expires_in,
            stream_url=issued.stream_url,
        ),
        trace_id=trace_id,
    )


@router.get(
    "/tasks/{task_id}/stream",
    summary="订阅任务事件流（SSE）",
    description=(
        "以 `text/event-stream` 持续推送一个任务的事件（详细设计 17.3 / 18.2）。\n\n"
        "**鉴权两条来路**：`Authorization` 请求头，或 `?token=<stream_token>`"
        "（浏览器 `EventSource` 只能走后者，令牌由 `stream-token` 接口签发）。\n\n"
        "**断线重连**：把最后一条事件的 `id` 通过 `Last-Event-ID` 请求头带回来，"
        "服务端从它之后补齐。流已被清理时先发一条 `snapshot` 并标 `replay_lost=true`。\n\n"
        "事件顺序与终止事件由 `done` 界定；`heartbeat` 只在空闲时出现。"
    ),
    response_class=StreamingResponse,
    responses=error_responses(
        ErrorCode.AUTHENTICATION_REQUIRED,
        ErrorCode.ACCESS_DENIED,
        ErrorCode.TASK_NOT_FOUND,
        ErrorCode.RATE_LIMITED,
        ErrorCode.INTERNAL_ERROR,
    ),
)
async def stream_task_events(
    task_id: str,
    request: Request,
    principal: StreamPrincipalDep,
    service: TaskServiceDep,
    bus: EventBusDep,
    settings: SettingsDep,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    # **归属校验与限流都在建连之前**：一旦开始返回流，HTTP 状态码就已经发出去了，
    # 之后再报 403 只能表现为"流里出现一条错误事件"——那与"任务跑失败了"长得一样
    if principal.token_task_id is not None and principal.token_task_id != task_id:
        raise AgentError(
            ErrorCode.ACCESS_DENIED,
            "订阅令牌与任务不匹配",
            details={"token_task_id": principal.token_task_id},
        )
    task = await service.get_task(user=principal.user, task_id=task_id)
    # 限流**在这里显式调用**而不是走 `dependencies=[...]`：那个依赖注入的是
    # `CurrentUser`（只认请求头），而这条端点的凭据有两条来路，
    # 走依赖会让"带订阅令牌的请求绕过限流"——而绕过是静默的
    limiter = request.app.state.rate_limiter
    result = await limiter.check(
        scope="task:stream",
        subject=principal.user.id,
        limit=settings.rate_limit_stream_per_minute,
        window_seconds=60,
    )
    if not result.allowed:
        raise AgentError(
            ErrorCode.RATE_LIMITED,
            f"订阅过于频繁，请 {result.reset_after_seconds} 秒后重试",
        )

    bind_context(task_id=task.id, conversation_id=task.conversation_id)
    frames = stream_frames(task=task, bus=bus, settings=settings, after_id=last_event_id)
    return StreamingResponse(
        _framed(frames),
        media_type="text/event-stream",
        headers={
            # 中间层不得缓存或缓冲：否则客户端要等到缓冲写满才看见第一批事件
            # （Nginx 侧还要配 proxy_buffering off，见详细设计 20.2）
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _framed(frames: AsyncIterator[SseFrame]) -> AsyncIterator[str]:
    """帧对象 → SSE 文本。**格式化集中在这里**，因此它只有一处实现、
    也只有一个地方会被测（`format_frame` 本身是纯函数）。"""
    async for frame in frames:
        yield format_frame(frame)
