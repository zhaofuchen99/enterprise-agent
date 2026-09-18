"""HTTP 契约模型：请求体、统一响应外壳、错误结构。

**统一响应外壳**：成功与失败共用同一个形状
`{code, message, data, trace_id, retryable}`。

这是把三份文档的三种写法合成一个的结果——需求规格 7.1 要求「响应包含 trace_id」、
7.3 定义了带 trace_id/retryable 的错误结构、开发流程 6.2 要求
「统一响应包装 {code, message, data}」。合成一件事而不是让成功/失败各一套，
前端就只需要写一套解析逻辑。已回写详细设计。

**长度上限为什么不写成 `max_length`**：FR-CHAT-001 规定问题长度「默认 4,000 字，
**可配置**」。若在模型上写死 4000，配置调大后校验反而更严，是自相矛盾的。
因此模型只约束下界，上界由路由按 `CHAT_MESSAGE_MAX_LENGTH` 实时校验。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field

from app.core.errors import ErrorCode, http_status_of
from app.core.ids import IdPrefix, id_pattern
from app.domain.task import Task, TaskStatus
from app.domain.user import User, UserRole


class SuccessCode(StrEnum):
    """成功响应的 `code` 取值（详细设计 17.1 / 17.3.1 / 17.4）。"""

    OK = "OK"
    ACCEPTED = "ACCEPTED"


class ApiResponse[T](BaseModel):
    """所有接口的统一外壳。"""

    code: str = Field(description="成功为 OK / ACCEPTED；失败为详细设计 19.1 的错误码")
    message: str
    data: T | None = None
    trace_id: str = Field(description="本次请求的追踪 ID，与日志、Trace 中的一致")
    retryable: bool = Field(default=False, description="客户端是否可直接重试")


class ErrorResponse(BaseModel):
    """错误响应。单独定义是为了让 OpenAPI 上每个接口都能挂出错误用例。"""

    code: ErrorCode
    message: str
    data: None = None
    trace_id: str
    retryable: bool


# --------------------------------------------------------------------- 认证
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    #: repr=False：口令绝不能进日志或错误堆栈
    password: str = Field(min_length=1, max_length=128, repr=False)


class UserProfile(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: UserRole

    @classmethod
    def from_domain(cls, user: User) -> Self:
        return cls(
            user_id=user.id,
            username=user.username,
            display_name=user.display_name,
            role=user.role,
        )


class LoginData(BaseModel):
    access_token: str = Field(repr=False)
    # S105 是误报：这里说的是 OAuth 的「令牌类型」，不是令牌本身
    token_type: Literal["Bearer"] = "Bearer"  # noqa: S105
    expires_in: int = Field(description="Access Token 有效期（秒）")
    user: UserProfile


# --------------------------------------------------------------------- 任务
class ChatRequest(BaseModel):
    conversation_id: Annotated[str | None, Field(pattern=id_pattern(IdPrefix.CONVERSATION))] = None
    message: str = Field(
        min_length=1,
        description="用户问题，长度上限由配置 CHAT_MESSAGE_MAX_LENGTH 决定（默认 4000）",
    )
    #: 详细设计 17.1 只定义了 markdown。新增取值要先回写文档，
    #: 不能靠「反正模型会忽略」蒙混过去——那会让契约和实现悄悄分家。
    response_format: Literal["markdown"] = "markdown"


class TaskCreatedData(BaseModel):
    task_id: str
    conversation_id: str
    trace_id: str
    status: TaskStatus
    stream_url: str


class TaskDetailData(BaseModel):
    """任务详情（详细设计 17.2）。

    17.2 要求返回「任务状态、意图、步骤进度、答案、证据、冲突、限制及时间」。
    前四样来自 `agent_task` 自己的列，后四样从 `plan_json` / `result_json`
    两个 JSON 列里取（`TaskOutcome` 落进去的）。

    **这几个字段是"可追溯"对外的唯一出口**：审查结论、证据、冲突、限制
    如果只在进程内可见，那么"答案为什么成立"就无从查起——
    而这正是本项目存在的理由。
    """

    task_id: str
    parent_task_id: str | None = None
    conversation_id: str
    trace_id: str
    status: TaskStatus
    intent: str | None = None
    query_text: str
    final_answer_md: str | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    trace_incomplete: bool = False
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    # --- 来自 plan_json（16.5 的计划摘要）---
    #: 步骤进度：每步的工具、目标、状态、是否"跑了但空"
    steps: list[dict[str, Any]] = Field(default_factory=list)
    #: `reflect` 的最终判定与理由（SUFFICIENT / EXPAND / BLOCKED…）。
    #: 它比答案本身更能回答"这个结论有多硬"——尤其是 EXPAND 过的时候。
    progress_decision: str | None = None
    progress_reason: str | None = None
    plan_revision: int = 0

    # --- 来自 result_json（16.5 的结构化结果）---
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    #: 已检出的多源冲突描述（13.3）。**未判定哪一方为准**——
    #: 判定属 Phase 9，这里的措辞不能让人以为已经判过了。
    conflicts: list[str] = Field(default_factory=list)
    #: 审查结论（14.2）。`None` 表示本任务没走到审查那一步
    review: dict[str, Any] | None = None
    #: 答案的限制：证据覆盖不到的部分、失败的数据源、未明确的前提
    limitations: list[str] = Field(default_factory=list)

    @classmethod
    def from_domain(cls, task: Task) -> Self:
        plan = task.plan_json or {}
        payload = task.result_json or {}
        return cls(
            steps=list(plan.get("steps") or []),
            progress_decision=plan.get("decision"),
            progress_reason=plan.get("reason"),
            plan_revision=int(plan.get("revision") or 0),
            evidence=list(payload.get("evidence") or []),
            conflicts=list(payload.get("conflicts") or []),
            review=payload.get("review"),
            limitations=list(payload.get("limitations") or []),
            task_id=task.id,
            parent_task_id=task.parent_task_id,
            conversation_id=task.conversation_id,
            trace_id=task.trace_id,
            status=task.status,
            intent=task.intent,
            query_text=task.query_text,
            final_answer_md=task.final_answer_md,
            error_code=task.error_code,
            error_message=task.error_message,
            trace_incomplete=task.trace_incomplete,
            queued_at=task.queued_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )


def error_responses(*codes: ErrorCode) -> dict[int | str, dict[str, Any]]:
    """给接口声明 OpenAPI 错误用例（开发流程 6.2 门禁：公开接口要有错误用例）。

    `4XX` 这条范围响应做两件事：如实说明「所有客户端错误都是同一个外壳」，
    以及**顶掉 FastAPI 自动补的 422**。FastAPI 见接口有参数就加一条
    `422 -> HTTPValidationError`，但本项目把参数校验失败映射成
    400 INVALID_ARGUMENT（需求规格 7.3），那条自动文档描述的是不会发生的行为。
    OpenAPI 里留一条错的，比留空更误导人。
    """
    responses: dict[int | str, dict[str, Any]] = {
        http_status_of(code): {"model": ErrorResponse, "description": code.value} for code in codes
    }
    responses["4XX"] = {"model": ErrorResponse, "description": "客户端错误统一外壳"}
    return responses
