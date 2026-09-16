"""Tool 通用契约（详细设计 9.1 / 9.2）。

**这个模块不得 import fastapi**：`app/tools/**` 属于 Worker 侧（分层检查器 L2），
Worker 进程要能脱离 Web 框架独立扩缩容。

## 为什么契约先于实现

Phase 4 只有 SQL 一个 Tool，写不写这套基类都能跑通。之所以现在就定下来，
是因为详设 9.3 的「Tool 执行器」要统一做超时、取消、脱敏、持久化这四件事，
而它必须对**所有** Tool 一视同仁——等第二个 Tool（Phase 5 的 RAG）写完再抽象，
两个 Tool 已经各自长出了不一样的结果形状，那时统一等于重写。

## 一条纪律：Tool 不自己决定重试

详设 9.3 明写「Tool 不自行决定是否重试。它只返回 `retryable` 和错误类别，
由 Graph 路由统一控制」。这条在代码里的落点是：`ToolResult.error` 里带的是
**错误类别**，不是「要不要重试」这个决定。SQL Tool 内部的**自修复**看似违反它，
实则不然——自修复是 **SQL 语义层**的定向修复（语法、别名、字段名），
预算来自 `loop.max_sql_repairs`，对调用方完全透明：调用方只看得到
「这个 Tool 成功/失败、尝试了几次」，看不到修复过程。两者的边界见
`app/tools/sql/tool.py` 的说明。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.domain.evidence import Evidence
from app.domain.user import PermissionScope

#: 工具名。取值与详细设计 9.2 的 `ToolResult.tool` 一致——
#: 这个字符串会写进 `agent_tool_call.tool_name` 与 Trace，改名等于历史数据失联。
ToolName = Literal["sql_query", "rag_retrieve", "web_search"]

#: 工具执行状态。
ToolStatus = Literal["SUCCEEDED", "FAILED", "PARTIAL"]

#: 错误类别（详细设计 9.4 的错误分类表）。
#: 决定**调用方**的默认动作，工具自己不改写它的语义。
ErrorClass = Literal[
    "VALIDATION",  # 参数或输出 Schema 无效 → 模型结构化重试 1 次
    "SECURITY",  # SQL 越权、注入、敏感数据 → 不重试，终止步骤
    "USER_INPUT",  # 时间或指标不明确 → 澄清
    "TRANSIENT",  # 网络抖动、429、短暂连接失败 → 指数退避重试 1 次
    "REPAIRABLE",  # SQL 语法、别名、字段类型错误 → 定向修复，最多 2 次
    "EMPTY_RESULT",  # SQL 空集、RAG 无相关证据 → 交 Reviewer 判断
    "TIMEOUT",  # 工具超时 → 可降级则继续，否则失败
    "INTERNAL",  # 未分类代码错误 → 失败并记录堆栈
]


class ToolContext(BaseModel):
    """一次工具调用的上下文（详细设计 9.1）。

    Attributes:
        user_id: 发起用户，用于审计与配额。
        task_id / step_id: 落 `agent_tool_call` 时的归属键。
        trace_id: 业务 trace ID，与 OTel 的 trace 并行（见 `observability.py`）。
        permission_scope: 数据权限范围。**由 `User.permission_scope()` 产出**，
            工具不得自行组装（见 `PermissionScope` 的说明）。
        deadline_at: 本步骤的截止时刻。超时判定用**绝对时刻**而不是「剩余秒数」：
            一串嵌套调用各扣各的剩余时间，会在每层重算时出现
            「每层都还剩一点、合起来早就超了」的漂移。
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    task_id: str
    step_id: str = ""
    trace_id: str = ""
    permission_scope: PermissionScope = Field(default_factory=PermissionScope)
    deadline_at: datetime | None = None

    def remaining_seconds(self, now: datetime) -> float | None:
        """距截止还有多少秒；未设截止时返回 None。"""
        if self.deadline_at is None:
            return None
        return (self.deadline_at - now).total_seconds()


class ToolError(BaseModel):
    """工具失败的结构化描述（详细设计 9.2）。

    `message` 面向用户，`safe_detail` 面向排查且**必须已脱敏**——
    详设 10.6 明确「SQL 结果不直接写应用日志」，那句报错里回显的 SQL
    本身可能带着客户名称之类的取值。
    """

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    error_class: ErrorClass
    retryable: bool = False
    safe_detail: str | None = None


class ToolResult(BaseModel):
    """工具统一结果（详细设计 9.2）。

    `payload` 用 `dict` 而非泛型：它要能穿过 LangGraph State 的序列化边界。
    **进 State 前仍须经 Pydantic 校验**（开发流程 5.3）——各 Tool 自己的
    结果模型（如 `SqlToolResult`）就是这个校验，`payload` 只是它的落点。
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: ToolName
    status: ToolStatus
    started_at: datetime
    finished_at: datetime
    summary: str
    payload: dict[str, Any] | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    error: ToolError | None = None

    @property
    def duration_ms(self) -> int:
        return int((self.finished_at - self.started_at).total_seconds() * 1000)


class BaseTool(Protocol):
    """Tool 接口（详细设计 9.1）。

    参数用 `BaseModel` 而不是 `dict`：Tool 的入参同样来自模型输出，
    进 State 之前必须先经 Pydantic 校验，这里是把那条纪律写进类型签名。
    """

    name: ToolName

    async def execute(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...


__all__ = [
    "BaseTool",
    "ErrorClass",
    "ToolContext",
    "ToolError",
    "ToolName",
    "ToolResult",
    "ToolStatus",
]
