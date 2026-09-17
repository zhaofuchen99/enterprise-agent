"""RAG 检索的入参、中间产物与结果（详细设计 11.7 / 9.1 / 9.2）。

## 为什么与 SQL Tool 一样另立一份 schema

`ToolResult.payload` 是 `dict`（9.2 的说明：它要穿过 LangGraph State 的序列化边界），
而"进 State 前必须经 Pydantic 校验"（开发流程 5.3）靠的正是各 Tool 自己的结果模型。
RAG 的结果形状与 SQL 差得远（一个是表格行，一个是带定位信息的分块），
共用一个模型只会让两边都长出用不到的字段。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.tools.rag.metadata import ChunkMetadata

#: 工具名。与 `app/tools/base.py` 的 `ToolName` 一致，**改名等于历史数据失联**
#: （这个字符串会写进 `agent_tool_call.tool_name` 与 Trace）。
TOOL_NAME: Literal["rag_retrieve"] = "rag_retrieve"

#: 改写后每个查询最多保留多少字。短查询是 Query Rewrite 的目的，
#: 超长的"改写"多半是把原问题抄了一遍再补几句，对召回没有帮助。
_MAX_QUERY_CHARS = 60


class RagQueryArgs(BaseModel):
    """`rag_retrieve` 的入参（详设 9.1 的 `BaseTool.execute(args, ctx)`）。

    Attributes:
        question: 用户原问题。**查询改写以它为准**，不是拿上一次改写的结果再改。
        objective: 该步骤的分析目标（来自 Planner）。与 question 分开同 SQL Tool：
            追问场景下二者不同，「那华东呢」是 question，「补齐华东的同口径制度依据」
            才是目标。改写时一起给模型，能让短问题也改出有信息量的查询。
        document_types: 文档类型白名单（11.7 第 1 步的 Intent 之一），空表示不限。
        departments: 部门白名单，空表示不限。
        as_of: 问题所问的时点，用于**生效区间过滤**（11.7 第 3 步）。
            None 表示不限——但 11.7 明写"有效期先作为标量过滤"，
            调用方在能确定时点时必须传：同名制度的两个版本共存时，
            不传就等于让 v1.0 与 v2.0 同时进候选。
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    objective: str = ""
    document_types: tuple[str, ...] = ()
    departments: tuple[str, ...] = ()
    as_of: date | None = None


class QueryRewrite(BaseModel):
    """Query Rewrite 的结构化输出（11.7 第 2 步）。

    **只给查询文本，不给理由**：理由是给模型自己理清思路用的，
    落进 State 只会让下游多读一段没有约束力的文字。这与 SQL Tool 的
    `explanation` 口径一致——那里也只要求说明采用了哪些业务口径。

    `queries` 的数量上限在检索侧按配置截断，不在这里限制：
    模型给 5 条时把它判成"输出不合规"并重试，代价是一次白花的模型调用，
    而多出来的两条本来就不影响结果（多召回一些排名靠后的候选而已）。
    """

    model_config = ConfigDict(extra="ignore")

    queries: tuple[str, ...] = Field(min_length=1)


class RetrievedChunk(BaseModel):
    """一条候选（11.7 第 7 步的 Top 8 之一）。

    `dense_score` 与 `fusion_score` **必须都留下**，它们的量纲完全不同：

    - `fusion_score` 是 RRF 分（`Σ 1/(k+rank)`，约 0.016–0.033），**只可比大小**；
    - `dense_score` 是余弦相似度（[0,1]），它才是**绝对量级**的相关性信号。

    只留一个的话，排查"为什么这条被召回"会立刻卡住：
    分不清它是排名凑巧靠前，还是真的语义接近。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    fusion_score: float
    #: **可空**：只被稀疏路召回的候选没有稠密分——它压根没被稠密路评估过。
    #: 补 0.0 会让它看起来像"语义完全不相关"，那是个我们并不知道的结论，
    #: 而下游（Reviewer、冲突检测）一旦读到就会把它当成事实。
    dense_score: float | None = None
    #: 融合后的名次，1 起
    rank: int
    metadata: ChunkMetadata


class RetrievalOutcome(BaseModel):
    """一次检索的完整产出（`Retriever.retrieve` 的返回）。

    与 `RagToolResult` 分开：这一层还没有 `call_id` / `evidence`（那是 Tool 的事），
    而**改写过程**是这一层才有的信息。合成一个模型的话，
    `Retriever` 就得凭空造一个 `call_id` 出来。
    """

    model_config = ConfigDict(frozen=True)

    #: 实际用于检索的查询。改写成功时是改写结果，降级时就是原问题本身——
    #: **调用方必须能看到实际用了什么**，「检索结果不对」与「改写改坏了」
    #: 是两条完全不同的排查路径。
    queries: tuple[str, ...]
    #: 改写被丢弃（模型不可用或输出不合规）而退化用原问题。降级路径的可见信号。
    rewrite_degraded: bool
    #: 融合后、取 Top 8 之前的候选数（11.7 第 5 步的上限是 30）
    candidate_count: int
    candidates: tuple[RetrievedChunk, ...]
    #: 本次生效的相关性阈值与实测最好余弦。**两个都要留下**：
    #: 阈值调错了与语料真的没有内容，症状都是"返回空"，而处置完全不同。
    relevance_threshold: float
    best_dense_score: float
    #: 问题里的主题词中，语料从未出现过的那些（未登录词）。
    #: 它是 `NO_RELEVANT_KNOWLEDGE` 的另一个判据，见 `Retriever._unseen_topics`——
    #: 两个判据是「或」的关系，所以**两个都要留下**：
    #: 「阈值调高了」与「语料真没见过这个词」症状相同，处置完全不同。
    unseen_topics: tuple[str, ...] = ()
    no_relevant_knowledge: bool
    duration_ms: int


class RagToolResult(BaseModel):
    """RAG Tool 的结果（进 `ToolResult.payload`）。"""

    model_config = ConfigDict(frozen=True)

    call_id: str
    queries: tuple[str, ...]
    rewrite_degraded: bool
    candidate_count: int
    chunks: tuple[RetrievedChunk, ...]
    relevance_threshold: float
    best_dense_score: float
    #: 见 `RetrievalOutcome.unseen_topics`
    unseen_topics: tuple[str, ...] = ()
    duration_ms: int
    warnings: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        """→ `ToolResult.payload`。

        **JSON 模式**：它会随 `ToolResult` 进 LangGraph State 并被序列化，
        裸 `date` / 枚举在那里会炸；用 `mode="json"` 让转换跟着模型走，
        而不是在这里手写一遍"哪些字段要转"。
        """
        return self.model_dump(mode="json")


__all__ = [
    "TOOL_NAME",
    "QueryRewrite",
    "RagQueryArgs",
    "RagToolResult",
    "RetrievalOutcome",
    "RetrievedChunk",
]
