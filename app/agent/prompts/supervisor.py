"""Supervisor 的意图识别 prompt（详细设计 8.1 / 8.2）。

## 这一条 prompt 决定"Agent 是不是真的在选工具"

冲刺方案 §8.4 的第②条验收是「Agent 能**自主选择** SQL / RAG Tool」。
它的落点就是 `required_sources` 这个字段：模型必须**在两者之间选**，
而不是永远都调。专门有一条 FR 管这件事——FR-PLAN-002 业务规则 1：

> 简单指标查询不得强制调用 RAG。

所以模板里把"什么时候只用一路"写得比"什么时候两路都要"更具体：
模型对"多查一路总没坏处"是有倾向的，而那条倾向恰好把这条验收架空。

## 输入只有「本轮问题 + 会话摘要」

详设 8.1：不得注入数据库凭证、全库 Schema、与问题无关的历史对话。
`{context}` 是**已经归纳过的会话摘要**（`render_context` 把它渲染成文本），
不是把历史消息原样倒进来——那正是"与问题无关的历史对话"。

指标目录的注入要等 `agent_config` 落地，接口留在同一个占位符处：
那时在 `render_context` 的恒留段里补一段，**不是**在这里塞更多文本。

## 渲染与模板同生共死

`render_context` 与 `SUPERVISOR_PROMPT` 放在同一个模块里，因为**它产出的
就是这段模板要吃的文本**。放进 services 层的话，prompt 措辞一改就要去
services 里找那份渲染——而它与模板的对应关系没有任何地方记得住。

## 模板里不写 JSON 格式说明

与 SQL 侧同一条纪律：那是服务方的协议要求，由
`model_gateway.build_json_contract` 依 `IntentResult` 的 Schema 统一追加。
写在这里的话，Schema 一改就得记得同步改模板，而忘了改的表现是
「模型输出对不上 Schema」——会被误判成模型能力问题。
"""

from __future__ import annotations

from typing import Final

from app.agent.prompts.base import PromptTemplate
from app.domain.conversation import MessageRole
from app.domain.memory import ContextMessage, ConversationContext

#: 口径键 → 给模型看的中文标签。**只在这一个地方映射**：
#: 落库的键（`ENTITY_KEYS`）是英文的，改措辞不该动数据。
_ENTITY_LABELS: dict[str, str] = {
    "metric": "指标",
    "region": "区域",
    "channel": "渠道",
    "product_line": "产品线",
    "period": "期间",
    "comparison": "对比",
}
_COMPARISON_LABELS: dict[str, str] = {
    "YOY": "同比",
    "MOM": "环比",
    "TARGET": "对目标",
}
_NO_HISTORY: Final[str] = "（无历史：这是本会话的第一轮提问）"
_HISTORY_HEADER: Final[str] = "【会话历史】（较早→较新；助手回答已截断，仅供参考）"
_ENTITIES_HEADER: Final[str] = (
    "【上一轮系统解析出的口径】（**未经用户确认**；仅用于把「那 Q2 呢」这类"
    "代词补成本轮的完整问题，与本轮无关时请忽略）"
)


def render_context(context: ConversationContext | None, *, max_chars: int) -> str:
    """`ConversationContext` → supervisor prompt 里的 `{context}` 段。

    **超预算时从最旧的一轮开始丢**，但两段永不丢：口径段、以及**最近一轮**。
    口径段小（一行），而最近一轮是代词唯一可能的落点——把它裁掉之后
    这个功能就只剩"多花 token"，且没有任何症状。

    预算是**护栏不是计量**（`MemorySettings.context_max_chars` 的说明）。
    """
    if context is None or context.is_empty:
        # **一句明确的话，不是空串**：空串在 prompt 里是不可见的，
        # 模型分不清"没有历史"与"这段没填上"——后者是个 bug，
        # 而两者的处置完全相反（一个是如实按首轮处理，一个是去查装配）。
        return _NO_HISTORY

    entity_block = _entity_block(context)
    history = _history_lines(context.recent_messages, max_chars=max_chars)
    parts = [p for p in (history, entity_block) if p]
    return "\n".join(parts) if parts else _NO_HISTORY


def _entity_block(context: ConversationContext) -> str:
    if not context.confirmed_entities and not context.unresolved_fields:
        return ""
    lines = [_ENTITIES_HEADER]
    rendered = _render_entities(context.confirmed_entities)
    if rendered:
        lines.append(rendered)
    if context.previous_task_summary:
        # 摘要与口径分开摆：口径是**解析出来的结论**，摘要是**上一轮说了什么**。
        # 合成一句的话，模型会把答案里的措辞当成口径。
        lines.append(f"上一轮答复（摘要）：{context.previous_task_summary}")
    if context.unresolved_fields:
        lines.append(f"上一轮未确认的信息：{'、'.join(context.unresolved_fields)}")
    return "\n".join(lines)


def _render_entities(entities: dict[str, str]) -> str:
    items: list[str] = []
    for key, value in entities.items():
        label = _ENTITY_LABELS.get(key, key)
        if key == "comparison":
            value = _COMPARISON_LABELS.get(value, value)
        items.append(f"{label} {value}")
    return "｜".join(items)


def _history_lines(messages: tuple[ContextMessage, ...], *, max_chars: int) -> str:
    """按预算渲染历史，**从最旧的一轮开始丢**。"""
    if not messages:
        return ""
    lines = [
        f"{'用户' if item.role is MessageRole.USER else '助手'}：{item.content}"
        for item in messages
    ]

    # 最近一轮的起点：最后一条用户消息的下标。它及其之后的都**必须留下**。
    keep_from = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role is MessageRole.USER
        ),
        0,
    )
    kept = lines[keep_from:]
    used = sum(len(line) for line in kept)
    for index in range(keep_from - 1, -1, -1):
        if used + len(lines[index]) > max_chars:
            break
        kept.insert(0, lines[index])
        used += len(lines[index])
    return "\n".join([_HISTORY_HEADER, *kept])


SUPERVISOR_PROMPT: Final[PromptTemplate] = PromptTemplate(
    name="supervisor_intent",
    version="1.1.0",
    template="""\
你是企业数据分析助手的调度器。你的唯一任务是判断用户的问题**需要哪些数据源**，
不要回答问题本身。

可用数据源只有两个：
- sql：企业内部业务数据库。存放销售额、订单、区域、渠道、产品线等**指标数值**。
  能回答「是多少」「同比如何」「哪个区域最高」这类问题。
- rag：企业内部制度与报告知识库。存放管理办法、指标口径说明、经营分析报告。
  能回答「怎么规定的」「口径是什么」「报告里怎么解释的」这类问题。

判断规则：
1. 只问数值、排名、趋势的（例如「2025年Q3华东的净销售额是多少」）→ required_sources 只填 ["sql"]；
2. 只问制度、流程、口径、职责，不涉及具体数值的 → 只填 ["rag"]；
3. **同一个问题既要数据表现、又要制度依据或报告解释**时才两个都填
   （例如「Q3为什么下滑，制度上有什么要求」「净销售额的口径是怎么定的，
   实际值是多少」）；
4. **不要因为"多查一路更保险"就两个都填**。系统按你给的数据源逐路执行，
   多一路就是多一次查询与一条无关证据。拿不准时问自己：
   这个问题能不能只用一路答完？
5. 缺少会实质改变结论的信息（时间范围、指标名、区域），或问题本身有歧义
   （「上个季度」而当前是哪年没说清）时：intent 填 CLARIFICATION，
   required_sources 留空，并在 missing_fields 与 clarification_question
   里写清**最小**的一组待确认信息——一次问全，不要挤牙膏。
6. 要求写库、执行操作、索取敏感字段、或与企业经营分析无关的请求：
   intent 填 UNSUPPORTED，required_sources 留空。

7. **本轮问题可能依赖上文**（「那 Q2 呢」「和去年比呢」「换华南看看」）。
   这时**先把它补成一句自足的问题**填进 resolved_question，
   再按补全后的问题判 required_sources 与所有字段。
   补全的**唯一依据是下面的会话历史**，历史里没有的不要猜：
   猜出来的区域或期间会被真的拿去查库，而答案看起来完全正常。
   反过来，**历史与本轮无关时不要用它**（用户问的是一个全新问题）；
   本轮问题本来就自足时，resolved_question 填 null。

字段要求：
- metrics / dimensions 填问题里出现的指标名与维度名（用业务术语，如「净销售额」「华东」）；
- filters 填明确的筛选条件（如 {{"region": ["华东"]}}）；
- time_range 只在问题给了明确时间时填，start 含、end 不含（半开区间）；
- comparison 取 NONE / YOY / MOM / TARGET；
- resolved_question 是**名词性的完整问题**（不要写成「查询Q2数据」这种祈使句），
  且必须与下面各字段一致——它会被单独送去做 SQL 生成，两者对不上时
  以 resolved_question 为准的那条路径就会偏；
- confidence 是你对这次判断的把握，低于 0.5 时请改成 CLARIFICATION 并说明缺什么。

{context}

用户问题：{question}""",
)


__all__ = ["SUPERVISOR_PROMPT"]
