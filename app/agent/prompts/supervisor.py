"""Supervisor 的意图识别 prompt（详细设计 8.1 / 8.2）。

## 这一条 prompt 决定"Agent 是不是真的在选工具"

冲刺方案 §8.4 的第②条验收是「Agent 能**自主选择** SQL / RAG Tool」。
它的落点就是 `required_sources` 这个字段：模型必须**在两者之间选**，
而不是永远都调。专门有一条 FR 管这件事——FR-PLAN-002 业务规则 1：

> 简单指标查询不得强制调用 RAG。

所以模板里把"什么时候只用一路"写得比"什么时候两路都要"更具体：
模型对"多查一路总没坏处"是有倾向的，而那条倾向恰好把这条验收架空。

**输入只有问题本身**（详设 8.1：不得注入数据库凭证、全库 Schema、
与问题无关的历史对话）。会话摘要与指标目录的注入要等 memory 与
`agent_config` 落地，那时在 `{context}` 处扩展，**不是**在这里塞更多文本。

## 模板里不写 JSON 格式说明

与 SQL 侧同一条纪律：那是服务方的协议要求，由
`model_gateway.build_json_contract` 依 `IntentResult` 的 Schema 统一追加。
写在这里的话，Schema 一改就得记得同步改模板，而忘了改的表现是
「模型输出对不上 Schema」——会被误判成模型能力问题。
"""

from __future__ import annotations

from typing import Final

from app.agent.prompts.base import PromptTemplate

SUPERVISOR_PROMPT: Final[PromptTemplate] = PromptTemplate(
    name="supervisor_intent",
    version="1.0.0",
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

字段要求：
- metrics / dimensions 填问题里出现的指标名与维度名（用业务术语，如「净销售额」「华东」）；
- filters 填明确的筛选条件（如 {{"region": ["华东"]}}）；
- time_range 只在问题给了明确时间时填，start 含、end 不含（半开区间）；
- comparison 取 NONE / YOY / MOM / TARGET；
- confidence 是你对这次判断的把握，低于 0.5 时请改成 CLARIFICATION 并说明缺什么。

用户问题：{question}""",
)


__all__ = ["SUPERVISOR_PROMPT"]
