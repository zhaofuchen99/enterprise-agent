"""SQL 生成与修复的 prompt 模板（开发流程 6.6 施工项 2、详细设计 10.8）。

**模板里不写 JSON 格式说明**：让模型「只输出 json」并给出格式示例是服务方的
协议要求，由 `model_gateway.build_json_contract` 依据 `SqlCandidate` 的 Schema
统一追加。写在这里的话，Schema 一改就得记得改模板，而忘了改的表现是
「模型输出对不上 Schema」——会被误判成模型能力问题。

**安全规则摘要必须进 prompt**，尽管校验器不看它：详设 10.5 的分层责任里，
LLM 那一层的职责正是「根据 Schema 生成候选 SQL、根据安全错误修正」。
把规则提前讲清楚，省下的是一次「生成 → 被拦 → 修复」的往返；
但**它不构成任何保证**——真正拦得住的是 `validator.py`。
模板里那句「这些规则由代码强制执行」不是客套，是给改这份 prompt 的人看的。

**改动模板必须升 `version`**（`PromptTemplate` 的约束）：`prompt_version`
会随调用记录落库，改文本不升版本会让历史记录指向两段不同的文本，
详设 22.9 的回归归因就断了。
"""

from __future__ import annotations

from typing import Final

from app.agent.prompts.base import PromptTemplate

#: 安全规则摘要（详设 10.4 的前 9 步 + 第 10 步的 LIMIT）。
#: 与校验器共用同一份文本，是为了让「模型看到的规则」与「代码执行的规则」
#: 不会各自漂移——两边不一致时，表现是模型反复生成必然被拒的 SQL，
#: 修复预算被白白烧光。
SAFETY_RULES: Final[str] = """\
以下规则由代码强制执行，违反的 SQL 会被直接拒绝且无法通过修复绕过：
1. 只能生成一条语句，且必须是 SELECT（或最终为 SELECT 的 WITH）；
2. 禁止 INSERT / UPDATE / DELETE / REPLACE / MERGE / DDL / 事务控制 / 存储过程；
3. 禁止注释、INTO OUTFILE、LOAD_FILE、系统变量、系统库（information_schema 等）；
4. 只能使用上面 Schema 段里列出的表和列，不得引用未列出的表；
5. 禁止 SELECT *，必须逐列写出（聚合函数如 COUNT(*) 不受此限）；
6. JOIN 只能使用上面列出的 JOIN 关系，且必须带 ON 条件，不得使用逗号连接或 CROSS JOIN；
7. 只能使用常规的聚合、日期与字符串函数，禁止 SLEEP / BENCHMARK 等；
8. 明细查询必须带 LIMIT，且不超过 {max_rows}。若不写，系统会自动补上，
   但显式写出更利于你表达真实意图；
9. 时间范围一律用半开区间，即「>= 起点 AND < 终点」。例如 2025 年 Q3 要写成
   order_date >= '2025-07-01' AND order_date < '2025-10-01'，不要用 BETWEEN
   或闭区间——那会把 9 月 30 日之后的数据算进来。"""

#: SQL 生成模板。
SQL_GENERATION_PROMPT: Final[PromptTemplate] = PromptTemplate(
    name="sql_generate",
    version="1.0.0",
    template="""\
你是一名严谨的数据分析师，负责把业务问题翻译成一条可在 MySQL 8 上执行的查询。

{schema}

安全规则：
{rules}

业务问题：{question}
分析目标：{objective}

要求：
- 严格使用上面给出的指标口径计算，不要自行发明算法；指标口径里写明的过滤条件必须出现在 WHERE 中；
- 时间条件用题目里明确给出的区间。题目没有给年份时，使用数据里最近的一年；
- 为了便于下游核对，把时间区间、区域等过滤条件写成命名参数（形如 :start_date），
  并在 parameters 里给出取值，不要把日期直接拼进 SQL 文本；
- selected_tables / selected_columns 填这条 SQL 真正用到的表与列（写全，用于安全校验留痕）；
- expected_columns 填结果集的列名，顺序与 SELECT 一致；
- metric_codes 填用到的指标 code；
- explanation 只说明采用了哪些业务口径（例如「净销售额按含税减折扣减退货」），
  不要写推理过程。""",
)

#: SQL 修复模板（详设 10.8）。
#:
#: 只包含「原 SQL + 脱敏后的错误 + 同一份 Schema + 规则摘要」四样东西，
#: 这是 10.8 的明文要求。**不回灌上一次的完整模型输出**：修复需要的
#: 是错在哪，不是模型上次想了什么。
SQL_REPAIR_PROMPT: Final[PromptTemplate] = PromptTemplate(
    name="sql_repair",
    version="1.0.0",
    template="""\
你上一次生成的 SQL 没有通过校验或执行失败，请修正后重新给出完整 SQL。

{schema}

安全规则：
{rules}

原始 SQL：
{original_sql}

错误原因：
{error}

业务问题：{question}
分析目标：{objective}

要求：
- 只针对上面的错误原因修改，不要重写整条查询的逻辑；
- 表名与列名必须与 Schema 段完全一致（含大小写）；
- 其余输出字段的要求与上一次相同。""",
)

__all__ = ["SAFETY_RULES", "SQL_GENERATION_PROMPT", "SQL_REPAIR_PROMPT"]
