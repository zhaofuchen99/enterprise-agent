# CLAUDE.md

本文件是项目级约定，AI 编码助手与人类协作者共用。改动架构或约定时**同步更新本文件**。

---

## 项目

**企业智能数据分析与决策 Agent** —— 把自然语言问题转化为可追溯的数据结论的多 Agent 系统。
SQL 查询、知识检索、结果校验与冲突识别在同一个任务循环里协作，全过程可观测、可重放、有界收敛。

技术栈：Python 3.12 · FastAPI · LangGraph · MySQL 8 · Redis · Qdrant · OTel · arq · uv

## 当前进度

> 每推进一个 Phase 就更新这里，新会话靠它接上下文。

| Phase | 状态 | 备注 |
|---|---|---|
| 0 项目初始化 | ✅ 完成 | 骨架、配置、错误码、结构化日志、健康探针、分层约束检查、CI、pre-commit 密钥防护 |
| 1 基础 API | ✅ 完成 | 统一响应外壳、全局异常映射（400/401/403/404/409/429/500）、`trc_` trace 中间件、本地 JWT 登录、`/api/agent/chat` 与 `/api/agent/tasks/{id}` 占位、OpenAPI 定制 |
| 1.5 基础设施接入 | ✅ 完成 | Redis 键与 Lua 脚本、Redis 限流 + 降级 + 熔断、Redis 任务仓储（临时）、arq 队列与 Worker 骨架（含自愈重启）、事件流、取消链路、对象存储抽象、OTel 与跨进程 trace。`make check` 全绿（267 测试）+ 7 条 integration 用例；验收记录见开发流程 11.1 |
| 2 数据库 | ✅ 完成 | Alembic 初始化、16 张表迁移（可升可回滚）、仓储层换 MySQL（用户/会话/任务）、`RedisTaskRepository` 已删除、MySQL 就绪探针、**业务演示库 8 表 + 反向构造 49.9 万行数据（11 条断言全过，含 EXPLAIN 索引验证）**。剩余项见下方登记的「后续扩展」 |
| 3 LLM 封装 | ✅ 完成 | **TBC-04 结案**：`deepseek-flash`（云）+ `bge-m3`（本地 Ollama，1024 维）。`ModelGateway`（OpenAI 兼容单实现）、Fake 替身、`PromptTemplate` 版本机制、两条独立重试预算（详设 9.4 的 TRANSIENT / VALIDATION）、密钥脱敏、OTel span 埋点、`make model-smoke`。新错误码 `MODEL_OUTPUT_INVALID`（已回写详设 19.1）。`make check` 285 测试全绿 + 集成 29 通过（1 条待密钥跳过） |
| 4 SQL Tool | ✅ 完成 | **第 1 批收尾**：`SchemaProvider`（YAML 目录，8 表 + 指标口径 + JOIN/函数白名单）、`SqlGenerator`、`SqlValidator`（详设 10.4 的 12 步 + 绑定参数完整性）、只读 `SqlExecutor`、自修复 ≤2、Evidence 生成、`make sql` / `make eval-sql`。**安全与越权 28 条 100% 阻断**；**金标 SQL 10/10**（结果集等价；列形状一致 9/10）。`make check` 426 测试 + 集成 43 全绿 |
| 5 RAG | 🔄 进行中（8/12） | **已完成**：⑦Qdrant collection 接入层（服务端 + 内存替身同契约）、③中文分词与稀疏向量（`Vocabulary` + `make tokenize`）、**①业务词典生成**（`make dict`，96 条，回读产物自检）、**②语料 88 篇 + 10 类缺陷注入**（含 4 份手工 Prompt Injection）、④PDF/DOCX 工具链、**⑥`parser.py` + `chunker.py`**（四格式解析 + 清洗 + 分块，`make chunk`；全语料 1887 块，页眉页脚零泄漏、跨页表格表头还原）、**③后半词表**（`rag_vocab` 仓储 + 构建 + 快照，`make vocab`；2781 词条）、**④`s3` 存储实现**（与 local 同一份契约测试）。**未完成**：⑤`ingestion.py`（staging → 抽样 → 原子发布）、⑪入库幂等、⑧`retriever.py`、⑩`NO_RELEVANT_KNOWLEDGE`、⑫金标 20 条 + Recall@8、`verify-corpus` 门禁。⑨Reranker 按 TBC-04 维持后置 |
| 6–9 冲刺切片 | ⬜ 未开始 | 第 2 批剩余：最小 Graph（6 节点）→ Evidence → Reviewer-lite |

> ⚠️ **当前按「秋招冲刺方案」执行**：`docs/秋招冲刺方案.md` 覆盖了开发流程第 6 章的 Phase 顺序。
> 近期做 **SQL + RAG 双源垂直切片**，**分两批交付**：
>
> - **第 1 批**：Phase 2 精简版（10 表）→ Phase 3 → Phase 4 SQL Tool
> - **第 2 批**：Phase 5 RAG 裁剪版 → 最小 Graph（6 节点）→ Evidence → Reviewer-lite
>
> **Search / SSE / Reranker / 完整 Conflict / 评测扩集 / 生产部署后置**，切片内不要顺手做。
>
> ⚠️ **2026-09-17 变更**：项目负责人裁决 **RAG 不裁剪**（冲刺方案 §4.2 已撤销），
> 语料回到 80+ 篇 / 10 类缺陷、完整入库发布流程、词表扩容机制、`verify-corpus` 门禁、
> Recall@8 校准**都要做**。唯一维持后置的是 **Reranker**——依据是详设 TBC-04 的决议记录，不是本节裁剪。
> 每次开工前先读冲刺方案的第 8 节（冻结范围与分批）和第 11 节（面试口径纪律——
> 第 1 批做完时 RAG 还没做，简历与口述不得提前声称）。

**开工前**：先 `make ps` 看有状态组件是否在跑，没起来就 `make up`。

**当前仍然存在的临时约定**：

1. **任务体是空实现**：任务会以 `SUCCEEDED` 且 `answer` 为 `null` 结束。
   这是预期行为，不是 bug——本阶段验证的是执行链路，分析能力在 Phase 7。
2. **演示账号要跑 `make seed` 才会写进 `app_user`**：登录依赖它。
   从零启动的完整流程是 `make up && make migrate && make seed && make run`。
   （Phase 1 是进程内自动播种，Phase 2 起改为显式命令——构造函数里写库
   会让「测试为什么改了数据库」变得难以解释。）
3. 演示账号（`analyst`/`admin`）的 ID 由用户名确定性派生，**多实例之间是同一个用户**；
   真实用户仍走随机 ULID。这样多实例的限流与并发配额才能被端到端验收。
4. **单元测试注入内存仓储，不连 MySQL**（`app/tests/fakes.py` 的
   `build_memory_repositories`）。仓储的 MySQL 实现由 `make test-integration`
   下的**同一批契约断言**覆盖——两条路径各有各的验证，不重叠也不留空。
5. **模型网关已接入装配点，但还没有业务消费者**——Phase 4 的 SQL Tool 是第一个。
   现在接进来是为了让「模型调用可被替身替换」这条门禁在 Phase 3 就有实证。
   排查模型连通性用 `make model-smoke`（需 `.env` 里的 `MODEL_API_KEY`；
   向量那半条只需本机 Ollama 在跑）。
6. **`.env` 里的 `MODEL_API_KEY` 需要你手动填**：它是密钥，不入库、不由脚本生成。
   其余模型配置（`deepseek-flash` / `bge-m3` / 两个 base_url / 45s 超时）已写好。
7. **`configs/schema_catalog.yaml` 是安全白名单**：SQL Tool 放行哪些表、列、JOIN、
   函数，全由它决定，代码里没有第二份。往里面加一行等于放开一条权限，改动要 review，
   并**同时升 `version`**（版本号会随结果进 Trace，用于回答「昨天还能查今天为什么被拦」）。
8. **`SqlAttempt` 还没有落库的调用方**：SQL Tool 把每次尝试（生成/修复/执行）组装成
   该对象放进 `ToolResult.payload["attempts"]`，形状对齐 `agent_tool_call` 表。
   落库在 Phase 6/7 的 Tool 执行器里做——那时才有 `task_id` / `step_id`。
9. **SQL 自修复的预算在 `LOOP__MAX_SQL_REPAIRS`**（不是 `SQL_TOOL__*`）：它属于
   5.4 那四类「相互独立、不可借用」的循环预算，与 `max_replans` 放在一起看。

### Phase 5 期间新立的临时约定

10. **语料由 `make corpus` 现生成，产物不入库**（`data/` 已 gitignore）。进版本库的是
    清单 `configs/corpus_manifest.yaml`（规格）+ `scripts/gen_corpus.py`（怎么生成）
    + `configs/corpus_handwritten/`（**手工**的 Prompt Injection 样本）。
    于是「语料是什么」与「语料怎么来的」都可复审，产物随时可重建。
    **改清单 = 换语料 = 所有阈值与评测结论失效**，因此改清单必须同时升 `version`，
    并重跑 Recall@8 校准与 `verify-corpus`。
    ⚠️ **语料冻结前必须先跑 `make seed-business`**：语料的数字是现查业务库的，
    业务库重灌后语料必须跟着重生成，否则报告数字与库会对不上。
11. **清单里的 `defects:` 标注不是证据，产物才是**。生成器在写完后**回读产物自检**
    （跨页表格的表头是否真的重复、扫描件是否真的无文本层、注入特征串是否真的在正文里），
    未生效直接非零退出。这条是踩坑换来的：第一版三份「跨页表格」标记齐全、
    一张都没跨页——`force_split` 只是把表挪到新页开头，**实测本版式下需 ≥35 行**
    才撑破一页。**新增缺陷注入时必须同步加自检**，否则它会以「已注入」的名义静默失效。
12. **`source_kind`（INTERNAL / EXTERNAL）必须由入库侧显式指定**，不要依赖列的
    `server_default='INTERNAL'`——那是给存量行的安全兜底。把外部材料误标成 INTERNAL，
    会让 SOURCE 冲突判定静默失效（FR-SEARCH-001 的业务规则就是靠它成立的）。
13. **业务词典是"输入 + 产物"两个文件，产物入库**：`configs/rag_terms.txt`（手工复合词，
    可带注释）→ `make dict` → `configs/rag_user_dict.txt`（产物，**无注释、无词频**）。
    与语料（`data/` 已 gitignore）的取舍**故意相反**：词典是**检索契约**的一部分，
    查询侧与入库侧必须切得一样；缺了它 `Tokenizer` 只退化并打一条 warning，
    而已入库 chunk 是用带词典的分词建的 —— 症状是"某些查询永远召回不到"，
    且不指向词典。语料可以随时重建，词典不一致则无法从表面察觉。
    **产物里不能写注释**：jieba 的 `load_userdict` 会把匹配不上词频/词性后缀的
    整行当成词条加进去（实测 `get_FREQ` 返回 1）。出处写在本文件与 `rag_terms.txt` 里。
    改词典 = 改切分 = 必须重跑检索回归集（详设 11.6.2）。
14. **`schema_catalog.yaml` 里的 `note` 会渲染进 prompt**（`app/tools/sql/schemas.py`）。
    写错不是文档问题，是**模型会照着错**——净销售额那条 note 原先写「含税口径差 1–2%」，
    按字面取含税实测差 18.7%，而真正落在 1–2% 的是「含税 − 折扣、未扣退货」。
    改 `note` 与改白名单一样要升 `version`。
15. **分块参数（`RAG__CHUNK_*`）在演示语料上当前不构成约束**。语料每篇正文仅 ~830 字、
    被 5–8 个标题切碎，**分块的实际边界由标题决定**：正文块落在 35–881 字（中位 85），
    真正影响粒度的是"要不要让块跨标题合并"，而不是那三个数字。
    11.3 要求这些值经评测集校准，校准排在 ⑫Recall@8——
    **在那之前不要调它们**：调了不会改变任何结果，只会让配置和口径各说各话。
16. **`app/tools/rag/parser.py` 是开发流程没点名的模块**。6.7 只列了 `chunker.py`
    与 `ingestion.py`，但 11.2 要求四种格式各自的解析策略、11.3 的清洗规则
    又必须跨块/跨页工作（页眉靠"跨页重复"认），都塞进 `ingestion.py` 会让那个文件
    同时承担文件校验、解析、清洗、分块、向量化、发布六件事。
    按职责拆开，行为与 11.1 的流程顺序一致（Parse → Normalize → Chunk）。
    **清洗掉的内容一律进 `ParsedDocument.dropped` 并在 `make chunk` 里显示**——
    清洗是"正确时无声、错误时也无声"的操作，多丢一行不会有任何症状。
17. **词表的读路径是 `Redis → 对象存储快照`，不读 MySQL**。
    `rag_vocab` 存的是 `token / token_id / df / created_at`（11.6.3 定义的列），
    而 IDF 的分母（chunk 总数）**不在其中**；硬从表里推分母只能拿"当前 chunk 数"
    顶替，而词表冻结后再入库新文档两者必然分叉，查询侧与入库侧的 IDF 就不可比了。
    于是 MySQL 是**词条账本**，快照 `system/vocab/{version}/vocab.json` 是**读取入口**。
    快照缺失时报错提示重跑 `make vocab`，**不降级**。
18. **`df` 的口径是「含该 token 的 chunk 数」，N = chunk 总数**，不是按源文件算。
    16.8 的列注释写"文档频率"，BM25 里的"文档"指被索引的单元，本项目里是 chunk。
    两种取法给出不同的 IDF 排序。**`df` 与 `token_id` 一样冻结**（`add` 不改任何列），
    否则老向量里的权重与它当时的 IDF 对不上，而新旧向量会一起被打分。

### 【后续扩展】登记

| 项 | 触发阶段 |
|---|---|
| ~~任务仓储换 MySQL~~ **已完成**（`RedisTaskRepository` 与 5 个索引键已删） | ✅ Phase 2 |
| `agent_task_step` / `agent_tool_call` / `agent_evidence` / `agent_trace_event` 的仓储实现（表已建，仓储待写；调用方在 Phase 6/7 才出现）。**`agent_tool_call` 的待写数据已经就位**：SQL Tool 的 `ToolResult.payload["attempts"]` 就是它的行 | Phase 6 / 7 |
| `schema_catalog` / `agent_config` 建表（切片内目录是 `configs/schema_catalog.yaml`，SQL Tool 已按它的形状写好 `SchemaCatalog`；接表只需换 `SchemaProvider` 的加载实现） | 后置 |
| `make cleanup` 的保留期策略与实现（详细设计 16.12） | Phase 2 收尾 |
| 事件流的 MySQL 权威重放（`agent_trace_event`）+ `sequence` 改由该表提供 | Phase 2 / 10 |
| `stream_url` 指向的 SSE 订阅端点（契约已固定，事件已可订阅） | Phase 10 |
| 用户消息落库（FR-CHAT-001 处理流程的「保存用户消息」，`agent_message` 表） | Phase 2 |
| 任务体换成 LangGraph（`TaskRunner._run_body` 一个函数） | Phase 7 |
| 节点内检查取消标记（`TaskRunner.is_cancel_requested` 目前只在领取与收尾时检查） | Phase 7 |
| `WAITING_CLARIFICATION` 的 `clarification_question` 字段 | Phase 6 |
| 任务详情的步骤进度、证据、冲突、限制（等各自 Schema 产出后增补） | Phase 6 / 9 |
| 对象存储 `s3` 实现（契约测试已就绪，加进参数表即被覆盖） | Phase 5 |
| 登录接口限流（当前配额按已认证用户计，登录不受保护，可被口令爆破） | Phase 12 |
| 未注册路径复用 `TASK_NOT_FOUND` 的语义含混（错误码表封闭所致） | 待定 |
| token / 成本进 **OTel Metrics**（详设 19.4.3）。Phase 3 先落 span 属性，`observability.py` 目前只有 tracer、没有 meter | Phase 11「节点埋点」 |
| `MODEL_TUNING__THINKING_ENABLED=true` 时，带 `tools` 的历史轮次必须回传 `reasoning_content`，否则 DeepSeek 返回 400。切片内不开思考模式，故未实现该回传路径 | 需要时 |

**冲刺期后置**（由 `docs/秋招冲刺方案.md` §10 冻结，切片内不要顺手做）：

| 项 | 备注 |
|---|---|
| **Reranker**（选型 + 阈值校准） | 依据详设 TBC-04 后置。实现 `reranker.py` 与 `RERANKER_ENABLED` 接缝，默认关闭走 RRF Top-K |
| ~~语料扩到 80+ 篇 + 缺陷注入全套（10 类）~~ | **已完成**（2026-09-17）：88 篇、10 类缺陷注入全部回读产物验证。剩余的「阈值必须基于该语料校准」属 ⑧`retriever.py` / ⑫Recall@8，**不得沿用任何小语料取值** |
| ~~文档版本发布流程（staging → smoke test → publish）~~ | **已移回 Phase 5**。staging 载体是 payload 状态位，不是 partition |
| 知识管理接口 FR-ADM-001、扫描件 OCR、跨页表格还原 | 入库走脚本不做后台；扫描件 PDF 在语料中保留但**明确标记不支持** |
| ~~词表扩容机制（`token_id` 只增不改的运维面）~~ | **已移回 Phase 5**。需验证"扩容后历史 chunk 无需重算仍可召回" |
| `agent_conflict` / `agent_review` 表与完整 5 类 Conflict 检测 | 切片内只做「报告数字 vs DB 数字」一类，且先不建表 |
| `agent_plan_revision` / `agent_finding` 表（切片内先存 `agent_task` 上的 JSON） | |
| SSE 订阅端点与订阅令牌（Phase 1.5 已完成事件流，剩余是暴露端点） | |
| `/trace` 接口与节点埋点（OTel 已接入，只差业务侧） | |
| 评测集扩到 115 条（先 40 题；**SQL 安全类 100% 阻断率的判定标准不缩**） | |
| `schema_catalog` / `agent_config` 建表（切片内暂用 YAML / 配置） | |
| 生产部署（Nginx / HTTPS / 备份 / 告警 / 回滚演练） | |

## 设计文档

`docs/` 下文档是**设计意图的记录，不是验收契约**：

| 文件 | 用途 |
|---|---|
| `企业智能数据分析与决策Agent-需求规格说明书.md` | 功能与非功能需求（FR 条目） |
| `企业智能数据分析与决策Agent-详细设计说明书.md` | 架构、Schema、接口、错误码表（19.1） |
| `企业智能数据分析与决策Agent-开发流程.md` | Phase 划分、验收命令、门禁、工作量估算 |
| **`秋招冲刺方案.md`** | **冲刺期当前的施工依据**——逐节裁决了简化方案，冻结切片范围与后置项。**效力优先于开发流程第 6 章** |
| `Agent项目需求.md` | 原始需求 |

> **需求条目（FR-*）不是"可以砍的生产工程化设计"。** 冲刺方案里推迟的 FR（如 FR-SSE-001、FR-CHAT-003 的多轮上下文）
> 是**推迟实现**，不是**取消需求**——对应实现补齐前，简历与面试中不得声称已具备该能力（见冲刺方案 §11）。

**偏离文档时**：先在回复里说明是哪一节、为什么、建议怎么改，得到确认后改代码，**并回写文档**保持两者一致。
不允许默默偏离，也不允许为了"忠于文档"硬做明显不合理的实现。

---

## 硬性约束

以下规则**由 CI 强制**（`make layering` / `scripts/check_layering.py`），不是风格建议。
违反时影响要到运行时才暴露，所以不接受"这次先这样"。

### 依赖方向

```
api  →  services  →  agent / domain / tools  →  repositories / infrastructure
```

只能向右。`domain/` 保持纯净——不依赖 FastAPI，不依赖具体数据库客户端。

### 进程边界（两条，检查器编号 L1/L2）

| 规则 | 内容 | 破坏后果 |
|---|---|---|
| **L1** | `app/main.py` 与 `app/api/**` 不得 import `agent.graph`、`agent.nodes.*`、`tools.*` | API 进程加载 LangGraph 与全部 Tool，启动变慢、内存翻倍 |
| **L2** | `app/worker.py` 与 `app/agent/**`、`app/tools/**` 不得 import `fastapi` | Worker 无法独立扩缩容 |
| **L3** | `app/domain/**` 不得依赖 FastAPI 或 DB 客户端 | 领域逻辑被基础设施绑死 |

检查器自身的测试在 `app/tests/test_layering.py`——**改检查器时必须让它继续通过**，
门禁自己失效比被检查的代码违规更危险。

### 其他纪律

- 所有进入 LangGraph State 的 LLM / Tool 输出**必须先经 Pydantic 校验**；禁止把裸 `dict` 写入 State
- 错误码只用详细设计 19.1 那张表里的，**不得新增同义码**
- 日志、SSE、Trace **默认脱敏**：禁止记录 Token、密码、密钥、敏感字段、模型思维链、SQL 原始行数据
- 无未标注的 TODO；未完成项标 `【后续扩展】` 并登记
- Redis 中的内容必须能由 MySQL 重建（缓存键带版本号，如 `cache:schema:{version}`）
- 循环预算（`LOOP__*`）四类相互独立**不可借用**；调整后必须重跑循环类评测集

---

## 常用命令

```bash
make bootstrap   # 首次：生成 .env + 装依赖
make up          # 起 redis / agent-mysql / business-mysql / minio
make run         # 同时起 api + worker  ← 本地开发必须用这个，不要只跑 make api
make check       # 提交前必跑：ruff + mypy + 分层约束 + pytest
make migrate     # 迁移 agent 运行库到最新（前置 make up）
make seed        # 灌入演示数据：agent 库的演示账号 + 业务库 49.9 万行（幂等）
make verify-business  # 只跑业务库的四条约束断言与 EXPLAIN 检查
make seed-business ROWS=20000  # 快速试跑业务库（改规模）
make test-integration  # 需要真实 Redis / MySQL（前置 make up；会自动建 agent_test 库）
make redis-cli   # 排查队列与 Stream
```

SQL Tool（Phase 4，**当前唯一能端到端演示的链路**）：

```bash
make sql Q="2025年华东地区Q3的净销售额是多少"          # 自然语言 → SQL → 真数据 → 证据
make sql Q="2025年Q3各区域的净销售额" REGION=华东      # 模拟 data_scope（受限用户只看得到华东）
make sql Q="帮我把表清空" SQL="DROP TABLE dim_region"  # 跳过生成、仍走全部校验：单独证明「安全由代码保证」
make eval-sql    # 金标评测（10 题），同时打印「内容正确率」与「列形状一致率」
```

> 危险 SQL 的演示**必须走 `--sql`**：模型在正常对话下不会写出 `DROP TABLE`，
> 这本身是第一层防线在工作。要证明「代码层拦得住」，就得绕过模型直接喂一条
> 危险 SQL 进同一个校验器——否则演示出来的只是「模型很乖」。

RAG 分词（Phase 5 进行中，**这是本阶段使用频率最高的调试命令**）：

```bash
make tokenize T="华东区域渠道折扣政策"        # → 华东 / 区域 / 渠道折扣 / 政策
make tokenize T="华东区域渠道折扣政策" NO_DICT=1  # 对照：不加载业务词典会切碎
make dict        # 生成业务词典（前置 make up + 业务库已灌数；产物入版本库）
make chunk P=data/corpus/SP-015.pdf          # 解析 + 分块，逐条核对
make chunk P=data/corpus SUMMARY=1           # 全语料只打汇总
make chunk P=data/corpus/SP-015.pdf TABLES_ONLY=1  # 只看表格块
make vocab       # 构建稀疏检索词表并导出快照（幂等，前置：make corpus）
```

`make chunk` 会打印**解析阶段清洗掉了什么**（页眉页脚、修订记录）。
清洗是唯一一类"正确时无声、错误时也无声"的操作——多丢一行不会有任何症状，
直到某天有人问"制度里明明写了"。

`make tokenize` 同时打印**被丢弃的 token**——「分词切错了」与「切对了但被过滤规则
丢了」是两种故障，只看得见保留结果时它们长得一样。

单独跑：`make lint` / `make typecheck` / `make layering` / `make test` / `make fmt`

**只跑 `make api` 会导致任务永远停在 `QUEUED`**，且从 API 日志几乎看不出原因。这是本项目最常见的自伤方式。

多实例验收用 `make api PORT=8001`（`PORT` 会覆盖 `API_PORT`）。

**排查一个任务**（Phase 1.5 起可用）：

```bash
make redis-cli
> ZCARD q:agent                       # 队列里还有多少没被领走
> HGETALL task:tsk_xxx:record         # 任务状态、worker_id、心跳时间
> XRANGE task:tsk_xxx:events - +      # 事件流（发布过什么）
> ZCARD idx:task:active:usr_xxx       # 该用户占用的并发配额
```

---

## 代码约定

- **语言**：注释、docstring、用户可见文案一律中文；代码标识符英文
- **风格**：`ruff` 格式化，行宽 100。`RUF001/002/003` 已关闭——全角标点在中文里是正确写法
- **类型**：`mypy --strict`。公开函数必须带类型标注
- **日志**：统一走 `app/infrastructure/logging.py`。字段固定为
  `timestamp、level、service、trace_id、conversation_id、task_id、step_id、node、tool、status、duration_ms、error_code、message`；
  用 `bind_context()` 绑定上下文字段，不要自己往 `extra` 塞任意键（会被 `ValueError` 拒绝）
- **配置**：新增配置项写入 `app/core/config.py`，必须有**默认值、上下限、环境覆盖规则**。
  嵌套配置用双下划线：`LOOP__MAX_TOTAL_STEPS=30`
- **测试**：单元测试不依赖真实外部组件；需要真实 Redis/MySQL/Qdrant 的用例打 `@pytest.mark.integration`。
  `make test` 默认**排除** integration，`make test-integration` 单独跑（前置 `make up`）。
  Redis 的替身是 `fakeredis`（它能跑真实 Lua 脚本，因此限流与仓储的原子性是被真实执行验证的）
- **服务角色**：`service` 字段（api / worker）由入口模块的常量决定，**不是配置项**——
  `make run` 下两个进程共用一份 `.env`，用环境变量区分必然失效
- **提交**：Conventional Commits，如 `feat(sql): 增加 AST 表字段白名单校验`

---

## 本机环境事实

这些是当前开发机的约束，换机器时需重新确认。

| 项 | 值 | 说明 |
|---|---|---|
| 项目路径 | `/home/zfc/projects/enterprise-agent` | WSL 原生 ext4。**不要放 `/mnt/c` 或 `/mnt/e`**——9p 协议小文件 I/O 慢 60–110 倍 |
| 内存 | 7.6GB（物理机 16GB） | **这就是 TBC-05 改判 Qdrant 的直接原因**：Milvus Standalone 需 8GB 起，本机跑不起来；Qdrant 实测仅占 300MB |
| Redis | 宿主端口 **6381** | 6379 被本机原生 redis 占用（存有另一个项目的数据，不可动）；6380 被 redis-stack 占用。容器内仍是 6379 |
| MinIO | 镜像用 `quay.io/minio/minio` | 本机 daemon 的 `docker.m.daocloud.io` 镜像源对 `minio/minio` 返回 403 |
| MySQL | agent **`3308`** / business `3307` | business 用只读账号，写操作必须被数据库拒绝。agent 用 3308 而非 3306：**Windows 侧另有一个独立安装的 MySQL 占着 `0.0.0.0:3306`**，wslrelay 因此无法为 3306 建立 localhost 转发（实测 3307/6379/6380/6381/9000 都转发，唯独没有 3306）。用 Windows 的图形客户端连 `localhost:3306` 会连到**那个** MySQL，看不到 `agent_task` 等表 |
| Qdrant | 宿主端口 **6333**（HTTP）/ 6334（gRPC） | 服务端形态，**不要用 SDK 的本地模式**——它单进程独占（见 TBC-05） |

### 从 Windows 侧的图形客户端连库（Navicat 等）

容器跑在 **WSL 内的原生 Docker** 里（`unix:///var/run/docker.sock`，非 Docker Desktop），
Windows 通过 `wslrelay` 转发 `127.0.0.1:<宿主端口>` 访问。**主机一律填 `localhost`**：

| 用途 | 端口 | 用户名 | 口令 | 库 |
|---|---:|---|---|---|
| Agent 运行库（读写） | **3308** | `agent` | `agent_pw` | `agent`（另有 `agent_test`） |
| 业务演示库（**只读**） | 3307 | `readonly` | `readonly_pw` | `business` |
| 业务演示库（要改数据时） | 3307 | `root` | `root_pw` | `business` |

三个口令是**本地开发占位值，不是密钥**——它们已经明文写在 `docker-compose.dev.yml` 与
`.env.example` 里（否则 `make bootstrap` 无法开箱可用）。生产部署必须换成密钥服务注入。

三条容易踩的：

1. **端口填 3306 会连到 Windows 上那个独立安装的 MySQL**，不是本项目的容器。
   能连上、但里面没有 `agent_task`，极易误判成"迁移没跑"。
2. **业务库用 `readonly` 连上只能 SELECT**（`GRANT SELECT, SHOW VIEW ON business.*`），
   在 Navicat 里改数据会报 `1142`。这是**设计如此**——SQL 安全由数据库权限保证，
   不依赖模型（详细设计 19.2）。要编辑就用上表的 `root`。
3. 连不上时先 `make up`：容器**没配 restart policy**，WSL 重启后不会自动起。
   `localhost` 转发也依赖 WSL 处于运行状态。

`Navicat for MySQL` 看不了 Redis，排查队列与事件流用 `make redis-cli`。

## 未决项

**当前没有未决项。**

| 编号 | 状态 | 决议 |
|---|---|---|
| **TBC-05** | ✅ **已结案（2026-09-17）** | **Qdrant 服务端**。原定 Milvus 2.4+，三候选同口径实测后改判：Chroma 无稀疏向量与融合排序，直接出局；Milvus Lite 与 Qdrant 本地模式**都是单进程独占**（`make run` 是 api + worker 双进程，无法共享）；Milvus Standalone 需 8GB 而本机 7.6GB，**生产形态在本机无法验证**。Qdrant 服务端实测：混合检索 4.4ms、内存 300MB、多进程并发正常。完整数据见详细设计 23.1.1，脚本 `scripts/spike_vector_store.py` |

> ⚠️ **对外解释选型时，前提必须说全**：否决 Milvus 的**是这台开发机的内存约束**，
> 不是 Milvus 本身不行。换台内存充裕的机器，Milvus Standalone 同样成立。
> 略过这个前提，结论就从"有数据支撑的选型"退化成"随便选了一个"。

**处理未决项的原则**：先做最小验证拿到数据再决策，不要靠读文档空猜。
`infrastructure/` 层必须把外部组件隔离干净，使换实现的成本控制在一个文件内。

---

## Phase 推进节奏

每个 Phase 按固定节奏，避免边写边改导致返工：

```
1. 读文档 —— 对照需求 FR 条目与详细设计章节，列出实现清单
2. 定 Schema —— 先写 Pydantic 模型与接口签名
3. 写测试骨架 —— 先写会失败的测试（含 Fake 依赖）
4. 实现
5. 本地验证 —— make check + 本阶段专用验收命令
6. 埋点 —— 补齐 Trace 事件、结构化日志
7. 更新配置 —— 新增配置项带默认值与上下限
8. 对照门禁 —— 逐条核对"进入下一阶段条件"
9. 提交 —— CI 全绿后合并
```

**完成的定义**：`make check` 全绿（含分层约束）+ 验收命令可重复执行并结果一致 + 需求条目逐条核对 +
Trace 与日志可按 `task_id` 定位 + 新增配置有默认值与上下限 + 设计偏离已回写文档 + 无未标注 TODO。

涉及基础设施的 Phase（1.5 / 5 / 7 / 10 / 11 / 14）额外要求：**降级路径已实现并测试**
（外部依赖不可用时行为已定义且有对应用例）。
