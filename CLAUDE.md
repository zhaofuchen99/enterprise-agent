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
| 5 RAG | ✅ 完成 | **已完成**：⑦Qdrant collection 接入层（服务端 + 内存替身同契约）、③中文分词与稀疏向量（`Vocabulary` + `make tokenize`）、**①业务词典生成**（`make dict`，96 条，回读产物自检）、**②语料 88 篇 + 10 类缺陷注入**（含 4 份手工 Prompt Injection）、④PDF/DOCX 工具链、**⑥`parser.py` + `chunker.py`**（四格式解析 + 清洗 + 分块，`make chunk`；全语料 **1871** 块，页眉页脚零泄漏、跨页表格表头还原）、**③后半词表**（`rag_vocab` 仓储 + 构建 + 快照，`make vocab`；2781 词条）、**④`s3` 存储实现**（与 local 同一份契约测试）、**⑤`ingestion.py` + ⑪入库幂等**（11.1 的九步全流程 + `make ingest`；`knowledge_document` 仓储、`ChunkMetadata` 双向映射、发布前抽样冒烟、失败整批回滚、原文件与入库报告归档）。全语料实测：**发布 83 篇 / 幂等跳过 3 篇 / 扫描件标记不支持 2 篇 / 0 失败，冒烟 249/249 全中**；Qdrant 点数 1871 与 `make chunk` 的汇总**逐块一致**（两条独立路径互为对照）、**⑧`retriever.py` + ⑩`NO_RELEVANT_KNOWLEDGE`**（11.7 的九步去掉第 ⑥⑧ 步；Query Rewrite 含降级、标量过滤、双路召回、单次 RRF、两判据相关性门禁、文档证据；`make retrieve`）。**⑫金标 20 条 + Recall@8**（`configs/eval_rag_golden.yaml` + `make eval-rag`）、**`verify-corpus` 门禁**（`make verify-corpus`，10 类缺陷逐条检出）。**实测**：Recall@8 **19/20 = 95%**（门禁 ≥85%）、定位一致率 17/20 = 85%、`verify-corpus` **10/10**（其中 2 类只验证了语料侧，冲突检出属 Phase 9）。⑨Reranker 与 11.7 第 ⑧ 步按 TBC-04 维持后置 |
| 6 最小 Graph 接入 | ✅ 完成 | **6 节点**：`supervisor / sql / rag / reflect / analysis / final`（`app/agent/`）。Supervisor 走模型出 `IntentResult`、**按 `required_sources` 真的在选工具**（FR-PLAN-002 业务规则 1 有专门用例钉着）；`reflect` 是**确定性**的任务循环判断点（某一路跑了但空 → 补另一路，最多一次）；`analysis` 把证据编号化交模型组织、`final` 用代码渲染引用与限制。已接入 `TaskRunner`（任务体由 `worker.py` 注入）。**实测**：端到端跑通「SQL 拿数字 + RAG 拿口径定义 + 报告数字与库不符被识别」 |
| 7 Evidence 冲突检测 | ✅ 完成（切片版） | **只做 VALUE 一类**（同口径数值超容差），`conflict` 节点（`app/agent/nodes/conflict.py`）。文档侧认**表格行**、SQL 侧认证据 `claim`，靠**指标目录**把表头映射到 `metric_code`；容差取「绝对 1 元 / 相对 0.1%」较大者（13.4 第 4 步）。冲突在 `analysis` **之前**算好并交给模型披露（详设 6.1 的顺序），`final` 单列「数据不一致（需人工核对）」并注明**未判定谁对**。**实测**：端到端检出「报告表格 11,039.58 万元 vs 库 111,967,031.73，差 1.42%」 |
| 8 Reviewer-lite | ✅ 完成（第一阶段） | **只做 14.1 的确定性检查**（六条：必需步骤是否跑过、claim 有无引用、引用是否存在、BLOCKING 冲突是否披露、敏感字段是否泄露、未解决问题是否列出）。**能 FAIL 任务**——14.3 的一票否决意味着"结论没有依据"时 `final` 输出"审查未通过"而不是把原答案放出去。`RETRY`/`CLARIFY` 不产出（要 retry_router 与状态位，属 Phase 8 完整版） |

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
⚠️ **`make ps` 看不到 Ollama**（它是裸容器，不归本项目 compose 管）——
做 RAG 入库/检索前额外确认一次：`curl -s http://127.0.0.1:11434/api/tags`。

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
19. **`payload.document_id` 是 `logical_key@version`，不是 `logical_key`**。
    按版本而不是按逻辑文档过滤是必须的：同名制度的 v1.0 与 v2.0 共用
    `logical_key`，按它过滤等于让两版互相串味（VERSION_PAIR 缺陷注入正测这件事）。
    这个值同时是 `chunk_id` 的派生种子、payload 的过滤键、以及
    `delete_document` 的选择器——**三处必须是同一个字符串**，
    否则"删掉这一版没发布成功的 point"会删到别处，而删除是不报错的。
    11.4 的 `ChunkMetadata` 没有 `logical_key`，本实现显式多存了一份：
    按 `@` 反解复合键，在 `logical_key` 里出现 `@` 的那天就会静默切错。
20. **`payload.status` 用 `domain.knowledge.DocumentStatus`，不用 11.4 写的
    `DRAFT / ACTIVE / ARCHIVED`**。11.1 的 staging 态是 `PROCESSING`、
    11.9 的过滤条件比的是它——那三个值是**文档生命周期**的写法，
    在入库流程里 DRAFT 根本不存在。两套取值并存必然漂移（过滤条件只认一个
    字符串，而枚举有两个），所以行与 payload 共用一个枚举。
21. **同版本号 + 不同内容 = 报错，不是覆盖**（`make ingest FORCE=1` 才允许重建）。
    `chunk_id` 由 `logical_key@version` 确定性派生，静默覆盖会让此前引用过
    `chk_xxx` 的证据指向另一段文字，而引用本身仍然打得开。反过来，
    **同一份内容挂在两个 `logical_key` 下也要拒绝**——那会让同一批向量
    在检索里并列出现，看起来像两个来源互相印证。
22. **文档级永久性失败（扫描件无文本层）返回 FAILED 报告而不抛异常**，
    且报告单列 `unsupported=True`。这不是"宽容"，是**退出码的判据**：
    语料里 2 份扫描件注定失败，若它们也算命令失败，`make ingest` 恒返回非零，
    退出码就没人看了；而"Ollama 没起导致 88 篇全挂"必须返回非零。
    **抛异常**留给"这次跑坏了"（校验不过、词表没覆盖、基础设施不可用）。
23. **`app/tests/db.py` 每个进程第一次会 `drop_all` + `create_all`**。
    只 `create_all` 不够：它的 `checkfirst` 只看"表在不在"、不看列全不全，
    于是模型加列后 `agent_test` 里的旧表照样通过，失败要到 INSERT 才发生，
    报的是 `Unknown column 'xxx' in 'field list'`——**读起来像代码写错了列名**。
    这条是 Phase 5 加 `source_kind` 时真踩到的。
24. **RAG 测试的 jieba 全局状态由 `app/tests/tools/rag/conftest.py` 隔离**，
    基线在**导入期**取（那时一次词典加载都还没发生）。
    `jieba.load_userdict` 改的是进程级词典，而 `Tokenizer` 明确不支持
    "不同实例用不同词典"——恢复成"本用例开始前"的状态在跨文件时是不够的，
    因为那时基线**已经是脏的**。症状是"单独跑通过、全量跑失败"，
    或者更糟：全量也通过，但通过的其实是别的用例加载的词。

25. **「低于阈值的候选剔除」（11.7 第 7 步）的阈值挂在稠密路的余弦上**，
    而不是 RRF 融合分。RRF 分是 `Σ 1/(k+rank)`，值域只有约 0.016–0.033——
    把 `score_threshold` 的默认值挂上去会把结果**全部**剔光；而 11.6.5
    正是靠「RRF 只依赖排名、不依赖分数绝对值」论证固定 IDF 成立的，
    在无量纲分数上挂绝对阈值与那条论证冲突。余弦才是系统里唯一量纲可比的分数。
    **逐候选的语义过滤要等重排器**：稀疏路独有的候选没有稠密分，
    补 0.0 等于断言"语义完全不相关"，那是个我们并不知道的结论
    （`RetrievedChunk.dense_score` 因此是可空的）。
26. **「语料里有没有」由两个判据「或」起来，缺一不可**（这是实测结论）：
    - **未登录主题词**：问题里的实义词若在词表里查不到，说明语料从未出现过它。
      88 篇语料上 12 条真问题 + 7 条不存在的问题**全部判对**（19/19）。
    - **稠密余弦地板**（`RAG__SCORE_THRESHOLD`，实测取 0.60）：**分不开两类**——
      真问题的下界 0.6659 低于不存在问题的上界 0.7056，那 0.04 的重叠是稠密模型
      各向异性的性质（`bge-m3` 对两段无关中文也给 0.5 以上），不是调参能消除的。
      它只负责拦明显无关的输入（"汽车保险费率"0.57、"员工股权激励"0.54）。
    两个判据**都要进 `safe_detail`**：「阈值调高了」与「语料没见过这个词」
    症状相同，处置完全不同。⚠️ 样本量 19 条，⑫ 会用金标 20 条重新校准。
27. **疑问词不算主题词**（`retriever._NON_TOPICAL`）。语料是陈述性文本，
    从不用疑问句式写句子，「怎么」「哪些」在词表里一律未登录——
    不排除它们的话**任何问句**都会被判成"语料没见过"，真问题会被一起挡掉。
    **这份表不参与分词**：`configs/rag_stopwords.txt` 管的是"哪些词不进稀疏向量"，
    改它会作废已入库的 1871 个稀疏向量（约定 13）。并入停用词表更整齐，
    但要连带重跑 `make vocab` + `make ingest`，已登记为【后续扩展】。
28. **RAG 的空结果是 `FAILED` + `NO_RELEVANT_KNOWLEDGE`，与 SQL 侧故意不同**。
    SQL 的 0 行返回 `SUCCEEDED`（那是关于数据的事实，可以照实说"没查到"）；
    RAG 的"没有相关知识"若也标成成功，`payload.chunks` 是空列表，
    下游只看得到"成功、零条"——与"工具没跑到"长得一模一样，
    生成节点就有机会把空结果补写成"公司暂无相关规定"。**那正是 11.8 要防的幻觉。**
    `error_class` 取 `EMPTY_RESULT`，Reviewer 仍可按 9.4 决定补证/澄清/受限回答。
29. **向量化服务挂掉时抛异常，绝不返回"没有相关知识"**。
    两者处置相反：服务不可用要重试或降级，"语料里没有"是要告诉用户
    "公司没有这条制度"。把前者伪装成后者，等于在故障时对着用户编一个结论，
    而它看起来完全正常（`NO_RELEVANT_KNOWLEDGE` 是个合法错误码）。
    **但改写失败要降级**：改写是"补充召回"，改不动不等于查不了。

30. **「语料里有没有这件事」由两个判据「或」起来，且未登录词那一路有**三个**
    条件（`retriever._unseen_topics`）。金标 20 条校准后定稿：
    1. **不在 `_NON_TOPICAL`**——疑问词不携带主题，而语料是陈述性的，
       不排除它们的话**任何问句**都会被判成"语料没见过"；
    2. **语料里没有被它包含的词**（`Vocabulary.covers`）——`归口` 不在词表里
       却在语料里出现 **69 次**，因为业务词典把「归口管理部门」收成了一个词。
       按"是不是一个 token"判定会漏掉它，按"是不是某个词的组成部分"才对；
    3. **在 jieba 的通用词典里**（`tokenizer.is_general_word`）——排除**切分伪 token**：
       「…经营月报里区域分布…」被切成 `月报 / 报里 / 区域分布`，
       `报里` 不在词表里，于是余弦 0.79 的问题被拒答。伪 token 是无界的。
    **已知漏网类：同义词**。「报备」语料里 0 次（制度写「备案」），也不被任何词包含，
    于是仍会被判成"语料没见过"而拒答（余弦 0.74，检索其实找得到）。
    纯词汇规则区分不了"语料没讲这件事"与"语料用了另一个说法"，
    只能靠语义（重排器 / Phase 8 Reviewer）。**这一例留在金标里当回归用例**（rag-06）。
31. **`RAG__SCORE_THRESHOLD` 已按金标 20 条校准为 0.60**，且**它是保守地板**，
    真正的判别力在未登录词那一路。实测（金标跑出来的）：
    真问题的最高余弦 0.61–0.86，语料中不存在的问题 0.51–0.71——**两类重叠**，
    那是稠密模型各向异性的性质，不是调参能消除的。
    任何"调阈值就能同时提召回又提拒答"的说法都与这份数据不符。
32. **`verify-corpus` 的两种标注不能混**：`[OK]` 是"在这里就检出了"，
    `[注入]` 是"语料侧确认注入了、但对应的冲突检出属 Phase 9"。
    VALUE / TIME 两类冲突现在标的是 `[注入]`——**把它们说成 `[OK]` 就是
    把"语料里有"说成"系统检得出"**，而 Phase 5 的门禁原文是"10 类缺陷逐条检出"，
    这个差别必须在面试口径里说清楚。SCOPE 是例外：它在这里就能检出，
    因为"文档声称的省份数 vs `dim_region` 的实际值"只要把两个数摆在一起就够了。
33. **`app/tests/tools/rag/conftest.py` 必须在 import 期先 `jieba.initialize()` 再取基线**。
    前缀词典是惰性构建的，未初始化时 `jieba.dt.FREQ` 是**空字典**；
    取一份空基线再去"恢复"它，等于把词频表清空，而 `initialize()` 因为
    `initialized=True` 不会重建。后果是分词退化成逐字切分，**而用例照样通过**
    （`all(len(t) > 1 for t in tokens)` 在空序列上恒为真）。
    这是本阶段真踩到的：它让"未登录词判据"静默失效，排查方向指向词表。

34. **`reflect` 是确定性的，不调模型**（冲刺方案 §8.1：先跑通再跑准）。
    它的判定只看 State 里的**事实**：还有没有 PENDING 步骤、哪条路跑了但空、
    演进预算还剩多少。第二条就是「Agent 能根据工具结果继续分析」的落点——
    SQL 判成只要查库而库按条件查不到时，它补一路 RAG 去找制度与报告的解释。
    **不接模型判定的理由**：接上之后"循环会不会收敛"就依赖云模型连通性，
    而那是验收标准第③条的核心，不该由外部服务决定。
35. **模型不可用时 supervisor 报错，不降级成"两路都查"**。
    降级看起来更稳，但它会在故障期间把 FR-PLAN-002 那条验收悄悄作废，
    而结果看起来完全正常（确实拿到了数据）。
36. **「查不到」与「查不成」在图里分开走**（`tool_nodes._normalize`）：
    SQL 的 0 行 / RAG 的 `NO_RELEVANT_KNOWLEDGE` 是**关于数据的事实**，
    只进 `step_results.empty`，不进 `errors`；其余失败两边都进。
    混在一起的话，最终答案会把"服务挂了"说成"公司没有这条数据"。
    ⚠️ SQL 的空结果走的是 `SUCCEEDED` + `payload["is_empty"]`（9.4 明写
    「SQL 空集不算失败」），与 RAG 走错误码**不是同一条路径**，两处都要认。
37. **`PermissionScope` 在 `_run_body` 里从用户记录装载**（`agent/runner.py`），
    **查不到用户就拒绝执行**。`agent_task` 表没有数据范围字段，
    而 `PermissionScope` 的空 `region_ids` 表示**不限**（TBC-03）——
    拿它当兜底等于让一个已删除用户的任务拿到全量数据，且不会有任何报错。

38. **冲突检测只做 VALUE 一类，另外四类各有各的缺前提**（`nodes/conflict.py` 列了清单）：
    `DEFINITION` / `SCOPE` 要文档侧的 `metric_code` 与 `scope`，而分块不带指标 code；
    `TIME` 要文档的统计期间，而 `event_time` 是**生效区间**（报告恒为空）；
    `SOURCE` 要判"方向相反"，属 Phase 9。**被问"冲突检测做了多少"时照这个答**，
    不要笼统说"做了冲突检测"。
39. **文档表头 → `metric_code` 靠指标目录，且精确名优先于别名**。
    `net_sales` 的别名里有「销售额」，而报告里那一列是**含税 − 折扣**口径——
    按别名匹配会把两列都映射到 `net_sales`，对同一个 SQL 数字报出**两条冲突**，
    其中一条是假的。`_prefer_exact` 是这条的正面防线：
    同一张表里某指标既有精确名列又有别名列时，别名列让位。
    匹配方式（exact/alias）进 `detected_difference`，读冲突的人据此判断结论有多硬。
40. **检测器不判谁对**（`resolution` 默认 `UNRESOLVED`，`severity` 取 WARNING 而非 BLOCKING）。
    判谁对要按 13.2 的证据优先级，那是 Reviewer/Phase 9 的事。
    默认成已处置会让一个未处置的冲突看起来已经处置过了。
41. **已知误报类：表格行不区分合计行与分项行**。实测里「分区域经营情况」表的一行会被
    当成区域的合计去比，而它其实是某个分项。模型能在 `claims` 里自己纠正
    （实测出现过），但**检测器这一层还没有这个判别力**——需要表格的合计标记或
    指标口径里的 grain 信息。同样登记为后续扩展。

42. **Reviewer-lite 只做 14.1 的六条，另外四条各有各的缺前提**（`nodes/reviewer.py` 列了表）。
    其中最值得记住的一条是「SQL 是否通过安全校验」——**它在架构上已被满足**
    （校验器在 Tool 内部，被拦下的根本到不了 Reviewer），
    重复检查只会在两处维护同一份副本，而两处迟早会漂移。
43. **`HYPOTHESIS` 无引用不判 FAIL，但要记一笔 INFO**。13.5 明写它本来就允许
    没有引用（"证据不足时的推测"）；判成阻断会让每一条含"可能原因"的答案都被拒。
    但完全静默也不对——"本答案含未验证推测"是读者该知道的事。
44. **敏感字段的判据只认英文标识符与赋值形态**（`password_hash` / `password=` /
    `api_key` / `bearer …`），**不认中文关键词**：语料里出现「密码」是正常的
    （制度会讲口令管理），按中文判会大面积误报，而误报会让这条检查被关掉。

45. **计划与结构化结果落 `agent_task` 的两列 JSON**（16.5 的 `plan_json` /
    `result_json`），入口是 `TaskOutcome`——**它放在 `domain/` 而不是
    `services/` 或 `agent/`**：两边都要用（agent 产出、services 落库），
    而依赖方向是单向的（`services → agent`），放任一侧都会让另一侧反向依赖。
46. **`intent` 也要落**（`agent_task.intent`）：它回答"这条任务是当查询问的
    还是当制度问的"，排查误答时第一个要看的字段。之前一直是空的——
    supervisor 判出来放进 State，而没有人把它写进那一列。

47. **执行产出落进 16.6 / 16.7 的五张表**（`repositories/agent_repo.py`），
    入口是 `TaskOutcome` 的五个字段，由 `TaskRunner._persist_artifacts` 一次写齐。
    **五张表一次写而不是分开写**：它们的不一致（证据落了、冲突没落）不会报错，
    只会让复盘时看到的图景缺一块。
48. **重投时整体替换，不是追加**。这几张表都没有天然唯一键，追加会造出两批
    并存的行而**从数据上分不出哪批属于最后一次执行**——复盘看到的是两份
    互相矛盾的证据清单，且没有任何地方报错。代价如实记下：跨重投的历史轨迹会丢。
49. **落库失败吞异常、只记日志**。它不该把一次成功的分析变成任务失败，
    但也不能静默——那是"复盘时发现表里没有产出"的唯一线索。
50. **`repositories/` 只认识 `domain/` 的对象**（分层方向使然）。所以
    `TaskStep` + `StepResult` 由 `services/` 合成 `StepRecord` 再传下来，
    `ReviewResult` 同理合成为 `ReviewRecord`。这个约束逼着仓储的入参只描述**行**。

### 【后续扩展】登记

| 项 | 触发阶段 |
|---|---|
| ~~任务仓储换 MySQL~~ **已完成**（`RedisTaskRepository` 与 5 个索引键已删） | ✅ Phase 2 |
| ~~`agent_task_step` / `agent_tool_call` / `agent_evidence` / `agent_conflict` / `agent_review` 的仓储实现~~ **已完成**（`repositories/agent_repo.py`，任务收尾时一次写齐五张表） | ✅ Phase 6 收尾 |
| `agent_trace_event` 的仓储实现（表已建，仓储待写；它是事件流的**权威重放**来源，属 Phase 10） | Phase 10 |
| **跨重投的执行轨迹会丢**：五张表在重投时整体替换（见 `agent_repo.py` 的说明），上几次失败的过程没有留痕。要留就得加 `attempt_no` 或一张执行流水表 | 需要时 |
| `schema_catalog` / `agent_config` 建表（切片内目录是 `configs/schema_catalog.yaml`，SQL Tool 已按它的形状写好 `SchemaCatalog`；接表只需换 `SchemaProvider` 的加载实现） | 后置 |
| `make cleanup` 的保留期策略与实现（详细设计 16.12） | Phase 2 收尾 |
| 事件流的 MySQL 权威重放（`agent_trace_event`）+ `sequence` 改由该表提供 | Phase 2 / 10 |
| `stream_url` 指向的 SSE 订阅端点（契约已固定，事件已可订阅） | Phase 10 |
| 用户消息落库（FR-CHAT-001 处理流程的「保存用户消息」，`agent_message` 表） | Phase 2 |
| ~~任务体换成 LangGraph~~ **已完成**（`TaskRunner` 收 `body=` 注入） | ✅ Phase 6 |
| 节点内检查取消标记（`TaskRunner.is_cancel_requested` 目前只在领取与收尾时检查） | Phase 7 |
| `WAITING_CLARIFICATION` 的**状态位**：澄清现在在答案文本里表达，任务终态仍是 SUCCEEDED。真正停在澄清态要 API/SSE 侧的配套 | Phase 6 收尾 |
| 图上的 `plan_extend` 与**模型的 EXPAND 判定**：现在由 `reflect` 确定性演进，`plan_deltas` 因此恒空、`investigation_chain.triggered_step_id` 恒为 None | Phase 7 |
| LangGraph **checkpointer**（断点续跑）：与"整任务重跑"是两种重试语义，并存会出"重投了一个跑了一半的任务" | 需要时 |
| ~~任务详情的步骤进度、证据、冲突、限制~~ **已完成**：`agent_task.plan_json` / `result_json` 两列（Phase 2 就建好了，一直没人写）+ 17.2 的返回字段 | ✅ Phase 6 收尾 |
| ~~对象存储 `s3` 实现~~ **已完成**（`make ingest` 起会真的用到它归档原文与报告） | ✅ Phase 5 |
| **11.7 第 ⑧ 步「邻近块扩展」**：原文条件是「同文档、同章节且**确有上下文缺口**时」，而缺口判定依赖重排器的相关性信号。没有它只能退化成「块短就扩」，在本语料上几乎恒真（正文块中位 85 字），等于无条件把候选翻倍。**随重排器一起后置**——这是时序调整，不是砍需求 | 与 Reranker 同批 |
| 文档侧的 `metric_code` / `definition_version` / `scope` 暂时留空（分块不带指标 code，按 `logical_key` 反推是把自由文本约定当语义用）。**13.4 的 DEFINITION / SCOPE 冲突因此暂时只覆盖 SQL 侧**，要等文档与指标目录挂钩 | Phase 9 |
| 把疑问词并入 `configs/rag_stopwords.txt`（更整齐，但改切分必须重跑 `make vocab` + `make ingest` + 检索回归集） | Phase 5 收尾 |
| **同义词导致的误拒**：`报备` vs 语料里的 `备案`（金标 rag-06，余弦 0.74 仍被拒答）。纯词汇规则解决不了，要靠重排器或 Phase 8 的 Reviewer 给语义信号 | 与 Reranker 同批 |
| **冲突检测的另外四类**（DEFINITION / TIME / SCOPE / SOURCE）：前提分别是文档侧 `metric_code`+`scope`、文档统计期间、方向判定，见 `nodes/conflict.py` 的清单 | Phase 9 |
| **表格合计行与分项行的区分**：现在两者都会被当成"该指标在该范围的值"去比，实测产生过误报 | Phase 9 |
| `agent_conflict` 表落库（现在冲突只在 State 与 `answer_payload` 里） | Phase 9 |
| **Reviewer 第二阶段（模型审查）**：是否回答问题、证据是否足够、推断是否越界（14.1 后半），以及 14.4 的 `retry_router` 与四类预算 | Phase 8 完整版 |
| `reindex` 接口：**原文件已归档、collection 已是可重建的派生数据，只差一条命令**。注意重建前要比对行的 `checksum` 与归档原文（`FORCE=1` 失败重建时两者会分叉，见详设 11.9 的落地记录） | Phase 5 收尾 |
| 扫描件 OCR：语料里 2 份（SP-016 / CM-010）已归档原文并标 `FAILED`，补齐 OCR 后可直接从归档重跑 | 后置 |
| Qdrant 的 `set_payload` 跨分片无事务保证 → 发布窗口内读者可能看到同一版本的部分 chunk。要严格就需把状态位提到文档级（见详设 11.1 的落地记录第 4 条） | 语料规模上去再评估 |
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

> 与向量相关的命令（`make ingest` 起）**还需要 Ollama 在跑**，而它不在 compose 里。
> 详见下方「本机环境事实」里的 Ollama 一段。

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
| Ollama（向量模型） | 宿主端口 **11434**，**是一个名叫 `ollama` 的独立 Docker 容器** | ⚠️ **不在 `docker-compose.dev.yml` 里，`make up` 不会拉起它**，`make ps` 也看不到它。模型实体在 docker 卷 `ollama-data`（宿主机 `/var/lib/docker/volumes/ollama-data/_data`，容器内 `/root/.ollama`），**库里只有 `bge-m3`**（1.1GB，2026-08-19 建）。见下方专段 |

### Ollama：一条未声明的依赖

本机有**两个** Ollama，别搞混：

| | 位置 | 模型库 |
|---|---|---|
| Windows 侧 | `D:\ollama`（含 `ollama app.exe`） | `C:\Users\DELL\.ollama\models` —— **空的，从没 pull 过** |
| WSL 侧（**项目用的是这个**） | Docker 容器 `ollama` | 卷 `ollama-data` 里只有 `bge-m3` |

那个容器**不是本项目建的**（创建于 2026-08-19，比本仓库第一次提交还早一个月），
是用 `docker run` 起的裸容器，没有 compose 标签，因此不受本项目编排：

- **`make up` 不会启动它**，`make ps` 也列不出它。RAG 入库前要另行确认它在跑；
  不在时 `make ingest` 会以"连不上 11434"失败，而报错**不指向 ollama**。
  自查一条命令：`curl -s http://127.0.0.1:11434/api/tags`
- 它带 `restart=unless-stopped`，能扛 Docker 守护进程重启；而本项目自己的容器
  **没有 restart policy**（WSL 重启后不会自动起）。两者行为相反，排查时容易判反。
- 卷 `ollama-data` 一旦被 `docker system prune --volumes` 之类删掉，模型就没了；
  而按详设 11.5 的硬规定**换 embedding 模型必须新建 collection 全量重建**——
  卷丢了，已入库的向量跟着作废。
- **收进 compose 之前先确认没有别的项目在用这个容器**（它是裸容器，可能被共享）。

> 排查提示：容器里的进程**会出现在 WSL 的 `ps` 里**（PID 命名空间分层，宿主可见子命名空间）。
> 所以 `ps` 里看到 `/bin/ollama serve` 不代表 WSL 装了原生 ollama——
> 那个路径是**容器内**的文件系统，`which ollama` 在 WSL 里是找不到的。

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
