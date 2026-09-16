# CLAUDE.md

本文件是项目级约定，AI 编码助手与人类协作者共用。改动架构或约定时**同步更新本文件**。

---

## 项目

**企业智能数据分析与决策 Agent** —— 把自然语言问题转化为可追溯的数据结论的多 Agent 系统。
SQL 查询、知识检索、结果校验与冲突识别在同一个任务循环里协作，全过程可观测、可重放、有界收敛。

技术栈：Python 3.12 · FastAPI · LangGraph · MySQL 8 · Redis · Milvus · OTel · arq · uv

## 当前进度

> 每推进一个 Phase 就更新这里，新会话靠它接上下文。

| Phase | 状态 | 备注 |
|---|---|---|
| 0 项目初始化 | ✅ 完成 | 骨架、配置、错误码、结构化日志、健康探针、分层约束检查、CI、pre-commit 密钥防护 |
| 1 基础 API | ✅ 完成 | 统一响应外壳、全局异常映射（400/401/403/404/409/429/500）、`trc_` trace 中间件、本地 JWT 登录、`/api/agent/chat` 与 `/api/agent/tasks/{id}` 占位、OpenAPI 定制 |
| 1.5 基础设施接入 | ✅ 完成 | Redis 键与 Lua 脚本、Redis 限流 + 降级 + 熔断、Redis 任务仓储（临时）、arq 队列与 Worker 骨架（含自愈重启）、事件流、取消链路、对象存储抽象、OTel 与跨进程 trace。`make check` 全绿（267 测试）+ 7 条 integration 用例；验收记录见开发流程 11.1 |
| 2 数据库 | ⬜ 未开始 | 下一步 |

**开工前**：先 `make ps` 看有状态组件是否在跑，没起来就 `make up`。

**Phase 1.5 之后仍然存在的临时约定**（Phase 2 拆掉）：

1. **用户与会话**仓储仍是进程内占位，重启即丢、多实例互不可见。
   （任务仓储已改为 Redis，跨实例可见——但它是临时的，Phase 2 换成 MySQL。）
2. **任务体是空实现**：任务会以 `SUCCEEDED` 且 `answer` 为 `null` 结束。
   这是预期行为，不是 bug——本阶段验证的是执行链路，分析能力在 Phase 7。
3. **Redis 此刻是创建任务的硬依赖**：任务仓储还是 Redis 实现，Redis 挂掉时
   `POST /api/agent/chat` 返回 503 `REDIS_UNAVAILABLE`。Phase 2 换成 MySQL 后，
   Redis 不可用只会影响限流与队列（限流有降级路径，任务由补偿扫描补投）。
4. 演示账号（`analyst`/`admin`）的 ID 由用户名确定性派生，**多实例之间是同一个用户**；
   真实用户仍走随机 ULID。这样多实例的限流与并发配额才能被端到端验收。

### 【后续扩展】登记

| 项 | 触发阶段 |
|---|---|
| 任务仓储换 MySQL（**并删除 `RedisTaskRepository` 与它的 5 个临时索引键**） | Phase 2 |
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

## 设计文档

`docs/` 下四份文档是**设计意图的记录，不是验收契约**：

| 文件 | 用途 |
|---|---|
| `企业智能数据分析与决策Agent-需求规格说明书.md` | 功能与非功能需求（FR 条目） |
| `企业智能数据分析与决策Agent-详细设计说明书.md` | 架构、Schema、接口、错误码表（19.1） |
| `企业智能数据分析与决策Agent-开发流程.md` | Phase 划分、验收命令、门禁、工作量估算 |
| `Agent项目需求.md` | 原始需求 |

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
make test-integration  # 需要真实 Redis（前置 make up）
make redis-cli   # 排查队列与 Stream
```

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
- **测试**：单元测试不依赖真实外部组件；需要真实 Redis/MySQL/Milvus 的用例打 `@pytest.mark.integration`。
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
| 内存 | 7.6GB（物理机 16GB） | **Milvus Standalone 需 8GB 起，本机跑不起来** |
| Redis | 宿主端口 **6381** | 6379 被本机原生 redis 占用（存有另一个项目的数据，不可动）；6380 被 redis-stack 占用。容器内仍是 6379 |
| MinIO | 镜像用 `quay.io/minio/minio` | 本机 daemon 的 `docker.m.daocloud.io` 镜像源对 `minio/minio` 返回 403 |
| MySQL | agent `3306` / business `3307` | business 用只读账号，写操作必须被数据库拒绝 |
| Milvus | 在 `rag` profile 下，默认不启动 | 见下方未决项 |

## 未决项

| 编号 | 内容 | 何时需要定 |
|---|---|---|
| TBC-04 | **模型选型**：主聊天模型、embedding、reranker。本机已装 Ollama，本地模型可省 API 成本 | Phase 3 开工前 |
| TBC-05 | **向量库**：Milvus Standalone 本机内存不足。候选为 Milvus Lite（嵌入式，`pymilvus` 同一客户端，只改 URI）、Qdrant、Chroma | Phase 5 开工前，**先跑最小用例实测再定** |

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
