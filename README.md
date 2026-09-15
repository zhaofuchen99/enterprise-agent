# 企业智能数据分析与决策 Agent

把「自然语言问题」转化为「可追溯的数据结论」的多 Agent 系统：SQL 查询、知识检索、
结果校验与冲突识别在同一个任务循环里协作，全过程可观测、可重放、有界收敛。

设计文档见 [`docs/`](docs/)：

| 文档 | 内容 |
|---|---|
| `企业智能数据分析与决策Agent-需求规格说明书.md` | 功能与非功能需求 |
| `企业智能数据分析与决策Agent-详细设计说明书.md` | 架构、Schema、接口、错误码 |
| `企业智能数据分析与决策Agent-开发流程.md` | 阶段划分、验收命令、门禁 |

---

## 快速开始（30 分钟内）

### 0. 前置条件

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.x | 由 `uv` 自动安装，无需手动准备 |
| uv | 最新 | <https://docs.astral.sh/uv/> |
| Docker + Compose | 最新稳定 | 承载有状态组件 |
| 内存 | ≥ 16GB | Milvus 单实例就要 4–8GB，见下方「已知限制」 |

### 1. 初始化

```bash
make bootstrap     # 生成 .env 并安装依赖
```

然后编辑 `.env`，**至少填上模型相关密钥**：

```
MODEL_PROVIDER / MODEL_NAME / MODEL_API_KEY
EMBEDDING_MODEL / EMBEDDING_API_KEY
```

> 密钥只从环境变量加载。`.env` 已在 `.gitignore` 中，任何情况下不得提交或写入 Compose 明文。

### 2. 起有状态组件

```bash
make up            # redis / agent-mysql / business-mysql / minio
```

### 3. 起服务

```bash
make run           # 同时起 api + worker
```

> **不要只跑 `make api`。** API 与 Worker 是两个进程，只起 API 会让任务永远停在
> `QUEUED`，且从 API 日志里几乎看不出原因——这是本项目最常见的自伤方式（开发流程 4.5）。

### 4. 验证

```bash
curl -s localhost:8000/health/live     # {"status":"ok","service":"api"}
curl -s localhost:8000/health/ready    # {"status":"ready","checked":["redis"]}
make check                             # lint + 类型 + 分层约束 + 测试
```

---

## 常用命令

```
make up         启动有状态组件          make check      提交前必跑（lint+类型+分层+测试）
make rag-up     额外启动 Milvus         make lint       仅静态检查
make down       停止全部                make typecheck  仅类型检查
make redis-cli  进入 Redis 排查         make layering   仅分层约束检查
make api        只起 API 进程           make test       仅单元测试
make worker     只起 Worker 进程        make fmt        格式化并自动修复
make run        同时起两个进程
```

排查任务卡住时：

```bash
make redis-cli
> KEYS q:agent                      # 队列积压
> XLEN task:tsk_xxx:events          # 某任务的事件流
> GET  task:tsk_xxx:heartbeat       # Worker 心跳是否新鲜
> LLEN q:agent                      # 队列深度
```

---

## 架构

### 分层与依赖方向

```
api  →  services  →  agent / domain / tools  →  repositories / infrastructure
```

依赖只能向右。领域 Schema **不依赖** FastAPI，也不依赖具体数据库客户端。

| 层 | 职责 | 禁止 |
|---|---|---|
| `api/` | 鉴权、参数校验、调用 service | 执行 SQL、调用模型、写业务规则、导入 `agent/` 编排实现 |
| `services/` | 编排、事务、事件、队列与限流封装 | 拼接 SQL 字符串、直接实例化 LangGraph |
| `agent/nodes/` | 读写 State、调用 Tool 接口 | 在 Node 内建数据库连接、导入 FastAPI |
| `domain/` | 纯数据模型与校验逻辑 | 依赖 FastAPI、依赖具体数据库客户端 |
| `repositories/` | 数据访问实现 | 包含业务判断 |
| `infrastructure/` | 外部系统客户端与协议实现 | 包含业务规则 |

### 两条进程边界硬约束

1. `app/main.py` 与 `app/api/**` 不得 import `agent.graph`、`agent.nodes.*`、`tools.*`；
2. `app/worker.py` 与 `app/agent/**`、`app/tools/**` 不得 import `fastapi`。

这两条**由 `scripts/check_layering.py` 在 CI 中强制**，而非靠人工 review。破坏后 API 进程
会加载 LangGraph 与全部 Tool，启动变慢、内存翻倍，且 Worker 无法独立扩缩容——影响要到
运行时才暴露。

```bash
make layering
```

### 目录结构

```
app/
├── main.py                     # API 进程入口
├── worker.py                   # Worker 进程入口
├── api/            鉴权、路由、参数校验
├── agent/          LangGraph 编排：graph / state / routing / nodes / prompts / schemas
├── tools/          sql（schema/generator/validator/executor）、rag、search
├── domain/         evidence / conflict / task / plan / loop / review（纯模型）
├── services/       task_runner / event_bus / rate_limit / ...
├── repositories/   数据访问实现
├── infrastructure/ db / redis / milvus / storage / model_gateway / observability / logging
├── core/           config / cache / errors / security
└── tests/
```

---

## 配置

全部配置项在 `app/core/config.py` 以 Pydantic Settings 声明，带默认值与上下限。
环境变量清单见 `.env.example`。嵌套配置用双下划线覆盖：

```bash
LOOP__MAX_TOTAL_STEPS=30
```

**循环预算**（`LOOP__*`）直接约束任务循环的收敛性。四类预算
（`MAX_SQL_REPAIRS` / `MAX_REPLANS` / `MAX_REVIEWER_EVIDENCE` / `MAX_EXPANSIONS`）
相互独立、不可借用；任何一项调整后必须重跑循环类评测集。

---

## 已知限制

| 限制 | 影响 | 处理 |
|---|---|---|
| 本机 WSL 内存 7.6GB | Milvus 需 4–8GB，与 MySQL/Redis/MinIO 并存会很紧张 | Milvus 置于 `rag` profile，Phase 0–4 不受影响；Phase 5 前需决定是否上调 WSL 内存或更换向量库 |
| 项目位于 WSL 原生 ext4 | Windows 侧需经 `\\wsl$\` 访问 | 有意为之：`/mnt/c` 走 9p，`uv sync` 与 `pytest` 会慢一个数量级 |
| `MODEL_*` / `EMBEDDING_*` 未选型（TBC-04） | Phase 3 起才真正需要 | 见开发流程 12.2 |

---

## 开发约定

- **分支**：主干开发 + 短生命周期特性分支（`feat/phase-04-sql-validator`）
- **提交**：Conventional Commits（`feat(sql): 增加 AST 表字段白名单校验`）
- **每个 Phase 的节奏**：读文档 → 定 Schema → 写会失败的测试 → 实现 → 本地验证 → 埋点 → 更新配置 → 对照门禁 → 提交
- **完成的定义**：`make check` 全绿、验收命令可重复执行、需求条目逐条核对、Trace 与结构化日志可按 `task_id` 定位、新增配置有默认值与上下限、设计偏离已回写文档、无未标注 TODO

### 日志与脱敏

结构化日志字段固定为
`timestamp、level、service、trace_id、conversation_id、task_id、step_id、node、tool、status、duration_ms、error_code、message`。

**日志、SSE、Trace 默认脱敏**：禁止记录 Token、密码、密钥、完整敏感字段、模型隐藏思维链、SQL 原始行数据。
