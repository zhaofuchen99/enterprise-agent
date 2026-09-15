# CLAUDE.md

本文件是项目级约定，AI 编码助手与人类协作者共用。改动架构或约定时**同步更新本文件**。

---

## 项目

**企业智能数据分析与决策 Agent** —— 把自然语言问题转化为可追溯的数据结论的多 Agent 系统。
SQL 查询、知识检索、结果校验与冲突识别在同一个任务循环里协作，全过程可观测、可重放、有界收敛。

技术栈：Python 3.12 · FastAPI · LangGraph · MySQL 8 · Redis · Milvus · OTel · arq · uv

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
make redis-cli   # 排查队列与 Stream
```

单独跑：`make lint` / `make typecheck` / `make layering` / `make test` / `make fmt`

**只跑 `make api` 会导致任务永远停在 `QUEUED`**，且从 API 日志几乎看不出原因。这是本项目最常见的自伤方式。

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
- **测试**：单元测试不依赖真实外部组件；需要真实 Redis/MySQL/Milvus 的用例打 `@pytest.mark.integration`
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
