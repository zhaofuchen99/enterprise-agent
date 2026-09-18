.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose -f docker-compose.dev.yml

.PHONY: help bootstrap up down ps logs redis-cli api worker run fmt lint typecheck \
        test test-integration layering check clean
.PHONY: migrate revision seed seed-business verify-business cleanup corpus corpus-list
.PHONY: dict tokenize chunk vocab
.PHONY: model-smoke sql eval-sql vector-spike

help:  ## 显示所有可用目标
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

bootstrap:  ## 首次启动：创建 .env 与虚拟环境
	@test -f .env || (cp .env.example .env && echo "已生成 .env，请填入 MODEL_* 与 EMBEDDING_* 密钥")
	uv sync
	@echo "完成。下一步：make up"

up:  ## 启动有状态组件（redis / agent-mysql / business-mysql / qdrant / minio）
	$(COMPOSE) up -d
	@echo "等待健康检查通过…"
	@$(COMPOSE) ps

rag-up:  ## 兼容旧写法：确保 Qdrant 已启动（它已并入默认 up，此目标可省略）
	$(COMPOSE) up -d qdrant

down:  ## 停止所有组件
	$(COMPOSE) down

ps:  ## 查看组件状态
	$(COMPOSE) --profile rag ps

logs:  ## 跟踪组件日志
	$(COMPOSE) logs -f

redis-cli:  ## 进入 Redis 排查队列与 Stream
	$(COMPOSE) exec redis redis-cli

api:  ## 只起 API 进程（多实例验收用 PORT=8001 指定端口）
	uv run uvicorn app.main:app --reload --host $${API_HOST:-0.0.0.0} --port $${PORT:-$${API_PORT:-8000}}

worker:  ## 只起 Worker 进程（带依赖抖动的自动重启，见 app/worker.py）
	uv run python -m app.worker

run:  ## 同时起 api + worker（本地开发必须用这个，见开发流程 4.5 调试纪律）
	@echo "启动 worker（后台）与 api（前台）…"
	@trap 'kill 0' EXIT INT TERM; \
		uv run python -m app.worker 2>&1 | sed 's/^/[worker] /' & \
		uv run uvicorn app.main:app --reload --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} 2>&1 | sed 's/^/[api]    /' & \
		wait

fmt:  ## 格式化并自动修复可修复的 lint 问题
	uv run ruff format app scripts
	uv run ruff check --fix app scripts

lint:  ## 静态检查（不修改文件）
	uv run ruff format --check app scripts
	uv run ruff check app scripts

typecheck:  ## 类型检查
	uv run mypy app scripts

test:  ## 单元测试（不含 integration，默认依赖都已用替身）
	uv run pytest

test-db:  ## 建测试库 agent_test（幂等；前置：make up）
	@$(COMPOSE) exec -T agent-mysql mysql -uroot -proot_pw -e "\
		CREATE DATABASE IF NOT EXISTS agent_test \
			CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci; \
		GRANT ALL PRIVILEGES ON agent_test.* TO 'agent'@'%'; \
		FLUSH PRIVILEGES;" 2>/dev/null
	@echo "agent_test 就绪（表结构由用例内的 create_all 建立）"

test-integration: test-db  ## 需要真实 Redis / MySQL 的用例（前置：make up）
	uv run pytest -m integration -v

layering:  ## 分层约束检查
	uv run python scripts/check_layering.py

check: lint typecheck layering test  ## 提交前必跑：lint + 类型 + 分层 + 测试

clean:  ## 清理缓存与虚拟环境
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache

# ------------------------------------------------------------ 数据库（Phase 2）
migrate:  ## 迁移 agent 运行库到最新版本（前置：make up）
	uv run alembic upgrade head

revision:  ## 按模型与库的差异自动生成迁移，需带 m="说明"
	@test -n "$(m)" || (echo "用法：make revision m=\"说明\"" && exit 1)
	uv run alembic revision --autogenerate -m "$(m)"

seed:  ## 灌入演示数据（Agent 库的演示账号 + 业务库的反向构造数据，均幂等）
	uv run python -m app.cli seed
	uv run python scripts/business_seed.py

seed-business:  ## 只灌业务库，带规模参数：make seed-business ROWS=20000
	uv run python scripts/business_seed.py --rows $${ROWS:-500000}

verify-business:  ## 只跑业务库的四条约束断言与 EXPLAIN 检查（不重新生成数据）
	uv run python scripts/business_seed.py --verify-only

corpus:  ## 生成演示语料（88 篇，含 10 类缺陷注入，前置：make up + 业务库已灌数）
	uv run python scripts/gen_corpus.py $(if $(SUBSET),--subset $(SUBSET),) $(if $(ONLY),--only $(ONLY),) --clean

corpus-list:  ## 只看语料清单与缺陷标注，不生成文件
	uv run python scripts/gen_corpus.py --list

# ------------------------------------------------------------------ RAG（Phase 5）
dict:  ## 生成业务分词词典（前置：make up + 业务库已灌数；产物入版本库）
	uv run python scripts/gen_dict.py

tokenize:  ## 逐条核对中文分词：make tokenize T="华东区域渠道折扣政策" [NO_DICT=1]
	@test -n "$(T)" || (echo '用法：make tokenize T="华东区域渠道折扣政策"' && exit 1)
	uv run python -m app.cli tokenize "$(T)" $(if $(NO_DICT),--no-dict,)

vocab:  ## 构建稀疏检索词表并导出快照（幂等，前置：make corpus）
	uv run python -m app.cli vocab $(if $(P),$(P),)
ingest:  ## 逐篇入库并发布：make ingest [P=data/corpus] [ONLY=SP-001] [FORCE=1]（前置：make corpus + make vocab + Ollama）
	uv run python -m app.cli ingest $(if $(P),$(P),) $(if $(ONLY),--only $(ONLY),) $(if $(FORCE),--force,)
retrieve:  ## 混合检索并产出文档证据：make retrieve Q="华东渠道折扣怎么规定" [AS_OF=2025-09-01]（前置：make ingest + Ollama）
	@test -n "$(Q)" || (echo '用法：make retrieve Q="问题" [AS_OF=YYYY-MM-DD]' && exit 1)
	uv run python -m app.cli retrieve "$(Q)" $(if $(AS_OF),--as-of $(AS_OF),) $(foreach t,$(TYPE),--doc-type $(t),) $(foreach d,$(DEPT),--department $(d),)

chunk:  ## 解析 + 分块，逐条核对：make chunk P=data/corpus/SP-015.pdf [SUMMARY=1] [LIMIT=5]
	@test -n "$(P)" || (echo '用法：make chunk P=data/corpus/SP-015.pdf' && exit 1)
	uv run python -m app.cli chunk "$(P)" $(if $(SUMMARY),--summary,) $(if $(LIMIT),--limit $(LIMIT),) $(if $(TABLES_ONLY),--table-only,) $(if $(TEXT_ONLY),--text-only,)

cleanup:  ## 按保留期清理过期数据（幂等，供 cron 调用，见详细设计 16.12）
	uv run python -m app.cli cleanup

model-smoke:  ## 打一次真实模型与向量化（前置：.env 已填 MODEL_API_KEY；向量模型需 Ollama 在跑）
	uv run python -m app.cli model-smoke

# ------------------------------------------------------------------ SQL Tool（Phase 4）
sql:  ## 跑一次自然语言 → SQL → 真数据 → 证据：make sql Q="2025年华东地区Q3净销售额"
	@test -n "$(Q)" || (echo '用法：make sql Q="问题" [REGION=华东]' && exit 1)
	uv run python -m app.cli sql "$(Q)" $(if $(REGION),$(foreach r,$(REGION),--region $(r)),)

eval-sql:  ## 跑金标 SQL 评测集，产出正确率（前置：make up + 业务库已灌数）
	uv run python scripts/eval_sql.py $(if $(ONLY),--only $(ONLY),)
eval-rag:  ## 跑 RAG 金标 20 条，产出 Recall@8 与定位一致率（前置：make ingest + Ollama）
	uv run python scripts/eval_rag.py $(if $(ONLY),--only $(ONLY),)
demo:  ## 端到端演示六条固化问题（前置：另一个终端跑 make run）
	uv run python scripts/demo.py $(if $(ONLY),--only $(ONLY),) $(if $(BASE),--base $(BASE),)
verify-corpus:  ## 断言 10 类缺陷真的注入了产物（不是清单标注），任一失败即非零（前置：make ingest）
	uv run python -m app.cli verify-corpus

vector-spike:  ## TBC-05 向量库选型实测（四个候选项同口径对比，需独立 venv，见脚本 docstring）
	@test -x /tmp/tbc05-venv/bin/python || { \
		echo "缺少实测环境，先按脚本 docstring 的「复现」一节建独立 venv："; \
		echo "  uv venv /tmp/tbc05-venv --python 3.12"; \
		echo "  uv pip install --python /tmp/tbc05-venv/bin/python pymilvus milvus-lite qdrant-client chromadb"; \
		exit 1; }
	/tmp/tbc05-venv/bin/python scripts/spike_vector_store.py
