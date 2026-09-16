.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose -f docker-compose.dev.yml

.PHONY: help bootstrap up down ps logs redis-cli api worker run fmt lint typecheck \
        test test-integration layering check clean

help:  ## 显示所有可用目标
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

bootstrap:  ## 首次启动：创建 .env 与虚拟环境
	@test -f .env || (cp .env.example .env && echo "已生成 .env，请填入 MODEL_* 与 EMBEDDING_* 密钥")
	uv sync
	@echo "完成。下一步：make up"

up:  ## 启动有状态组件（redis / agent-mysql / business-mysql / minio）
	$(COMPOSE) up -d
	@echo "等待健康检查通过…"
	@$(COMPOSE) ps

rag-up:  ## 额外启动 Milvus（内存占用大，Phase 5 才需要）
	$(COMPOSE) --profile rag up -d milvus

down:  ## 停止所有组件
	$(COMPOSE) --profile rag down

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

test-integration:  ## 需要真实 Redis 的用例（前置：make up）
	uv run pytest -m integration app/tests/integration -v

layering:  ## 分层约束检查
	uv run python scripts/check_layering.py

check: lint typecheck layering test  ## 提交前必跑：lint + 类型 + 分层 + 测试

clean:  ## 清理缓存与虚拟环境
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
