# API 与 Worker 共用镜像，仅启动命令不同（开发流程 6.1 第 9 项）
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# curl 供 healthcheck 使用；build-essential 供 asyncmy 等 C 扩展编译
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential default-libmysqlclient-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 先只拷贝依赖描述文件，让依赖层可缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
COPY scripts ./scripts
COPY configs ./configs

# 以非 root 运行
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 默认启动 API；Worker 用 `docker run ... arq app.worker.WorkerSettings` 覆盖
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
