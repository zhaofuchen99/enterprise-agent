"""Qdrant 挂掉时的**异常形状**（详设 9.4 / 19.1）。

为什么单独一个文件而不是放进 `test_vector_store_contract.py`：那份契约测试
参数化跑**两个实现**（内存替身 + 真 Qdrant），验的是"两边行为一致"；
而这里验的是 **Qdrant 实现独有**的一处故障分类——内存替身不会连接失败。

**不需要容器**：连一个没人监听的端口就够了（连接被拒是即时的）。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import AgentError, ErrorCode
from app.infrastructure.vector_store import ChunkFilter, QdrantVectorStore

#: 一个没人监听的本地端口。用 `127.0.0.1:1` 而不是随便挑一个高位端口：
#: 高位端口有被别的进程占用的可能，而"连上了"会让这条用例测的东西消失。
_DEAD_URL = "http://127.0.0.1:1"


def _store() -> QdrantVectorStore:
    return QdrantVectorStore(url=_DEAD_URL, collection="whatever")


async def test_a_transport_failure_becomes_a_retryable_upstream_error() -> None:
    """连不上 Qdrant → `UPSTREAM_UNAVAILABLE`（503 / 可重试），**不是"内部错误"**。

    不包的话原始异常会一路穿到 `graph.ainvoke` 与 `TaskRunner.execute` 的
    `except Exception`，任务以 `INTERNAL_ERROR` 收场。后果不只是措辞：
    - `INTERNAL_ERROR` 的处置是"告警并返回通用提示"，客户端不会重试；
    - 而"业务库断连"与"模型 429"走的都是可重试的 5xx —— 同一个
      「外部依赖挂了」在三处有三种待遇，排查时看不出它们是一回事。

    实测的异常链（`docker stop ea-qdrant`）：
    `ResponseHandlingException <- httpx.ConnectError <- OSError <- ConnectionRefusedError`。
    """
    store = _store()
    try:
        with pytest.raises(AgentError) as caught:
            await store.search_dense(
                [0.1] * 1024, limit=1, chunk_filter=ChunkFilter(status="ACTIVE")
            )
    finally:
        await store.aclose()

    assert caught.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert caught.value.retryable is True
    assert caught.value.details is not None
    assert caught.value.details["error_class"] == "TRANSIENT"
    # 排查时要能看出"是哪一层、哪个操作"挂了
    assert caught.value.details["store"] == "qdrant"
    assert caught.value.details["operation"] == "search_dense"


async def test_our_own_bug_is_not_disguised_as_an_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """**我们自己的 bug 必须原样冒出去**，不能被伪装成"可重试的上游故障"。

    这是这条包装的边界，也是它最容易写错的地方——一个 `except Exception`
    会让 `TypeError` / `KeyError` 这类编程错误也变成"Qdrant 抖了一下"：
    它会重试、会失败，而排查方向被指向 Qdrant。

    ⚠️ **这个坑在写这版时就踩过一次**：实验脚本少传一个参数抛了 `TypeError`，
    看起来与"Qdrant 挂了"一模一样（见 `_QDRANT_FAILURES` 的说明）。
    """

    class _BuggyClient:
        async def query_points(self, *_args: Any, **_kwargs: Any) -> Any:
            raise TypeError("payload 反解写错了")

        async def close(self) -> None:
            """`aclose` 会关客户端——替身少了它，用例会以另一种错误失败。"""

        async def get_collections(self) -> Any:
            # 构造 `AsyncQdrantClient` 时会探一次版本，替身不必真连
            return None

    store = _store()
    monkeypatch.setattr(store, "_client", _BuggyClient())
    try:
        with pytest.raises(TypeError):
            await store.search_dense(
                [0.1] * 1024, limit=1, chunk_filter=ChunkFilter(status="ACTIVE")
            )
    finally:
        await store.aclose()


async def test_fetch_is_wrapped_too() -> None:
    """`fetch` 也在同一组包装里（11.7 第 ⑧ 步的邻近块扩展走它）。

    漏掉它的症状是"检索路径修好了、补块路径照样报内部错误"——
    而那两条路径在用户看来都是"RAG 挂了"。
    """
    store = _store()
    try:
        with pytest.raises(AgentError) as caught:
            await store.fetch(ChunkFilter(document_ids=("doc@v1.0",)), limit=1)
    finally:
        await store.aclose()

    assert caught.value.code is ErrorCode.UPSTREAM_UNAVAILABLE
    assert caught.value.details is not None
    assert caught.value.details["operation"] == "fetch"
