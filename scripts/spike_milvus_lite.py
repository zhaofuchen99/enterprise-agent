"""TBC-05 最小验证：Milvus Lite 是否满足详细设计 11.5 / 11.6 / 11.7。

背景：本机 7.6GB 内存，Milvus Standalone 需 8GB 起，跑不起来（CLAUDE.md 本机环境事实）。
详细设计要求「先跑最小用例实测再定」，本脚本即该实测，用于在
Milvus Lite / Qdrant / Chroma 之间做决策。

回答三个问题：
  1. 嵌入式 Milvus Lite 能不能跑（不占 8GB、不用 Docker）？
  2. 能不能按 11.5 建 DENSE + SPARSE_FLOAT_VECTOR 双路 collection？
  3. 能不能按 11.7 做 dense+sparse 的 RRF 混合检索？

复现：
    uv pip install milvus-lite
    uv run python scripts/spike_milvus_lite.py

实测结果（2026-09-16）：三项全部通过，嵌入式方案能承载设计 11.5 的全部字段。

**这不是 TBC-05 的决议**——向量库选型仍在 Phase 5 开工前决定，届时需连同 Qdrant / Chroma
一起做同口径对比。本脚本的用处仅在于排除一种风险：设计里的 SPARSE_FLOAT_VECTOR 与
RRF 混合检索**不是**只有重型部署方案才能满足，因此 RAG 不存在"向量库选不出来"这类隐藏工期炸弹。
"""

from __future__ import annotations

import os
import resource
import shutil
import tempfile
import time

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

DIM = 768  # 取一个真实 embedding 量级的维度，避免小维度掩盖问题
N_CHUNKS = 2000  # 80+ 文档量级的上限估计，留足余量


def main() -> int:
    db_dir = os.path.join(tempfile.mkdtemp(prefix="milvus_lite_spike_"), "kb.db")
    client = MilvusClient(db_dir)

    # --- 1. 按 11.5 建双路 collection ---
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    # 标量字段走 enable_dynamic_field，对应 11.4 的 ChunkMetadata（状态/日期/权限过滤）
    client.create_collection("enterprise_knowledge_chunks_v1", schema=schema)

    # --- 2. 入库 ---
    rows = [
        {
            "chunk_id": f"chk_{i:06d}",
            "text": "销售政策 > 渠道折扣 > 华东区域 " * 8,
            "dense_vector": [0.01 * (i % 97)] * DIM,
            # 稀疏向量：真实实现里由 jieba 分词 + rag_vocab 映射得到（11.6）
            "sparse_vector": {1: 0.5, 7: 0.3, (i % 500) + 100: 0.9},
            "document_type": "POLICY",
            "status": "ACTIVE",
        }
        for i in range(N_CHUNKS)
    ]
    t0 = time.perf_counter()
    client.insert("enterprise_knowledge_chunks_v1", rows)
    client.flush("enterprise_knowledge_chunks_v1")
    insert_s = time.perf_counter() - t0

    # --- 3. 建索引（稀疏索引用 SPARSE_INVERTED_INDEX，见 11.6）---
    params = client.prepare_index_params()
    params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="COSINE")
    params.add_index(
        field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP"
    )
    client.create_index("enterprise_knowledge_chunks_v1", index_params=params)
    client.load_collection("enterprise_knowledge_chunks_v1")

    # --- 4. 按 11.7 做 RRF 混合检索 ---
    dense_req = AnnSearchRequest(data=[[0.01] * DIM], anns_field="dense_vector", param={}, limit=8)
    sparse_req = AnnSearchRequest(
        data=[{1: 0.5, 7: 0.3}], anns_field="sparse_vector", param={}, limit=8
    )
    t0 = time.perf_counter()
    hits = client.hybrid_search(
        "enterprise_knowledge_chunks_v1",
        [dense_req, sparse_req],
        ranker=RRFRanker(60),
        limit=8,
        output_fields=["text"],
    )
    search_ms = (time.perf_counter() - t0) * 1000

    # --- 5. 标量过滤（11.4 的 metadata 过滤）---
    filtered = client.search(
        "enterprise_knowledge_chunks_v1",
        data=[[0.01] * DIM],
        limit=5,
        filter='document_type == "POLICY" && status == "ACTIVE"',
    )

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    disk_mb = (
        sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(db_dir) for f in fs)
        / 1024
        / 1024
    )

    print("=" * 64)
    print(f"入库 {N_CHUNKS} 条（{DIM} 维 dense + sparse）  耗时 {insert_s:.1f}s")
    print(f"RRF 混合检索                                耗时 {search_ms:.1f}ms")
    print(f"混合检索返回 {len(hits[0])} 条，标量过滤返回 {len(filtered[0])} 条")
    print(f"进程峰值 RSS  {rss_mb:.0f} MB   （Milvus Standalone 需 8000 MB 起）")
    print(f"落盘占用      {disk_mb:.1f} MB")
    print("=" * 64)
    print("结论：SPARSE_FLOAT_VECTOR + SPARSE_INVERTED_INDEX + hybrid_search(RRF) 全部可用。")
    print("      嵌入式方案可承载设计 11.5 的全部字段，本机内存不构成 RAG 的工期障碍。")
    print("      注意：这是风险勘察，不是 TBC-05 决议——选型仍在 Phase 5 开工前做同口径对比后决定。")

    shutil.rmtree(os.path.dirname(db_dir), ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
