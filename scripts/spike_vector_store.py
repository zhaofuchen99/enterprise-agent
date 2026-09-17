"""TBC-05 决议实测：Milvus Lite / Qdrant / Chroma 同口径对比。

背景见 CLAUDE.md「本机环境事实」与详细设计 11.5 / 11.7：本机 7.6GB 内存跑不动
Milvus Standalone（需 8GB 起），而设计要求的 **dense + sparse 双路召回 + RRF 融合**
不是所有嵌入式向量库都提供。本脚本**取代**了早期的 `scripts/spike_milvus_lite.py`
（那份只验了 Milvus Lite 一家，结论已被这里的同口径对比覆盖，脚本本身已删除）——
选型不能只验一个候选，"能跑"和"比别的更合适"是两个问题。

它做的事：三个候选跑同一份语料、同一组操作，把能力与资源开销放在一张表里比。
**并发能力也在其中**：最初两版脚本都漏了这条，而它恰恰是最终改判的决定性因素。

## 口径（三个候选完全一致，否则对比无意义）

| 项 | 值 | 说明 |
|---|---|---|
| 分块数 | 2000 | 80+ 文档量级的上限估计（详设 11.5 的规模） |
| dense 维度 | 1024 | `bge-m3` 的真实维度，不是随便取的小数 |
| 稀疏向量 | `{token_id: weight}` | 与 11.6 的产出形状一致 |
| 标量过滤 | `document_type == POLICY && status == ACTIVE` | 11.4 的 metadata 过滤 |
| 混合检索 | dense top20 + sparse top20 → RRF → 20 | 11.7 的在线检索路径 |

## 为什么用子进程分别跑

峰值 RSS 是**进程级**指标。三个库放进同一个进程里，谁先跑谁把 numpy / grpc
之类的运行时拉起来，后跑的就白捡一个已经预热的内存基线，量出来的差异是假的。
所以每个候选各起一个子进程，父进程只做汇总。

## 复现

四个候选项里有三个**不是项目依赖**（Milvus、Chroma 已被否决），因此要用独立 venv 跑：

    uv venv /tmp/tbc05-venv --python 3.12
    uv pip install --python /tmp/tbc05-venv/bin/python \
        pymilvus milvus-lite qdrant-client chromadb
    make up          # 起 Qdrant 服务端，qdrant_server 那一行需要它
    /tmp/tbc05-venv/bin/python scripts/spike_vector_store.py

`make vector-spike` 是同一件事的快捷方式。服务端常驻内存不在这张表里，
用 `docker stats --no-stream ea-qdrant` 读——那是容器级指标，与客户端进程 RSS 不是一回事。

**结论写回 `docs/秋招冲刺方案.md` §4.3 与详细设计 23.1.1。**
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

DIM = 1024
N_CHUNKS = 2000
TOP_K = 20

#: 语料用真实业务词条拼装，避免「全是一模一样的字符串」让稀疏索引的
#: df 统计退化成常量——那样量出来的检索耗时不代表真实分布。
_DEPTS = ["销售部", "市场部", "财务部", "供应链部"]
_TYPES = ["POLICY", "REPORT", "PRODUCT", "METRIC", "OTHER"]
_STATUS = ["ACTIVE", "ARCHIVED", "DRAFT"]
_PHRASES = [
    "华东区域渠道折扣上限为三成，超出部分需大区总监审批",
    "净销售额口径为含税收入扣除退货与折让后的金额",
    "经销商返利按季度结算，达成率低于八成不予发放",
    "2025 年 Q3 华南大区毛利率同比提升两个百分点",
    "电商渠道佣金比例按品类分档，最高不超过百分之十五",
]


def build_corpus() -> list[dict[str, Any]]:
    """构造确定性语料。**同一份语料必须能被三个候选逐字节复用**。

    全部由下标推导、不掷骰子：三个候选拿到的必须是逐字节相同的输入，
    否则量出来的延迟差异里混着数据差异，对比就没意义了。
    """
    rows = []
    for i in range(N_CHUNKS):
        # dense 向量按文档分组给一点结构，纯随机向量会让近邻检索失去意义
        group = i % 50
        dense = [((group * 7 + j) % 97) / 97.0 for j in range(DIM)]
        rows.append(
            {
                "chunk_id": f"chk_{i:06d}",
                "document_id": f"doc_{group:04d}",
                "text": f"销售政策 > 渠道折扣 > {_PHRASES[i % len(_PHRASES)]}（第 {i} 条）",
                "dense": dense,
                # token_id 空间取 1000 起，模拟 11.6.3 的「只增不改」词表
                "sparse": {1000 + (i % 300): 0.9, 1000 + (i % 17): 0.4, 1300 + group: 0.7},
                "document_type": _TYPES[i % len(_TYPES)],
                "status": _STATUS[i % len(_STATUS)],
                "department": _DEPTS[i % len(_DEPTS)],
            }
        )
    return rows


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            # 并发写盘时文件可能刚好被换掉，stat 失败不应让整个测量中断
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(dirpath, name))
    return total / 1024 / 1024


def timed(fn: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


# --------------------------------------------------------------------------- Milvus Lite


def run_milvus_lite(workdir: str) -> dict[str, Any]:
    from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

    rows = build_corpus()
    db_file = os.path.join(workdir, "kb.db")
    client = MilvusClient(db_file)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    client.create_collection("kb_chunks", schema=schema)

    payload = [
        {
            "chunk_id": r["chunk_id"],
            "text": r["text"],
            "dense_vector": r["dense"],
            "sparse_vector": r["sparse"],
            "document_type": r["document_type"],
            "status": r["status"],
            "department": r["department"],
        }
        for r in rows
    ]

    def _insert() -> None:
        client.insert("kb_chunks", payload)
        client.flush("kb_chunks")

    _, insert_ms = timed(_insert)

    params = client.prepare_index_params()
    params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="COSINE")
    params.add_index(
        field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP"
    )
    client.create_index("kb_chunks", index_params=params)
    client.load_collection("kb_chunks")

    query_dense = rows[3]["dense"]
    query_sparse = rows[3]["sparse"]

    _, dense_ms = timed(
        lambda: client.search("kb_chunks", data=[query_dense], limit=TOP_K, output_fields=["text"])
    )
    _, filtered_ms = timed(
        lambda: client.search(
            "kb_chunks",
            data=[query_dense],
            limit=TOP_K,
            filter='document_type == "POLICY" && status == "ACTIVE"',
        )
    )
    _, sparse_ms = timed(
        lambda: client.search(
            "kb_chunks", data=[query_sparse], anns_field="sparse_vector", limit=TOP_K
        )
    )

    def _hybrid() -> Any:
        reqs = [
            AnnSearchRequest(data=[query_dense], anns_field="dense_vector", param={}, limit=TOP_K),
            AnnSearchRequest(
                data=[query_sparse], anns_field="sparse_vector", param={}, limit=TOP_K
            ),
        ]
        return client.hybrid_search(
            "kb_chunks", reqs, ranker=RRFRanker(60), limit=TOP_K, output_fields=["text"]
        )

    hits, hybrid_ms = timed(_hybrid)

    return {
        "dense_search": True,
        "sparse_search": True,
        "hybrid_rrf": len(hits[0]) > 0,
        "scalar_filter": True,
        "insert_ms": insert_ms,
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "filtered_ms": filtered_ms,
        "hybrid_ms": hybrid_ms,
        "hits": len(hits[0]),
        "disk_mb": dir_size_mb(workdir),
        "note": "嵌入式文件库，pymilvus 同一客户端；换 Milvus 服务端只改 URI",
    }


# --------------------------------------------------------------------------- Qdrant


def run_qdrant(workdir: str) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models

    rows = build_corpus()
    client = QdrantClient(path=os.path.join(workdir, "qdrant"))

    client.create_collection(
        "kb_chunks",
        vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    points = [
        models.PointStruct(
            id=i,
            vector={
                "dense": r["dense"],
                "sparse": models.SparseVector(
                    indices=list(r["sparse"].keys()), values=list(r["sparse"].values())
                ),
            },
            payload={
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "document_type": r["document_type"],
                "status": r["status"],
                "department": r["department"],
            },
        )
        for i, r in enumerate(rows)
    ]

    def _insert() -> None:
        # 分批：单次 upsert 2000 条在本地模式下会撑大峰值内存
        for start in range(0, len(points), 500):
            client.upsert("kb_chunks", points=points[start : start + 500])

    _, insert_ms = timed(_insert)

    query_dense = rows[3]["dense"]
    query_sparse = models.SparseVector(
        indices=list(rows[3]["sparse"].keys()), values=list(rows[3]["sparse"].values())
    )
    policy_filter = models.Filter(
        must=[
            models.FieldCondition(key="document_type", match=models.MatchValue(value="POLICY")),
            models.FieldCondition(key="status", match=models.MatchValue(value="ACTIVE")),
        ]
    )

    dense_hits, dense_ms = timed(
        lambda: client.query_points(
            "kb_chunks", query=query_dense, using="dense", limit=TOP_K, with_payload=True
        )
    )
    _, filtered_ms = timed(
        lambda: client.query_points(
            "kb_chunks",
            query=query_dense,
            using="dense",
            limit=TOP_K,
            query_filter=policy_filter,
        )
    )
    _, sparse_ms = timed(
        lambda: client.query_points("kb_chunks", query=query_sparse, using="sparse", limit=TOP_K)
    )

    def _hybrid() -> Any:
        return client.query_points(
            "kb_chunks",
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=TOP_K),
                models.Prefetch(query=query_sparse, using="sparse", limit=TOP_K),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=TOP_K,
            with_payload=True,
        )

    hybrid_hits, hybrid_ms = timed(_hybrid)

    return {
        "dense_search": True,
        "sparse_search": True,
        "hybrid_rrf": len(hybrid_hits.points) > 0,
        "scalar_filter": len(dense_hits.points) > 0,
        "insert_ms": insert_ms,
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "filtered_ms": filtered_ms,
        "hybrid_ms": hybrid_ms,
        "hits": len(hybrid_hits.points),
        "disk_mb": dir_size_mb(workdir),
        "note": "本地模式与嵌入式相同；服务端模式需额外容器（+300MB 量级）",
    }


# --------------------------------------------------------------------------- Chroma


def run_chroma(workdir: str) -> dict[str, Any]:
    import chromadb

    rows = build_corpus()
    client = chromadb.PersistentClient(path=os.path.join(workdir, "chroma"))
    collection = client.create_collection("kb_chunks", metadata={"hnsw:space": "cosine"})

    def _insert() -> None:
        for start in range(0, len(rows), 500):
            batch = rows[start : start + 500]
            collection.add(
                ids=[r["chunk_id"] for r in batch],
                embeddings=[r["dense"] for r in batch],
                documents=[r["text"] for r in batch],
                metadatas=[
                    {
                        "document_type": r["document_type"],
                        "status": r["status"],
                        "department": r["department"],
                    }
                    for r in batch
                ],
            )

    _, insert_ms = timed(_insert)

    query_dense = rows[3]["dense"]
    _, dense_ms = timed(lambda: collection.query(query_embeddings=[query_dense], n_results=TOP_K))
    hits, filtered_ms = timed(
        lambda: collection.query(
            query_embeddings=[query_dense],
            n_results=TOP_K,
            where={"$and": [{"document_type": "POLICY"}, {"status": "ACTIVE"}]},
        )
    )

    # Chroma 的公开 API 只有稠密向量：没有 sparse_vector 字段，也没有
    # 稀疏检索或融合排序查询。这里如实探一次，避免"没看到文档"就下结论。
    sparse_supported = hasattr(collection, "query_sparse")

    return {
        "dense_search": True,
        "sparse_search": sparse_supported,
        "hybrid_rrf": False,
        "scalar_filter": len(hits["ids"][0]) > 0,
        "insert_ms": insert_ms,
        "dense_ms": dense_ms,
        "sparse_ms": None,
        "filtered_ms": filtered_ms,
        "hybrid_ms": None,
        "hits": len(hits["ids"][0]),
        "disk_mb": dir_size_mb(workdir),
        "note": "公开 API 无稀疏向量与融合排序，需自建 BM25 + 自行融合",
    }


def run_qdrant_server(workdir: str) -> dict[str, Any]:
    """Qdrant **服务端**——这正是 TBC-05 最终选中的形态。

    它必须出现在这张表里，否则"选中的方案"反而是唯一没有实测数据支撑的行。
    地址取 `QDRANT_URL`（默认 `http://127.0.0.1:6333`），需要先 `make up`。

    它不落盘到 `workdir`：数据在容器卷里，因此 `disk_mb` 记 0 并在注里说明。
    内存也**不在这里测**——那是容器级指标，用 `docker stats --no-stream ea-qdrant` 读，
    比在客户端进程里估准得多。
    """
    from qdrant_client import QdrantClient, models

    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    rows = build_corpus()
    client = QdrantClient(url=url)
    if client.collection_exists("kb_chunks_spike"):
        client.delete_collection("kb_chunks_spike")

    client.create_collection(
        "kb_chunks_spike",
        vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    points = [
        models.PointStruct(
            id=i,
            vector={
                "dense": r["dense"],
                "sparse": models.SparseVector(
                    indices=list(r["sparse"].keys()), values=list(r["sparse"].values())
                ),
            },
            payload={
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "document_type": r["document_type"],
                "status": r["status"],
                "department": r["department"],
            },
        )
        for i, r in enumerate(rows)
    ]

    def _insert() -> None:
        for start in range(0, len(points), 500):
            client.upsert("kb_chunks_spike", points=points[start : start + 500])

    _, insert_ms = timed(_insert)

    query_dense = rows[3]["dense"]
    query_sparse = models.SparseVector(
        indices=list(rows[3]["sparse"].keys()), values=list(rows[3]["sparse"].values())
    )
    policy_filter = models.Filter(
        must=[
            models.FieldCondition(key="document_type", match=models.MatchValue(value="POLICY")),
            models.FieldCondition(key="status", match=models.MatchValue(value="ACTIVE")),
        ]
    )

    _, dense_ms = timed(
        lambda: client.query_points(
            "kb_chunks_spike", query=query_dense, using="dense", limit=TOP_K
        )
    )
    _, filtered_ms = timed(
        lambda: client.query_points(
            "kb_chunks_spike",
            query=query_dense,
            using="dense",
            limit=TOP_K,
            query_filter=policy_filter,
        )
    )
    _, sparse_ms = timed(
        lambda: client.query_points(
            "kb_chunks_spike", query=query_sparse, using="sparse", limit=TOP_K
        )
    )

    def _hybrid() -> Any:
        return client.query_points(
            "kb_chunks_spike",
            prefetch=[
                models.Prefetch(query=query_dense, using="dense", limit=TOP_K),
                models.Prefetch(query=query_sparse, using="sparse", limit=TOP_K),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=TOP_K,
        )

    hits, hybrid_ms = timed(_hybrid)
    client.delete_collection("kb_chunks_spike")

    return {
        "dense_search": True,
        "sparse_search": True,
        "hybrid_rrf": len(hits.points) > 0,
        "scalar_filter": True,
        "insert_ms": insert_ms,
        "dense_ms": dense_ms,
        "sparse_ms": sparse_ms,
        "filtered_ms": filtered_ms,
        "hybrid_ms": hybrid_ms,
        "hits": len(hits.points),
        # 数据在容器卷里，不在 workdir；内存用 `docker stats ea-qdrant` 读
        "disk_mb": 0.0,
        "note": f"{url} 服务端形态；内存读 `docker stats --no-stream ea-qdrant`，非本进程 RSS",
    }


CANDIDATES = {
    "milvus_lite": run_milvus_lite,
    "qdrant": run_qdrant,
    "qdrant_server": run_qdrant_server,
    "chroma": run_chroma,
}

#: 服务端形态不参与并发探针：它本来就是为解决并发而存在的（Valkey/Qdrant 这类
#: 独立进程天然支持多客户端），对它做"第二个进程能否打开"没有意义——
#: 客户端连的是同一个服务端，不存在"打开目录"这件事。
_NO_CONCURRENCY_PROBE = frozenset({"qdrant_server"})


def _hold_open(name: str, workdir: str, seconds: float) -> int:
    """把库开在一个**活着的引用**上并保持住，供父进程探测并发。

    这里必须让 client 一直有引用。第一版探针直接复用基准测试的开库代码，
    而那种写法把 client 留在被调函数的局部作用域里——函数一返回，client 被 GC，
    **锁跟着释放**，第二个进程于是能开进来，探针得到"支持并发"的**假阳性**。
    当时 Qdrant 就被误判成了支持并发，和手工严格测试的结论相反。

    教训：测"资源被占住"时，占住资源的那个对象不能在测到之前先死掉。
    """
    client: Any
    if name == "milvus_lite":
        from pymilvus import DataType, MilvusClient

        client = MilvusClient(os.path.join(workdir, "kb.db"))
        if not client.has_collection("kb"):
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=8)
            client.create_collection("kb", schema=schema)
    elif name == "qdrant":
        from qdrant_client import QdrantClient, models

        client = QdrantClient(path=os.path.join(workdir, "qdrant"))
        if not client.collection_exists("kb"):
            client.create_collection(
                "kb",
                vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
            )
    elif name == "chroma":
        import chromadb

        client = chromadb.PersistentClient(path=os.path.join(workdir, "chroma"))
        # collection 名要求 3–512 字符，`kb` 会被拒（Chromadb InvalidArgumentError）
        client.get_or_create_collection("kb_chunks")

    print("__HOLDING__" + json.dumps({"workdir": workdir}, ensure_ascii=False), flush=True)
    time.sleep(seconds)
    del client  # 显式释放，表明"锁是被这个引用持有的"
    return 0


def _child(name: str, hold_seconds: float = 0.0) -> int:
    """子进程入口：跑一个候选，或只把库占住（`hold_seconds > 0`）。

    **清理不放在 `finally` 里**：`finally` 中出现 `return` 会吞掉正在传播的异常
    （ruff B012），而占住模式需要提前返回——两者放一起，会让"候选跑失败"
    变成一个静默的 0 退出码。
    """
    workdir = tempfile.mkdtemp(prefix=f"tbc05_{name}_")

    if hold_seconds:
        # 占住模式：**不跑基准**，直接开库并保持住。
        # 自己不清理 workdir：父进程测完才删，否则锁随目录一起消失，测的是空气。
        try:
            return _hold_open(name, workdir, hold_seconds)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    result: dict[str, Any] = {"candidate": name}
    try:
        result.update(CANDIDATES[name](workdir))
        result["ok"] = True
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["peak_rss_mb"] = round(peak_rss_mb(), 1)

    shutil.rmtree(workdir, ignore_errors=True)
    print("__RESULT__" + json.dumps(result, ensure_ascii=False))
    return 0


def probe_concurrency(name: str, hold_seconds: float = 12.0) -> tuple[bool, str]:
    """先开一个进程占住库，再在**另一个进程**里尝试打开同一目录。

    这一项是 TBC-05 最终改判的决定性依据，却也是最容易漏测的一项：三个候选的
    单进程基准都漂漂亮亮，只有把第二个进程拉进来才会暴露锁。

    返回 `(是否支持多进程, 说明)`。
    """
    holder = subprocess.Popen(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--candidate",
            name,
            "--hold",
            str(hold_seconds),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    workdir = ""
    try:
        assert holder.stdout is not None
        for line in holder.stdout:
            if line.startswith("__HOLDING__"):
                workdir = json.loads(line[len("__HOLDING__") :])["workdir"]
                break
        if not workdir:
            return False, "占用进程未能启动"

        # 第二个进程：拿着同一个目录去开库
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--probe-open", name, workdir],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and "__OPENED__" in proc.stdout:
            return True, "两个进程可同时打开"
        reason = (proc.stdout + proc.stderr).strip().splitlines()
        return False, reason[-1][:90] if reason else "未知原因"
    finally:
        holder.terminate()
        holder.wait(timeout=30)
        shutil.rmtree(workdir, ignore_errors=True)


def _probe_open(name: str, workdir: str) -> int:
    """第二个进程：只尝试打开已存在的库，不做写入。

    **只打开、不建 collection**：这里的目的是复现"第二个进程能不能进来"，
    顺带建表会把"打开失败"和"建表失败"两种原因混在一起。
    """
    try:
        if name == "milvus_lite":
            from pymilvus import MilvusClient

            MilvusClient(os.path.join(workdir, "kb.db"))
        elif name == "qdrant":
            from qdrant_client import QdrantClient

            QdrantClient(path=os.path.join(workdir, "qdrant"))
        elif name == "chroma":
            import chromadb

            chromadb.PersistentClient(path=os.path.join(workdir, "chroma"))
        else:
            # 不认识的候选直接失败。**不能静默打印 __OPENED__**：
            # 那会让"没测"看起来像"测过了且通过"，是最坏的一种假阳性。
            print(f"未知候选：{name}")
            return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
    print("__OPENED__")
    return 0


def _fmt(value: Any, unit: str = "", digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def main() -> int:
    parser = argparse.ArgumentParser(description="TBC-05 向量库选型同口径对比")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), help=argparse.SUPPRESS)
    parser.add_argument("--hold", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--probe-open", nargs=2, metavar=("NAME", "DIR"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.probe_open:
        return _probe_open(args.probe_open[0], args.probe_open[1])
    if args.candidate:
        return _child(args.candidate, args.hold)

    results: dict[str, dict[str, Any]] = {}
    for name in CANDIDATES:
        print(f"→ 跑 {name} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--candidate", name],
            capture_output=True,
            text=True,
            check=False,
        )
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("__RESULT__")), None)
        if line is None:
            results[name] = {"ok": False, "error": (proc.stderr or "无输出").strip()[-300:]}
            continue
        results[name] = json.loads(line[len("__RESULT__") :])

    # 并发探针单独跑：它的结论不是从上面那次基准里读出来的，而是要再起两个进程
    concurrency: dict[str, tuple[bool, str]] = {}
    for name in CANDIDATES:
        if name in _NO_CONCURRENCY_PROBE:
            concurrency[name] = (True, "服务端形态，天然支持多客户端")
            continue
        print(f"→ 并发探针 {name} ...", flush=True)
        concurrency[name] = probe_concurrency(name)

    rows: list[tuple[str, str, Any]] = [
        ("稀疏向量（11.5 硬要求）", "sparse_search", "bool"),
        ("RRF 混合检索（11.7）", "hybrid_rrf", "bool"),
        ("标量过滤（11.4）", "scalar_filter", "bool"),
        ("峰值 RSS", "peak_rss_mb", "mb"),
        ("落盘占用", "disk_mb", "mb"),
        ("入库 2000 条", "insert_ms", "ms"),
        ("dense top20", "dense_ms", "ms"),
        ("sparse top20", "sparse_ms", "ms"),
        ("过滤 + dense top20", "filtered_ms", "ms"),
        ("混合 RRF top20", "hybrid_ms", "ms"),
    ]

    names = list(CANDIDATES)
    label_w = 26
    cell_w = 15
    line_w = label_w + cell_w * len(names)
    rule = "-" * line_w

    print()
    print("=" * line_w)
    print("TBC-05 向量库选型 · 同口径对比（2000 chunk / 1024 维 / dense+sparse）")
    print("=" * line_w)
    print("指标".ljust(label_w) + "".join(n.ljust(cell_w) for n in names))
    print(rule)
    for label, key, kind in rows:
        cells = []
        for name in names:
            r = results.get(name, {})
            if not r.get("ok"):
                cells.append("失败".ljust(cell_w))
                continue
            value = r.get(key)
            unit = {"mb": " MB", "ms": " ms"}.get(kind, "")
            cells.append(_fmt(value, unit, 1).ljust(cell_w))
        print(label.ljust(label_w) + "".join(cells))
    # 并发单独一行：它的取值不来自基准结果，而来自上面那轮双进程探测
    cells = []
    for name in names:
        ok, _reason = concurrency.get(name, (False, "未测"))
        cells.append(("✅" if ok else "❌ 单进程独占").ljust(cell_w))
    print("多进程并发".ljust(label_w) + "".join(cells))
    print(rule)
    for name in names:
        ok, reason = concurrency.get(name, (False, "未测"))
        if not ok:
            print(f"{name} 并发失败原因: {reason}")
    print(rule)
    for name in names:
        r = results.get(name, {})
        detail = r.get("note") or r.get("error") or ""
        print(f"{name}: {detail}")
    print("=" * line_w)
    print("提示：延迟为单次测量，受机器负载影响，只能定性比较；")
    print("      内存与并发结论可复现。服务端形态的内存请读 docker stats，不是本进程 RSS。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
