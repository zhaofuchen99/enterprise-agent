"""重排阈值校准（开发流程 3.4.1 的「阈值必须用真实数据跑出来」）。

```bash
make calibrate-rerank            # 需要 RERANKER_ENABLED=true（前置：make ingest + Ollama）
```

## 它回答的问题只有一个

`RAG__RERANK_SCORE_THRESHOLD` 该取多少？判据是**两类样本能不能分开**：

- **有答案的 20 条**（`cases`）：最高重排分若低于阈值，这条会被整个剔空 →
  一次**误拒**，用户拿到"公司没有这条制度"；
- **语料中不存在的 3 条**（`absent_cases`）：最高重排分若高于阈值，
  就会带着一堆无关片段往下走 → 一次**编造**的机会。

所以理想情况是「有答案类的最低分 > 不存在类的最高分」，两者之间就是可用的
阈值区间。**重叠的话没有任何阈值是安全的**——那时要如实报出来，
而不是挑一个"看起来还行"的数（`RAG__SCORE_THRESHOLD` 的历史就是这样：
真问题与不存在问题的余弦重叠 0.04，只能退到"两个判据取或"）。

## 为什么单独一个脚本，而不是塞进 `make eval-rag`

`eval-rag` 回答的是「召回够不够」（Recall@8）。校准回答的是「线划在哪」，
它要的是**每条样本的分数分布**，而不是一个比率。混在一起的代价是
调阈值时得从一堆命中率数字里反推分档——而那正是拍脑袋的来源。

⚠️ **与 `eval-rag` 一样，K 取线上的 `rag.rerank_top_k`**：拿一个更宽的 K
去校准，得到的阈值解释不了线上行为。
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings, get_settings
from app.domain.user import PermissionScope, UserRole
from app.infrastructure.cache import VersionedCache
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.model_gateway import build_model_gateway
from app.infrastructure.redis import create_client
from app.infrastructure.storage import build_object_storage
from app.infrastructure.vector_store import build_vector_store
from app.repositories.vocab_repo import SqlVocabRepository
from app.tools.rag.golden import GoldenCase, load_golden
from app.tools.rag.schemas import RagQueryArgs
from app.tools.rag.tool import build_retriever


@dataclass(frozen=True, slots=True)
class Sample:
    """一条样本的分数。**`has_answer` 是分组依据**（有答案 / 语料中不存在）。"""

    case_id: str
    question: str
    has_answer: bool
    best_score: float | None
    kept: int
    doc_hit: bool


def _expected_docs(case: GoldenCase) -> tuple[str, ...]:
    return tuple(case.expect_docs)


async def run(settings: Settings) -> list[Sample]:
    """跑完全部样本（20 条有答案 + 3 条语料中不存在）。"""
    golden = load_golden()
    engine = create_engine(settings)
    redis = create_client(settings)
    storage = build_object_storage(settings)
    vector_store = build_vector_store(settings)
    gateway = build_model_gateway(settings)
    samples: list[Sample] = []
    try:
        retriever = await build_retriever(
            settings,
            gateway,
            vector_store=vector_store,
            storage=storage,
            vocab=SqlVocabRepository(create_session_factory(engine)),
            cache=VersionedCache(redis, default_ttl_seconds=settings.rag.vocab_cache_ttl_seconds),
        )
        for case in golden.cases:
            outcome = await retriever.retrieve(
                RagQueryArgs(question=case.question, as_of=case.as_of),
                # 每条用例可以指定身份：语料里有 CONFIDENTIAL 文档，
                # 用 ANALYST 去检索会被**正确地**挡在外面（同 `eval_rag.py`）
                scope=PermissionScope(role=case.role),
            )
            found = {chunk.metadata.document_id for chunk in outcome.candidates}
            samples.append(
                Sample(
                    case_id=case.id,
                    question=case.question,
                    has_answer=True,
                    best_score=outcome.best_rerank_score,
                    kept=len(outcome.candidates),
                    doc_hit=bool(found & set(_expected_docs(case))),
                )
            )
        for absent in golden.absent_cases:
            outcome = await retriever.retrieve(
                RagQueryArgs(question=absent.question),
                # 不存在类固定用 ADMIN：这些问题的主题在整个语料里都不存在，
                # 若被权限挡住而没拒答，测出来的就不是"语料里有没有"
                # （`make verify-corpus` 的拒答检查同此口径）
                scope=PermissionScope(role=UserRole.ADMIN),
            )
            samples.append(
                Sample(
                    case_id=absent.id,
                    question=absent.question,
                    has_answer=False,
                    best_score=outcome.best_rerank_score,
                    kept=len(outcome.candidates),
                    doc_hit=False,
                )
            )
        return samples
    finally:
        # 与 `eval_rag.py` 逐行对齐的释放顺序：先关共享资源，最后关连接池
        await gateway.aclose()
        await vector_store.aclose()
        await redis.aclose()
        await engine.dispose()


def _range(samples: list[Sample]) -> tuple[float | None, float | None]:
    """返回（有答案类的最低分, 不存在类的最高分）。

    前者是"阈值不能高过它"，后者是"阈值不能低过它"。前者大于后者时
    区间为空——**那就是分不开**，要如实报出来。
    """
    answered = [s.best_score for s in samples if s.has_answer and s.best_score is not None]
    absent = [s.best_score for s in samples if not s.has_answer and s.best_score is not None]
    return (min(answered) if answered else None, max(absent) if absent else None)


def _render(sample: Sample) -> str:
    score = "—" if sample.best_score is None else f"{sample.best_score:.4f}"
    mark = "✓" if sample.doc_hit else ("·" if sample.has_answer else "拒答")
    group = "有答案" if sample.has_answer else "不存在"
    return (
        f"  {sample.case_id}  {score}  保留 {sample.kept:>2} 条"
        f"  [{group}] {mark}  {sample.question}"
    )


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    if not settings.reranker_enabled:
        print("重排未启用（RERANKER_ENABLED=false），没有可校准的分数。")
        print("先在 .env 里打开重排并填好 RERANKER_MODEL / BASE_URL / API_KEY。")
        return 1

    samples = asyncio.run(run(settings))
    print(f"重排模型：{settings.reranker_model}")
    print(f"当前阈值：{settings.rag.rerank_score_threshold}（逐候选剔除）")
    print(f"评测 K：{settings.rag.rerank_top_k}（与线上一致）")
    print()
    for sample in samples:
        print(_render(sample))

    low, high = _range(samples)
    print()
    print("─" * 72)
    if low is None or high is None:
        print("样本不足或全部没有分数，无法给出区间。")
        return 1
    print(f"有答案类最低分 {low:.4f} ｜ 不存在类最高分 {high:.4f}")
    if low > high:
        midpoint = (low + high) / 2
        print(f"✅ 两类可分离：阈值取 ({high:.4f}, {low:.4f}] 之间，建议 {midpoint:.4f}")
        return 0
    print(
        "❌ 两类**重叠**：没有任何阈值能同时做到「不误拒真问题」与「拦住不存在的问题」。\n"
        "   此时不要硬挑一个数——要么保留现有的两判据（余弦地板 + 未登录词）作为否决，\n"
        "   要么承认重排分只够用来排序、判不了「有没有这件事」。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
