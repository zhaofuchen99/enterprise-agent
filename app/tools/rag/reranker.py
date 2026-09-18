"""重排（详细设计 11.7 的第 ⑥ 步 + 第 ⑦ 步的后半，开发流程 6.7 施工项 9）。

```text
… → ⑤ RRF 融合（≤30 条） → **⑥ 重排** → **⑦ 取 Top 8、低分剔除** → ⑧ 邻近块扩展 …
```

## 它在 RRF **之后**，不是替代它

融合只按排名（11.6.5 的固定 IDF 论证正建立在「RRF 不依赖分数绝对值」上），
重排给出的是**有量纲**的相关性分。两者各司其职：RRF 负责"两路召回的候选
怎么并成一条队"，重排负责"这条队里谁真的回答了这个问题"。

## 为什么「剔除」必须由它来做

11.7 的落地记录把逐候选的语义剔除明确判给重排器：稀疏路独有的候选
**没有**稠密分（它压根没被稠密路评估过），补 0.0 等于断言"语义完全不相关"——
那是个我们并不知道的结论（见 `RetrievedChunk.dense_score` 的可空语义）。
cross-encoder 把 (问题, 候选) **一起**过一遍模型，于是每个候选都拿到一个
可比的分数，逐条判定这才站得住。

## 关闭与失败走**同一条**代码路径

`RERANKER_ENABLED=false`（默认值）与"调用了但失败"的处置完全一样：
按 RRF 序取 Top-K。这不是巧合，而是"重排是增强步骤"这句话的实现——
它在，排序更好；它不在，系统照常工作。

但两者留下的**原因必须不同**（`DISABLED` / `UNAVAILABLE`）：
"召回质量下降了"是排查时的第一类问题，而"没开"与"打了服务但失败了"
是两条完全不同的排查路径，合成一个"没生效"就分不出来了。

## 失败时不写 `rerank_score`

那个字段的含义是"模型给的分数"，失败时它**没有值**。填一个假的进去，
下游与读 Trace 的人会把 RRF 的名次当成模型的判断——而它恰好看起来
完全正常（一个 [0,1] 的数）。

## 失败不抛给调用方，但也不静默

检索是一次任务的必经路径，而重排只是它的增强步骤——把一次重排超时
升级成任务失败，代价与收益完全不相称。所以这里吞掉 `AgentError` 并降级，
但**打一条 warning 并把原因放进返回值**：它必须能被看见。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import AgentError
from app.infrastructure.model_gateway import ModelGateway
from app.tools.rag.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

#: 重排没生效的原因（进 `RetrievalOutcome.rerank_skipped_reason`）。
#: **取值刻意是"能不能自己修"的分界**：`DISABLED` 是配置没打开，
#: `UNAVAILABLE` / `INCOMPLETE` 是服务侧的问题，处置完全不同。
SKIP_DISABLED = "DISABLED"
SKIP_UNAVAILABLE = "UNAVAILABLE"
SKIP_INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class RerankResult:
    """一次重排的完整产出（含"没生效"的情况）。

    Attributes:
        chunks: 最终候选，**始终是已排好序、已剔除、已截到 Top-K 的**。
            没生效时它就是 RRF 序的前 K 条——调用方不必分两种情况写。
        applied: 重排分是否真的生效。**它是"这次排序能不能代表语义"的唯一判据**，
            也是相关性门禁选哪条规则的开关（见 `retriever.retrieve`）。
        skipped_reason: 没生效的原因（见上面的三个常量）；生效时为 None。
        threshold: 本次生效的剔除阈值；没生效时为 None。
        pruned: 被阈值剔掉的条数。
        best_score: 实测最高重排分；没生效时为 None。**与 `threshold` 一起留下**：
            阈值调高了与语料真没有内容，症状都是"剔空了"，处置完全不同。
    """

    chunks: tuple[RetrievedChunk, ...]
    applied: bool
    skipped_reason: str | None = None
    threshold: float | None = None
    pruned: int = 0
    best_score: float | None = None


class Reranker:
    """11.7 第 ⑥⑦ 步。依赖注入与 `Retriever` 同形，便于单元测试用替身。"""

    def __init__(self, *, settings: Settings, gateway: ModelGateway) -> None:
        self._settings = settings
        self._gateway = gateway

    async def rerank(self, *, question: str, candidates: Sequence[RetrievedChunk]) -> RerankResult:
        """对候选重排并剔除。

        Args:
            question: **原问题**，不是改写后的查询。11.7 第 ⑥ 步原文是
                「Reranker 对**原问题**与候选重排」——改写是为了召回，
                而判定相关性要以用户真正问的那件事为准。
            candidates: RRF 融合后的候选（≤ `rag.max_candidates`）。
        """
        tuning = self._settings.rag
        if not self._settings.reranker_enabled:
            return _unchanged(candidates, tuning.rerank_top_k, SKIP_DISABLED)
        if not candidates:
            # 没有候选就没什么可排的。`applied=True`：这是一次**成功地**
            # 判定"没有候选"，而 `applied=False` 的含义是"这次排序没有语义"。
            return RerankResult(
                chunks=(),
                applied=True,
                threshold=tuning.rerank_score_threshold,
                best_score=None,
            )

        passages = [_passage(chunk, tuning.rerank_max_chars) for chunk in candidates]
        try:
            scores = await self._gateway.rerank(question, passages)
        except AgentError as exc:
            logger.warning(
                "重排未生效，退回 RRF 顺序：%s",
                exc.message,
                extra={"error_code": exc.code.value, "candidates": len(candidates)},
            )
            return _unchanged(
                candidates, tuning.rerank_top_k, f"{SKIP_UNAVAILABLE}:{exc.code.value}"
            )

        if len(scores) != len(candidates):
            # 真实网关会在这里之前就报缺条（`_scores_by_index`）。这条兜底是给
            # 替身与将来的实现留的：**长度对不上时按顺序硬配是最危险的做法**，
            # 它会让每个候选拿到别人的分数——排序看起来完全正常。
            logger.warning(
                "重排返回的分数条数与候选数不符：%d != %d",
                len(scores),
                len(candidates),
                extra={"candidates": len(candidates)},
            )
            return _unchanged(candidates, tuning.rerank_top_k, SKIP_INCOMPLETE)

        scored = [
            chunk.model_copy(update={"rerank_score": score})
            for chunk, score in zip(candidates, scores, strict=True)
        ]
        # 并列时按 `chunk_id` 兜底排序：只按分数排的话，并列项的顺序取决于
        # 输入顺序，同一批候选会时好时坏（与 `_fuse` 同一条理由）
        ordered = sorted(scored, key=lambda chunk: (-(chunk.rerank_score or 0.0), chunk.chunk_id))
        threshold = tuning.rerank_score_threshold
        kept = [chunk for chunk in ordered if (chunk.rerank_score or 0.0) >= threshold]
        return RerankResult(
            chunks=_renumber(kept[: tuning.rerank_top_k]),
            applied=True,
            threshold=threshold,
            pruned=len(ordered) - len(kept),
            # 最高分取的是**全部候选**里的最高，不是留下的那些的：
            # "剔空了"要看的是它离阈值有多远
            best_score=ordered[0].rerank_score if ordered else None,
        )


def _passage(chunk: RetrievedChunk, max_chars: int) -> str:
    """送进 cross-encoder 的文本。

    **只给正文，不拼标题与章节路径**：语料的分块本身带层级标题（入库时的
    清洗与还原都做过），再拼一遍是拿"我们的实现细节"去引导模型。
    如果校准结果显示标题确实有信息量（某些问法靠章节名就能定位），
    再改这里是**一次可测量的实验**，而不是先验地加进去。
    """
    return chunk.text[:max_chars]


def _unchanged(candidates: Sequence[RetrievedChunk], top_k: int, reason: str) -> RerankResult:
    """没生效时的产出：按原序（RRF 序）取 Top-K，并如实记下原因。

    **`rerank_score` 留空**，见模块 docstring：那个字段是"模型给的分数"，
    没算过就没有值。
    """
    return RerankResult(
        chunks=_renumber(candidates[:top_k]),
        applied=False,
        skipped_reason=reason,
    )


def _renumber(chunks: Sequence[RetrievedChunk]) -> tuple[RetrievedChunk, ...]:
    """重排名次从 1 起连续编。

    `rank` 的语义是"最终第几条"，所以它必须由**最后一步**赋值——
    重排之后还留着 RRF 的名次，读的人会以为那是融合序。
    """
    return tuple(chunk.model_copy(update={"rank": index}) for index, chunk in enumerate(chunks, 1))


__all__ = [
    "SKIP_DISABLED",
    "SKIP_INCOMPLETE",
    "SKIP_UNAVAILABLE",
    "RerankResult",
    "Reranker",
]
