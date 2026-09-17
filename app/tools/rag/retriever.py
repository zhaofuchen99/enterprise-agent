"""在线检索（详细设计 11.7 的九步）。

```text
① Intent 提取过滤条件        → RagQueryArgs 的 document_types / departments / as_of
② Query Rewrite 1–3 条短查询  → QUERY_REWRITE_PROMPT（失败降级为原问题）
③ 权限 / ACTIVE / 有效期标量过滤 → ChunkFilter
④ Dense TopK=20 + Sparse TopK=20
⑤ RRF 合并，候选 ≤ 30
⑥ 重排                        → **按 TBC-04 后置**
⑦ 取 Top 8，低于阈值的候选剔除
⑧ 邻近块扩展                  → **随重排器后置**，见下
⑨ 输出证据                    → `tools/rag/evidence.py`
```

## 第 ⑥⑧ 步为什么不在这一版里（这是时序调整，不是砍需求）

**第 ⑥ 步（重排）** 的后置依据是详设 TBC-04 的决议记录，不是冲刺期的裁剪——
它需要另选一个 cross-encoder 模型，而本机 7.6GB 内存下再跑一个模型
会挤压已经跑着的 `bge-m3`（见 CLAUDE.md 的「本机环境事实」）。
11.7 第 5 步到第 7 步之间因此是直接的：RRF 的 Top 30 → 取 Top 8。

**第 ⑧ 步（邻近块扩展）** 的条件原文是「同文档、同章节且**确有上下文缺口**时」。
缺口判定要的正是重排器给出的相关性信号——没有它，只能退化成
「块长小于 `chunk_min_chars` 就扩」，而这条在本语料上**几乎恒真**
（正文块中位仅 85 字，见 CLAUDE.md 临时约定 15），等于无条件把候选翻倍，
在 Recall@8 上引入的全是噪声。所以它跟着重排器一起后置，理由与它同源。

## 第 ⑦ 步的阈值挂在哪一路分数上（这一条是本实现自己决定的）

11.7 说「低于**校准阈值**的候选剔除」，而 `RAG__SCORE_THRESHOLD` 的默认值是 0.35。
**把 0.35 挂在 RRF 融合分上会把结果全部剔光**：RRF 分是 `Σ 1/(k+rank)`，
k=60 时整个值域只有约 0.016–0.033。11.6.5 也正是靠「RRF 只依赖排名、
不依赖分数绝对值」来论证固定 IDF 方案成立的——在一个被刻意做成无量纲的
分数上挂绝对阈值，与那条论证直接冲突。

系统里唯一**量纲可比**的分数是稠密路的余弦相似度（bge-m3 + COSINE，落在 [0,1]）。
因此这里的做法是：

- **融合仍然只按排名**（RRF，量纲不变）；
- **相关性判定整体做一次**：所有查询的稠密结果里最高的那个余弦就是
  「这一问在语料里到底有没有东西」的度量，低于阈值即
  `NO_RELEVANT_KNOWLEDGE`（11.8 第 1 条 / 详设 9.4 的 `EMPTY_RESULT`）。

**为什么是"整体判定"而不是"逐候选剔除"**：逐候选需要每个候选的语义分，
而稀疏路独有的候选**没有**稠密分（它压根没被稠密路评估过）。给它们补 0.0
等于断言"语义完全不相关"，那是个我们并不知道的结论。逐候选的语义过滤
是重排器的职责——它给每个候选一个可比的分数。这一版把阈值用在它唯一
站得住的位置上，并在 `RetrievalOutcome` 里把**阈值与实测最好余弦都留下**：
「阈值调错了」与「语料真没有」症状相同，处置完全不同。
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from app.core.config import Settings
from app.core.errors import AgentError, ErrorCode
from app.domain.knowledge import DocumentStatus
from app.domain.user import PermissionScope
from app.infrastructure.model_gateway import ModelGateway
from app.infrastructure.vector_store import ChunkFilter, ScoredPoint, VectorStore
from app.tools.rag.metadata import ChunkMetadata
from app.tools.rag.prompts import QUERY_REWRITE_PROMPT
from app.tools.rag.schemas import QueryRewrite, RagQueryArgs, RetrievalOutcome, RetrievedChunk
from app.tools.rag.tokenizer import (
    Tokenizer,
    Vocabulary,
    build_sparse,
    is_general_word,
    normalize_numbers,
)

#: 从文本里取"数字锚点"。**这是 11.7 第 2 步「必须保留实体和时间」里
#: 唯一能靠代码判定的那一半**：年份、季度、金额都是数字，
#: 而专有名词（区域名、产品线）需要一份实体表才能判定，见 `_keeps_numbers`。
_NUMBER = re.compile(r"\d+")

#: 问题里的**非主题词**：疑问词、语气词、泛指动词。用于 `_unseen_topics`。
#:
#: **为什么需要这份表**：语料是**陈述性**文本（制度、报告、口径说明），
#! 它从不用疑问句式写句子——于是「怎么」「哪些」「是什么」这些词在词表里
#! 一律未登录，而它们不携带任何主题信息。不排除它们的话，**任何问句**
#! 都会被判成"语料从没见过这个问题里的词"，`NO_RELEVANT_KNOWLEDGE` 会
#! 把真问题一起挡掉。
#:
#: **它不是第二份停用词表，也不参与分词**：`configs/rag_stopwords.txt`
#: 管的是"哪些词不进稀疏向量"，改动它会改变已入库的 1871 个 chunk 的
#: 稀疏向量与词表 IDF（CLAUDE.md 临时约定 13：改切分必须重跑入库与回归集）。
#: 这份表只在这一个判定里用，把两件事分开，改它不需要重建索引。
#: 把疑问词并入停用词表是更整齐的做法，但那要连带重跑 `make vocab` + `make ingest`，
#: 已登记为【后续扩展】。
#:
#: ⚠️ **这是一份闭类词的枚举，漏一个的代价是"把好问题判成没有"**。
#: 实测踩过：第一版只列了「怎么/怎样/如何」，漏了**「怎么样」**，
#: 于是「2025 年第三季度的整体经营业绩怎么样」这条余弦 0.80 的**高度相关**
#: 问题被判成 `NO_RELEVANT_KNOWLEDGE`。因此
#: `app/tests/tools/rag/test_retriever.py` 里有一份**问句形式的清单**逐条钉住它，
#: 补词时可以照着那个清单扩。
#:
#: 方向是**宁可多列**：多列一个虚词只会让判据少触发一次（还有余弦那一路兜底），
#: 少列一个则会直接拒答一条真问题。
_NON_TOPICAL: frozenset[str] = frozenset(
    {
        # 疑问代词与疑问副词（闭类）
        "怎么",
        "怎么样",
        "怎么办",
        "怎样",
        "怎的",
        "如何",
        "哪些",
        "哪个",
        "哪一些",
        "哪里",
        "什么",
        "什么样",
        "为什么",
        "为何",
        "是否",
        "多少",
        "几个",
        "多久",
        "多长时间",
        "何时",
        "什么时候",
        "谁",
        # 语气词与结构助词（单字在 `Tokenizer._keep` 里已被滤掉，这里收双字的）
        "吗",
        "呢",
        "的",
        "了",
        "是",
        "有",
        "的话",
        # 泛指动词：任何问句里都可能出现，不携带主题
        "请问",
        "介绍",
        "说明",
        "解释",
        "告诉",
        "讲讲",
        "说说",
        "列举",
    }
)


class Retriever:
    """混合检索（11.7 第 2–7 步）。

    依赖全部由构造函数注入，因此单元测试可以用 `FakeModelGateway` +
    `InMemoryVectorStore` 把整条链路跑完，不必连 Qdrant 也不必打网络。
    装配见 `tool.build_rag_retrieve_tool`。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        gateway: ModelGateway,
        vector_store: VectorStore,
        tokenizer: Tokenizer,
        vocabulary: Vocabulary,
    ) -> None:
        self._settings = settings
        self._gateway = gateway
        self._store = vector_store
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary

    async def retrieve(self, args: RagQueryArgs, *, scope: PermissionScope) -> RetrievalOutcome:
        """跑完 11.7 的第 2–7 步。

        Args:
            args: 入参见 `RagQueryArgs`。
            scope: **唯一**的数据权限来源（同 SQL Tool 的理由：代码层必须只有一个
                入口，否则"哪儿算错了"会变成一个查不清的问题）。
                文档权限走 `roles` 与 payload 的 `allowed_roles` 求交集，
                与 SQL 的 `region_ids` 是两套东西——见 `ChunkFilter.roles`。
        """
        started = time.monotonic()
        tuning = self._settings.rag

        queries, degraded = await self._rewrite(args)
        chunk_filter = ChunkFilter(
            # **显式写出来**：`ChunkFilter.status` 的默认值就是 ACTIVE，
            # 但 11.9 那条纪律值得在这条最关键的读路径上被看见——
            # 「失败版本不得被在线查询命中」的兑现处就在这里。
            status=DocumentStatus.ACTIVE.value,
            document_types=args.document_types,
            departments=args.departments,
            effective_at=args.as_of,
            roles=(scope.role.value,),
        )

        rank_lists, dense_scores = await self._search(queries, chunk_filter=chunk_filter)
        best_dense = max(dense_scores.values(), default=0.0)
        fused = _fuse(rank_lists, rrf_k=tuning.rrf_k)[: tuning.max_candidates]
        # 覆盖检查看**原问题**而不是改写结果：改写可能把「碳积分」换成更通用的
        # 说法（那正是它的职责），而"语料从没见过这个词"这件事不该因此被掩盖——
        # 用户问的就是那个词。
        unseen = self._unseen_topics(args.question)

        outcome = RetrievalOutcome(
            queries=queries,
            rewrite_degraded=degraded,
            candidate_count=len(fused),
            candidates=(),
            relevance_threshold=tuning.score_threshold,
            best_dense_score=best_dense,
            unseen_topics=unseen,
            no_relevant_knowledge=best_dense < tuning.score_threshold or bool(unseen),
            duration_ms=_elapsed_ms(started),
        )
        if outcome.no_relevant_knowledge:
            # **不返回任何候选**，而不是"返回候选但打个标记"：11.8 要求
            # 「检索为空时显式返回 NO_RELEVANT_KNOWLEDGE」，而只要候选还在，
            # 下游就有机会把它当成证据用——"不让生成节点补写制度"这条纪律
            # 靠的是**没有东西可写**，不是靠调用方自觉。
            return outcome
        return outcome.model_copy(
            update={"candidates": _to_chunks(fused, dense_scores, tuning.rerank_top_k)}
        )

    # ------------------------------------------------------------------ ② 改写
    async def _rewrite(self, args: RagQueryArgs) -> tuple[tuple[str, ...], bool]:
        """Query Rewrite（11.7 第 2 步），返回（实际使用的查询, 是否降级）。

        **原问题永远参与检索**：改写的作用是补充召回，不是替换。改写模型把
        时间或区域写歪时，只拿改写结果去查会让整次检索问错问题，
        而原问题是无损的——多一路召回的成本只是每路多 20 条候选，
        融合之后它们会自然地排在后面。

        **失败一律降级成"只有原问题"**，不抛给调用方：检索是不可用的外部依赖
        （模型网关）的下游，而"改不动"不等于"查不了"。这是开发流程对基础设施
        Phase 要求的降级路径（DoD：「外部依赖不可用时行为已定义且有对应用例」）。
        """
        try:
            result = await self._gateway.invoke_structured(
                QUERY_REWRITE_PROMPT,
                QueryRewrite,
                question=args.question,
                objective=args.objective or args.question,
            )
        except AgentError:
            return (args.question,), True

        kept = [
            query
            for query in (q.strip() for q in result.value.queries)
            if query and _keeps_numbers(args.question, query)
        ][: self._settings.rag.max_rewrites]
        if not kept:
            # 全部改写都丢了数字锚点 → 整批不用。**整批丢而不是逐条丢**：
            # 留一条"改得还行"的和用原问题在信息上没有区别，
            # 而混进去会让"这次检索到底问了什么"多一层解释成本。
            return (args.question,), True
        return _dedupe((args.question, *kept)), False

    def _unseen_topics(self, question: str) -> tuple[str, ...]:
        """问题里的**主题词**中，语料从未出现过的那些（未登录词）。

        这是 `NO_RELEVANT_KNOWLEDGE` 的第二个判据，而且是**比余弦更干净的那个**。
        实测（88 篇语料、12 条真问题 + 7 条语料中不存在的问题）：

        | | 最高余弦 | 主题词未登录 |
        |---|---|---|
        | 真问题 | 0.6659 – 0.8570 | 全部为 0 个 |
        | 语料中不存在 | 0.5396 – 0.7056 | 全部 ≥ 1 个 |

        **余弦分不开这两类**：真问题的下界 0.6659 低于不存在问题的上界 0.7056。
        原因是稠密向量的余弦有一个很高的底噪——`bge-m3` 对两段毫无关系的中文
        也能给出 0.5 以上，那是模型各向异性的性质，不是语料的特点。
        任何单一阈值都会在某一侧出错，而错在哪一侧的代价并不对称：
        误判"没有"会让真问题拒答，误判"有"会让生成节点拿到一堆无关片段。

        未登录词则分得很干净，因为它是**词表事实**而不是相似度估计：
        「跨境」「出海」「直播」「食堂」这些词在 88 篇语料里一次都没出现过，
        词表里自然没有它们——而词表是入库时逐词统计出来的，不是猜的。

        **两个判据是「或」的关系**，各自补另一方的短板：未登录词拦不住
        "用词都在词表里、但语料没讲过这件事"的问题（重叠词组合出来的新话题），
        余弦拦不住"语义相近但其实是别的事"。余弦那一路取保守的低地板，
        见 `rag.score_threshold` 的配置说明。

        ### 三个条件同时成立才算"主题词缺失"

        1. **不在 `_NON_TOPICAL`**：疑问词不携带主题，而语料是陈述性的，
           任何一个疑问词对它来说都是未登录的（见那份表的说明）；
        2. **语料里没有被它包含的词**（`Vocabulary.covers`）：语料从未用过它；
        3. **在 jieba 的通用词典里**（`is_general_word`）：排除**切分伪 token**。

        后两条都是实测补上的，各挡住一类**误拒**：

        - `「2025 年 8 月的经营月报里区域分布情况如何」` 被 jieba 切成
          `…月报 / 报里 / 区域分布…`，`报里` 不在词表里 → 余弦 0.79 的问题被拒答。
          伪 token 是无界的（任何切分抖动都会造出新的），只能从"它是不是一个词"挡。
        - `「直营渠道的价格管理由哪个部门归口负责」` 里的 `归口` 不在词表里，
          但它在语料里出现 **69 次**——业务词典把「归口管理部门」收成了一个词。
          按"是不是一个 token"判断会漏掉它，按"是不是某个词的组成部分"才对。

        **已知的漏网类：同义词**。「报备」在语料里一次都没出现（制度写「备案」），
        也不被任何词包含，于是仍会被判成"语料没见过"而拒答——
        而那条问题的余弦是 0.74，检索其实找得到。纯词汇规则区分不了
        "语料没讲过这件事"与"语料用的是另一个说法"，那一类只能靠语义
        （重排器 / Phase 8 的 Reviewer）。这一例留在金标集里当回归用例。
        """
        return tuple(
            token
            for token in self._tokenizer.cut(question)
            if token not in _NON_TOPICAL
            and not self._vocabulary.covers(token)
            and is_general_word(token)
        )

    # ------------------------------------------------------------------ ④⑤ 召回
    async def _search(
        self, queries: Sequence[str], *, chunk_filter: ChunkFilter
    ) -> tuple[list[list[ScoredPoint]], dict[str, float]]:
        """双路召回（11.7 第 4 步）。

        返回（每个查询的两路排名列表, chunk_id → 稠密余弦）。

        **两条路分开调、不在库里融合**：`VectorStore.hybrid_rrf` 是"一次查询
        融合两路"的便利方法（入库冒烟用它），而这里有 1–4 个查询，
        要的是把 **2×N 路排名一次性融合**。先各自融合再融合一次
        （"RRF 的 RRF"）会让内层已经按排名压缩过一遍，
        同一个分块在不同查询里的相对强弱就此丢失。

        向量化**一次批量发出去**而不是逐条：`bge-m3` 在本机单进程推理，
        批量请求省下的是 N-1 次往返，而查询文本都很短。
        """
        tuning = self._settings.rag
        vectors = await self._gateway.embed(list(queries))
        if len(vectors) != len(queries):
            # `INTERNAL_ERROR` 而不是 `INVALID_ARGUMENT`：条数不符是**网关违约**，
            # 不是调用方的输入问题。用后者会让排查方向跑到用户输入上去。
            raise AgentError(
                ErrorCode.INTERNAL_ERROR,
                f"查询向量化返回条数不符：请求 {len(queries)} 条，返回 {len(vectors)} 条",
                details={"requested": len(queries), "returned": len(vectors)},
            )

        rank_lists: list[list[ScoredPoint]] = []
        dense_scores: dict[str, float] = {}
        for query, vector in zip(queries, vectors, strict=True):
            dense_hits = await self._store.search_dense(
                vector, limit=tuning.dense_top_k, chunk_filter=chunk_filter
            )
            for hit in dense_hits:
                # 同一个 chunk 被多个查询命中时取**最高**余弦：门禁要回答的是
                # "语料里最近的东西有多近"，取平均会把最好的那一路稀释掉
                dense_scores[hit.chunk_id] = max(dense_scores.get(hit.chunk_id, 0.0), hit.score)
            sparse_hits = await self._store.search_sparse(
                build_sparse(
                    self._tokenizer.cut(query),
                    self._vocabulary,
                    dim=tuning.sparse_dim,
                ),
                limit=tuning.sparse_top_k,
                chunk_filter=chunk_filter,
            )
            rank_lists.append(dense_hits)
            rank_lists.append(sparse_hits)
        return rank_lists, dense_scores


# ------------------------------------------------------------------ 融合与组装


def _fuse(
    rank_lists: Sequence[Sequence[ScoredPoint]], *, rrf_k: int
) -> list[tuple[str, float, dict[str, Any]]]:
    """RRF 融合（11.7 第 5 步）：`score = Σ 1/(k + rank)`，rank 从 1 起。

    **与 Qdrant 的 `Fusion.RRF` 同公式**（`InMemoryVectorStore.hybrid_rrf`
    已经把这个公式钉在契约测试里）。同一个分块被多路命中时分数累加——
    这正是"两路都认为它相关"应该得到的加成。

    并列时按 `chunk_id` 兜底排序：只按分数排的话，并列项的顺序取决于
    字典遍历顺序，同一个问题会时好时坏。
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            payloads[hit.chunk_id] = hit.payload
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(chunk_id, score, payloads[chunk_id]) for chunk_id, score in ordered]


def _to_chunks(
    fused: Sequence[tuple[str, float, dict[str, Any]]],
    dense_scores: dict[str, float],
    top_k: int,
) -> tuple[RetrievedChunk, ...]:
    """融合结果 → Top K 条候选（11.7 第 7 步）。

    payload 反解失败的分块**整条跳过**：`ChunkMetadata` 的字段都是证据定位
    必需的（没有 `checksum` 就说不清引用的是哪一份文件），
    返回一条定位不全的候选，下游会把它当成一条正常证据来引用。
    跳过的同时不静默——条数差会反映在 `candidate_count` 与 `len(candidates)` 上。
    """
    chunks: list[RetrievedChunk] = []
    for rank, (chunk_id, score, payload) in enumerate(fused[:top_k], start=1):
        try:
            metadata = ChunkMetadata.from_payload(payload)
        except ValueError:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=str(payload.get("text", "")),
                fusion_score=score,
                # None 而不是 0.0：稀疏路独有候选**没有**稠密分，
                # 补 0 等于断言"语义完全不相关"，那是我们并不知道的结论
                dense_score=dense_scores.get(chunk_id),
                rank=rank,
                metadata=metadata,
            )
        )
    return tuple(chunks)


def _dedupe(queries: Sequence[str]) -> tuple[str, ...]:
    """去重且保序。改写模型经常把原问题原样抄一条回来。"""
    seen: dict[str, None] = {}
    for query in queries:
        seen.setdefault(query, None)
    return tuple(seen)


def _keeps_numbers(question: str, rewrite: str) -> bool:
    """改写是否保留了原问题里的**全部数字锚点**（11.7 第 2 步的「必须保留时间」）。

    「2025 年华东 Q3」被改写成「三季度区域业绩」时，检索照常跑、照常返回结果，
    **只是不再是我们问的那件事**——没有任何报错。这条检查就是为它存在的。

    归一化之后再比：`normalize_numbers` 把中文数词折成阿拉伯数字，
    于是「2025 年第三季度」与「2025 年 Q3」都得到 `{2025, 3}`，不会因为
    表述差异被误判成"丢了时间"。

    **专有名词（区域名、产品线）不在检查范围内**——没有一份实体表就判定不了，
    而现造一份近义词表只会把误报变成新的噪声源。那一半由"原问题永远参与检索"
    兜底（见 `_rewrite`）。
    """
    required = set(_NUMBER.findall(normalize_numbers(question)))
    return required <= set(_NUMBER.findall(normalize_numbers(rewrite)))


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


__all__ = ["Retriever"]
