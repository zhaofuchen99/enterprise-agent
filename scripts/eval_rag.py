"""RAG 检索金标评测（开发流程 6.7 的「Recall@8 ≥ 85%」门禁）。

```bash
make eval-rag              # 全部 20 条
make eval-rag ONLY=rag-01,rag-02
```

## 判定的三层，逐层收紧

对每条金标：

1. **命中@8**：Top 8 里出现了 `expect_docs` 里的任一份（按 `document_id`）；
2. **定位一致**：命中的那条分块的 `section_path` 含 `expect_sections` 里的任一个；
3. **版本正确**：`as_of` 给定时，命中的是那一版（由 `expect_docs` 的
   `@version` 直接表达——命中别的版本就不是命中）。

**Recall@8 取第 1 层**，第 2 层单独报「定位一致率」。两层分开报而不是合成一个
分数，是因为它们的失败原因完全不同：命中率低要查召回（分词、词表、embedding），
定位不一致要查分块与标题路径——合成一个数字就分不出来了。

## 为什么不用 `rerank_top_k` 之外的口径

`rerank_top_k` 默认 8，11.7 第 7 步定的就是 Top 8。**评测的 K 必须与线上一致**：
用一个更宽的 K 去测，得到的数字解释不了线上行为，而它会看起来更好看。

## 拒答那 3 条不进这个分母

`absent_cases` 由 `make verify-corpus` 判定（期望 `NO_RELEVANT_KNOWLEDGE`）。
把"该拒答的"和"该召回的"混进同一个比率，两个方向都说不清。
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.infrastructure.cache import VersionedCache
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.model_gateway import build_model_gateway
from app.infrastructure.redis import create_client
from app.infrastructure.storage import build_object_storage
from app.infrastructure.vector_store import build_vector_store
from app.repositories.vocab_repo import SqlVocabRepository
from app.tools.base import ToolResult
from app.tools.rag.golden import GoldenCase, load_golden
from app.tools.rag.schemas import RagQueryArgs
from app.tools.rag.tool import build_rag_retrieve_tool
from app.tools.sql.tool import build_cli_context


@dataclass(frozen=True)
class CaseOutcome:
    case: GoldenCase
    hit_docs: tuple[str, ...]
    locator_ok: bool
    queries: tuple[str, ...]
    error: str | None = None

    @property
    def hit(self) -> bool:
        return bool(self.hit_docs)


def evaluate(case: GoldenCase, result: ToolResult) -> CaseOutcome:
    """一次检索结果 → 判定。

    只读 `payload.chunks`（Tool 的结构化产物），不读 `result.evidence`：
    evidence 会把 `locator` 的键名写死在这里，而那是**引用格式**，
    它的稳定性由 `test_retriever.py` 单独钉住，不该再被评测脚本依赖一遍。
    """
    if result.error is not None:
        return CaseOutcome(case, (), False, (), result.error.code)

    payload = result.payload or {}
    chunks = payload.get("chunks") or []
    found = [
        chunk["metadata"]["document_id"]
        for chunk in chunks
        if chunk["metadata"]["document_id"] in case.expect_docs
    ]
    locator_ok = False
    if case.expect_sections:
        locator_ok = any(
            any(anchor in section for section in chunk["metadata"].get("section_path") or [])
            for chunk in chunks
            if chunk["metadata"]["document_id"] in case.expect_docs
            for anchor in case.expect_sections
        )
    else:
        locator_ok = bool(found)
    return CaseOutcome(
        case=case,
        hit_docs=tuple(found),
        locator_ok=locator_ok,
        queries=tuple(payload.get("queries") or ()),
    )


async def run(settings: Settings, cases: Sequence[GoldenCase]) -> list[CaseOutcome]:
    engine = create_engine(settings)
    redis = create_client(settings)
    storage = build_object_storage(settings)
    vector_store = build_vector_store(settings)
    gateway = build_model_gateway(settings)
    outcomes: list[CaseOutcome] = []
    try:
        tool = await build_rag_retrieve_tool(
            settings,
            gateway,
            vector_store=vector_store,
            storage=storage,
            vocab=SqlVocabRepository(create_session_factory(engine)),
            cache=VersionedCache(redis, default_ttl_seconds=settings.rag.vocab_cache_ttl_seconds),
        )
        base_ctx = build_cli_context(region_ids=(), timeout_seconds=settings.task_timeout_seconds)
        for case in cases:
            # 每条用例可以指定身份：语料里有 CONFIDENTIAL 文档，它们的
            # `allowed_roles` 只有 ADMIN，用 ANALYST 去检索会被**正确地**挡在外面。
            # 不区分身份的话，那条用例会报"未命中"，看起来像召回缺陷。
            ctx = base_ctx.model_copy(
                update={
                    "permission_scope": base_ctx.permission_scope.model_copy(
                        update={"role": case.role}
                    )
                }
            )
            try:
                result = await tool.execute(
                    RagQueryArgs(question=case.question, as_of=case.as_of), ctx
                )
            except AgentError as exc:
                # 单条失败不中断整轮：一条打不通不该让后面 19 条的数字都拿不到
                outcomes.append(CaseOutcome(case, (), False, (), exc.code.value))
                continue
            outcomes.append(evaluate(case, result))
    finally:
        await gateway.aclose()
        await vector_store.aclose()
        await redis.aclose()
        await engine.dispose()
    return outcomes


def render(outcome: CaseOutcome) -> str:
    mark = "✓" if outcome.hit else "✗"
    detail = "、".join(outcome.hit_docs) if outcome.hit_docs else "未命中"
    locator = "" if outcome.locator_ok else "｜**定位不一致**"
    error = f"｜{outcome.error}" if outcome.error else ""
    line = f"{mark} {outcome.case.id}  {detail}{locator}{error}"
    if not outcome.hit:
        # 未命中时把实际用的查询打出来：**改写改坏了**与**召回不到**
        # 是两条完全不同的排查路径，而结果里只看得见"未命中"
        line += f"\n    实际查询：{'|'.join(outcome.queries) or '（无）'}"
        line += f"\n    期望：{'、'.join(outcome.case.expect_docs)}｜{outcome.case.proves}"
    return line


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RAG 检索金标评测（Recall@8）")
    parser.add_argument("--only", default="", help="只跑指定 ID，逗号分隔，如 rag-01,rag-02")
    parser.add_argument("--path", default=None, help="金标文件路径")
    args = parser.parse_args(argv)

    golden = load_golden(args.path) if args.path else load_golden()
    cases = list(golden.cases)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case.id in wanted]

    settings = get_settings()
    outcomes = asyncio.run(run(settings, cases))

    print(f"金标 {golden.version}｜{len(outcomes)} 条｜Top {settings.rag.rerank_top_k}")
    print(f"相关性阈值 {settings.rag.score_threshold}（⑫ 之前是保守地板，见 config.py）")
    print()
    for outcome in outcomes:
        print(render(outcome))

    hits = sum(1 for outcome in outcomes if outcome.hit)
    located = sum(1 for outcome in outcomes if outcome.locator_ok)
    total = len(outcomes)
    print()
    print("─" * 72)
    if total:
        print(f"Recall@8    {hits}/{total} = {hits / total:.1%}（门禁 ≥ 85%）")
        print(f"定位一致率  {located}/{total} = {located / total:.1%}")
    if hits < total:
        print("\n未命中明细（按 proves 分类，便于判断是召回问题还是分块问题）：")
        for outcome in outcomes:
            if not outcome.hit:
                print(f"  - {outcome.case.proves}")
    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main())
