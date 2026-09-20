"""独立验证与运维入口（开发流程 4.2 的 `app/cli.py`）。

`make seed` / `make cleanup` 都走这里。做成子命令而不是一堆散落脚本，
是因为这些动作**必须复用应用自己的配置与仓储**：另写一份连接串与
SQL，就会出现「CLI 灌进了 A 库、服务读的是 B 库」这类只在演示时才发现的问题。

**这个模块不得 import `app/main.py`**：main 会加载 FastAPI，
而本模块要能在没有 Web 框架的进程里跑（也避免把 API 的启动成本
转嫁到一条运维命令上）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Coroutine, Sequence
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

from sqlalchemy import text

from app.agent.prompts import SMOKE_PROMPT
from app.agent.schemas import SmokeAnswer
from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.domain.knowledge import SourceKind
from app.infrastructure.cache import VersionedCache
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import SERVICE_CLI, setup_logging
from app.infrastructure.model_gateway import EmbeddingDimensionError, build_model_gateway
from app.infrastructure.redis import create_client
from app.infrastructure.storage import build_object_storage
from app.infrastructure.vector_store import build_vector_store
from app.repositories.knowledge_repo import SqlKnowledgeDocumentRepository
from app.repositories.user_repo import SqlUserRepository, seed_demo_users
from app.repositories.vocab_repo import SqlVocabRepository
from app.tools.base import ToolResult
from app.tools.rag import verify
from app.tools.rag.chunker import Chunk, chunk_document
from app.tools.rag.golden import load_golden
from app.tools.rag.ingestion import (
    DocumentMetadata,
    IngestionReport,
    ingest_document,
)
from app.tools.rag.parser import SUFFIX_TO_FORMAT, ParsedDocument, parse_document
from app.tools.rag.schemas import RagQueryArgs
from app.tools.rag.tokenizer import Tokenizer, load_stopwords, normalize, normalize_numbers
from app.tools.rag.tool import build_rag_retrieve_tool, build_retriever
from app.tools.rag.vocabulary import build_vocabulary, export_snapshot, load_vocabulary
from app.tools.sql.schemas import SqlQueryArgs
from app.tools.sql.tool import build_cli_context, build_sql_query_tool

#: 子命令处理器的返回值：同步的直接给退出码，异步的给协程（分发处统一 await）。
#: 两者都是合法的，见 `main` 的分发说明。
HandlerResult = int | Coroutine[Any, Any, int]

#: 语料目录。`chunk` 用它定位生成报告（取标题与文档键），找不到就退化成文件名。
_CORPUS_ROOT = Path("data/corpus")


async def _seed(settings: Settings, args: argparse.Namespace) -> int:
    """灌入演示账号，幂等。

    业务演示库（`fact_sales_order_item` 等 8 张表）**不在这里**：那套数据是
    「模拟企业既有系统」，建表与灌数属于 DBA 侧的供给，走
    `scripts/business_seed.py`（`make seed` 会把两者依次跑一遍）。
    放进 CLI 会让读者以为应用运行时也会写那个库——而它连的是只读账号。
    """
    engine = create_engine(settings)
    try:
        sessions = create_session_factory(engine)
        created = await SqlUserRepository(sessions).upsert_demo_users(seed_demo_users(settings))
        print(f"演示账号：新增 {created} 个（已存在的跳过）")
        print("业务演示库请另行执行：make seed-business（或 make seed 一并跑）")
    finally:
        await engine.dispose()
    return 0


async def _cleanup(settings: Settings, args: argparse.Namespace) -> int:
    """按保留期清理过期数据（详细设计 16.12）。

    **幂等**，供 cron 每日调用；不引入分布式调度器，多实例重复执行也不出错。
    """
    print("【后续扩展】`cleanup` 尚未实现：需先确定各表的保留期策略（详细设计 16.12）")
    return 0


async def _model_smoke(settings: Settings, args: argparse.Namespace) -> int:
    """打一次真实模型 + 一次真实向量化，把观测到的结果打印出来。

    **为什么需要它**：Phase 3 之后的所有节点都要过 `ModelGateway`，
    而它连的是外部服务。把「密钥对不对、端点通不通、返回的 JSON 能不能被
    Pydantic 吃下」这三件事留到第一个业务节点去发现，
    排查时会同时面对「prompt 写得对不对」和「配置对不对」两个未知数。
    自检把后者单独摘出来先验掉。

    **只打一次、不做重试放大**：网关自身的重试仍然生效，但这里不额外循环——
    自检要如实反映一次调用的代价（延迟、token），而不是压测吞吐。
    """
    gateway = build_model_gateway(settings)
    failures: list[str] = []

    def report_failure(label: str, exc: AgentError | EmbeddingDimensionError) -> None:
        """报告一次失败。**必须带 details**。

        自检命令的全部价值在于给出可行动的线索。只打印 `AgentError.message`
        （面向用户的通用文案，如「向量化服务返回错误」）等于什么都没说——
        真正有用的 `status_code` / `attempts` 都在 `details` 里。
        本项目自己的诊断工具尤其不能犯这个错。
        """
        if isinstance(exc, AgentError):
            failures.append(f"{label}：{exc.code.value} - {exc.message}")
            print(f"  ✗ {label}失败：{exc.code.value} - {exc.message}")
            print(f"    详情（仅诊断）：{exc.details}")
        else:
            failures.append(f"{label}：{exc}")
            print(f"  ✗ {label}失败：{exc}")

    try:
        print(f"主模型   ：{settings.model_name} @ {settings.model_base_url}")
        try:
            result = await gateway.invoke_structured(
                SMOKE_PROMPT, SmokeAnswer, question="请判断：1 + 1 是否等于 2？"
            )
        except AgentError as exc:
            report_failure("结构化调用", exc)
        else:
            print(f"  ✓ 结构化调用成功：{result.value.model_dump()}")
            print(
                f"    耗时 {result.duration_ms}ms ｜ 请求次数 {result.attempts} ｜ "
                f"token 入/出 {result.usage.prompt_tokens}/{result.usage.completion_tokens} ｜ "
                f"prompt {SMOKE_PROMPT.name}({result.prompt_version})"
            )

        # 打印**生效的**端点而不是配置里的那一项：`EMBEDDING_BASE_URL` 缺省时
        # 会回落到 `MODEL_BASE_URL`，于是向量化请求会被悄悄发到聊天模型的服务商那里。
        # 这一行是发现那类「配漏了但没报错」问题的第一现场。
        effective_embedding_url = settings.embedding_base_url or settings.model_base_url
        print(f"向量模型：{settings.embedding_model} @ {effective_embedding_url}")
        try:
            vectors = await gateway.embed(["华东地区 Q3 净销售额"])
        except (AgentError, EmbeddingDimensionError) as exc:
            report_failure("向量化", exc)
        else:
            print(f"  ✓ 向量化成功：{len(vectors)} 条 × {len(vectors[0])} 维")
    finally:
        await gateway.aclose()

    if failures:
        print("\n自检未通过：")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\n自检通过。")
    return 0


async def _sql(settings: Settings, args: argparse.Namespace) -> int:
    """跑一次完整的自然语言 → SQL → 真数据 → 证据（开发流程 6.6 的验证命令）。

    **这条命令是 Phase 4 的门禁本身**：详设 21.7 要求「SQL Tool 可独立通过
    API/CLI 调用，不依赖 Graph」。因此它走的是与将来 Worker 完全相同的
    `SqlQueryTool`，而不是一份为演示写的简化逻辑——那种写法只能证明
    「演示脚本能跑」，证明不了「工具能跑」。

    `--region` 模拟 `app_user.data_scope_json`：CLI 没有登录用户，
    但数据权限的注入路径必须被真实执行过一次才算数（详设 10.4 第 11 步）。
    不传即全量，与演示库里 admin 账号的语义一致。
    """
    gateway = build_model_gateway(settings)
    tool = build_sql_query_tool(settings, gateway)
    ctx = build_cli_context(
        region_ids=tuple(args.region), timeout_seconds=settings.task_timeout_seconds
    )

    try:
        if args.sql:
            result = await tool.run_sql(args.sql, ctx)
        else:
            result = await tool.execute(SqlQueryArgs(question=args.question), ctx)
    finally:
        await tool.aclose()

    if args.sql:
        print(f"给定 SQL（跳过生成，仍走完整校验）：{args.sql}")
    else:
        print(f"问题：{args.question}")
    if args.region:
        print(f"数据范围：{'、'.join(args.region)}（模拟 data_scope）")
    print(f"状态：{result.status}｜{result.summary}｜耗时 {result.duration_ms}ms")
    print()

    if result.status == "FAILED" and result.error is not None:
        error = result.error
        print(f"✗ 已拒绝：{error.message}")
        print(f"  错误码 {error.code}｜类别 {error.error_class}｜可重试 {error.retryable}")
        if error.safe_detail:
            print(f"  详情（仅诊断）：{error.safe_detail}")
        _print_attempts(result.payload)
        return 1

    payload = result.payload or {}
    print("规范化 SQL：")
    print(f"  {payload.get('normalized_sql', '')}")
    print(f"  fingerprint {payload.get('sql_fingerprint', '')}")
    print()

    columns = [column["name"] for column in payload.get("columns", [])]
    if columns:
        print(f"结果（{payload.get('row_count', 0)} 行）：")
        print("  " + " | ".join(columns))
        for row in payload.get("rows") or []:
            print("  " + " | ".join("" if v is None else str(v) for v in row))
    else:
        print("结果：未返回任何列")
    print()

    for warning in payload.get("warnings", []):
        print(f"⚠ {warning}")
    if payload.get("warnings"):
        print()

    if result.evidence:
        print(f"证据（{len(result.evidence)} 条）：")
        for item in result.evidence:
            print(f"  · {item.claim}")
            print(f"    定位 {item.locator}｜口径 {item.metric_code}@{item.definition_version}")
        print()

    _print_attempts(result.payload)
    return 0


def _tokenize(settings: Settings, args: argparse.Namespace) -> int:
    """逐条核对切分结果（开发流程 6.7 施工项 3）。

    **这是 Phase 5 使用频率最高的调试命令**（见详设 11.6.2）：检索召回不到东西时，
    第一个要回答的问题是「这句话到底被切成了什么」。

    因此它不只打印保留的 token，还打印两件别的命令看不到的事：

    1. **被丢弃的 token**——「切错了」与「切对了但被过滤规则丢了」是两种故障，
       只看得见保留结果时它们长得一样；
    2. **`--no-dict` 的对照**——不加载业务词典再切一遍。词典有没有生效、
       某条术语是不是词典覆盖不到，只有对照着看才判得出来。
       这也是词典产物（`configs/rag_user_dict.txt`）唯一的现场验证手段。

    稀疏向量**不在这里打印**：它需要一份已构建的词表（`rag_vocab`），
    而词表要在语料分块之后才能建。词表就位后这条命令会补上向量输出。
    """
    stopwords = load_stopwords(settings.rag.stopword_path)
    tokenizer = Tokenizer(
        user_dict_path=None if args.no_dict else settings.rag.user_dict_path,
        stopwords=stopwords,
    )
    text: str = args.text
    normalized = normalize(text)
    numbered = normalize_numbers(normalized)

    print(f"原文      ：{text}")
    if normalized != text:
        print(f"归一化    ：{normalized}")
    if numbered != normalized:
        print(f"数字/日期 ：{numbered}")

    kept, dropped = tokenizer.cut_explained(text)
    print(f"分词      ：{' / '.join(kept) if kept else '（空）'}")
    if dropped:
        # 单字与纯标点也在这里：它们是稀疏维度的主要消耗者，
        # 出现在这里说明"被主动丢掉了"，而不是"分词切不出来"
        shown = " ".join(repr(t) for t in dropped)
        print(f"丢弃      ：{shown}")

    if not args.no_dict:
        if not Path(settings.rag.user_dict_path).exists():
            print(
                f"\n⚠ 业务词典 {settings.rag.user_dict_path} 不存在，以上结果来自通用分词。"
                "\n  生成：make dict（需 make up + make seed-business 已执行）"
            )
        else:
            baseline = Tokenizer(user_dict_path=None, stopwords=stopwords)
            base_kept, _ = baseline.cut_explained(text)
            if base_kept != kept:
                print(
                    f"\n对照（不加载业务词典）：{' / '.join(base_kept) if base_kept else '（空）'}"
                )
                print("  ↑ 两者不同即说明业务词典在本句上起了作用")
    return 0


def _chunk(settings: Settings, args: argparse.Namespace) -> int:
    """解析 + 分块，把结果逐条打出来（详细设计 11.3 的现场验证命令）。

    **为什么需要它**：11.3 的分块参数是一组目标值，而分块是那种"参数错了
    要到检索评测才看得出来"的环节——那时你面对的是 Recall@8 掉了几个点，
    完全不知道是切大了、切小了，还是标题路径没挂上。这条命令把中间产物摊开：
    每块的标题路径、页码、字符数、正文，以及**解析阶段清洗掉了什么**。

    入库元数据（标题 / 文档键）优先取语料生成报告，取不到就退化成文件名。
    标题不是装饰：它是标题路径的根，没有它，
    「这段话出自哪份文件」这个问题在检索侧就没有答案。
    """
    root = Path(args.path)
    targets = _collect_documents(root)
    if not targets:
        print(f"没有可解析的文件：{root}")
        return 1

    failures = 0
    for file in targets:
        meta = _document_meta(file)
        parsed = parse_document(file)
        chunks = chunk_document(
            parsed,
            document_key=meta.key,
            settings=settings,
            title=meta.title,
        )
        if args.summary:
            _print_chunk_summary(meta, parsed, chunks)
            continue
        failures += _print_chunks(meta, parsed, chunks, args)
    return 1 if failures else 0


def _collect_documents(root: Path) -> list[Path]:
    """收集待处理文件。目录按扩展名过滤并排序——**同一批文件必须每次同序**，
    否则 `chunk_id` 依赖的文档顺序变了，两次运行的输出对不上。"""
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.rglob("*") if p.suffix.lower() in SUFFIX_TO_FORMAT and p.is_file()),
        key=lambda p: p.name,
    )


class _DocMeta(NamedTuple):
    id: str
    title: str
    key: str


def _document_meta(file: Path) -> _DocMeta:
    """取文档元数据：语料生成报告 → 文件名。

    **不硬编码"标题就是文件名"**：语料里的标题与文件名本来就不同
    （`SP-015.pdf` 的标题是「区域折扣授权额度表」），
    用文件名当标题会让标题路径悄悄变成一串编号。
    """
    item = _corpus_entries().get(_resolve(file))
    if item is not None:
        return _DocMeta(
            id=item["id"],
            title=item["title"],
            key=f"{item['logical_key']}@{item['version']}",
        )
    return _DocMeta(id=file.stem, title=file.stem, key=f"{file.stem}@v1.0")


@lru_cache(maxsize=1)
def _corpus_entries() -> dict[str, dict[str, Any]]:
    """语料生成报告，按**解析后的绝对路径**索引。

    做成缓存而不是每篇文件读一次：`chunk` 的旧写法在 88 篇上要重复解析
    同一份 JSON 88 次，而 `ingest` 需要的字段更多。缓存后两条命令共用同一份。

    用绝对路径而不是文件名作键：报告里写的是相对路径（`data/corpus/SP-001.pdf`），
    而调用方给进来的可能是绝对路径或从别处传的相对路径，
    只按字符串比会静默查不到——查不到的症状是"标题退化成文件名"，
    然后这个文件名会一路进标题路径、进向量库。
    """
    report = _CORPUS_ROOT / "corpus_report.json"
    if not report.exists():
        return {}
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {_resolve(Path(item["path"])): item for item in payload.get("documents") or []}


def _resolve(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:  # pragma: no cover - 路径不存在时 resolve 也可能抛
        return str(path)


def _print_chunk_summary(meta: _DocMeta, parsed: ParsedDocument, chunks: list[Chunk]) -> None:
    status = "有文本层" if parsed.has_text_layer else "**无文本层（扫描件，标记 FAILED）**"
    print(
        f"{meta.id:14} {len(chunks):4d} 块｜{parsed.page_count} 页｜{status}"
        f"｜清洗 {len(parsed.dropped)} 项"
    )
    if chunks:
        sizes = sorted(len(c.text) for c in chunks)
        print(
            f"{'':14} 长度 min/中位/max = {sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}"
            f"｜表格块 {sum(1 for c in chunks if c.is_table)}"
        )


def _print_chunks(
    meta: _DocMeta, parsed: ParsedDocument, chunks: list[Chunk], args: argparse.Namespace
) -> int:
    """打印单篇的分块明细。返回失败计数（供退出码）。"""
    kinds: dict[str, int] = {}
    for block in parsed.blocks:
        kinds[block.kind] = kinds.get(block.kind, 0) + 1
    print(f"文档：{meta.id}  标题：{meta.title}")
    print(f"      键 {meta.key}")
    print(f"解析：{parsed.page_count} 页｜块 " + " / ".join(f"{k} {v}" for k, v in kinds.items()))
    if not parsed.has_text_layer:
        print("⚠ 无文本层（图片型扫描件）。切片内不做 OCR，入库应标记 FAILED 并写明原因。")
        return 0
    if parsed.dropped:
        # **清洗结果必须显示**：清洗是"正确时无声、错误时也无声"的操作，
        # 多丢一行不会有任何症状，直到某天有人问"制度里明明写了"
        print(f"清洗丢弃 {len(parsed.dropped)} 项：")
        for item in parsed.dropped:
            print(f"  - {item}")
    print()

    shown = [c for c in chunks if _keep_chunk(c, args)]
    if args.limit:
        shown = shown[: args.limit]
    for order, chunk in enumerate(shown, start=1):
        page = f"p{chunk.page_no}" if chunk.page_no else "—"
        tag = "表" if chunk.is_table else "文"
        print(f"[{order}] {chunk.chunk_id} {page} {tag} {len(chunk.text)}字")
        print(f"    路径 {' > '.join(chunk.section_path)}")
        for line in chunk.text.splitlines():
            print(f"    {line}")
        print()
    print(
        f"分块 {len(chunks)} 块" + (f"（显示 {len(shown)}）" if len(shown) != len(chunks) else "")
    )
    print()
    return 0


def _keep_chunk(chunk: Chunk, args: argparse.Namespace) -> bool:
    if args.table_only:
        return chunk.is_table
    if args.text_only:
        return not chunk.is_table
    return True


async def _vocab(settings: Settings, args: argparse.Namespace) -> int:
    """构建稀疏检索词表（详细设计 11.6.3 / 11.6.4）。

    **为什么构建是独立的一步**：11.6.4 选了固定 IDF，而 IDF 依赖全库统计，
    两者要同时成立只有一条路——先把词表冻结，再入库。见
    `app/tools/rag/vocabulary.py` 的模块说明。

    命令是**幂等**的：`rag_vocab` 只增不改，已发布的 token_id 与 df 原样沿用，
    重复跑只会把新出现的词接在后面。因此可以放心重跑，
    也可以在语料扩充后重跑来吸收新词。
    """
    engine = create_engine(settings)
    try:
        sessions = create_session_factory(engine)
        repository = SqlVocabRepository(sessions)
        existing = await repository.load()

        files = _collect_documents(Path(args.path))
        if not files:
            print(f"没有可解析的文件：{args.path}")
            return 1

        tokenizer = Tokenizer.from_settings(settings)
        texts: list[str] = []
        for file in files:
            meta = _document_meta(file)
            parsed = parse_document(file)
            chunks = chunk_document(
                parsed, document_key=meta.key, settings=settings, title=meta.title
            )
            texts.extend(chunk.text for chunk in chunks)

        result = build_vocabulary(
            texts,
            tokenizer=tokenizer,
            sparse_dim=settings.rag.sparse_dim,
            existing=existing,
        )
        added = await repository.add(result.added)
        # **回读落库结果**，而不是相信内存里的那份：`add` 的语义是"已存在的跳过"，
        # 而"内存里的词表"与"库里真实存在的词表"在并发跑两条 make vocab 时会分叉。
        # 回读拿到的是下次分配 id 的真正起点。
        stored = await repository.load()
        version = await repository.version()
        key = await export_snapshot(
            build_object_storage(settings), result.vocabulary, version=version
        )
    finally:
        await engine.dispose()

    print(f"语料：{len(files)} 篇 / {result.total_chunks} 个 chunk（IDF 的分母）")
    print(f"词表：历史 {len(existing)} 条 → 本次新增 {added} 条 → 合计 {len(result.vocabulary)} 条")
    if result.added:
        preview = "、".join(entry.token for entry in result.added[:12])
        print(f"  新增样例：{preview}{' …' if len(result.added) > 12 else ''}")
    max_id = max(entry.token_id for entry in stored)
    print(f"  token_id 上限 {max_id} / 维度 {settings.rag.sparse_dim}")
    print(f"版本 {version}")
    print(f"快照 {key}")
    return 0


async def _ingest(settings: Settings, args: argparse.Namespace) -> int:
    """逐篇入库（详细设计 11.1 的九步 / 11.9 的幂等与发布）。

    **前置条件有两条，缺一条都会在跑到一半才失败**：
    `make corpus`（文件与元数据都在 `corpus_report.json` 里）与
    `make vocab`（稀疏向量的 token_id 来自冻结的词表快照）。

    一条命令跑完整个语料而不是逐篇调，是因为入库的失败模式大多是**全局的**
    （Ollama 没起、词表没建、Qdrant 连不上），逐篇跑会把这些错误重复 88 次。
    但**单篇失败不中断整批**：语料里本来就有 2 份扫描件注定失败（11.2 明确
    "标记不支持"），一份失败把整批停下来会让另外 86 篇永远入不了库。
    """
    targets = _collect_documents(Path(args.path))
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        targets = [
            file
            for file in targets
            if (_corpus_entries().get(_resolve(file)) or {}).get("id") in wanted
        ]
    if not targets:
        print(f"没有可入库的文件：{args.path}")
        return 1

    entries = _corpus_entries()
    engine = create_engine(settings)
    redis = create_client(settings)
    storage = build_object_storage(settings)
    vector_store = build_vector_store(settings)
    gateway = build_model_gateway(settings)
    reports: list[IngestionReport] = []
    failures: list[tuple[str, str]] = []
    try:
        sessions = create_session_factory(engine)
        documents = SqlKnowledgeDocumentRepository(sessions)
        vocabulary = await load_vocabulary(
            SqlVocabRepository(sessions),
            VersionedCache(redis, default_ttl_seconds=settings.rag.vocab_cache_ttl_seconds),
            storage,
            ttl_seconds=settings.rag.vocab_cache_ttl_seconds,
        )
        tokenizer = Tokenizer.from_settings(settings)
        # 建 collection 放在循环外：它是幂等的，但每篇一次就是 88 次往返，
        # 而更重要的是——它**会在维度不匹配时立刻报错**（11.5），
        # 放在循环外能让这个致命错误在第 1 篇之前就暴露，而不是第 88 篇之后。
        await vector_store.ensure_collection(dim=settings.embedding_dim)

        print(
            f"入库 {args.path}：{len(targets)} 篇｜词表 {len(vocabulary)} 词条"
            f"｜force={bool(args.force)}"
        )
        for index, file in enumerate(targets, start=1):
            item = entries.get(_resolve(file))
            if item is None:
                failures.append((file.name, "语料报告里没有这篇的元数据，先跑 make corpus"))
                print(f"[{index:3d}/{len(targets)}] {file.name:16} 跳过：报告里无元数据")
                continue
            try:
                report = await ingest_document(
                    file,
                    _metadata_from_entry(item),
                    settings=settings,
                    storage=storage,
                    vector_store=vector_store,
                    documents=documents,
                    gateway=gateway,
                    tokenizer=tokenizer,
                    vocabulary=vocabulary,
                    force=bool(args.force),
                )
            except AgentError as exc:
                # 已知错误（校验不过、同版本不同内容、词表未覆盖）：记下继续。
                # 整批停下来会让"第 3 篇版本号写错了"变成"后面 85 篇都没入"
                failures.append((file.name, str(exc)))
                print(f"[{index:3d}/{len(targets)}] {file.name:16} 失败：{exc}")
                continue
            reports.append(report)
            if not report.ok:
                failures.append((file.name, report.error_summary or "入库失败"))
            print(f"[{index:3d}/{len(targets)}] {_format_ingest_line(item['id'], report)}")
    finally:
        await gateway.aclose()
        await vector_store.aclose()
        await redis.aclose()
        await engine.dispose()

    # 退出码按"这次跑坏了没有"给，而不是"有没有文档是 FAILED"：
    # 语料里 2 份扫描件注定 FAILED（11.2 允许"明确标记不支持"），
    # 让它们把整条命令判成失败的话，`make ingest` 恒返回非零——
    # 而 Phase 5 的门禁恰恰要求"标记不支持"算是通过。
    # 反过来说，"Ollama 没起导致 88 篇全挂"必须返回非零，所以"不支持"与
    # "真的坏了"要分开数（`report.unsupported`）。
    _print_ingest_summary(reports, failures)
    return 1 if _ingest_failure_count(reports, failures) else 0


def _ingest_failure_count(reports: list[IngestionReport], failures: list[tuple[str, str]]) -> int:
    """**真的坏了**的篇数：抛异常的，加上 FAILED 且非"文件不支持"的。

    数错这里的代价不是退出码难看，而是门禁失去意义：
    算多了 → 语料里那 2 份扫描件让 `make ingest` 永远非零，
    于是没人再看退出码；算少了 → "Ollama 没起"变成一次安静的成功。
    """
    unsupported = sum(1 for r in reports if not r.ok and r.unsupported)
    return len(failures) - unsupported


def _metadata_from_entry(item: dict[str, Any]) -> DocumentMetadata:
    """语料报告的一条 → 入库元数据。

    `source_kind` **显式转换而不透传字符串**：报告里是 `"EXTERNAL"`，
    而 `DocumentMetadata` 收的是 `SourceKind`。直接透传字符串看着也能过
    （Pydantic 会转），但转换失败时的报错落在"入库第 47 篇"上，
    而这一处是**所有文档的来源标记唯一产生的地方**——FR-SEARCH-001 的
    SOURCE 冲突判定全靠它。在这里转，错了就在启动时错。
    """
    return DocumentMetadata(
        logical_key=item["logical_key"],
        version=item["version"],
        title=item["title"],
        document_type=item["type"],
        source_kind=SourceKind(item["source_kind"]),
        classification=item.get("classification") or "INTERNAL",
        department=item.get("department"),
        effective_from=_parse_date(item.get("effective_from")),
        effective_to=_parse_date(item.get("effective_to")),
        published_at=_parse_datetime(item.get("published_at")),
    )


def _parse_date(value: Any) -> date | None:
    return date.fromisoformat(str(value)) if value else None


def _parse_datetime(value: Any) -> datetime | None:
    """`published_at` 在清单里可能只写到日（`2025-11-05`），补成当天零点。

    解析失败**直接报错而不是当作没有**：一个拼错的日期被当成 `None`
    意味着这篇文档变成"长期有效"，而它可能正是要通过生效区间排除掉的那一版。
    """
    if not value:
        return None
    text = str(value)
    parsed = (
        datetime.fromisoformat(text) if "T" in text else datetime.fromisoformat(f"{text}T00:00:00")
    )
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _format_ingest_line(doc_id: str, report: IngestionReport) -> str:
    if report.skipped:
        return f"{doc_id:14} 已入库（指纹相同，跳过）"
    smoke_ok = sum(1 for item in report.smoke if item.matched)
    parts = [
        f"{doc_id:14}",
        f"{report.status.value:10}",
        f"{report.chunk_count:4d} 块",
        f"{report.page_count:3d} 页",
    ]
    if report.smoke:
        parts.append(f"冒烟 {smoke_ok}/{len(report.smoke)}")
    if report.dropped:
        parts.append(f"清洗 {len(report.dropped)} 项")
    if report.previous_checksum:
        parts.append("重建")
    if report.error_summary:
        parts.append(f"⚠ {report.error_summary}")
    else:
        # 冒烟没全中要显眼：它意味着"发布出去了，但召回不到自己"，
        # 而这正是 11.7 的检索质量在这个语料上最早能露出的马脚
        failed = [item for item in report.smoke if not item.matched]
        if failed:
            parts.append(f"⚠ 冒烟未命中 {len(failed)} 条：{failed[0].chunk_id}")
    return "｜".join(parts)


def _print_ingest_summary(reports: list[IngestionReport], failures: list[tuple[str, str]]) -> None:
    active = sum(1 for r in reports if r.ok and not r.skipped)
    skipped = sum(1 for r in reports if r.skipped)
    unsupported = sum(1 for r in reports if not r.ok and r.unsupported)
    broken = sum(1 for r in reports if not r.ok and not r.unsupported)
    print()
    print("─" * 72)
    aborted = len(failures) - broken - unsupported
    print(
        f"汇总：发布 {active} 篇｜幂等跳过 {skipped} 篇"
        f"｜不支持 {unsupported} 篇｜入库失败 {broken} 篇｜异常中止 {aborted} 篇"
    )
    # **chunk 合计只数本次真的写了向量的那些**：跳过的文档把库里的旧值带回来了，
    # 把它算进来会让"合计"在一条什么都没做的命令上显示一个非零数字
    chunks = sum(r.chunk_count for r in reports if not r.skipped)
    dropped = sum(len(r.dropped) for r in reports if not r.skipped)
    smoke_total = sum(len(r.smoke) for r in reports)
    smoke_hit = sum(1 for r in reports for item in r.smoke if item.matched)
    print(f"chunk 合计 {chunks}｜清洗丢弃 {dropped} 项｜冒烟命中 {smoke_hit}/{smoke_total}")
    if failures:
        # **失败明细必须打全**：语料里 2 份扫描件注定失败，
        # 而"失败 2 篇"这个数字看不出它们是"预期内的不支持"还是"真的坏了"
        print("明细（未发布的每一篇）：")
        for name, reason in failures:
            print(f"  - {name}：{reason}")


async def _retrieve(settings: Settings, args: argparse.Namespace) -> int:
    """跑一次完整的自然语言 → 混合检索 → 文档证据（开发流程 6.7 的验证命令）。

    **这条命令是 Phase 5 检索侧的门禁本身**：它走的是与将来 Worker 完全相同的
    `RagRetrieveTool`，而不是一份为演示写的简化逻辑——那种写法只能证明
    "演示脚本能跑"，证明不了"工具能跑"。

    `--as-of` 与 `--doc-type` 模拟 Intent 提取出来的过滤条件。
    不传 `--as-of` 就是"不限时点"，那时同名制度的两个版本会同时进候选——
    这正是 16.11.2 的 VERSION_PAIR 缺陷要暴露的场景，所以默认值留给调用方定。
    """
    engine = create_engine(settings)
    redis = create_client(settings)
    storage = build_object_storage(settings)
    vector_store = build_vector_store(settings)
    gateway = build_model_gateway(settings)
    try:
        sessions = create_session_factory(engine)
        tool = await build_rag_retrieve_tool(
            settings,
            gateway,
            vector_store=vector_store,
            storage=storage,
            vocab=SqlVocabRepository(sessions),
            cache=VersionedCache(redis, default_ttl_seconds=settings.rag.vocab_cache_ttl_seconds),
        )
        ctx = build_cli_context(region_ids=(), timeout_seconds=settings.task_timeout_seconds)
        result = await tool.execute(
            RagQueryArgs(
                question=args.question,
                document_types=tuple(args.doc_type or ()),
                departments=tuple(args.department or ()),
                as_of=_parse_date_arg(args.as_of),
            ),
            ctx,
        )
    finally:
        await gateway.aclose()
        await vector_store.aclose()
        await redis.aclose()
        await engine.dispose()

    _print_retrieval(settings, args, result)
    return 0 if result.status != "FAILED" else 1


def _print_retrieval(settings: Settings, args: argparse.Namespace, result: ToolResult) -> None:
    payload = result.payload or {}
    print(f"问题：{args.question}")
    queries = payload.get("queries") or []
    degraded = payload.get("rewrite_degraded")
    print(f"实际查询（{len(queries)} 条{'，**改写已降级**' if degraded else ''}）：")
    for query in queries:
        print(f"  - {query}")
    if args.as_of:
        print(f"生效时点：{args.as_of}")
    print(
        f"相关性：最高余弦 {payload.get('best_dense_score', 0):.4f}"
        f"｜阈值 {payload.get('relevance_threshold')}"
        f"｜候选 {payload.get('candidate_count')} 条"
        f"｜耗时 {result.duration_ms}ms"
    )
    # 重排那一行**只在它生效时打**，但要打清楚：没生效时"召回变差了"的
    # 第一嫌疑就是它（开没开、还是打了服务但失败了），而结果里本来看不出来
    print(f"重排：{_rerank_line(payload)}")
    print()

    if result.status == "FAILED":
        error = result.error
        assert error is not None
        print(f"✗ {error.code}：{error.message}")
        print(f"  类别 {error.error_class}｜可重试 {error.retryable}")
        if error.safe_detail:
            print(f"  详情（仅诊断）：{error.safe_detail}")
        return

    # **逐条打印定位信息与分数**：检索质量变差时，第一件要看的是
    # "召回的是哪几块、为什么是它们"。只打正文的话，两份制度的相似段落
    # 长得几乎一样，光看正文分不出召回错了哪一份。
    for chunk in payload.get("chunks") or []:
        dense = chunk.get("dense_score")
        print(
            f"[{chunk['rank']}] {chunk['chunk_id']}"
            # 第 ⑧ 步补回来的行块没有检索分，打 `—` 而不是 0.0000：
            # 两者必须长得不一样，否则"没评过"会被读成"评了、垫底"
            f"｜融合 {_score_text(chunk.get('fusion_score'))}"
            f"｜余弦 {_score_text(dense)}"
            # 重排分**逐条打出来**：它是"为什么这条排在前面"的直接答案，
            # 而候选顺序变了却看不出原因时，第一件要查的就是它
            f"｜重排 {_score_text(chunk.get('rerank_score'))}"
        )
        meta = chunk.get("metadata") or {}
        path = " > ".join(meta.get("section_path") or [])
        page = f"p{meta['page_no']}" if meta.get("page_no") else "—"
        # 表格行块**打出它在整张表里的位置**：表格按行分块，光看这一条
        # 分不出它是"整张表"还是"表里的一行"，而两者的差别正是
        # "能不能对它求和"。`—` 表示这不是表格的一部分。
        row = chunk.get("table_row")
        table = f"｜表第 {row[0]}/{row[1]} 行（{meta.get('table_caption')}）" if row else ""
        print(
            f"    {path}｜{page}｜{meta.get('source_kind')}/{meta.get('classification')}"
            f"｜生效 {meta.get('effective_from') or '—'} ~ {meta.get('effective_to') or '—'}"
            f"{table}"
        )
        for line in str(chunk.get("text", "")).splitlines()[:4]:
            print(f"    {line}")
        print()
    print(f"证据 {len(result.evidence)} 条（source_type={_evidence_sources(result)}）")


def _evidence_sources(result: ToolResult) -> str:
    return "、".join(sorted({item.source_type for item in result.evidence})) or "—"


def _score_text(value: float | None) -> str:
    """分数列的可空打印。**空值与 0 必须长得不一样**：0 表示"模型判它不相关"，
    空表示"这次没有这个分数"（重排没开或失败）——那是两件事。"""
    return "—" if value is None else f"{value:.4f}"


def _rerank_line(payload: dict[str, Any]) -> str:
    """重排那一行的文案。**"没开"与"打了但失败了"要一眼分得开**：

    - 没开：证据按 RRF 顺序取，这是一条**正确**的路径（11.7 第 ⑤ 步的直接结果）；
    - 开了但失败：说明服务或配置出了问题，而候选顺序已经悄悄退回去了。

    两者在结果里都表现为"没有重排分"，只看分数分不出来——所以原因串要打出来。
    """
    if payload.get("rerank_applied"):
        best = payload.get("best_rerank_score")
        return (
            f"已生效｜最高分 {'—' if best is None else f'{best:.4f}'}"
            f"｜阈值 {payload.get('rerank_threshold')}"
            f"｜剔除 {payload.get('rerank_pruned')} 条"
        )
    reason = payload.get("rerank_skipped_reason") or "—"
    return f"未生效（{reason}）｜证据按 RRF 顺序取"


def _parse_date_arg(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


async def _verify_corpus(settings: Settings, args: argparse.Namespace) -> int:
    """逐条检出 10 类缺陷注入（开发流程 6.7 的门禁）。

    **断言的是产物，不是清单标注**（CLAUDE.md 约定 11）：清单里写
    `defects: [CROSS_PAGE_TABLE]` 只是一句声明，第一版语料就是三份标注齐全、
    一张都没跨页。所以这里的输入全部来自 `make corpus` 与 `make ingest` 的产物。

    **两类标注要分清**（见 `tools/rag/verify.py` 的模块说明）：
    `[OK]` 是"在这里就检出了"，`[注入]` 是"语料侧确认注入了、但对应的冲突
    检出属 Phase 9"。把后者标成 `[OK]` 就是把"语料里有"说成"系统检得出"。
    """
    documents = verify._load_manifest()
    engine = create_engine(settings)
    redis = create_client(settings)
    storage = build_object_storage(settings)
    vector_store = build_vector_store(settings)
    gateway = build_model_gateway(settings)
    try:
        sessions = create_session_factory(engine)
        documents_repo = SqlKnowledgeDocumentRepository(sessions)
        records = await documents_repo.list_documents()
        chunks = await _scroll_all_chunks(vector_store)
        # 直接用 `Retriever` 而不是 `RagRetrieveTool`：这条命令要的是
        # "召回/拒答的判定"，不需要 ToolResult 那层封装（证据、call_id、错误码
        # 都不是它要的）。走工具的话还得把 payload 反解一遍，
        # 而那条反解路径正是评测脚本刻意避开的东西。
        retriever = await build_retriever(
            settings,
            gateway,
            vector_store=vector_store,
            storage=storage,
            vocab=SqlVocabRepository(sessions),
            cache=VersionedCache(redis, default_ttl_seconds=settings.rag.vocab_cache_ttl_seconds),
        )
        golden = load_golden()
        corpus_text = "\n".join(chunk["payload"].get("text") or "" for chunk in chunks)
        provinces = await _region_province_counts(settings)

        results = [
            verify.check_furniture(chunks),
            await verify.check_version_pair(documents, chunks, retriever),
            verify.check_value_conflict(documents, chunks),
            verify.check_time_conflict(documents, chunks),
            verify.check_scope_conflict(documents, chunks, provinces),
            verify.check_source_pair(documents, records, chunks),
            verify.check_scanned(documents, records, chunks),
            verify.check_cross_page_table(documents, chunks),
            verify.check_prompt_injection(documents, chunks),
            await verify.check_absent(golden.absent_cases, corpus_text, retriever),
        ]
    finally:
        await gateway.aclose()
        await vector_store.aclose()
        await redis.aclose()
        await engine.dispose()

    return _print_verify_results(results)


async def _scroll_all_chunks(vector_store: Any) -> list[dict[str, Any]]:
    """把全部 chunk 连同 payload 取回来。

    **不走 `VectorStore` 接口**：那个接口刻意没有"列出全部"（把向量库当目录
    服务用是错的方向，见 `vector_store.py` 的说明）——日常读路径都该走
    MySQL 的 `knowledge_document` + 检索。这里是**一次性的人工核验**，
    与检索的质量无关，所以要的是"库里到底有什么"，绕开接口去 Qdrant 取正是意图。
    """
    points: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        batch, offset = await vector_store._client.scroll(
            vector_store.collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            # **要显式关掉向量**：默认会把 1024 维的稠密向量一起取回来，
            # 1871 个 chunk 就是几百万个浮点数——本条命令一个都用不上
            with_vectors=False,
        )
        points.extend({"payload": dict(point.payload or {})} for point in batch)
        if offset is None:
            return points


async def _region_province_counts(settings: Settings) -> dict[str, int]:
    """业务库 `dim_region` 的真实省份数，供 SCOPE 冲突比对。

    这一条是 verify-corpus 里**唯一需要读业务库的断言**，而它必须在：
    SCOPE 冲突的定义就是"文档说的"与"库里是的"不一致，
    只看文档那一侧永远判不出冲不冲突。
    """
    engine = create_engine(settings, url=settings.database_url_business_ro)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(text("SELECT region_name, province_count FROM dim_region"))
            ).all()
        return {str(row[0]): int(row[1]) for row in rows}
    finally:
        await engine.dispose()


def _print_verify_results(results: Sequence[Any]) -> int:
    width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.mark:6} {result.name:<{width}}  {result.expected:14} {result.detail}")
    failed = [result for result in results if not result.ok]
    injected = [result for result in results if result.ok and result.injected_only]
    print()
    print("─" * 72)
    print(
        f"检出 {len(results) - len(failed)}/{len(results)} 类"
        f"｜其中 {len(injected)} 类只验证了语料侧（冲突检出属 Phase 9）"
    )
    if failed:
        print("未通过：")
        for result in failed:
            print(f"  - {result.name}：{result.detail}")
    return 1 if failed else 0


def _print_attempts(payload: dict[str, Any] | None) -> None:
    """打印每次尝试。**失败的尝试也要打**——详设 6.6 施工项 5 要求每次尝试
    都落 `agent_tool_call`，而在落库之前，这里是唯一能看到「修复了几次、
    每次为什么失败」的地方。"""
    attempts: list[dict[str, Any]] = (payload or {}).get("attempts") or []
    if not attempts:
        return
    print(f"尝试记录（{len(attempts)} 次，落 agent_tool_call）：")
    for item in attempts:
        detail = item.get("error_summary") or ""
        print(
            f"  #{item.get('attempt_no')} {item.get('stage')} {item.get('status')}"
            f"｜{item.get('error_code') or '—'}｜{detail}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli", description="项目运维与验证命令")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed", help="灌入演示账号与业务演示数据（幂等）")
    sub.add_parser("cleanup", help="按保留期清理过期数据（幂等，供 cron 调用）")
    sub.add_parser("model-smoke", help="打一次真实模型与向量化，验证连通性与结构化输出")
    sql = sub.add_parser("sql", help="跑一次自然语言 → SQL → 真数据 → 证据（Phase 4 门禁）")
    sql.add_argument(
        "question",
        nargs="?",
        default="",
        help="业务问题，如「2025 年华东地区 Q3 净销售额是多少」",
    )
    sql.add_argument(
        "--sql",
        default="",
        metavar="SQL",
        help="直接给一条 SQL：跳过生成，但仍走全部 12 步校验与只读执行。"
        "用于单独验证「安全由代码保证」与跑金标 SQL",
    )
    sql.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="名称",
        help="模拟数据权限范围（可重复）。不传即全量，对应 admin 账号",
    )
    sql.set_defaults(region=[])
    tokenize = sub.add_parser("tokenize", help="逐条核对中文分词结果（Phase 5 调试命令）")
    tokenize.add_argument(
        "text",
        nargs="?",
        default="",
        help="待切分的文本，如「华东区域渠道折扣政策」",
    )
    tokenize.add_argument(
        "--no-dict",
        action="store_true",
        help="不加载业务词典：用于判断某个术语是不是靠词典才切对的",
    )
    tokenize.set_defaults(no_dict=False)
    chunk = sub.add_parser("chunk", help="解析 + 分块，逐条核对结果（详细设计 11.3）")
    chunk.add_argument("path", nargs="?", default="", help="文件或目录，如 data/corpus/SP-015.pdf")
    chunk.add_argument("--summary", action="store_true", help="只打每篇的汇总，不打明细")
    chunk.add_argument(
        "--limit", type=int, default=0, metavar="N", help="每篇最多显示 N 块，0 为全部"
    )
    group = chunk.add_mutually_exclusive_group()
    group.add_argument("--table-only", action="store_true", help="只看表格块")
    group.add_argument("--text-only", action="store_true", help="只看正文块")
    chunk.set_defaults(summary=False, limit=0, table_only=False, text_only=False)
    vocab = sub.add_parser("vocab", help="构建稀疏检索词表并导出快照（幂等，只增不改）")
    vocab.add_argument("path", nargs="?", default="data/corpus", help="语料目录，默认 data/corpus")
    ingest = sub.add_parser("ingest", help="逐篇入库并发布（详细设计 11.1 / 11.9）")
    ingest.add_argument("path", nargs="?", default="data/corpus", help="语料目录，默认 data/corpus")
    ingest.add_argument(
        "--only",
        default="",
        help="只入库指定 ID，逗号分隔，如 SP-001,MD-008",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="同一版本号下内容变了时允许原地重建。默认拒绝——"
        "改内容必须升版本号，否则引用过该版 chunk_id 的证据会指向另一段文字",
    )
    ingest.set_defaults(only="", force=False)
    retrieve = sub.add_parser("retrieve", help="混合检索并产出文档证据（详细设计 11.7）")
    retrieve.add_argument(
        "question", nargs="?", default="", help="业务问题，如「华东区域渠道折扣政策怎么规定」"
    )
    retrieve.add_argument(
        "--as-of",
        default="",
        metavar="YYYY-MM-DD",
        help="问题所问的时点，用于生效区间过滤。不传即不限时点——"
        "那时同名制度的两个版本会同时进候选（VERSION_PAIR 的场景）",
    )
    retrieve.add_argument(
        "--doc-type", action="append", default=[], metavar="类型", help="文档类型过滤（可重复）"
    )
    retrieve.add_argument(
        "--department", action="append", default=[], metavar="部门", help="部门过滤（可重复）"
    )
    retrieve.set_defaults(as_of="", doc_type=[], department=[])
    sub.add_parser("verify-corpus", help="逐条检出 10 类缺陷注入（详细设计 6.7 的门禁）")
    args = parser.parse_args(argv)

    if args.command == "tokenize" and not args.text:
        parser.error('tokenize 需要一段文本，例如：tokenize "华东区域渠道折扣政策"')

    if args.command == "chunk" and not args.path:
        parser.error("chunk 需要一个文件或目录，例如：chunk data/corpus/SP-015.pdf")

    if args.command == "retrieve" and not args.question:
        parser.error('retrieve 需要一个问题，例如：retrieve "华东区域渠道折扣政策怎么规定"')

    if args.command == "sql" and not args.question and not args.sql:
        # 两个入口都没有时**必须报错**，不能让 `make sql` 空跑一次模型。
        # argparse 的 `nargs="?"` 做不到「二者恰有其一」，因此在这里补一条校验。
        parser.error(
            'sql 需要一个问题，或用 --sql 直接给一条 SQL，例如：sql "2025 年华东 Q3 净销售额"'
        )

    settings = get_settings()
    setup_logging(SERVICE_CLI, settings.log_level)

    handlers: dict[str, Callable[[Settings, argparse.Namespace], HandlerResult]] = {
        "seed": _seed,
        "cleanup": _cleanup,
        "model-smoke": _model_smoke,
        "sql": _sql,
        "tokenize": _tokenize,
        "chunk": _chunk,
        "vocab": _vocab,
        "ingest": _ingest,
        "retrieve": _retrieve,
        "verify-corpus": _verify_corpus,
    }
    # 所有子命令都收 (settings, args)：让签名统一，代价只是几个用不到 args 的
    # 函数多一个参数；不统一的话分发处就得按命令名分支，加一个命令改一次。
    #
    # 分发容忍同步实现：`tokenize` 是纯计算，给它套一层 `async def` 只会让读的人
    # 去找它究竟在等什么。**参数与返回值约定仍然统一**，差别只在要不要 await。
    outcome = handlers[args.command](settings, args)
    if isinstance(outcome, Coroutine):
        # 显式标注而不是直接 return：`asyncio.run` 的返回类型是 Any，
        # 直接 return 会让 mypy 的这个检查点在整条链路上失效
        code: int = asyncio.run(outcome)
        return code
    return outcome


if __name__ == "__main__":
    sys.exit(main())
