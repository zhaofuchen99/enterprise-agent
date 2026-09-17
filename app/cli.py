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
from pathlib import Path
from typing import Any, NamedTuple

from app.agent.prompts import SMOKE_PROMPT
from app.agent.schemas import SmokeAnswer
from app.core.config import Settings, get_settings
from app.core.errors import AgentError
from app.infrastructure.db import create_engine, create_session_factory
from app.infrastructure.logging import SERVICE_CLI, setup_logging
from app.infrastructure.model_gateway import EmbeddingDimensionError, build_model_gateway
from app.infrastructure.storage import build_object_storage
from app.repositories.user_repo import SqlUserRepository, seed_demo_users
from app.repositories.vocab_repo import SqlVocabRepository
from app.tools.rag.chunker import Chunk, chunk_document
from app.tools.rag.parser import SUFFIX_TO_FORMAT, ParsedDocument, parse_document
from app.tools.rag.tokenizer import Tokenizer, load_stopwords, normalize, normalize_numbers
from app.tools.rag.vocabulary import build_vocabulary, export_snapshot
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
    report = _CORPUS_ROOT / "corpus_report.json"
    if report.exists():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("documents") or []:
            if Path(item["path"]).resolve() == file.resolve():
                return _DocMeta(
                    id=item["id"],
                    title=item["title"],
                    key=f"{item['logical_key']}@{item['version']}",
                )
    return _DocMeta(id=file.stem, title=file.stem, key=f"{file.stem}@v1.0")


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
    args = parser.parse_args(argv)

    if args.command == "tokenize" and not args.text:
        parser.error('tokenize 需要一段文本，例如：tokenize "华东区域渠道折扣政策"')

    if args.command == "chunk" and not args.path:
        parser.error("chunk 需要一个文件或目录，例如：chunk data/corpus/SP-015.pdf")

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
