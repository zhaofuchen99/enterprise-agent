"""生成 jieba 业务分词词典（开发流程 6.7 施工项 1 / 详细设计 11.6.2）。

产出 `configs/rag_user_dict.txt`，由 `Tokenizer` 在进程启动时加载。

## 为什么词典要"生成"而不是"手写一份"

详设 11.6.2 要求词典**与 `schema_catalog` 一同版本化发布**。手写一份的后果是：
业务库加了产品线、指标目录加了新指标，词典不会跟着变，
而**没有任何东西会报错**——表现是"某个产品的检索忽然变差了"，
排查方向会先跑到向量库、再跑到分词，最后才发现是词典漏了一个词。

所以词典的三个来源里有两个是自动的，且都以**已发布的配置/数据**为准：

1. **业务库维度表取值**：`dim_region` / `dim_channel` / `dim_product_line` /
   `dim_product`。取的是**全部取值**，不做人工筛选——"哪些产品名重要"
   不该由人来判断，库里有的就是业务上存在的。走只读账号，与 SQL Tool 同一条路径。
2. **`schema_catalog.yaml` 的指标名与别名**：指标目录是 SQL 侧的口径来源，
   语料与用户提问都按它措辞，两边必须切得一样。
3. **`configs/rag_terms.txt` 手工术语**：自动来源覆盖不到的复合词。

## 产物为什么入版本库（而语料不入）

`data/corpus/` 是 gitignore 的产物，词典却是提交的。差别在**它是不是检索契约的一部分**：

- 词典决定查询侧与入库侧切出什么 token。两侧用同一份词典时结果一致；
  一旦不一致（比如新克隆的仓库没有词典文件），`Tokenizer` 会静默退化成通用分词，
  而**入库的 chunk 是用带词典的分词建的**——稀疏向量从此对不上，
  表现为"某些查询永远召回不到东西"，且两边代码看起来都对。
- 它只有一百多行文本，diff 可读，与 `schema_catalog.yaml` 的评审方式一致。

结论：**提交，且以本脚本为唯一写入方**。改词典要走 review，
并重跑 RAG 检索回归集（11.6.2 明写）。

## 关于产物文件里的注释（实测结论）

**不能写。** jieba 的 `load_userdict` 对每一行跑
`re_userdict.match(line)` → `^(.+?)( [0-9]+)?( [a-z]+)?$`，
匹配不上词频/词性后缀的行会被 `.+?` 整个吃掉、当作**词条**加进词典。
实测 `# 来源：schema_catalog v2026.09.17-1` 这一行，
`jieba.get_FREQ()` 返回 1——它真的成了一个词。
于是文件头再写一遍来源说明，等于往词典里塞一个垃圾词条。

所以：**出处写在本模块与 `configs/rag_terms.txt` 的注释里，产物只写词。**

用法：
    uv run python scripts/gen_dict.py            # 生成并自检
    uv run python scripts/gen_dict.py --check    # 只比对，不写文件（CI/提交前用）
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import jieba
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.tools.rag.tokenizer import normalize, normalize_numbers

ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = ROOT / "configs" / "rag_user_dict.txt"
TERMS_PATH = ROOT / "configs" / "rag_terms.txt"
CATALOG_PATH = ROOT / "configs" / "schema_catalog.yaml"


@dataclass
class Terms:
    """按来源分组的候选词。**分组是为了报告**：出问题时要知道该去改哪一处。"""

    regions: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    product_lines: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    handwritten: list[str] = field(default_factory=list)

    def auto_groups(self) -> dict[str, list[str]]:
        return {
            "区域": self.regions,
            "渠道": self.channels,
            "产品线": self.product_lines,
            "品类": self.categories,
            "产品名": self.products,
            "指标名": self.metrics,
        }


async def load_dimension_terms() -> dict[str, list[str]]:
    """从业务库读维度表取值。用只读账号——与 SQL Tool、语料生成器同一条路径。"""
    settings = get_settings()
    engine = create_async_engine(settings.database_url_business_ro)
    queries = {
        "regions": "SELECT region_name AS v FROM dim_region",
        "channels": "SELECT channel_name AS v FROM dim_channel",
        "product_lines": "SELECT product_line_name AS v FROM dim_product_line",
        "categories": "SELECT DISTINCT category AS v FROM dim_product_line",
        "products": "SELECT product_name AS v FROM dim_product",
    }
    result: dict[str, list[str]] = {}
    try:
        async with engine.connect() as conn:
            for key, sql in queries.items():
                rows = (await conn.execute(text(sql))).scalars().all()
                result[key] = sorted({str(v).strip() for v in rows if str(v).strip()})
    finally:
        await engine.dispose()
    return result


def load_metric_terms() -> list[str]:
    """指标名与别名。

    **别名必须一起收**：`schema_catalog` 里 `净销售额` 的别名含 `收入`、`销售额`，
    用户按别名提问时要落到同一个 token 上；只收规范名会让别名问法在稀疏路上失配。
    """
    payload = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    terms: set[str] = set()
    for metric in payload.get("metrics") or []:
        if metric.get("name"):
            terms.add(str(metric["name"]))
        for alias in metric.get("aliases") or []:
            terms.add(str(alias))
    return sorted(terms)


def load_handwritten_terms(path: Path = TERMS_PATH) -> list[str]:
    """读手工术语表。`#` 开头为注释、空行忽略——**这份文件是给人看的输入**，
    不像产物那样受 jieba 解析规则约束，所以可以带注释。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    terms: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped not in terms:
            terms.append(stripped)
    return terms


async def collect_terms() -> Terms:
    dims = await load_dimension_terms()
    return Terms(
        regions=dims["regions"],
        channels=dims["channels"],
        product_lines=dims["product_lines"],
        categories=dims["categories"],
        products=dims["products"],
        metrics=load_metric_terms(),
        handwritten=load_handwritten_terms(),
    )


def canonical(term: str) -> str:
    """词条在入库/查询链路上真正会遇到的形态。

    **必须与 `Tokenizer.cut` 的前两步完全一致**（`normalize_numbers(normalize(text))`），
    否则词典里的词与待切文本不是同一个字符串，永远匹配不上：
    比如 `KA` 这类全角/大写写法，`normalize` 之后变小写 `ka`，
    词典里若存着 `KA`，加载成功、日志无异常、检索却一条都召回不到。
    **不在这里抄一份归一化逻辑**——直接调真实现，两边不可能漂移。
    """
    return normalize_numbers(normalize(term))


def build_dict_text(terms: Terms) -> tuple[str, list[str]]:
    """拼产物文本，返回 (文本, 被剔除的重复词)。"""
    ordered: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for group in (*terms.auto_groups().values(), terms.handwritten):
        for term in group:
            word = canonical(term)
            # 归一化后可能是空串（理论上不会，但空行会污染产物）或纯标点
            if not word or not any(ch.isalnum() for ch in word):
                continue
            if word in seen:
                duplicates.append(word)
                continue
            seen.add(word)
            ordered.append(word)
    return "".join(f"{w}\n" for w in ordered), duplicates


def baseline_split(term: str) -> list[str]:
    """通用词典（未加载业务词典）下的切分结果。

    **必须在加载业务词典之前调用**：`jieba.load_userdict` 改的是进程级全局状态，
    加载之后再问"本来会怎么切"，问到的已经是加过词典的结果。
    """
    return list(jieba.lcut(term))


def verify(words: Iterable[str], *, baseline: dict[str, list[str]]) -> list[str]:
    """回读产物自检：加载词典后，每个词条都必须能被切成**它自己**。

    这条检查是必要的，理由与语料生成器的缺陷注入自检相同（CLAUDE.md 约定 11）：
    **"清单里写了"不是证据，"产物里生效了"才是**。一个没生效的词条不会报错，
    只会让某个查询悄悄召回不到东西，而排查方向会先跑到向量库上去。

    返回未生效的词条清单（空表示全部通过）。
    """
    failures: list[str] = []
    for word in words:
        if jieba.lcut(word) != [word]:
            failures.append(f"{word} -> {jieba.lcut(word)}（期望切成它自己）")
        elif baseline.get(word) == [word]:
            # 通用词典本来就切得对：这个词条是死条目，占着位置却不产生任何效果。
            # 不报错（自动来源按规则全收，不该因为"碰巧切得对"就漏掉某个产品名），
            # 但要在报告里点出来，让手工表的维护者把它删掉。
            failures.append(f"__dead__{word}")
    return failures


async def main_async(args: argparse.Namespace) -> int:
    sources = " + ".join(
        (TERMS_PATH.relative_to(ROOT).as_posix(), CATALOG_PATH.relative_to(ROOT).as_posix())
    )
    print(f"词典来源：{sources} + 业务库维度表")
    terms = await collect_terms()

    auto_counts = terms.auto_groups()
    print("  " + " / ".join(f"{k} {len(v)}" for k, v in auto_counts.items()))
    print(f"  手工术语 {len(terms.handwritten)}")

    dict_text, duplicates = build_dict_text(terms)
    words = [line for line in dict_text.splitlines() if line]

    # 基线必须在 load_userdict 之前算：它问的是"没有业务词典时会怎么切"
    baseline = {w: baseline_split(w) for w in words}
    dead_auto = {w for w, cut in baseline.items() if cut == [w]}
    hand_words = {canonical(t) for t in terms.handwritten}

    # 自检验证的是**即将写出的内容**，不是磁盘上的旧文件——
    # 否则"改完词典但没生效"这种情况恰好会被漏掉（旧文件是好的，新内容没人验）。
    jieba.load_userdict(io.StringIO(dict_text))
    problems = verify(words, baseline=baseline)
    failures = [p for p in problems if not p.startswith("__dead__")]
    dead_hand = sorted(
        w
        for w in (p[len("__dead__") :] for p in problems if p.startswith("__dead__"))
        if w in hand_words
    )

    print(f"\n词条合计 {len(words)}（自动来源里 {len(dead_auto)} 条通用词典已切得对，不影响结果）")
    if duplicates:
        print(f"去重 {len(set(duplicates))} 条：{'、'.join(sorted(set(duplicates)))}")

    if failures:
        print(f"\n**{len(failures)} 个词条加载后仍未生效**——词典不可用：")
        for item in failures:
            print(f"  - {item}")
        return 1

    if dead_hand:
        # 手工表里的死条目**报错**：手工表是按"通用词典切错"这个判据收词的，
        # 一条死条目意味着判据不成立（词写错了、或通用词典已经改好），
        # 留着会让后来的人以为它有作用。
        print(
            f"\n**{len(dead_hand)} 个手工术语是死条目**（通用词典已能正确切分），请从 "
            f"{TERMS_PATH.relative_to(ROOT)} 删除："
        )
        for word in dead_hand:
            print(f"  - {word}")
        return 1

    current = DICT_PATH.read_text(encoding="utf-8") if DICT_PATH.exists() else ""
    if current == dict_text:
        print(f"\n产物无变化：{DICT_PATH.relative_to(ROOT)}")
        return 0

    if args.check:
        print(f"\n**产物与词典来源不一致**：{DICT_PATH.relative_to(ROOT)} 需要重新生成")
        print("  执行 make dict 后一并提交；词典决定切分结果，改动必须随检索回归集一起 review。")
        return 1

    DICT_PATH.write_text(dict_text, encoding="utf-8")
    print(f"\n已写入 {DICT_PATH.relative_to(ROOT)}（{len(words)} 条）")
    print("自检通过：全部词条加载后都能切成它自己。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 jieba 业务分词词典（详设 11.6.2）")
    parser.add_argument(
        "--check",
        action="store_true",
        help="不写文件，只在产物与来源不一致时非零退出（提交前/CI 用）",
    )
    return asyncio.run(main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
