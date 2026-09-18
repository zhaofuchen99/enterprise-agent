"""中文分词与稀疏向量构建（详细设计 11.6）。

## 这一层为什么必须自建

详设 3.3 拒绝了 Elasticsearch，理由是「Qdrant 已承担 Dense + Sparse 混合检索，
80+ 文档规模不需要第二套搜索集群」。**这个决定有一份明码代价**：
Qdrant 的稀疏检索同样不自带中文分词，它接受的是 `{token_id: weight}`，
token 从哪来、id 怎么分配、权重怎么算，全是本系统的责任。
不能用"向量库支持 BM25"一笔带过——那正是详设 3.3 明确点名要避免的含混说法。

对照一下默认分词的实测结果就清楚了：

    华东地区渠道折扣政策  →  华东地区 / 渠道 / 折扣 / 政策

「渠道折扣」被切成了两个词。制度里它是个完整术语，检索时也必须是一个 token，
否则用户搜「渠道折扣」会把所有含「渠道」或「折扣」的段落一并召回，
精确性直接丢失。这就是 11.6.2 要求加载业务自定义词典的原因。

## 处理链路（11.6.1）

    归一化 → jieba（通用词典 + 业务词典）→ 停用词过滤
      → token → token_id 映射 → 稀疏向量 {token_id: weight}

## 固定 IDF（11.6.4）

稀疏向量在**入库时就被固定下来**，而 BM25 的 IDF 依赖全库统计信息——
新文档进来会让 IDF 变化，若每次都用最新值重算，已入库的向量就与新的不可比。
本实现选**固定 IDF**：IDF 来自一份冻结的快照，入库与查询两侧读同一份。

这个取舍之所以成立，前提是 11.6.5 的 RRF：**融合只依赖排名、不依赖分数绝对值**，
所以 IDF 的偏差不会被放大到结果里。如果哪天改成加权分数融合，
这个选择就必须重新评估——两处是一起成立的，不是各自独立的决定。
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import jieba

from app.core.config import Settings

logger = logging.getLogger(__name__)

#: 词表快照的格式版本。**改格式必须升它**：快照是冻结产物，
#: 老快照配上新代码却按新格式解析，会得到一份"能跑但 id 全错"的词表。
SNAPSHOT_FORMAT_VERSION = 1

#: 未登录词（不在快照里的 token）的 IDF 取值。
#:
#: 取 `log(N/1)` 量级的高值而不是 0：新词恰恰是最有区分度的词（制度编号、
#: 新产品线名），给 0 等于让它们在稀疏检索里彻底消失。
#: 这里用一个"看起来像只出现在 1 篇文档里"的固定值，是刻意的保守选择。
_UNSEEN_TOKEN_DF = 1

#: 归一化的替换表：全角标点与空白 → 半角。
#: **只处理与检索相关的字符**：全角句号、逗号、括号在中文字符串里与半角
#: 在语义上是一回事，但字面上不同，不归一就会出现"搜半角找不到全角"。
_PUNCT_NORMALIZATION = {
    "，": ",",
    "。": ".",
    "、": ",",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "「": '"',
    "」": '"',
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "－": "-",
    "—": "-",
    "～": "~",
}

#: 数字归一：中文数字 → 阿拉伯数字。
#: 制度里"三成""百分之十五"与报告里"30%""15%"是同一件事，
#: 不归一就永远检索不到一起（这是 16.11.2 的"报告数字与 DB 数字差异"
#: 那类缺陷能被检出的前提之一）。
_CN_DIGITS = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}

#: 数词单位。`万` 单独处理是因为它是"进位"而不是"倍乘"——
#: 见 `_parse_cn_numeral`。
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10_000}

#: 可能构成数词的字符集合（数字 + 单位）。正则用它圈出候选区间，
#: 再交给 `_parse_cn_numeral` 判断到底是不是数词——**两者必须一起改**。
_CN_NUMERAL_CHARS = "".join(_CN_DIGITS) + "".join(_CN_UNITS)

#: 日期归一：`2025年7月1日` / `2025-07-01` / `2025/7/1` → `2025-07-01`
_DATE_PATTERNS = (
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"), r"\1-\2-\3"),
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月"), r"\1-\2"),
    (re.compile(r"(\d{4})[/.](\d{1,2})[/.](\d{1,2})"), r"\1-\2-\3"),
)


def normalize(text: str) -> str:
    """归一化（11.6.1 第一步）。

    顺序有讲究：**先做 Unicode 全角/半角折叠，再套业务替换表**。
    反过来的话，`NFKC` 会把我刚替换好的半角字符又按它自己的规则处理一遍，
    结果是替换表里那些刻意保留的差异被抹平——这类"看起来等价"的顺序问题
    不会报错，只会让某些查询悄悄召回不到东西。
    """
    # NFKC 会把全角字母数字折成半角，并统一大部分兼容字符
    folded = unicodedata.normalize("NFKC", text)
    for fullwidth, halfwidth in _PUNCT_NORMALIZATION.items():
        folded = folded.replace(fullwidth, halfwidth)
    for pattern, replacement in _DATE_PATTERNS:
        folded = pattern.sub(replacement, folded)
    folded = folded.replace("　", " ")  # 全角空格
    # 空白压缩：制表符、换行、连续空格统一成单个空格
    folded = re.sub(r"\s+", " ", folded)
    # 大小写统一放在最后：前面几步可能引入新的大写字符
    return folded.strip().lower()


def normalize_numbers(text: str) -> str:
    """中文数字 → 阿拉伯数字（11.6.1 的"数字与日期归一"）。

    **逐字替换是错的**，这一点值得单独说：把「三」→「3」这种一对一映射直接
    套上去，「百分之十五」会变成「百分之十5」——
    一个既不是中文也不是数字、**永远匹配不到任何东西**的 token。
    这类错误不会报错，只会让某些查询悄悄召回不到东西，
    比不做归一更糟。所以这里按**数词序列整体解析**。

    **与 `normalize` 分开是刻意的**：这一步会改变词形，让分词结果偏离原文，
    调试时看不出为什么某个词切没了。入库与查询两侧都要调用它，
    但调用方要能单独关掉做对照——`make tokenize` 就是这么用的。
    """
    # 先把「百分之X」整体换成 `X%`：它与报告里的 "15%" 是同一个意思，
    # 不统一则跨文档检索必然失配（16.11.2 那类数字差异缺陷也就无从比对）
    text = re.sub(r"百分之\s*([\d.]+)", r"\1%", text)
    text = re.sub(
        rf"百分之([{_CN_NUMERAL_CHARS}]+)",
        lambda m: f"{_parse_cn_numeral(m.group(1))}%",
        text,
    )
    return re.sub(
        rf"[{_CN_NUMERAL_CHARS}]+",
        lambda m: _replace_numeral(m.group(0)),
        text,
    )


def _replace_numeral(matched: str) -> str:
    """单个中文数词序列 → 阿拉伯数字；解析不出来就原样返回。

    **中文数词与普通字在字面上不可分**，所以一定会误伤：
    「一线城市」会变成「1线城市」，「一」在这里不是数词。
    这类误伤无法用规则彻底消除（要消歧必须上模型，代价远超收益）。

    之所以可以接受，是因为**归一化是对称的、纯函数**：
    入库与查询两侧对同一段文本必然得到同一个结果，
    所以「1线城市」这个 token 依然能被正常召回。
    **误伤影响的是可读性，不是检索能力**——
    真正危险的是"同一段文本两边归一化结果不同"，而纯函数从构造上排除了它。

    `_parse_cn_numeral` 返回 None（含非数词字符）时原样返回，
    不猜——猜错会引入不确定性，而这里最不需要的就是不确定性。
    """
    value = _parse_cn_numeral(matched)
    return matched if value is None else str(value)


def _parse_cn_numeral(text: str) -> int | None:
    """解析中文数词，支持到「万」量级；解析不了返回 None。

    能处理的形态：`十五`(15) `二十`(20) `一百零五`(105) `三千五百`(3500) `两`(2)。
    **不处理「零」开头的分数、也不处理小数点**——语料里没有，
    而且真出现时"原样保留"是安全的退化（见 `_replace_numeral`）。
    """
    total = 0
    section = 0
    number = 0
    for ch in text:
        digit = _CN_DIGITS.get(ch)
        if digit is not None:
            number = int(digit)
            continue
        unit = _CN_UNITS.get(ch)
        if unit is None:
            return None
        if unit == 10_000:
            # 万：把当前 section 进位到 total，再开新的 section
            total += (section + number) * unit
            section = 0
        else:
            # 「十五」的「十」前面没有数字，按 1 计
            section += (number or 1) * unit
        number = 0
    return total + section + number


class Tokenizer:
    """jieba + 业务词典 + 停用词（11.6.1 第二、三步）。

    **词典加载是全局副作用**：`jieba.load_userdict` 改的是 jieba 的进程级词典，
    不是这个对象的私有状态。因此多个 `Tokenizer` 实例会互相影响——
    这里用 `_loaded_paths` 记录已加载的路径来避免重复加载，
    但**不支持"不同的 Tokenizer 用不同的词典"**。

    之所以接受这个限制而不是给每个实例一份词典副本：一份 80 篇规模的语料，
    词典是全局一致的（它由 `schema_catalog` 版本化发布，11.6.2），
    要做多份词典得先把 jieba 的全局状态隔离掉，代价远大于收益。
    真需要时应该换分词器，而不是给 jieba 打补丁。
    """

    _loaded_paths: ClassVar[set[str]] = set()

    def __init__(self, *, user_dict_path: str | None = None, stopwords: Iterable[str] = ()) -> None:
        if user_dict_path:
            self._load_user_dict(user_dict_path)
        self._stopwords = frozenset(stopwords)

    @classmethod
    def from_settings(cls, settings: Settings) -> Tokenizer:
        return cls(
            user_dict_path=settings.rag.user_dict_path,
            stopwords=load_stopwords(settings.rag.stopword_path),
        )

    @classmethod
    def _load_user_dict(cls, path: str) -> None:
        resolved = str(Path(path).resolve())
        if resolved in cls._loaded_paths:
            return
        # 文件不存在时**不报错**：词典是可选增强，开发机上还没生成时
        # 应该退化成通用分词并继续，而不是让整个 Worker 起不来。
        # 但"没加载到"这件事必须能从日志看出来，否则检索变差没人能定位。
        if not Path(resolved).exists():
            # 这条 warning 不是装饰。词典缺失的表现是：**查询侧切分与入库侧不一致**
            # （已入库的 chunk 是用带词典的分词建的），结果是某些词永远召回不到，
            # 而两边的代码看起来都对、没有任何异常。没有这行日志，
            # 排查会从向量库一路试到 embedding 模型，最后才想到是少了个文件。
            logger.warning(
                "业务词典不存在，退化为通用分词：%s。检索精度会下降且难以察觉，"
                "请执行 make dict 生成（需业务库已灌数）",
                resolved,
            )
            return
        jieba.load_userdict(resolved)
        cls._loaded_paths.add(resolved)
        logger.info("已加载业务词典：%s", resolved)

    def cut(self, text: str) -> list[str]:
        """切分成 token 序列（不含停用词与纯标点）。"""
        kept, _ = self.cut_explained(text)
        return kept

    def cut_explained(self, text: str) -> tuple[list[str], list[str]]:
        """切分并说明**丢弃了什么**，返回 (保留, 丢弃)。

        这条公开路径存在的理由是排查方向：某个词检索不到时，「分词把它切错了」
        与「分词切对了但被过滤规则丢了」是两种完全不同的故障，
        而 `cut()` 把两者抹成同一个结果——只看到一个空列表。
        `make tokenize` 靠它把这两类原因分开显示。
        """
        normalized = normalize_numbers(normalize(text))
        kept: list[str] = []
        dropped: list[str] = []
        for token in jieba.lcut(normalized):
            (kept if self._keep(token) else dropped).append(token)
        return kept, dropped

    def _keep(self, token: str) -> bool:
        token = token.strip()
        if not token or token in self._stopwords:
            return False
        # 纯标点与纯空白：切分后会留下 `,` `.` `(` 这类碎片
        if not any(ch.isalnum() for ch in token):
            return False
        # 中文单字不进向量：它们极少是有效检索词，却是稀疏维度的主要消耗者，
        # 留在里面只会稀释真正有区分度的词。
        # **ASCII 单字除外**——`q3` 拆出来的 `q`、产品型号里的字母
        # 往往是精确检索的目标，丢掉它们正违背了稀疏路存在的意义。
        return not (len(token) == 1 and not token.isascii())


def is_general_word(token: str) -> bool:
    """这个 token 在 jieba 的**通用词典**里是不是一个真词。

    供检索侧的「语料从未出现过这个词」判定使用（见
    `retriever._unseen_topics`）——**为什么需要它**：只说"词表里没有"
    是不够的，切分本身会产生**伪 token**。实测一例：

        「2025 年 8 月的经营月报里区域分布情况如何」
        → jieba 切成 …月报 / 报里 / 区域分布…

    `报里` 不在词表里（语料当然没用过这个词），于是它被判成"语料没见过
    的主题词"，一条余弦 0.79 的**高度相关问题**被拒答。而这类伪 token 是
    **无界**的：任何切分抖动都会造出新的，靠维护一份排除表堵不住。

    加一道「通用词典认不认识」之后，判据变成一句可检验的话：
    **一个通用词典认识的词，我们的语料一次都没用过**。
    `报里` 连通用词典都不认识 → 出局；`食堂` 认识而语料没有 → 仍然触发。

    代价是那些通用词典也不认识的**复合词**（`碳积分`、`带货`、`月报`、
    `区域分布`）不再单独触发。这是**故意偏保守**的一侧：它们的漏判由
    相关性门禁的余弦那一路兜，而误判会直接拒掉一条好问题。

    **必须显式 `initialize()`**：前缀词典是惰性构建的，没构建时
    `get_FREQ` 对**任何**词都返回 None，判据会静默失效（全部判成"不认识"
    等于永不触发）。它幂等，只有第一次真的建词典。
    """
    jieba.initialize()
    return jieba.get_FREQ(token) is not None


def load_stopwords(path: str) -> frozenset[str]:
    """读停用词表。文件不存在或为空时返回空集。

    **空表不是错误**：11.6.1 明确"停用词过滤"是可选步骤，理由是
    BM25 的 IDF 本身就会抑制高频词。真正的停用词表要等语料冻结后
    按实际词频统计生成，现在硬塞一份通用表反而会把业务词误杀。
    """
    file = Path(path)
    if not file.exists():
        return frozenset()
    return frozenset(
        line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()
    )


class Vocabulary:
    """`token → token_id` 映射与冻结的 IDF（11.6.3 / 11.6.4）。

    **入库与查询必须用同一份映射**，否则两个稀疏向量不可比：
    同一个词在文档侧是 id 42、在查询侧是 id 77，点积恒为 0，
    表现为"检索永远召回不到东西"，而两边各自的代码看起来都对。

    `token_id` **只增不改**：新增词分配新 id，已发布 chunk 的稀疏向量因此
    无需重算。删词或复用 id 会让历史向量悄悄指向别的词——
    同样表现为检索结果漂移，且**无法从数据上察觉**。
    """

    def __init__(
        self,
        token_ids: Mapping[str, int],
        *,
        document_count: int,
        df: Mapping[str, int] | None = None,
    ) -> None:
        self._token_ids = dict(token_ids)
        #: 冻结快照对应的语料规模，IDF 分母来自它
        self._document_count = max(document_count, 1)
        self._df = dict(df or {})
        self._idf = {token: self._idf_of(token) for token in self._token_ids}

    @property
    def document_count(self) -> int:
        return self._document_count

    def __len__(self) -> int:
        return len(self._token_ids)

    def id_of(self, token: str) -> int | None:
        return self._token_ids.get(token)

    def covers(self, token: str) -> bool:
        """语料里**有没有以这个词为组成部分**的 token。

        **`id_of(token) is None` 不等于"语料没见过这个词"**，这是实测踩到的：
        「归口」在 88 篇语料的 chunk 文本里出现 **69 次**，而它的 `token_id` 是
        `None`——因为业务词典把「归口管理部门」收成了一个词（`rag_terms.txt`），
        jieba 从此不再单独切出「归口」。于是按"不在词表"判定，
        一条余弦 0.84 的**高度相关问题**（「直营渠道的价格管理由哪个部门归口负责」）
        会被判成 `NO_RELEVANT_KNOWLEDGE`。

        分词把长词收成一个 token 是**业务词典的正常工作方式**，不是异常。
        判据因此要说成"语料里有没有以它为组成部分的词"，而不是"它是不是一个 token"。

        **两个方向都要查，缺一个就会误拒**（两条都是实测踩到的）：

        - **问题词是语料某个词的一部分**：`归口` ⊂ `归口管理部门`（语料里出现 69 次）。
        - **语料某个词是问题词的一部分**：`华东` ⊂ `华东地区`——用户写「华东地区」，
          语料一律写「华东区域」，jieba 把前者切成一个整词，于是 `.id_of()` 是 None。
          只查第一个方向的话，这条余弦 0.71 的**真问题**会被拒答。

        第二个方向加一道"被包含的语料词至少两个字"的闸：语料词表里有 `q`
        这种单字母 token（产品型号切出来的），它几乎出现在任何字符串里，
        不加闸等于让判据恒为真。

        ⚠️ **它解决不了同义词**：「报备」在语料里一次都没出现（制度写的是「备案」），
        也不与任何词互相包含，于是仍然会被判成"语料没见过"。那一类只能靠语义
        （重排器 / Phase 8 的 Reviewer），纯词汇规则区分不了"语料没讲过这件事"
        与"语料用的是另一个说法"。已知的这一例会长期留在金标集里当回归用例。
        """
        if token in self._token_ids:
            return True
        if any(token in known for known in self._token_ids):
            return True
        return any(known in token for known in self._token_ids if len(known) > 1)

    def idf(self, token: str) -> float:
        """该 token 的 IDF。未登录词返回一个保守的高值，理由见 `_UNSEEN_TOKEN_DF`。"""
        return self._idf.get(token, self._idf_of(token))

    def _idf_of(self, token: str) -> float:
        """`log(N / df)` 的平滑形式，恒为正。

        加 1 平滑而不是直接 `log(N/df)`：后者在"词出现在全部文档里"时等于 0，
        与"未登录词"的取值撞在一起，两种完全不同的含义无法区分。
        """
        df = max(self._df.get(token, _UNSEEN_TOKEN_DF), 1)
        return math.log((self._document_count + 1) / (df + 1)) + 1.0

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "document_count": self._document_count,
            "tokens": {
                token: {"token_id": token_id, "df": self._df.get(token, _UNSEEN_TOKEN_DF)}
                for token, token_id in sorted(self._token_ids.items())
            },
        }

    @classmethod
    def from_snapshot(cls, payload: Mapping[str, Any]) -> Vocabulary:
        version = payload.get("format_version")
        if version != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"词表快照格式版本不符：期望 {SNAPSHOT_FORMAT_VERSION}，实际 {version}。"
                "快照是冻结产物，格式变更后必须重新导出，不能按新格式解析老快照。"
            )
        tokens = payload.get("tokens") or {}
        return cls(
            {token: int(entry["token_id"]) for token, entry in tokens.items()},
            document_count=int(payload.get("document_count", 1)),
            df={token: int(entry.get("df", _UNSEEN_TOKEN_DF)) for token, entry in tokens.items()},
        )

    def dump(self, path: str | Path) -> None:
        """导出快照（开发流程 6.7 施工项 3 的"快照导出"）。"""
        Path(path).write_text(
            json.dumps(self.to_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        return cls.from_snapshot(json.loads(Path(path).read_text(encoding="utf-8")))


def build_sparse(
    tokens: Sequence[str],
    vocabulary: Vocabulary,
    *,
    dim: int,
) -> dict[int, float]:
    """token 序列 → 稀疏向量 `{token_id: weight}`（11.6.1 最后一步）。

    权重是 **归一化词频 × 冻结 IDF**（11.6.4）。两个细节：

    1. **词频取对数**：一个词出现 10 次不代表它比出现 1 次重要 10 倍。
       线性词频会让长文档（或反复强调某术语的制度）在稀疏路上一家独大。
    2. **按最大词频归一化**（而非按向量模长）：这样权重落在 (0, 1]，
       与 Qdrant 稀疏路的点积可比。用模长归一化会把"文档越长权重越小"
       引进来，而长度差异在 RRF 阶段本来就不该有影响。

    不在词表里的 token **分配不到 id，直接丢弃**——查询侧与入库侧用的是
    同一份词表，所以"丢弃"是对称的，不会造成单边失配。
    未登录 token 的 id 分配发生在入库阶段（`VocabularyBuilder`），
    查询阶段遇到未登录词就是真的检索不到，这是固定 IDF 方案的已知代价。
    """
    counts = Counter(tokens)
    if not counts:
        return {}
    max_count = max(counts.values())
    weights: dict[int, float] = {}
    for token, count in counts.items():
        token_id = vocabulary.id_of(token)
        if token_id is None:
            continue
        if not 0 <= token_id < dim:
            # 超出配置的维度上限说明词表分配出了问题，**报错而不是静默丢弃**：
            # 静默丢弃的表现是"这个词检索不到"，排查方向会跑到分词上去。
            raise ValueError(
                f"token_id {token_id}（token={token!r}）超出稀疏向量维度 {dim}。"
                "rag.sparse_dim 需要扩容，或词表分配存在缺陷。"
            )
        tf = 1.0 + math.log(count)
        weights[token_id] = tf / (1.0 + math.log(max_count)) * vocabulary.idf(token)
    return weights
