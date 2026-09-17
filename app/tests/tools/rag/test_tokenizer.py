"""分词与稀疏向量（详设 11.6 / 开发流程 6.7 的验证命令）。

覆盖三块：归一化的确定性、业务词典的实际效果、词表扩容后历史向量仍可召回。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.tools.rag.tokenizer import (
    SNAPSHOT_FORMAT_VERSION,
    Tokenizer,
    Vocabulary,
    build_sparse,
    load_stopwords,
    normalize,
    normalize_numbers,
)
from scripts.gen_dict import Terms, build_dict_text, load_handwritten_terms

# `_isolate_jieba` 夹具已移到 `app/tests/tools/rag/conftest.py`：
# 它恢复的必须是**进程初始**状态，而不是"本用例开始前"的状态——
# 后者在同文件内够用，跨文件就不够了（见那个 conftest 的 docstring）。


def _write(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


# ------------------------------------------------------------------ 归一化


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("２０２５", "2025"),  # 全角数字
        ("ＡＢＣ", "abc"),  # 全角字母 + 大小写
        ("华东  区域\n\t渠道", "华东 区域 渠道"),  # 空白压缩
        ("（含税）", "(含税)"),  # 全角括号
        ("2025年7月1日", "2025-7-1"),  # 中文日期
        ("2025/07/01", "2025-07-01"),  # 斜杠日期
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_is_idempotent() -> None:
    """归一化必须幂等。

    入库侧对原文调一次、查询侧对用户输入调一次，若 `normalize(x)` 与
    `normalize(normalize(x))` 不同，同一段文本在两侧就会得到不同的向量——
    这是"检索莫名召回不到"里最难查的一类原因。
    """
    for raw in ("２０２５年７月１日，华东", "ＡＢＣ  渠道\n折扣", "百分之十五"):
        once = normalize(raw)
        assert normalize(once) == once


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("百分之十五", "15%"),
        ("百分之三", "3%"),
        ("十五", "15"),
        ("二十", "20"),
        ("一百零五", "105"),
        ("三千五百", "3500"),
        ("两万", "20000"),
        ("华东", "华东"),  # 非数词不动
    ],
)
def test_normalize_numbers(raw: str, expected: str) -> None:
    assert normalize_numbers(raw) == expected


def test_normalize_numbers_parses_sequence_not_characters() -> None:
    """**逐字替换是错的**，这条用例专门钉住它。

    「百分之十五」逐字替换会得到「百分之十5」——一个既不是中文也不是数字、
    永远匹配不到任何东西的 token。错误的替换不会报错，
    只会让某些查询悄悄召回不到东西。
    """
    assert "十5" not in normalize_numbers("百分之十五")
    assert normalize_numbers("百分之十五") == "15%"


def test_normalize_numbers_matches_arabic_form() -> None:
    """中文数词与阿拉伯数字必须归一到同一个形态。

    制度写「三成」、报告写「30%」时，两者表达同一件事；
    不统一则跨文档检索必然失配，16.11.2 那类"报告数字与 DB 数字差异"
    的缺陷也就无从比对。
    """
    assert normalize_numbers("百分之三十") == normalize_numbers("30%") == "30%"


# ------------------------------------------------------------------ 分词


def test_business_dict_keeps_compound_term_together(tmp_path: Path) -> None:
    """业务词典的实际效果——这是拒绝 ES 之后必须自证的能力。

    不加载词典时 jieba 会把「渠道折扣」切成「渠道 / 折扣」，
    于是用户搜这个完整术语会把所有含「渠道」或「折扣」的段落一并召回，
    精确性丢失（详设 11.6.2）。
    """
    dictionary = _write(tmp_path / "dict.txt", "渠道折扣 100000\n")

    default_tokens = Tokenizer().cut("华东区域渠道折扣政策")
    dict_tokens = Tokenizer(user_dict_path=dictionary).cut("华东区域渠道折扣政策")

    assert "渠道折扣" not in default_tokens  # 默认切分切开了
    assert "渠道折扣" in dict_tokens  # 加载业务词典后是一个完整 token


def test_missing_dict_file_does_not_raise() -> None:
    """词典是可选的增强。

    开发机上词典还没生成时应该退化成通用分词并继续，
    而不是让整个 Worker 起不来——**启不来的代价远大于切得不准**。
    """
    assert Tokenizer(user_dict_path="/nonexistent/dict.txt").cut("华东渠道折扣")


def test_stopwords_are_filtered(tmp_path: Path) -> None:
    stopword_file = tmp_path / "stop.txt"
    stopword_file.write_text("的\n了\n", encoding="utf-8")

    tokenizer = Tokenizer(stopwords=load_stopwords(str(stopword_file)))

    assert "的" not in tokenizer.cut("华东的渠道折扣了")


def test_missing_stopword_file_yields_empty_set() -> None:
    """停用词表为空是合法状态（11.6.1 把它列为可选步骤）。

    BM25 的 IDF 本身就会抑制高频词，硬塞一份通用停用词表反而会误杀业务词。
    """
    assert load_stopwords("/nonexistent/stop.txt") == frozenset()


def test_punctuation_and_single_chars_are_dropped() -> None:
    """纯标点与中文单字不进稀疏向量。

    单字几乎不携带信息，却是维度的主要消耗者——它们会占满稀疏向量的位置，
    让真正有区分度的词在点积里被稀释。
    """
    tokens = Tokenizer().cut("华东、区域（含税）。")

    assert all(any(ch.isalnum() for ch in t) for t in tokens)
    assert all(len(t) > 1 for t in tokens)


def test_cut_explained_separates_dropped_tokens() -> None:
    """`cut` 把两种故障抹成同一个空结果，`cut_explained` 把它们分开。

    某个词检索不到时，「分词切错了」与「切对了但被过滤规则丢了」
    排查方向完全不同。`make tokenize` 靠这条区分显示丢弃项。
    """
    kept, dropped = Tokenizer().cut_explained("华东、区域（含税）")

    assert "华东" in kept and "含税" in kept
    # 丢弃项是**归一化之后**的形态：全角「、」「）」先被折成半角，再因不含字母数字被丢。
    # 断言写成半角，是因为用户看到的调试输出就是这个形态——
    # 若这里能断言出全角，说明归一化那一步没跑到。
    assert "," in dropped and ")" in dropped


# ------------------------------------------------ 业务词典产物（详设 11.6.2）

#: 设计文档与 `tokenizer.py` 反复引用的那个例子（「渠道折扣」必须是一个 token）。
_DESIGN_EXAMPLE = "华东区域渠道折扣政策"


def test_business_dict_artifact_is_present_and_effective(settings: Settings) -> None:
    """**断言的对象是仓库里那份产物文件**，不是"生成脚本能跑通"。

    这条用例是一次真实事故的回归：`configs/rag_user_dict.txt` 曾经是 0 字节，
    而 `Tokenizer` 对缺失的词典**静默退化**——不抛异常、旧版本连日志都没有，
    表现只是"某些查询召回不到东西"。测试全绿、`make sql` 正常，
    检索却在悄悄变差。生成脚本当时根本不存在，所以"脚本跑通"这种断言也写不出来。
    """
    path = Path(settings.rag.user_dict_path)
    assert path.exists(), f"业务词典产物缺失：{path}（执行 make dict 生成）"
    assert path.stat().st_size > 0, f"业务词典产物为空：{path}"

    tokens = Tokenizer(user_dict_path=str(path)).cut(_DESIGN_EXAMPLE)
    assert "渠道折扣" in tokens, f"设计文档要求的复合术语未生效：{tokens}"


def test_default_tokenizer_loads_the_committed_dict(settings: Settings) -> None:
    """装配点（`from_settings`）读到的就是那份产物。

    上面那条用例手工传了路径；这条走真实配置路径，
    防的是"产物在、但配置指向别处"——那种情况下文件检查全过，运行时仍然是通用分词。
    """
    tokens = Tokenizer.from_settings(settings).cut(_DESIGN_EXAMPLE)

    assert "渠道折扣" in tokens


def test_business_dict_has_no_comment_or_blank_lines(settings: Settings) -> None:
    """产物里不能有注释行与空行——**jieba 会把注释整行当成词条**。

    实测：`jieba.load_userdict` 对不匹配词频/词性后缀的行，
    会用 `.+?` 把整行吃下来 `add_word`。`# 来源：...` 那一行的
    `jieba.get_FREQ()` 返回 1，它真的成了一个词。
    于是"在文件头写清来源"这个看起来完全无害的动作，往词典里塞了垃圾。
    出处写在 `scripts/gen_dict.py` 与 `configs/rag_terms.txt` 的注释里，不写在产物里。
    """
    lines = _dict_lines(settings)

    assert lines, "产物不应为空"
    assert all(line.strip() for line in lines), "产物不应有空行"
    assert not [line for line in lines if line.lstrip().startswith("#")], "产物不应有注释行"


def test_business_dict_entries_are_in_canonical_form(settings: Settings) -> None:
    """产物里的每个词条都必须等于它自己的归一化形态。

    查询与入库两侧都是先 `normalize_numbers(normalize(text))` 再分词，
    所以词典存 `KA` 而文本归一成 `ka` 时，加载成功、日志无异常、**一条也召回不到**。
    手改产物文件（补一个全角或大写词）就会踩到这个坑，这条用例把它挡住。
    """
    non_canonical = [w for w in _dict_lines(settings) if normalize_numbers(normalize(w)) != w]

    assert not non_canonical, f"词条未按归一化形态存储：{non_canonical}"


def _dict_lines(settings: Settings) -> list[str]:
    return Path(settings.rag.user_dict_path).read_text(encoding="utf-8").splitlines()


def test_generator_dedupes_sources_and_canonicalizes() -> None:
    """生成器：跨来源去重，且词条按归一化形态写出。

    「净销售额」同时是指标名（自动来源）与手工表的常见写法，
    抄重了只会让产物里出现两行同样的词——jieba 不报错，产物也看不出问题。
    """
    terms = Terms(
        regions=["华东", "华东"],
        metrics=["净销售额"],
        handwritten=["KA", "净销售额"],
    )
    text, duplicates = build_dict_text(terms)

    # 顺序 = 自动来源（区域 → 指标）在前，手工表在后
    assert text.splitlines() == ["华东", "净销售额", "ka"]
    # 重复项要被点名（`华东` 在来源里出现两次，`净销售额` 跨来源重复）——
    # 报告里看不见重复，手工表抄重了就没有任何反馈
    assert duplicates == ["华东", "净销售额"]


def test_handwritten_terms_skip_comments_and_blanks(tmp_path: Path) -> None:
    """手工术语表是**给人看的输入**，因此可以带注释——注释不得进产物流水线。"""
    path = tmp_path / "terms.txt"
    path.write_text("# 分组说明\n\n渠道折扣\n  清仓折扣  \n渠道折扣\n", encoding="utf-8")

    assert load_handwritten_terms(path) == ["渠道折扣", "清仓折扣"]


# ------------------------------------------------------------------ 词表


def _vocab(count: int = 10) -> Vocabulary:
    return Vocabulary(
        {"华东": 1, "渠道折扣": 2, "销售额": 3},
        document_count=count,
        df={"华东": 5, "渠道折扣": 2, "销售额": 8},
    )


def test_token_ids_are_stable() -> None:
    """`token_id` 只增不改（11.6.3）。

    删词或复用 id 会让历史 chunk 的稀疏向量**悄悄指向别的词**——
    表现为检索结果漂移，且无法从数据上察觉。
    """
    vocab = _vocab()

    assert vocab.id_of("华东") == 1
    assert vocab.id_of("渠道折扣") == 2


def test_unknown_token_gets_no_id() -> None:
    assert _vocab().id_of("不存在的词") is None


def test_idf_is_positive_and_orders_rare_above_common() -> None:
    """IDF 恒为正，且稀有词的 IDF 高于常见词。

    `log(N/df)` 无平滑时，"出现在全部文档里"的词 IDF 为 0，
    与"未登录词"的取值撞在一起，两种完全不同的含义无法区分。
    """
    vocab = _vocab()

    assert vocab.idf("渠道折扣") > vocab.idf("销售额") > 0
    assert vocab.idf("未登录词") > 0


def test_snapshot_round_trips() -> None:
    vocab = _vocab()

    restored = Vocabulary.from_snapshot(vocab.to_snapshot())

    assert restored.id_of("华东") == vocab.id_of("华东")
    assert restored.document_count == vocab.document_count
    assert restored.idf("渠道折扣") == pytest.approx(vocab.idf("渠道折扣"))


def test_snapshot_rejects_unknown_format_version(tmp_path: Path) -> None:
    """快照是**冻结产物**，格式变更后必须重新导出。

    按新格式解析老快照不会报错，只会得到一份 id 全错的词表——
    那时检索失效，而没有人会怀疑到格式版本上。
    """
    path = tmp_path / "snap.json"
    path.write_text(
        json.dumps({"format_version": SNAPSHOT_FORMAT_VERSION + 1, "tokens": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="格式版本"):
        Vocabulary.load(path)


# ------------------------------------------------------------------ 稀疏向量


def test_build_sparse_maps_tokens_to_ids() -> None:
    weights = build_sparse(["华东", "渠道折扣", "华东"], _vocab(), dim=2**24)

    assert set(weights) == {1, 2}
    assert all(w > 0 for w in weights.values())


def test_build_sparse_ignores_unknown_tokens() -> None:
    """不在词表里的 token 丢弃，不分配临时 id。

    查询侧与入库侧用同一份词表，所以"丢弃"是对称的；
    若这里给未登录词临时分配 id，两侧分配结果必然不同，
    点积恒为 0 且**看不出为什么**。
    """
    weights = build_sparse(["华东", "从未见过的词"], _vocab(), dim=2**24)

    assert set(weights) == {1}


def test_build_sparse_dampens_repetition() -> None:
    """词频取对数：出现 4 次不等于权重是 1 次的 4 倍。

    线性词频会让反复强调某术语的制度在稀疏路上一家独大，
    把真正相关的其他文档挤出去。

    **必须在同一个文档内比较两个词的比值**。第一版这条用例拿"只含一个词的
    文档"去比 1 次与 4 次的权重，结果两者都等于 1.0——因为最大词频归一化
    会把单 token 文档整个压成 1。实现是对的，是断言写错了：
    归一化之后，文档内的**绝对量级**没有意义，只有**词之间的相对权重**有意义。
    """
    vocab = _vocab()
    # 华东 ×4、渠道折扣 ×1，两个词的 IDF 不同，需要先除掉 IDF 的影响
    weights = build_sparse(["华东", "华东", "华东", "华东", "渠道折扣"], vocab, dim=2**24)

    idf_ratio = vocab.idf("华东") / vocab.idf("渠道折扣")
    observed = (weights[1] / weights[2]) / idf_ratio

    assert 1.0 < observed < 4.0, "词频必须是次线性的：线性时 observed 会等于 4"


def test_build_sparse_rejects_id_beyond_dimension() -> None:
    """超出维度上限必须**报错而不是静默丢弃**。

    静默丢弃的表现是"这个词检索不到"，排查方向会跑到分词上去找，
    而真正的原因（`rag.sparse_dim` 需要扩容）完全不在视野里。
    """
    with pytest.raises(ValueError, match="维度"):
        build_sparse(["华东"], _vocab(), dim=1)


def test_build_sparse_on_empty_input() -> None:
    assert build_sparse([], _vocab(), dim=2**24) == {}


# ---------------------------------------------- 词表扩容（开发流程 6.7 的验证项）


def test_expanding_vocabulary_keeps_old_chunks_retrievable() -> None:
    """**词表扩容后，历史 chunk 无需重算稀疏向量仍可召回**（11.6.3）。

    这是"固定 IDF + token_id 只增不改"这套方案的核心承诺，
    也是开发流程 6.7 明确要求的验证项。做法：

    1. 用初始词表给一个 chunk 建稀疏向量；
    2. 扩容：新增若干 token（分配新 id，老的 id 不动）；
    3. 用**原样的旧向量**去检索新词表下的文档向量，仍然命中。

    如果 `token_id` 是位置相关的（比如按字母序重排、或删词后复用 id），
    第 3 步会失败——而且失败得很安静：分数降为 0，表现为"检索质量下降"
    而不是"报错"。这条用例就是钉住那个安静失败。
    """
    initial = Vocabulary({"华东": 1, "渠道折扣": 2}, document_count=10, df={"华东": 5})
    chunk_sparse = build_sparse(["华东", "渠道折扣"], initial, dim=2**24)
    assert set(chunk_sparse) == {1, 2}

    expanded = Vocabulary(
        {"华东": 1, "渠道折扣": 2, "净销售额": 3, "返利": 4},
        document_count=10,
        df={"华东": 5, "渠道折扣": 2, "净销售额": 8, "返利": 1},
    )
    # 查询侧用扩容后的词表构建查询向量
    query_sparse = build_sparse(["华东", "渠道折扣"], expanded, dim=2**24)

    assert set(chunk_sparse) <= set(query_sparse), "旧向量的 id 必须仍然存在"
    overlap = sum(w * query_sparse.get(tid, 0.0) for tid, w in chunk_sparse.items())
    assert overlap > 0, "历史 chunk 的稀疏向量必须仍能与查询产生正的点积"

    # 新词拿到的是新 id，没有挤掉老词的位置
    assert expanded.id_of("华东") == 1
    assert expanded.id_of("净销售额") == 3
