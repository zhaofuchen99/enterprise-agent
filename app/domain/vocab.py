"""稀疏检索词表条目（详细设计 11.6.3）。

放在 `domain/` 而不是 `tools/rag/`：**词表是检索的口径，不是某个工具的内部状态**。
`tools/rag` 要用它、`repositories` 要存它，而分层约束只允许前者依赖后者，
所以它必须落在两边都能依赖的位置。放进 `tools/rag/` 会让仓储反向依赖工具层。

## `df` 的口径是「索引单元频率」，不是「源文件频率」

`rag_vocab.df` 的列注释在详细设计 16.8 里写的是"文档频率"。在 BM25 里
"文档"指的是**被索引的那个单元**，而在本项目里被索引的单元是 **chunk**：
稀疏向量一个个挂在 chunk 上，检索也是一次在一个 chunk 上打分。

于是 `df` = 含该 token 的 chunk 数，IDF 的分母 = chunk 总数。
按源文件算（88 篇）会得到一个几乎不随内容变化的分母——
一个词只要在某篇文档里出现过，按源文件算就是"出现 1 次"，
哪怕它在那篇文档的 90 个 chunk 里到处都是。

这不是术语之争：两种取法会给出**不同的 IDF 排序**，
而这个排序直接决定稀疏路召回谁。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VocabEntry(BaseModel):
    """词表里的一行：token → id 与它的索引单元频率。

    Attributes:
        token: 归一化之后的词形（与 `Tokenizer.cut` 的产物一致）。
        token_id: **只增不改**。删词或复用 id 会让历史 chunk 的稀疏向量
            悄悄指向别的词——检索结果漂移，且无法从数据上察觉（11.6.3）。
        df: 含该 token 的 chunk 数。**同样是冻结的**：入库时算出的稀疏向量
            用的是当时的 df，事后更新它会让老向量与新向量不可比。
    """

    model_config = ConfigDict(frozen=True)

    token: str = Field(min_length=1, max_length=128)
    token_id: int = Field(ge=0)
    df: int = Field(ge=0)
