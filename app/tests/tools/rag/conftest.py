"""RAG 测试的共享夹具。

## 为什么必须有 `_isolate_jieba`，以及它为什么在这里而不是某个测试文件里

`jieba.load_userdict` 改的是**进程级**全局词典，而 `Tokenizer` 的注释已经
明说了它"不支持不同的 Tokenizer 用不同的词典"。于是同一次 pytest 进程里，
先跑的用例加载的业务词典会留给后面所有用例：

- 「单独跑通过、全量跑失败」；
- 更糟的一种：**全量跑也通过，但通过的其实是另一个用例加载的词**，
  断言失去了它声称的意义。

原来这份夹具写在 `test_tokenizer.py` 里，恢复的是"**本用例开始前**的样子"。
那在同一个文件内够用，跨文件就不够了——`test_ingestion.py`
（字母序在前）会先用 `Tokenizer.from_settings` 加载真实的
`configs/rag_user_dict.txt`，等 `test_tokenizer.py` 的夹具跑起来时，
它保存的"基线"**已经是脏的**了。实测的表现正是那条：

    FAILED test_tokenizer.py::test_business_dict_keeps_compound_term_together
    AssertionError: assert '渠道折扣' not in ['华东', '区域', '渠道折扣', '政策']

所以基线要在**导入期**取——那一刻 Module 级的词典加载一次都还没发生，
它一定是"只加载过 jieba 自带词典"的状态。夹具在**每个用例之前**恢复它，
于是无论前面跑过什么，每个用例都从干净状态开始。
"""

from __future__ import annotations

from collections.abc import Iterator

import jieba
import pytest

from app.tools.rag.tokenizer import Tokenizer

#: 进程最初始的 jieba 词典状态。**在导入期取**，见模块 docstring。
#:
#: ⚠️ **必须先 `initialize()` 再取**：前缀词典是惰性构建的，
#: 未初始化时 `jieba.dt.FREQ` 是**空字典**。取一份空基线再去"恢复"它，
#: 等于把整个词频表清空——而 `initialize()` 因为 `initialized` 已经是 True，
#: 不会重建。后果是后续所有分词退化成逐字切分，**而用例照样通过**：
#: 像 `all(len(t) > 1 for t in tokens)` 这种断言在空序列上恒为真。
#: 实测的表现是"未登录词判据全部失效"（`get_FREQ` 对任何词都返回 None），
#: 排查方向会跑到词表上去。
#:
#: 两份都要存：`load_userdict` 同时改 `FREQ`（词频表，决定切分）
#: 与 `user_word_tag_tab`（词性标注表）。只恢复前者的话，
#: 词性相关的行为仍然带着上一个用例的痕迹。
jieba.initialize()
_PRISTINE_FREQ = dict(jieba.dt.FREQ)
_PRISTINE_TAGS = dict(jieba.dt.user_word_tag_tab)


@pytest.fixture(autouse=True)
def _isolate_jieba() -> Iterator[None]:
    """每个用例开始前，把 jieba 的全局词典与 `Tokenizer._loaded_paths` 复原。

    `_loaded_paths` 是同一个问题的另一半：不清它的话，第二个用例传同一个路径
    会被"已经加载过"跳过，而词典其实已经被上一个用例清掉了——
    于是那一次入库是用通用词典切的，而它以为自己加载了业务词典。
    """
    jieba.dt.FREQ.clear()
    jieba.dt.FREQ.update(_PRISTINE_FREQ)
    jieba.dt.user_word_tag_tab.clear()
    jieba.dt.user_word_tag_tab.update(_PRISTINE_TAGS)
    Tokenizer._loaded_paths.clear()
    yield
