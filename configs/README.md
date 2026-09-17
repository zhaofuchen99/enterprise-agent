# configs/

| 文件 | 用途 | 填充阶段 |
|---|---|---|
| `rag_user_dict.txt` | jieba 业务词典。格式 `词 [词频] [词性]`，每行一个 | Phase 5 |
| `rag_stopwords.txt` | 停用词，每行一个 | Phase 5 |

两个文件当前为空占位（配置项 `RAG__USER_DICT_PATH` / `RAG__STOPWORD_PATH` 指向它们，
是一对普通默认值，不必写进 `.env`）。

**注意**：词典不是手写的，而是由 `dim_region` / `dim_product_line` / `dim_channel`
与指标目录脚本提取生成（开发流程 3.3 并行组 F），否则会漏词且无法随维度表演进。
词表扩容后必须验证历史 chunk 仍可被召回，无需重算稀疏向量。
