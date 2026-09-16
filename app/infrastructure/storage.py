"""对象存储抽象（开发流程 6.3 施工项 4）。

知识库的原始文件要留底：详细设计 18.3 的「Redis 里的东西都能由 MySQL 与
**原始文件**重建」，其中的「原始文件」就存在这里。Phase 5 建 RAG 流水线时，
文档解析、分块、重嵌入都要回到原始文件重跑，因此存储层必须先能独立跑通。

**本阶段只实现 `local`**：`s3` 实现按施工项清单留到 Phase 5。
契约测试 `app/tests/infrastructure/test_storage_contract.py` 已经写成
参数化的形式，Phase 5 补上 s3 实现后，把它加进参数表就自动被同一套断言覆盖——
契约是被**执行**验证过的，而不只是被声明过。

**为什么不直接用路径拼字符串**：`key` 来自上传接口的参数，用户可控。
`../../etc/passwd` 这类 key 如果直接拼进路径就能读写到根目录之外，
这是对象存储适配器最常见的一类漏洞。`LocalObjectStorage` 在拼路径之前
先把 key 规范化并校验它没有逃出根目录。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class ObjectStorage(Protocol):
    """对象存储接口。

    只保留四个动作。**没有列目录**：调用方需要枚举对象时应该查 MySQL
    里的 `knowledge_document` 表，而不是去列存储桶——对象存储的列表操作
    又慢又不保证顺序，把它当目录服务用是错的方向。
    """

    async def put(self, key: str, data: bytes, *, content_type: str = "") -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def exists(self, key: str) -> bool: ...

    async def delete(self, key: str) -> None: ...


class InvalidObjectKeyError(ValueError):
    """key 不合法（越出根目录、绝对路径、含空字节等）。"""


class LocalObjectStorage:
    """本地文件系统实现（`STORAGE_BACKEND=local`）。

    先落临时文件再 `replace`：`replace` 在同分区上是原子的，因此
    `get` 永远看不到写了一半的文件。直接往目标路径写的话，
    并发的读取方会读到截断的内容——而文件一旦被解析成 chunk 写进向量库，
    这个错误就被固化下来了。
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    async def put(self, key: str, data: bytes, *, content_type: str = "") -> None:
        # 本地实现无处存放 content_type，参数只是为了让两个实现在调用方看来一致
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

    async def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    async def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    async def delete(self, key: str) -> None:
        # 不存在也算删除成功：删除接口的语义是「让它不在」，
        # 而不是「它必须在并且被删掉」。这样重试删除是安全的。
        self._resolve(key).unlink(missing_ok=True)

    def _resolve(self, key: str) -> Path:
        """key -> 根目录下的绝对路径，并确保没有逃出根目录。

        `Path.resolve()` 会把 `..` 与符号链接一并展开，因此在它之后做
        前缀判断能同时挡住「用 .. 爬出去」与「软链到别处」两种走法。
        只做字符串替换（比如把 '..' 抹掉）挡不住后者。
        """
        if not key or "\x00" in key:
            raise InvalidObjectKeyError("对象 key 不能为空或含空字节")
        candidate = (self._root / key).resolve()
        # is_relative_to 而非 startswith 字符串比较：
        # 后者会把 `/data/storage-evil` 误判成在 `/data/storage` 之下
        if candidate != self._root and not candidate.is_relative_to(self._root):
            raise InvalidObjectKeyError(f"对象 key 越出存储根目录：{key}")
        return candidate
