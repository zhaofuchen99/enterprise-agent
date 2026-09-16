"""对象存储的**契约测试**（开发流程 6.3 验收命令 5）。

参数化的形式是有意的：`s3` 实现按施工项清单留到 Phase 5，
届时把它加进 `_IMPLEMENTATIONS` 就自动被同一套断言覆盖。
契约于是是**被执行验证过**的，而不只是被声明过。

**越界用例是这里最重要的部分**。`key` 来自上传接口的参数、用户可控，
`../../etc/passwd` 这类 key 拼进路径就能读写到根目录之外——
对象存储适配器最经典的一类漏洞。只把 `..` 字符串抹掉挡不住软链，
因此实现里用的是 `Path.resolve()` 之后的 `is_relative_to` 判断，
下面的用例同时覆盖了这两种走法。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from app.infrastructure.storage import InvalidObjectKeyError, LocalObjectStorage, ObjectStorage


@pytest.fixture
def make_storage(tmp_path: Path) -> Callable[[], ObjectStorage]:
    """参数化入口。Phase 5 补上 s3 实现后，在这里加一个分支即可。

    返回的是 `ObjectStorage` 而不是 `LocalObjectStorage`：
    用例只该依赖协议，依赖具体类型会让「换实现自动被覆盖」这件事失效。
    """

    def _local() -> ObjectStorage:
        return LocalObjectStorage(tmp_path / "objects")

    return _local


@pytest.fixture
def storage(make_storage: Callable[[], ObjectStorage]) -> ObjectStorage:
    return make_storage()


# ------------------------------------------------------------------ 基本契约
async def test_put_then_get_round_trips(storage: ObjectStorage) -> None:
    await storage.put("docs/2026/report.md", "华东销售分析".encode())

    assert await storage.get("docs/2026/report.md") == "华东销售分析".encode()


async def test_missing_key_raises_on_get(storage: ObjectStorage) -> None:
    with pytest.raises(FileNotFoundError):
        await storage.get("docs/不存在.md")


async def test_exists_reflects_reality(storage: ObjectStorage) -> None:
    assert await storage.exists("k") is False
    await storage.put("k", b"v")
    assert await storage.exists("k") is True


async def test_put_creates_intermediate_directories(storage: ObjectStorage) -> None:
    """key 里带层级是常态（按租户/日期分目录），实现必须自己建目录。"""
    await storage.put("a/b/c/d.md", b"x")

    assert await storage.exists("a/b/c/d.md") is True


async def test_put_overwrites_existing_object(storage: ObjectStorage) -> None:
    await storage.put("k", b"old")
    await storage.put("k", b"new")

    assert await storage.get("k") == b"new"


async def test_delete_removes_the_object(storage: ObjectStorage) -> None:
    await storage.put("k", b"v")

    await storage.delete("k")

    assert await storage.exists("k") is False


async def test_delete_is_idempotent(storage: ObjectStorage) -> None:
    """删除的语义是「让它不在」，不是「它必须在并被删掉」——
    否则重试删除会失败，而重试在分布式下是必然发生的。"""
    await storage.delete("从来就不存在")

    await storage.put("k", b"v")
    await storage.delete("k")
    await storage.delete("k")


async def test_binary_content_is_preserved(storage: ObjectStorage) -> None:
    payload = bytes(range(256))

    await storage.put("blob.bin", payload)

    assert await storage.get("blob.bin") == payload


async def test_empty_object_round_trips(storage: ObjectStorage) -> None:
    await storage.put("empty", b"")

    assert await storage.exists("empty") is True
    assert await storage.get("empty") == b""


# ------------------------------------------------------------------ 越界防护
@pytest.mark.parametrize(
    "key",
    [
        "../escape.md",
        "docs/../../escape.md",
        "/etc/passwd",
        "a/b/../../../../../../etc/passwd",
    ],
)
async def test_keys_escaping_the_root_are_rejected(storage: ObjectStorage, key: str) -> None:
    with pytest.raises(InvalidObjectKeyError):
        await storage.put(key, b"x")


async def test_escaping_key_is_rejected_on_read_and_delete(storage: ObjectStorage) -> None:
    """写被挡住但读没挡，等于把「能不能读」交给了一个只写路径的校验。"""
    with pytest.raises(InvalidObjectKeyError):
        await storage.get("../secret")
    with pytest.raises(InvalidObjectKeyError):
        await storage.delete("../secret")
    with pytest.raises(InvalidObjectKeyError):
        await storage.exists("../secret")


async def test_sibling_directory_with_a_shared_prefix_is_rejected(tmp_path: Path) -> None:
    """按字符串前缀判断的实现会在这里失守。

    `/tmp/x/objects-evil` 以 `/tmp/x/objects` 开头，`startswith` 会放行，
    而它其实是另一个目录。因此实现用的是 `is_relative_to`。
    """
    storage = LocalObjectStorage(tmp_path / "objects")

    with pytest.raises(InvalidObjectKeyError):
        await storage.put("../objects-evil/k", b"x")


async def test_symlink_out_of_the_root_is_rejected(tmp_path: Path) -> None:
    """软链绕过：`..` 能靠字符串检查挡，符号链接不行，只有 resolve 之后判断才挡得住。"""
    root = tmp_path / "objects"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)

    storage = LocalObjectStorage(root)

    with pytest.raises(InvalidObjectKeyError):
        await storage.put("link/escaped.md", b"x")


@pytest.mark.parametrize("key", ["", "a\x00b"])
async def test_empty_or_null_byte_keys_are_rejected(storage: ObjectStorage, key: str) -> None:
    """空 key 会解析成根目录本身；空字节在 C 层会截断路径。两者都必须挡住。"""
    with pytest.raises(InvalidObjectKeyError):
        await storage.put(key, b"x")


# ------------------------------------------------------------------ 落盘行为
async def test_no_temporary_files_are_left_behind(tmp_path: Path) -> None:
    """先写临时文件再原子替换，临时文件不能留在目录里。"""
    storage = LocalObjectStorage(tmp_path / "objects")
    await storage.put("docs/a.md", b"x")

    leftovers = [
        p.name for p in (tmp_path / "objects" / "docs").iterdir() if p.name.startswith(".")
    ]

    assert leftovers == []


async def test_bytes_are_flushed_to_disk_not_buffered(tmp_path: Path) -> None:
    """put 返回后内容必须已经落在磁盘上。

    另起一个存储实例（模拟另一个进程）去读：还在缓冲区里的实现会在这里失败，
    而那种失败在真实场景中表现为「上传成功但解析任务找不到文件」。
    """
    storage = LocalObjectStorage(tmp_path / "objects")
    await storage.put("docs/a.md", b"written")

    another_process = LocalObjectStorage(tmp_path / "objects")

    assert await another_process.get("docs/a.md") == b"written"
