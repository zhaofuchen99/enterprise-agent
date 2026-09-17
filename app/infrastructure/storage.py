"""对象存储抽象（开发流程 6.3 施工项 4）。

知识库的原始文件要留底：详细设计 18.3 的「Redis 里的东西都能由 MySQL 与
**原始文件**重建」，其中的「原始文件」就存在这里。Phase 5 建 RAG 流水线时，
文档解析、分块、重嵌入都要回到原始文件重跑，因此存储层必须先能独立跑通。

**两个实现都在**（Phase 5 补齐 `s3`）：`local` 供本地开发与单测，
`s3` 指向 MinIO / 任何 S3 兼容服务。两者被**同一份契约测试**覆盖
（`app/tests/infrastructure/test_storage_contract.py` 的参数化表）——
契约是被**执行**验证过的，而不只是被声明过。

**为什么不直接用路径拼字符串**：`key` 来自上传接口的参数，用户可控。
`../../etc/passwd` 这类 key 如果直接拼进路径就能读写到根目录之外，
这是对象存储适配器最常见的一类漏洞。`LocalObjectStorage` 在拼路径之前
先把 key 规范化并校验它没有逃出根目录。
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ObjectNotFoundError(FileNotFoundError):
    """对象不存在。

    **继承 `FileNotFoundError`**：`local` 实现原本就是抛它，调用方的
    `except FileNotFoundError` 也必须继续成立；`s3` 把 `NoSuchKey` 归一成
    同一个类型之后，契约测试才可能对两个实现写同一条断言——
    否则"取不存在的 key"这件事就得按实现各写一遍，而那正是契约测试要消灭的东西。
    """


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


def validate_key(key: str) -> None:
    """key 必须是**相对路径形态的标识**：非空、无空字节、非绝对路径、不含 `..`。

    **这是应用层契约，不是某个后端的实现细节**，所以两个实现共用这一个函数。
    两边的理由不同，规则必须相同：

    - 对 `local`，它是安全边界：`../../etc/passwd` 拼进路径就能读写到根目录之外，
      这是对象存储适配器最经典的一类漏洞；
    - 对 `s3`，S3 的 key 是扁平字符串、`..` 并无穿越语义，**看起来可以不校验**。
      但不校验就意味着同一个 key 在 local 上被拒、在 s3 上被接受——
      而契约测试的全部意义就是消灭这种"换个后端行为就变"的差异。
      库里的 key 由 `logical_key` 拼出来，含 `..` 一定是上游出了问题，
      不该因为部署到了 s3 就变成"能跑"。

    真正的边界（bucket 策略、凭证权限）仍在服务端；这个函数管的是**调用方**。
    """
    if not key or "\x00" in key:
        raise InvalidObjectKeyError("对象 key 不能为空或含空字节")
    path = PurePosixPath(key)
    if path.is_absolute() or ".." in path.parts:
        raise InvalidObjectKeyError(f"对象 key 不得是绝对路径或含 ..：{key}")


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
        try:
            return self._resolve(key).read_bytes()
        except FileNotFoundError as exc:
            # 归一成契约定义的类型，让"取不存在的 key"在两个实现上是同一条断言
            raise ObjectNotFoundError(f"对象不存在：{key}") from exc

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
        validate_key(key)
        candidate = (self._root / key).resolve()
        # is_relative_to 而非 startswith 字符串比较：
        # 后者会把 `/data/storage-evil` 误判成在 `/data/storage` 之下
        if candidate != self._root and not candidate.is_relative_to(self._root):
            raise InvalidObjectKeyError(f"对象 key 越出存储根目录：{key}")
        return candidate


class S3ObjectStorage:
    """S3 兼容实现（MinIO / AWS S3 / 阿里云 OSS，`STORAGE_BACKEND=s3`）。

    ## 每次调用新建一个 client

    `aioboto3` 的 client 是异步上下文管理器，要长期持有就得自己管生命周期
    （在 `__aenter__` 里建、在 `aclose()` 里关）。而 `ObjectStorage` 协议
    **故意没有 `aclose`**（见协议说明：只有四个动作），硬加一个会让所有调用方
    都背上关闭责任。

    代价是每次调用重建连接池。在入库这个量级（几十份文档、每份几个对象）下
    可以忽略；真到了需要长连接的量级，该做的是给协议补生命周期方法，
    而不是在这里偷偷缓存一个可能已经失效的 client。

    ## key 校验与 local 共用

    见 `validate_key`：S3 自己的 key 是扁平字符串、`..` 并无穿越语义，
    但**校验的理由不是安全而是契约**——同一个 key 在 local 被拒、
    在 s3 被接受，正是契约测试要消灭的差异。
    """

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        # MinIO 不校验 region，但 botocore 的签名流程要求有一个值
        self._region = region

    def _client(self) -> Any:
        import aioboto3

        return aioboto3.Session().client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name=self._region,
        )

    async def put(self, key: str, data: bytes, *, content_type: str = "") -> None:
        from botocore.exceptions import ClientError

        validate_key(key)
        extra = {"ContentType": content_type} if content_type else {}
        try:
            async with self._client() as client:
                await client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)
        except ClientError as exc:
            if not _is_no_such_bucket(exc):
                raise
            # 桶不存在时**就地建桶再重试一次**。这不是"容错"，是让
            # `docker-compose.dev.yml` 起完 MinIO 之后第一次入库就能跑通——
            # compose 里没有建桶的步骤，而 bucket 名是应用配置（MINIO_BUCKET）。
            # 不这么做，新机器上的第一次 `make ingest` 会以 NoSuchBucket 失败，
            # 看起来像"数据丢了"而不是"桶没建"。
            async with self._client() as client:
                await client.create_bucket(Bucket=self._bucket)
                await client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)
            logger.info("对象存储桶不存在，已创建：%s", self._bucket)

    async def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        validate_key(key)
        try:
            async with self._client() as client:
                response = await client.get_object(Bucket=self._bucket, Key=key)
                body: bytes = await response["Body"].read()
        except ClientError as exc:
            # 归一成与 local 同一个异常类型，契约测试才能对两个实现写同一条断言
            if _is_not_found(exc):
                raise ObjectNotFoundError(f"对象不存在：{key}") from exc
            raise
        return body

    async def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        validate_key(key)
        try:
            async with self._client() as client:
                await client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if _is_not_found(exc):
                return False
            raise
        return True

    async def delete(self, key: str) -> None:
        # S3 的 DeleteObject 对不存在的 key 本来就返回成功，
        # 与 local 实现的语义一致（删除是幂等的，见 LocalObjectStorage.delete）
        validate_key(key)
        async with self._client() as client:
            await client.delete_object(Bucket=self._bucket, Key=key)


def _is_no_such_bucket(exc: Any) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return code in {"NoSuchBucket", "404"}


def _is_not_found(exc: Any) -> bool:
    """`ClientError` 是不是"对象不存在"。

    `head_object` 没有权限时也返回 403 而不抛 `NoSuchKey`，
    所以不能只认 `NoSuchKey` 一种——头部的 `head_object` 更常给出
    HTTP 404 而不是错误码。两者都认，才能让 `exists` 在两种服务上行为一致。
    """
    error = getattr(exc, "response", {}).get("Error", {})
    status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
    return error.get("Code") in {"NoSuchKey", "404"} or status == 404


def build_object_storage(settings: Settings) -> ObjectStorage:
    """按配置装配对象存储。**这里是唯一的装配点**。

    仓储层有一条同样的纪律（`build_sql_repositories` 放在 `repositories/`），
    理由一样：装配散在多处时，"API 写 local、Worker 读 s3"这类不一致
    要到演示当天才会暴露，而且看起来像数据丢了。
    """
    if settings.storage_backend == "s3":
        # 配置校验已经保证这三个非空（`Settings._require_minio_settings`），
        # 这里再断言一次是为了让类型收敛，不是为了再报一次错
        assert settings.minio_endpoint and settings.minio_access_key and settings.minio_secret_key
        return S3ObjectStorage(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
        )
    return LocalObjectStorage(settings.storage_local_root)
