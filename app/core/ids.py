"""带前缀的、按创建时间字典序递增的主键生成。

**为什么是 26 位**：详细设计 16.3 / 16.5 把主键统一定义为 `CHAR(26)`，
而接口示例又写作 `tsk_01J...`（17.1）。两者只有在前缀也计入 26 位时才同时成立，
因此本模块把 ID 固定为 `前缀(4) + 时间戳(10) + 随机(12) = 26`：

    tsk_  01J7QV8K2M  X4B9ZP0R7NQD
    └┬─┘ └────┬───┘  └─────┬────┘
    前缀     48 位毫秒时间   60 位随机

时间戳在高位，所以同一前缀下 ID 与创建时间同序，`ORDER BY id` 等价于按时间排序；
随机部分 60 位，单机演示规模下碰撞概率可忽略。

字符集用 Crockford Base32（去掉 I/L/O/U），避免人工抄录时把 1/I、0/O 看混。
"""

from __future__ import annotations

import secrets
import time
from enum import StrEnum

#: Crockford Base32
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_TIMESTAMP_CHARS = 10
_RANDOM_CHARS = 12
_TIMESTAMP_BITS = 48
_RANDOM_BITS = _RANDOM_CHARS * 5  # 60


class IdPrefix(StrEnum):
    """ID 前缀。

    前缀不只是好看——日志里 `tsk_xxx` 与 `cnv_xxx` 一眼分清，
    也能在参数传错时靠长度/前缀一眼看出，而不是等到查库为空才发现。
    """

    USER = "usr"
    CONVERSATION = "cnv"
    TASK = "tsk"
    TRACE = "trc"


def _encode(value: int, length: int) -> str:
    """把整数编码成定长 Base32（高位在前，不足左侧补 0）。"""
    chars: list[str] = []
    for _ in range(length):
        value, remainder = divmod(value, 32)
        chars.append(_ALPHABET[remainder])
    return "".join(reversed(chars))


def new_id(prefix: IdPrefix) -> str:
    """生成 `{prefix}_{22 位 Base32}`，总长固定 26。"""
    timestamp_ms = int(time.time() * 1000) & ((1 << _TIMESTAMP_BITS) - 1)
    body = _encode(timestamp_ms, _TIMESTAMP_CHARS) + _encode(
        secrets.randbits(_RANDOM_BITS), _RANDOM_CHARS
    )
    return f"{prefix}_{body}"


def new_conversation_id() -> str:
    return new_id(IdPrefix.CONVERSATION)


def new_task_id() -> str:
    return new_id(IdPrefix.TASK)


def new_trace_id() -> str:
    """每个请求一个 trace_id，API / 队列 / Worker 全程沿用（详细设计 19.4.1）。"""
    return new_id(IdPrefix.TRACE)


def id_pattern(prefix: IdPrefix) -> str:
    """`new_id` 产出的格式对应的正则，供接口层做入参校验。

    在这里生成而不是在接口层手写，是为了让格式只有一处定义：
    ID 长度或字符集改了，校验规则跟着改，不会留下一个对不上的旧正则。
    """
    return rf"^{prefix}_[0-9A-HJKMNP-TV-Z]{{{_TIMESTAMP_CHARS + _RANDOM_CHARS}}}$"
