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

import hashlib
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
    #: 工具调用与证据（Phase 4 起）。两者都是「答案为什么成立」的追溯入口，
    #: `tcl_` / `evd_` 在日志里互不混淆——证据的 locator 里同时出现这两个 ID，
    #: 前缀相同的话一眼看不出谁指谁（详见 `SqlToolResult.evidence`）。
    TOOL_CALL = "tcl"
    EVIDENCE = "evd"

    # 业务演示库的维度表（Phase 2 的 `scripts/business_seed.py`）。
    # 它们属于「企业已有数据」，本可以自定编号方案；沿用同一套前缀 + 26 位
    # 约束，是为了 SQL Tool 在做 JOIN 时两边的键长得一样，
    # 排查时不必在脑子里切换两种 ID 形态。
    REGION = "rgn"
    CHANNEL = "chn"
    PRODUCT_LINE = "pln"
    PRODUCT = "prd"
    CUSTOMER = "cst"
    ORDER = "ord"


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


def deterministic_id(prefix: IdPrefix, seed: str) -> str:
    """由种子确定性派生 ID，格式与长度同 `new_id`。

    **只给演示夹具用**。真实实体的 ID 必须随机——可预测的主键意味着
    别人能枚举出你的任务、会话，`new_id` 的 60 位随机部分正是为此存在的。

    之所以需要它：开发流程 6.3 的多实例验收要求两个 API 实例看到**同一个用户**，
    而两个进程各自播种演示账号时若拿到不同的随机 ID，令牌互不认、限流计数
    也各算各的，那条验收命令从根上跑不起来。演示账号是夹具不是实体，
    固定下来才是它该有的样子。Phase 2 接入 MySQL 后真实用户走 `new_id`。
    """
    digest = hashlib.sha256(f"{prefix.value}:{seed}".encode()).digest()
    body = _encode(int.from_bytes(digest[:12], "big"), _TIMESTAMP_CHARS + _RANDOM_CHARS)
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
