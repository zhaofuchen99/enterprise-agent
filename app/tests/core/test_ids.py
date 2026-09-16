"""ID 生成：格式、唯一性、按时间有序。"""

from __future__ import annotations

import re
import time

from app.core.ids import IdPrefix, id_pattern, new_conversation_id, new_task_id, new_trace_id


def test_id_is_fixed_26_chars_with_prefix() -> None:
    """详细设计 16.3 / 16.5 把主键定义为 CHAR(26)，接口示例又写作 tsk_01J...。

    两个要求只有在前缀计入长度时才同时成立，这条用例把该约定钉死：
    日后有人把 ULID 换成 26 位标准格式（前缀另外拼），长度会变成 30，这里会红。
    """
    task_id = new_task_id()
    assert len(task_id) == 26
    assert task_id.startswith("tsk_")
    assert len("tsk_") + 22 == 26


def test_all_prefixes() -> None:
    assert new_conversation_id().startswith("cnv_")
    assert new_trace_id().startswith("trc_")


def test_ids_are_lexicographically_time_ordered() -> None:
    """时间戳在高位，因此 ORDER BY id 等价于按创建时间排序。

    Phase 2 建表后不会再用自增列，这条性质是分页与范围扫描能走索引的前提。
    """
    first = new_task_id()
    time.sleep(0.002)  # 时间戳是按毫秒编码的，跨过一个毫秒边界才可比
    second = new_task_id()
    assert first < second


def test_ids_are_unique() -> None:
    ids = {new_task_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_id_pattern_matches_generated_ids() -> None:
    pattern = re.compile(id_pattern(IdPrefix.CONVERSATION))
    for _ in range(50):
        assert pattern.match(new_conversation_id())


def test_id_pattern_rejects_malformed_ids() -> None:
    pattern = re.compile(id_pattern(IdPrefix.CONVERSATION))
    for bad in (
        "cnv_short",
        "tsk_0000000000000000000000",  # 前缀不对
        "cnv_00000000000000000000000",  # 多了 1 位
        "cnv_000000000000000000000",  # 少了 1 位
        "cnv_000000000000000000000I",  # Crockford Base32 里没有 I
        "cnv_000000000000000000000O",  # 也没有 O
    ):
        assert not pattern.match(bad), bad
