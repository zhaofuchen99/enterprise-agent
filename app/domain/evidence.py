"""证据领域模型（详细设计 13.1）。

**这是「答案有证据」这条能力的落点**，也是 SQL 与 RAG 两个来源之所以能被
一致对待的原因：Analysis 与 Reviewer 读的是同一个 `Evidence`，不必关心
它来自一次 `SELECT` 还是某篇制度的第 3 节。切片内先由 SQL Tool 产出，
Phase 5 的 RAG 往同一张表里写。

## 与文档的两处对齐说明

1. **`source_type` 取 `SQL / DOCUMENT / WEB`**（按详设 13.1），而
   `app/infrastructure/models/evidence.py` 的字段注释原写作
   `SQL / RAG / SEARCH`。两者是同一件事的两种叫法，**以 13.1 为准**——
   13.1 是 Schema 定义处，13.2「证据优先级」与 11.4 的 `ChunkMetadata`
   也都按「文档 / 网页」描述。已同步修正该处注释，表结构未变（`String(16)` 装得下两者）。
2. `locator` / `scope` 在文档里是裸 `dict`，这里**保持裸 dict 不收紧**：
   它们的键按来源类型变化（SQL 是 `call_id + sql_fingerprint + result_slice`，
   文档是 `document_id + version + chunk_id`），用一个联合类型去描述只会
   让每一种来源都要处理另外几种的字段。约束放在产出侧（`tools/sql/evidence.py`）。

**证据必须可定位、可复算**：`content_hash` 不是校验和装饰，它让「同一份文件
两个版本」「同一条 SQL 两次执行」可以被判定为同一份证据，这是 13.4 冲突检测
第 1 步「按 metric_code 或实体主题分组」的前提。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: 证据来源。切片内只会产出 `SQL`；`DOCUMENT` 待 Phase 5。
EvidenceSource = Literal["SQL", "DOCUMENT", "WEB"]

#: 可靠性等级（详细设计 12.3）：内部业务库与已发布制度高于外部来源。
#: 「外部不得覆盖内部事实」（FR-SEARCH-001）在 Reviewer 侧就是靠它判的。
Reliability = Literal["HIGH", "MEDIUM", "LOW"]


class TimeRange(BaseModel):
    """闭开区间 `[start, end)`。

    **半开不是随手定的**：`order_date >= '2025-07-01' AND order_date < '2025-10-01'`
    是 SQL 里表示「Q3」的唯一无歧义写法，闭区间会把 9/30 当天的数据算两次或漏掉。
    冲突检测（13.4 第 2 步「规范化时区、日期闭开区间」）比较的正是这个类型，
    取值约定必须在这里一次说清，否则两个来源各按各的理解表达同一段时间。
    """

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


class Evidence(BaseModel):
    """一条证据（详细设计 13.1）。

    Attributes:
        id: `evd_` 前缀的 26 位 ID。
        source_type: SQL / DOCUMENT / WEB。
        title: 短标题，用于展示与日志。**不含原始行数据**（19.4 脱敏纪律）。
        claim: 这条证据支撑的具体陈述——「答案有证据」里被引用的就是它。
        locator: 定位信息，键按来源类型变化，见模块 docstring。
        event_time: 业务时间区间（不是抓取时间）。
        retrieved_at: 取到这条证据的时刻。与 `event_time` 必须分开：
            「9 月 30 日截止的报告」与「11 月 3 日读到的这份报告」是两件事，
            13.4 的 TIME 冲突检测比对的是前者。
        metric_code: 对应指标目录的 code，冲突检测按它分组。
        definition_version: 指标口径版本。口径不同即 DEFINITION 冲突（13.4 第 5 步）。
        scope: 维度范围，如 `{"region": ["华东"]}`。SCOPE 冲突的比对依据。
        reliability: 可靠性等级。
        content_hash: 证据内容的哈希，用于判同与去重。
        access_level: 访问级别（TBC-07 的 INTERNAL / CONFIDENTIAL）。
            SQL 证据取所涉列的**最高**敏感级别——一条证据的密级由它最敏感的那部分决定。
    """

    id: str = Field(pattern=r"^evd_[0-9A-HJKMNP-TV-Z]{22}$")
    source_type: EvidenceSource
    title: str
    claim: str
    locator: dict[str, object]
    event_time: TimeRange | None = None
    retrieved_at: datetime
    metric_code: str | None = None
    definition_version: str | None = None
    scope: dict[str, str | list[str]] = Field(default_factory=dict)
    reliability: Reliability = "MEDIUM"
    content_hash: str = Field(min_length=64, max_length=64)
    access_level: str = "INTERNAL"

    def group_key(self) -> tuple[str, str]:
        """冲突检测的分组键（13.4 第 1 步）：按指标分组，无指标时退化为来源+标题。"""
        return (self.metric_code or "", self.title)


__all__ = ["Evidence", "EvidenceSource", "Reliability", "TimeRange"]
