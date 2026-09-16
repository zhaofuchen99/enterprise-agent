"""ORM 列类型约定（详细设计 16.1）。

三条全局约定在这里落地一次，避免 16 张表各写一遍：
  - 主键 `CHAR(26)` ULID；
  - 时间统一 `DATETIME(3)` 且按 UTC 写入；
  - 状态用 `VARCHAR` + 应用侧枚举（不用 MySQL ENUM：加一个状态值就要改表结构）。

**为什么这里不建物理外键**：详细设计 16.3–16.9 为每张表列了索引，但**从未列出外键约束**；
16.12 的数据清理是按保留期批量物理删除，级联语义也没有定义。
在 `CHAR(26)` 上建外键会给高写入表（`agent_trace_event`、`agent_tool_call`）
额外引入一层索引与行锁，而收益在这个规模下接近于零。
因此关联完整性由应用层（仓储与用例）保证，**不依赖数据库**——
这是一处有意的偏离，登记在 `docs/秋招冲刺方案.md` §12 的回写清单里。

**为什么时间列直接用 MySQL 方言的 `DATETIME`**：毫秒精度（`fsp=3`）是详细设计 16.1
的硬要求，而通用 `sqlalchemy.DateTime` 不接受 `fsp`，`mysql_fsp` 这个方言参数也不被
`Column` 接受（两者都试过）。因此 ORM 模型**有意绑定 MySQL 方言**——
本项目只针对 MySQL 8，换方言本来就要重做迁移脚本。
影响是：仓储的单元测试不能改用 SQLite 跑，需要真实 MySQL 的用例一律打
`@pytest.mark.integration`（开发流程 5.6），纯逻辑用 Fake 仓储验证。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from sqlalchemy import CHAR, JSON
from sqlalchemy.dialects.mysql import DATETIME
from sqlalchemy.orm import mapped_column

#: ULID 主键。格式见详细设计 16.1：前缀(4) + 时间戳(10) + 随机(12) = 26。
ulid_pk = Annotated[str, mapped_column(CHAR(26), primary_key=True)]

#: 指向其他表 ULID 的关联列（无物理外键，见模块 docstring）。
ulid_ref = Annotated[str, mapped_column(CHAR(26))]
ulid_ref_opt = Annotated[str | None, mapped_column(CHAR(26))]

#: UTC 时间，毫秒精度。按 UTC 写入、按 UTC 读出，时区换算只在展示层做。
utc_dt = Annotated[datetime, mapped_column(DATETIME(fsp=3))]
utc_dt_opt = Annotated[datetime | None, mapped_column(DATETIME(fsp=3))]

#: 可空的 JSON 对象列。用 `Any` 而不是 `object`：读出来就是待校验的原始结构，
#: 由 Pydantic 模型在边界处负责收敛类型（开发流程 5.3：裸 dict 不得进 State）。
json_obj_opt = Annotated[dict[str, Any] | None, mapped_column(JSON)]

#: 可空的 JSON 数组列（文档里以 `_ids_json` / `_roles_json` 命名的那些）。
json_list_opt = Annotated[list[Any] | None, mapped_column(JSON)]
