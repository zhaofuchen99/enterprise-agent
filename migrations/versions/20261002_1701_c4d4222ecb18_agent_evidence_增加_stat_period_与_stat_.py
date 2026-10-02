"""agent_evidence 增加 stat_period 与 stat_cutoff（TIME 冲突的两个判据字段）

Revision ID: c4d4222ecb18
Revises: 1616f9868a5a
Create Date: 2026-10-02 17:01:53.121179
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = 'c4d4222ecb18'
down_revision: str | None = '1616f9868a5a'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `stat_period` 是**补一个已经漏掉的列**，不是本批新增的能力：它在 2026-09-27
    # 落 TIME 期间判据时就进了 `Evidence`（`app/domain/evidence.py`），却没进这张表
    # ——于是"落库的证据"与"答案里引用的证据"在字段集合上就对不上了，而
    # `_evidence_of` 的 docstring 明写两者应当逐字段对称。本批要加的 `stat_cutoff`
    # 与它配对使用，两列一起补。
    #
    # 两列都可空：制度、产品资料没有统计期间，非报告类文档也没有截止日——
    # 空值是"不知道"这个合法语义的载体，不是待补的数据。
    op.add_column("agent_evidence", sa.Column("stat_period", sa.String(length=16), nullable=True))
    op.add_column("agent_evidence", sa.Column("stat_cutoff", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_evidence", "stat_cutoff")
    op.drop_column("agent_evidence", "stat_period")
