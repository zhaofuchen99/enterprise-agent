"""knowledge_document 增加 source_kind（SOURCE 冲突的来源内外载体）

Revision ID: 1616f9868a5a
Revises: e02f04e7e6cf
Create Date: 2026-09-17 13:36:51.594843
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = '1616f9868a5a'
down_revision: str | None = 'e02f04e7e6cf'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `server_default='INTERNAL'` 不只是为了新插入的行——MySQL 给已有行加 NOT NULL 列时
    # 必须有一个值可填，否则这张表非空时 ALTER 直接失败。默认值取 INTERNAL 是因为
    # 存量文档（若有）都来自内部制度体系，标成 EXTERNAL 会让它们在 SOURCE 冲突判定里
    # 被当作「外部信号」，方向正好是反的。
    op.add_column(
        "knowledge_document",
        sa.Column("source_kind", sa.String(length=16), server_default="INTERNAL", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("knowledge_document", "source_kind")
