"""init: транскрипты, версии правил, классификации, материал discover, эмбеддинги

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "transcripts",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("metadata", JSONB, nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "rule_versions",
        sa.Column("version", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("yaml", sa.Text, nullable=False),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("base_version", sa.Integer, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "active_rules",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("version", sa.Integer, sa.ForeignKey("rule_versions.version"), nullable=False),
        sa.CheckConstraint("id = 1", name="active_rules_single_row"),
    )
    op.create_table(
        "classifications",
        sa.Column("file_id", sa.Text, primary_key=True),
        sa.Column("tree_version", sa.Integer, nullable=False),
        sa.Column("node_uid", sa.Text, nullable=False),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("run_id", sa.Text, nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "material",
        sa.Column("node_uid", sa.Text, primary_key=True),
        sa.Column("file_id", sa.Text, primary_key=True),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column(
            "added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_table(
        "discover_state",
        sa.Column("node_uid", sa.Text, primary_key=True),
        sa.Column("last_discover_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "embeddings",
        sa.Column("file_id", sa.Text, primary_key=True),
        sa.Column("vector", Vector(1024), nullable=False),
    )


def downgrade() -> None:
    for t in (
        "embeddings",
        "discover_state",
        "material",
        "classifications",
        "active_rules",
        "rule_versions",
        "transcripts",
    ):
        op.drop_table(t)
