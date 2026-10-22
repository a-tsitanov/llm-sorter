"""Описание таблиц (SQLAlchemy Core) и фабрики engine/sessionmaker."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

EMBED_DIM = 1024

metadata = MetaData()

transcripts = Table(
    "transcripts",
    metadata,
    Column("id", Text, primary_key=True),
    Column("text", Text, nullable=False),
    Column("metadata", JSONB, nullable=False),
    Column("received_at", DateTime(timezone=True), nullable=False),
)

rule_versions = Table(
    "rule_versions",
    metadata,
    Column("version", Integer, primary_key=True, autoincrement=True),
    Column("yaml", Text, nullable=False),
    Column("source", Text, nullable=False),
    Column("base_version", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

active_rules = Table(
    "active_rules",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("version", Integer, ForeignKey("rule_versions.version"), nullable=False),
    CheckConstraint("id = 1", name="active_rules_single_row"),
)

classifications = Table(
    "classifications",
    metadata,
    Column("file_id", Text, primary_key=True),
    Column("tree_version", Integer, nullable=False),
    Column("node_uid", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("run_id", Text, nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

material = Table(
    "material",
    metadata,
    Column("node_uid", Text, primary_key=True),
    Column("file_id", Text, primary_key=True),
    Column("kind", Text, nullable=False),
    Column("added_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

discover_state = Table(
    "discover_state",
    metadata,
    Column("node_uid", Text, primary_key=True),
    Column("last_discover_at", DateTime(timezone=True), nullable=True),
)

embeddings = Table(
    "embeddings",
    metadata,
    Column("file_id", Text, primary_key=True),
    Column("vector", Vector(EMBED_DIM), nullable=False),
)


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
