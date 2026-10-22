"""Общая инфраструктура интеграционных тестов: отдельная БД *_test с миграциями."""

import asyncio
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url

from llm_sorter.adapters.outbound.postgres import make_engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DB = "postgresql+asyncpg://sorter:sorter@localhost:5432/sorter_test"
TABLES = (
    "embeddings, discover_state, material, classifications, active_rules, "
    "rule_versions, transcripts"
)


def assert_test_db(url: str) -> None:
    """TRUNCATE разрешён только в базе, имя которой оканчивается на _test."""
    name = make_url(url).database or ""
    if not name.endswith("_test"):
        raise RuntimeError(f"refusing to touch non-test database {name!r}")


async def _ensure_database(url: str) -> None:
    target = make_url(url)
    admin = make_engine(
        target.set(database="postgres").render_as_string(hide_password=False),
    ).execution_options(isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target.database}
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    finally:
        await admin.dispose()


async def truncate(url: str) -> None:
    assert_test_db(url)
    engine = make_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Создаёт тестовую БД при отсутствии (init.sql не перезапускается на старом томе)
    и применяет миграции."""
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DB)
    assert_test_db(url)
    asyncio.run(_ensure_database(url))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")
    return url


@pytest.fixture
async def clean_db(test_db_url: str) -> str:
    await truncate(test_db_url)
    return test_db_url
