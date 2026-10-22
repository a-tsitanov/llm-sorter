"""Защита от TRUNCATE рабочей БД; инфраструктура не нужна (без маркера integration)."""

import pytest

from tests.integration.conftest import assert_test_db


def test_guard_refuses_non_test_database() -> None:
    with pytest.raises(RuntimeError):
        assert_test_db("postgresql+asyncpg://sorter:sorter@localhost:5432/sorter")
    assert_test_db("postgresql+asyncpg://sorter:sorter@localhost:5432/sorter_test")
