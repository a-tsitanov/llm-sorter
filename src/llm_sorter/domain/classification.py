from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Status(StrEnum):
    CLASSIFIED = "classified"
    IN_OTHER = "in_other"
    GUARD_FAILED = "guard_failed"


class Transcript(_Frozen):
    # id попадает в пути файлов и id workflow: только безопасные символы, без ".."
    id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime

    @field_validator("id")
    @classmethod
    def _no_parent_ref(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("id must not contain '..'")
        return v


class Classification(_Frozen):
    file_id: str
    tree_version: int
    node_uid: str
    path: str
    status: Status
    run_id: str | None = None  # токен запуска для идемпотентного повтора finalize

    @property
    def final(self) -> bool:
        """classified и guard_failed неизменяемы; in_other можно переобработать."""
        return self.status != Status.IN_OTHER


class FinalClassificationExists(Exception):
    """Попытка перезаписать финальную классификацию."""
