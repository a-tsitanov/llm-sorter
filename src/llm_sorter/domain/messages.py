"""Сообщения между слоями: входящий транскрипт, результат и записи stream `other`."""

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Transcript(_Frozen):
    # id попадает в routing и логи: только безопасные символы, без ".."
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


Reason = Literal["no_match", "gate_failed", "model_error", "leaf"]


class TopicRef(_Frozen):
    id: str
    path: str
    title: str


class Classified(_Frozen):
    id: str
    text: str
    metadata: dict[str, Any]
    topic: TopicRef
    trail: list[str]
    gate: Literal["passed", "skipped"] = "passed"
    tree_rev: int
    classified_at: datetime


class OtherItem(_Frozen):
    """Сообщение, застрявшее на узле `node` (или материал листа при reason=leaf)."""

    kind: Literal["item"] = "item"
    id: str
    text: str
    metadata: dict[str, Any]
    received_at: datetime
    node: str
    trail: list[str]
    reason: Reason
    tree_rev: int
    reprocess: int = 0
    final: bool = False
    at: datetime

    def transcript(self) -> Transcript:
        return Transcript(
            id=self.id, text=self.text, metadata=self.metadata, received_at=self.received_at
        )


class Resolved(_Frozen):
    """Эти id перегнаны дообучением: прежние копии в stream больше не в буфере."""

    kind: Literal["resolved"] = "resolved"
    ids: list[str]
    at: datetime


class Considered(_Frozen):
    """Эти id узла уже были в discover без результата: в порог не засчитываются."""

    kind: Literal["considered"] = "considered"
    node: str
    ids: list[str]
    at: datetime


class Marker(_Frozen):
    """Метка дообучения: прочитав свою метку, оно знает, что догнало конец stream."""

    kind: Literal["marker"] = "marker"
    token: str
    at: datetime


AnyRecord = OtherItem | Resolved | Considered | Marker
StreamRecord = Annotated[AnyRecord, Field(discriminator="kind")]
_RECORD: TypeAdapter[AnyRecord] = TypeAdapter(StreamRecord)


def dump_record(r: AnyRecord) -> bytes:
    return r.model_dump_json().encode()


def load_record(b: bytes) -> AnyRecord:
    """ValueError на мусоре (pydantic ValidationError - подкласс ValueError)."""
    try:
        return _RECORD.validate_json(b)
    except (ValidationError, json.JSONDecodeError) as e:
        raise ValueError(str(e)) from e
