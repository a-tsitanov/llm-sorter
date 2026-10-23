from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llm_sorter.domain.messages import (
    Considered,
    OtherItem,
    Resolved,
    Transcript,
    dump_record,
    load_record,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("file_id", ["a", "call-01.2026_x", "A.B"])
def test_transcript_id_accepts_safe_names(file_id: str) -> None:
    assert Transcript(id=file_id, text="", received_at=NOW).id == file_id


@pytest.mark.parametrize("file_id", ["", "../etc", "a/b", "a..b", "..", "a b", "ф", "a\\b"])
def test_transcript_id_rejects_unsafe_names(file_id: str) -> None:
    with pytest.raises(ValidationError):
        Transcript(id=file_id, text="", received_at=NOW)


def _item() -> OtherItem:
    return OtherItem(
        id="m1",
        text="текст",
        metadata={"source": "tg"},
        received_at=NOW,
        node="finance",
        trail=["finance"],
        reason="no_match",
        tree_rev=3,
        at=NOW,
    )


@pytest.mark.parametrize(
    "record",
    [
        _item(),
        Resolved(ids=["m1", "m2"], at=NOW),
        Considered(node="root", ids=["m3"], at=NOW),
    ],
)
def test_stream_record_roundtrip(record: OtherItem | Resolved | Considered) -> None:
    assert load_record(dump_record(record)) == record


def test_item_serialized_with_kind() -> None:
    assert b'"kind":"item"' in dump_record(_item())


def test_item_back_to_transcript() -> None:
    t = _item().transcript()
    assert (t.id, t.text, t.metadata, t.received_at) == ("m1", "текст", {"source": "tg"}, NOW)


def test_load_record_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        load_record(b'{"kind": "nope"}')
