from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from llm_sorter.domain.actions import Action, TranslateAction
from llm_sorter.domain.classification import Transcript

NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("file_id", ["a", "call-01.2026_x", "A.B"])
def test_transcript_id_accepts_safe_names(file_id: str) -> None:
    assert Transcript(id=file_id, text="", received_at=NOW).id == file_id


@pytest.mark.parametrize("file_id", ["", "../etc", "a/b", "a..b", "..", "a b", "ф", "a\\b"])
def test_transcript_id_rejects_unsafe_names(file_id: str) -> None:
    with pytest.raises(ValidationError):
        Transcript(id=file_id, text="", received_at=NOW)


@pytest.mark.parametrize("lang", ["en", "rus", "pt-BR", "zh-Hant"])
def test_translate_to_accepts_language_tags(lang: str) -> None:
    assert TranslateAction(type="translate", to=lang).to == lang


@pytest.mark.parametrize("lang", ["", "EN", "e", "english", "en/../x", "en-", "en-B"])
def test_translate_to_rejects_garbage(lang: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Action).validate_python({"type": "translate", "to": lang})
