from datetime import UTC, datetime, timedelta

from llm_sorter.adapters.inbound.history import filter_history, latest_snapshots
from llm_sorter.domain.messages import ClassificationSnapshot, TopicRef, Transcript

NOW = datetime(2026, 11, 11, tzinfo=UTC)


def snapshot(id_: str, revision: int, topics: list[str], minute: int = 0) -> ClassificationSnapshot:
    return ClassificationSnapshot(
        transcript=Transcript(id=id_, text="текст", received_at=NOW),
        topics=[TopicRef(id=t, path=f"root.{t}", title=t) for t in topics],
        tree_rev=revision,
        classified_at=NOW + timedelta(minutes=minute),
    )


def test_filter_after_latest_snapshot_removes_old_assignments() -> None:
    records = [
        snapshot("a", 1, ["cards"]),
        snapshot("a", 2, ["flights"]),
        snapshot("b", 1, ["cards", "flights"]),
    ]
    latest = latest_snapshots(records)
    assert [r.transcript.id for r in filter_history(latest, "cards", None)] == ["b"]
    assert {r.transcript.id for r in filter_history(latest, "root.flights", None)} == {"a", "b"}


def test_latest_revision_wins_even_if_older_cache_publishes_later() -> None:
    records = [snapshot("a", 2, ["cards"]), snapshot("a", 1, ["flights"], minute=1)]
    assert latest_snapshots(records)[0].topics[0].id == "cards"
