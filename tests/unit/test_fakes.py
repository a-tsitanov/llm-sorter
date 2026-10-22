from tests.fakes import FakeMaterial


async def test_material_counts_both_kinds_and_mark_resets():
    m = FakeMaterial()
    await m.add("n", "f1", "other")
    await m.add("n", "f2", "leaf")
    assert await m.count_new("n") == 2
    assert await m.due_nodes(2) == ["n"]
    assert await m.other_size("n") == 1
    await m.mark_discovered("n")
    assert await m.count_new("n") == 0
    assert await m.due_nodes(1) == []
    await m.add("n", "f3", "leaf")
    assert await m.count_new("n") == 1
    assert await m.other_size("n") == 1


async def test_fake_material_mark_cutoff():
    from datetime import UTC, datetime

    m = FakeMaterial()
    await m.add("n", "a", "other")
    cutoff = datetime.now(UTC)
    while datetime.now(UTC) <= cutoff:  # часы должны уйти вперёд, иначе b == cutoff
        pass
    await m.add("n", "b", "other")
    await m.mark_discovered("n", at=cutoff)
    assert await m.count_new("n") == 1
