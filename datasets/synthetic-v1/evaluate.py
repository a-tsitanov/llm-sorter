#!/usr/bin/env python3
"""Сравнить итог классификации с gold_path. Usage: evaluate.py [DATABASE_URL]

URL по умолчанию — из AppConfig (DATABASE_URL / .env). Темы, выращенные discover,
называются иначе, чем эталонные, поэтому каждая тема сопоставляется с эталоном
по большинству своих файлов (majority mapping).
"""
import asyncio
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

HERE = Path(__file__).parent


def load_gold() -> dict[str, dict[str, str]]:
    with (HERE / "manifest.csv").open(encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


async def load_results(url: str) -> dict[str, tuple[str, str]]:
    engine = create_async_engine(url)
    try:
        async with engine.connect() as c:
            query = text("select file_id, status, path from classifications")
            rows = (await c.execute(query)).all()
    finally:
        await engine.dispose()
    return {r[0]: (r[1], r[2]) for r in rows}


def topic_of(status: str, path: str) -> str | None:
    """Тема, в которую файл попал окончательно; None — other или guard_failed."""
    return path if status == "classified" else None


def main() -> int:
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        from llm_sorter.config import AppConfig

        url = AppConfig().database_url
    gold = load_gold()
    res = asyncio.run(load_results(url))
    missing = sorted(set(gold) - set(res))

    by_topic: dict[str, list[str]] = defaultdict(list)
    for fid, (status, path) in res.items():
        if fid in gold and (t := topic_of(status, path)) is not None:
            by_topic[t].append(fid)

    print(f"файлов в эталоне {len(gold)}, обработано {len(res)}, не дошли {len(missing)}")
    print("\nстатусы по kind:")
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    for fid, g in gold.items():
        kinds[g["kind"]][res[fid][0] if fid in res else "missing"] += 1
    for k, cnt in sorted(kinds.items()):
        print(f"  {k:10s} " + " ".join(f"{s}={n}" for s, n in sorted(cnt.items())))

    print("\nтемы (путь -> основной gold_path, чистота):")
    covered: Counter[str] = Counter()
    correct = total = 0
    for topic, fids in sorted(by_topic.items()):
        golds = Counter(gold[f]["gold_path"] for f in fids)
        major, n = golds.most_common(1)[0]
        covered[major] += n
        correct += n
        total += len(fids)
        rest = ", ".join(f"{g}={c}" for g, c in golds.most_common()[1:4])
        print(f"  {topic:45s} {len(fids):3d}  -> {major:24s} {n / len(fids):.2f}  {rest}")
    if total:
        print(f"\nчистота тем (взвешенная): {correct / total:.2f}")

    leaves = sorted({g["gold_path"] for g in gold.values()} - {"other"})
    print("\nпокрытие эталонных тем (файлов normal попало в тему с этим большинством):")
    for leaf in leaves:
        n_gold = sum(1 for g in gold.values() if g["gold_path"] == leaf and g["kind"] == "normal")
        print(f"  {leaf:24s} {covered[leaf]:3d}/{n_gold}")

    noise = [f for f, g in gold.items() if g["gold_path"] == "other" and f in res]
    leaked = [f for f in noise if res[f][0] == "classified"]
    print(f"\nшум/edge попал в тему: {len(leaked)}/{len(noise)}")
    for f in leaked:
        print(f"  {f} -> {res[f][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
