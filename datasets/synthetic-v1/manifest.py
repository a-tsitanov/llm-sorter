#!/usr/bin/env python3
"""Write manifest.csv (id, gold_path, kind, lang, words) sorted by id."""
import csv
import json
from pathlib import Path

base = Path(__file__).resolve().parent
rows = []
for f in base.joinpath("transcripts").glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    m = d["metadata"]
    rows.append((d["id"], m["gold_path"], m["kind"], m["lang"], len(d["text"].split())))
rows.sort()
with open(base / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "gold_path", "kind", "lang", "words"])
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {base / 'manifest.csv'}")
