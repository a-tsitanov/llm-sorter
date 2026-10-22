#!/usr/bin/env python3
"""Validate synthetic-v1 transcripts. Usage: validate.py [dir]  (dir defaults to ./transcripts next to this script)."""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

LEAVES = [
    "finance/crypto/trading", "finance/crypto/wallets", "finance/banking/loans",
    "finance/banking/cards", "health/dental", "health/insurance",
    "travel/flights", "travel/hotels", "tech/internet", "tech/mobile",
]
GOLD = set(LEAVES) | {"other"}
LANGS = {"ru", "en", "de", "es", "uk", "kk"}
KINDS = {"normal", "near_miss", "noise", "edge"}
ID_RE = re.compile(r"^[a-z0-9-]+$")
EXPECTED = {"normal": 12, "near_miss": 15, "noise": 15, "edge": 6}


def check(path: Path):
    errs, rec = [], None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"invalid JSON: {e}"], None
    if not isinstance(d, dict):
        return ["top level is not an object"], None
    if set(d) != {"id", "text", "metadata"}:
        errs.append(f"keys must be exactly id/text/metadata, got {sorted(d)}")
        return errs, None
    i, text, md = d["id"], d["text"], d["metadata"]
    if not isinstance(i, str) or not ID_RE.match(i):
        errs.append(f"bad id {i!r}")
    elif i != path.stem:
        errs.append(f"id {i!r} != filename {path.stem!r}")
    if not isinstance(text, str):
        errs.append("text is not a string")
        text = ""
    if not isinstance(md, dict):
        errs.append("metadata is not an object")
        return errs, None
    gp, lang, kind = md.get("gold_path"), md.get("lang"), md.get("kind")
    if gp not in GOLD:
        errs.append(f"bad gold_path {gp!r}")
    if lang not in LANGS:
        errs.append(f"bad lang {lang!r}")
    if kind not in KINDS:
        errs.append(f"bad kind {kind!r}")
    if kind == "near_miss" and not (isinstance(md.get("note"), str) and md["note"].strip()):
        errs.append("near_miss requires non-empty note")
    if kind in ("noise", "edge") and gp != "other":
        errs.append(f"{kind} must have gold_path 'other'")
    if kind in ("normal", "near_miss") and gp == "other":
        errs.append(f"{kind} must not have gold_path 'other'")
    words = len(text.split())
    if kind != "edge" and words < 100:
        errs.append(f"only {words} words (<100)")
    rec = {"id": i, "gold_path": gp, "kind": kind, "lang": lang, "words": words}
    return errs, rec


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    tdir = base / "transcripts" if (base / "transcripts").is_dir() else base
    files = sorted(tdir.glob("*.json"))
    n_err, recs = 0, []
    for f in files:
        errs, rec = check(f)
        for e in errs:
            print(f"ERROR {f.name}: {e}")
        n_err += len(errs)
        if rec:
            recs.append(rec)
    print(f"\nFiles: {len(files)}  errors: {n_err}\n")

    ok = [r for r in recs if r["gold_path"] in GOLD and r["kind"] in KINDS]
    kinds = ["normal", "near_miss", "noise", "edge"]
    print("gold_path x kind")
    print(f"{'gold_path':26}" + "".join(f"{k:>11}" for k in kinds))
    c = Counter((r["gold_path"], r["kind"]) for r in ok)
    for gp in LEAVES + ["other"]:
        print(f"{gp:26}" + "".join(f"{c[(gp, k)]:>11}" for k in kinds))
    print("\nby lang:", dict(sorted(Counter(r["lang"] for r in recs).items())))
    w = [r["words"] for r in recs]
    if w:
        print(f"words: min {min(w)}  median {statistics.median(w)}  max {max(w)}")

    print("\nGroup completeness")
    ck = Counter(r["kind"] for r in ok)
    incomplete = False
    for leaf in LEAVES:
        have = c[(leaf, "normal")]
        if have != EXPECTED["normal"]:
            incomplete = True
            print(f"  INCOMPLETE normal {leaf}: {have}/{EXPECTED['normal']}")
    for k in ("near_miss", "noise", "edge"):
        if ck[k] != EXPECTED[k]:
            incomplete = True
            print(f"  INCOMPLETE {k}: {ck[k]}/{EXPECTED[k]}")
    if not incomplete:
        print("  all groups complete")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
