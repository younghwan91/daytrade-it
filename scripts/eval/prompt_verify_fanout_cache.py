"""Two correctness checks before recommending a sweep-only fan-out tightening.

team-lead's questions:

  (1) The judgment cache is keyed per pair. Can tightening fan-out for FUTURE
      judging make already-judged pairs look unjudged, or unjudged pairs look
      judged?
  (2) Day-completeness accounting must use the same threshold the judging run
      used. If it does not, days silently never read as complete.

Reads files only. No DB, no network, no Anthropic call.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/eval")
from judge_batch import DEFAULT_MODEL, cache_row_valid, pair_input_hash  # noqa: E402

from daytrade_it.application.news import mention_prefilter as mp  # noqa: E402

PAIRS = Path("data/eval/universe_pairs_n300.jsonl")
CACHE = Path("data/eval/cache/universe_v2.jsonl")


def set_threshold(n: int | None) -> None:
    if n is None:
        os.environ.pop(mp._FANOUT_ENV_VAR, None)
    else:
        os.environ[mp._FANOUT_ENV_VAR] = str(n)


def main() -> None:
    pairs = [json.loads(line) for line in PAIRS.open()]
    print(f"pairs: {len(pairs)}  (module default FANOUT_MAX = {mp.FANOUT_MAX})")

    # ---- (1) is the cache key independent of the fan-out threshold? ----
    print("\n=== (1) cache key vs fan-out threshold ===")
    sample = pairs[:2000]
    hashes = {}
    for n in (None, 11, 2, 1):
        set_threshold(n)
        hashes[str(n)] = [pair_input_hash(p, DEFAULT_MODEL) for p in sample]
    base = hashes["None"]
    for k, v in hashes.items():
        same = sum(1 for a, b in zip(base, v, strict=True) if a == b)
        print(f"  threshold={k:>4s}: {same}/{len(base)} input hashes identical to default")
    set_threshold(None)

    # and is it independent of related_n itself?
    mutated = [{**p, "related_n": (p.get("related_n") or 1) + 50} for p in sample]
    same = sum(
        1
        for a, b in zip(base, [pair_input_hash(p, DEFAULT_MODEL) for p in mutated], strict=True)
        if a == b
    )
    print(f"  related_n mutated by +50: {same}/{len(base)} input hashes identical")
    print("  -> the hash covers (model, system, user) only; fan-out is not in it.")

    # ---- (2) does the denominator follow the judging threshold? ----
    print("\n=== (2) completeness denominator vs judging threshold ===")
    cache_rows: dict[str, list[dict]] = {}
    with CACHE.open() as f:
        for line in f:
            r = json.loads(line)
            cache_rows.setdefault(r["key"], []).append(r)
    judged_keys = set(cache_rows)

    kept: dict[str, set[str]] = {}
    for n in (11, 2):
        set_threshold(n)
        kept[str(n)] = {
            p["key"]
            for p in pairs
            if mp.should_judge(
                p.get("name"),
                p.get("code"),
                p.get("title"),
                p.get("body"),
                p.get("related_n"),
                enabled=True,
            )
        }
        print(f"  prefilter N={n}: keeps {len(kept[str(n)]):,} of {len(pairs):,} pairs")
    set_threshold(None)

    only11 = kept["11"] - kept["2"]
    print(f"\n  pairs kept at N=11 but dropped at N=2: {len(only11):,}")
    unjudged_gap = len([k for k in only11 if k not in judged_keys])
    print(f"    of which not yet judged: {unjudged_gap:,}")
    print(
        "\n  If judging runs at N=2 while the denominator file is the N=11 build,\n"
        f"  those {unjudged_gap:,} pairs are counted as owed but never submitted --\n"
        "  every affected day is permanently short of 100%."
    )

    # per-day worst case
    by_day: Counter[str] = Counter()
    day_kept11: Counter[str] = Counter()
    for p in pairs:
        if p["key"] in only11 and p["key"] not in judged_keys:
            by_day[p["day"]] += 1
    for p in pairs:
        if p["key"] in kept["11"]:
            day_kept11[p["day"]] += 1
    worst = sorted(
        ((d, by_day[d], day_kept11[d]) for d in day_kept11),
        key=lambda t: -(t[1] / t[2] if t[2] else 0),
    )[:5]
    print("\n  worst days under that mismatch (unreachable / N=11 denominator):")
    for d, miss, tot in worst:
        print(f"    {d}  {miss:5d} / {tot:5d} = {miss / tot:.1%} unreachable")

    # ---- the fix: same file on both sides ----
    print("\n=== the fix ===")
    print("  `refilter_pairs.py` reads the same env var, so regenerating the pairs")
    print("  file at N=2 makes the judging set and the denominator the same object:")
    set_threshold(2)
    n2 = [
        p
        for p in pairs
        if mp.should_judge(
            p.get("name"),
            p.get("code"),
            p.get("title"),
            p.get("body"),
            p.get("related_n"),
            enabled=True,
        )
    ]
    set_threshold(None)
    still = sum(1 for p in n2 if p["key"] not in judged_keys)
    print(f"  N=2 pairs file: {len(n2):,} rows, {still:,} still unjudged -> that is the")
    print("  denominator AND the submit set, so a day can reach 100%.")
    judged_in_n2 = sum(
        1
        for p in n2
        if any(cache_row_valid(r, p, DEFAULT_MODEL) for r in cache_rows.get(p["key"], []))
    )
    print(
        f"  already-judged rows still valid under N=2: {judged_in_n2:,} "
        f"({judged_in_n2 / len(n2):.1%} of the N=2 file)"
    )


if __name__ == "__main__":
    main()
