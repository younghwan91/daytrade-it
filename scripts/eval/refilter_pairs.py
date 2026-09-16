#!/usr/bin/env python3
"""Re-apply the adopted pre-LLM filter to an already-built pairs file.

    uv run python scripts/eval/refilter_pairs.py \\
        --pairs data/eval/universe_pairs_n300.jsonl \\
        --out   data/eval/universe_pairs_n300_prefiltered.jsonl

`build_universe_events.py --prefilter` writes the same file, but it rebuilds
the pool from the DB and the C1 backfill, which takes hours. When only the
*filter* changed, this is the equivalent shortcut: `related_n` in the pairs
file is exactly the fan-out `should_judge` reads, and the filter is pure, so
filtering the frozen pool gives byte-identical rows.

Reads and writes files only -- no DB, no network, no Anthropic call.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from daytrade_it.application.news.mention_prefilter import should_judge, skip_reason


def refilter(lines: list[str]) -> tuple[list[str], Counter[str]]:
    """Kept lines and a per-rule skip histogram, in input order."""
    kept: list[str] = []
    reasons: Counter[str] = Counter()
    for line in lines:
        row = json.loads(line)
        if should_judge(
            row.get("name"),
            row.get("code"),
            row.get("title"),
            row.get("body"),
            row.get("related_n"),
            enabled=True,
        ):
            kept.append(line)
        else:
            reason = skip_reason(row.get("title"), row.get("related_n"), enabled=True)
            reasons[(reason or "?").split(":")[0]] += 1
    return kept, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lines = args.pairs.read_text().splitlines(keepends=True)
    kept, reasons = refilter(lines)

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text("".join(kept))
    tmp.replace(args.out)

    total = len(lines)
    skipped = total - len(kept)
    print(f"total {total:,} kept {len(kept):,} ({len(kept) / total:.2%}) skipped {skipped:,}")
    for rule, n in reasons.most_common():
        print(f"  {rule:<10} {n:,} ({n / total:.2%})")


if __name__ == "__main__":
    main()
