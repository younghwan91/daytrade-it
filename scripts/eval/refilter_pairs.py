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

**The threshold lives in the FILE, not in the environment.** `judge_batch`
re-applies `should_judge` at submit time and reads
`DAYTRADE_NEWS_PREFILTER_MAX_CODES`, but the completeness denominator does not:
`judge_days.verify()` returns `len(pairs)` and `pairs_day_summary.pairs_kept` is
the row count of the `--prefiltered` file. Setting the env var for judging while
leaving an N=11 pairs file as the denominator therefore counts pairs that will
never be submitted, and those days can never reach 100%. Measured on the n300
pool: 93,063 pairs are kept at N=11 and dropped at N=2, 48,526 of them unjudged,
leaving the worst day 42.8% unreachable.

So regenerate the pairs file at the threshold you intend and pass **that one
file** to both the judging run and the accounting::

    DAYTRADE_NEWS_PREFILTER_MAX_CODES=2 \\
      uv run python scripts/eval/refilter_pairs.py \\
        --pairs data/eval/universe_pairs_n300.jsonl \\
        --out   data/eval/universe_pairs_n300_prefiltered_n2.jsonl

The sweep/batch tools run at 2; the **live daemon keeps 11**
(`news_signal_daemon.sh`) and `FANOUT_MAX` stays 11. That split is deliberate --
on the dashboard pool live trades, 82.3% of BUYs are already fan-out 1, so
tightening wins nothing there, while on the sweep pool it halves the judging
bill. Do not "fix" the inconsistency by changing the module default; that would
change live on sweep-pool evidence. See
`docs/research/2026-09-16-prompt-v3-oracle-mining.md`, appendices 11 and 14.
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
