"""Per-day judging ledger for a pairs file: what is left, and what it costs.

    uv run python scripts/eval/pairs_day_summary.py \\
        --pairs data/eval/universe_pairs_n300.jsonl \\
        --prefiltered data/eval/universe_pairs_n300_prefiltered.jsonl \\
        --out data/eval/universe_pairs_n300_summary.csv

One row per trading day, **most recent first** (that is the order a budget is
spent in, so it is the order the file is written in):

| column | meaning |
|---|---|
| `day` | trading day |
| `pairs_total` | (article, ticker) pairs in the universe that day, before the pre-LLM filter |
| `pairs_kept` | pairs the adopted prefilter keeps -- the judging denominator |
| `already_judged` | kept pairs with a **valid** cache row for the current input |
| `missing` | `pairs_kept - already_judged` -- what a top-up has to pay for |
| `est_cost_usd` | Batch-discounted Haiku 4.5 cost of exactly those `missing` requests |

The cost is not a per-pair constant: it renders each missing pair's real
request and counts its characters, so a day of long headlines costs more than a
day of short ones. `--chars-per-token 1.321` is the measured value from A6's
`count_tokens` calibration; `--est-output-tokens 68` the measured judge output.

Reads files only -- no DB, no network, no Anthropic call.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_batch import (  # noqa: E402
    CACHE,
    DEFAULT_MODEL,
    build_request,
    cache_row_valid,
    estimate_cost,
)

#: A6 calibration: mean input 1,124 tokens at 1.321 chars/token, 68 output tokens.
CHARS_PER_TOKEN = 1.321
OUTPUT_TOKENS = 68

FIELDS = ("day", "pairs_total", "pairs_kept", "already_judged", "missing", "est_cost_usd")


def group_by_day(path: Path) -> dict[str, list[dict]]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_day[str(row["day"])].append(row)
    return dict(by_day)


def count_by_day(path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with path.open() as f:
        for line in f:
            if line.strip():
                counts[str(json.loads(line)["day"])] += 1
    return dict(counts)


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["key"]] = row
    return rows


def summarize(
    totals: dict[str, int],
    kept_by_day: dict[str, list[dict]],
    cache: dict[str, dict],
    model: str,
    chars_per_token: float,
    output_tokens: int,
) -> list[dict]:
    """One dict per day, most recent first."""
    rows: list[dict] = []
    for day in sorted(set(totals) | set(kept_by_day), reverse=True):
        kept = kept_by_day.get(day, [])
        missing = [p for p in kept if not cache_row_valid(cache.get(p["key"], {}), p, model)]
        est = estimate_cost(
            [build_request(p, model) for p in missing], chars_per_token, output_tokens
        )
        rows.append(
            {
                "day": day,
                "pairs_total": totals.get(day, len(kept)),
                "pairs_kept": len(kept),
                "already_judged": len(kept) - len(missing),
                "missing": len(missing),
                "est_cost_usd": round(est["usd"], 6),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=Path, required=True, help="unfiltered pairs jsonl")
    ap.add_argument(
        "--prefiltered",
        type=Path,
        default=None,
        help="prefiltered pairs jsonl (default: --pairs, i.e. no filter applied)",
    )
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN)
    ap.add_argument("--est-output-tokens", type=int, default=OUTPUT_TOKENS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    totals = count_by_day(args.pairs)
    kept_by_day = group_by_day(args.prefiltered or args.pairs)
    cache = load_cache(args.cache)
    rows = summarize(
        totals,
        kept_by_day,
        cache,
        args.model,
        args.chars_per_token,
        args.est_output_tokens,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    total_missing = sum(r["missing"] for r in rows)
    total_cost = sum(r["est_cost_usd"] for r in rows)
    kept = sum(r["pairs_kept"] for r in rows)
    print(
        f"{len(rows)} days -> {args.out}\n"
        f"pairs_total {sum(r['pairs_total'] for r in rows):,}, kept {kept:,}, "
        f"already_judged {kept - total_missing:,}, missing {total_missing:,}\n"
        f"estimated remaining cost ${total_cost:.2f} "
        f"(${total_cost / len(rows):.3f} per day over {len(rows)} days)"
    )


if __name__ == "__main__":
    main()
