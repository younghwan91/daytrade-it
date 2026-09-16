"""Split judged days into the census cohort and the stratified cohort, reading coverage LIVE.

    uv run python scripts/eval/cohort_split.py --out data/eval/cohort_split.json

Why this exists as a script rather than an ad-hoc query: the two cohorts must not
be pooled for a within-day contrast, and the first attempt at this split was
computed from a summary CSV written *before* the last judging tranche landed. A
day that was 99% judged read as 24.8% and would have been quarantined as
contaminated. So coverage is recomputed here from the judgment cache and the
pairs file every time — never from a cached summary, which can predate the
tranche it is describing.

The cohorts:

* **census** — days bought whole with ``judge_batch --days``. Every kept pair was
  submitted, so the within-day sample is a census and is unbiased by construction.
* **stratified** — days judged by ``--stratify month,sector,session``. Sector and
  session are outcome-correlated and ``persistence`` correlates with sector, so the
  within-day bucket mix is distorted *by the sampling design*. These cannot be
  pooled with census days for a within-day contrast; carry them as a separate
  stratum and report whether the sign agrees (A6 Addendum J9).

Classification is by live coverage. The two cohorts normally separate with a wide
empty band (census 0.98+, stratified 0.32-), so if that band closes the script
warns and exits non-zero rather than emitting a plausible-looking wrong answer --
that is the shape the stale-CSV error took.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

CENSUS_MIN = 0.90  # census days land at 0.983+; stratified at 0.32 or below
MIN_SEPARATION = 0.30  # the empty band the two cohorts must leave between them

__all__ = ["classify", "live_coverage"]


def live_coverage(pairs_path: Path, cache_path: Path) -> dict[str, tuple[int, int]]:
    """Per day, (judged kept pairs, kept pairs) recomputed from the cache right now."""
    judged_keys = set()
    with cache_path.open() as fh:
        for line in fh:
            if line.strip():
                judged_keys.add(json.loads(line)["key"])
    kept: dict[str, int] = {}
    hit: dict[str, int] = {}
    with pairs_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            d = r["day"]
            kept[d] = kept.get(d, 0) + 1
            if r["key"] in judged_keys:
                hit[d] = hit.get(d, 0) + 1
    return {d: (hit.get(d, 0), n) for d, n in kept.items()}


def classify(coverage: dict[str, tuple[int, int]], census_min: float = CENSUS_MIN) -> dict:
    """Days at or above `census_min` coverage are census; the rest are stratified."""
    census, strat = [], []
    for day, (j, n) in sorted(coverage.items()):
        (census if n and j / n >= census_min else strat).append(day)
    return {"census": census, "stratified": strat}


def _frac(coverage: dict[str, tuple[int, int]], days: list[str]) -> list[float]:
    return [coverage[d][0] / coverage[d][1] for d in days if coverage[d][1]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pairs", type=Path, default=Path("data/eval/universe_pairs_n300_prefiltered.jsonl")
    )
    ap.add_argument("--cache", type=Path, default=Path("data/eval/cache/universe_v2.jsonl"))
    ap.add_argument("--census-min", type=float, default=CENSUS_MIN)
    ap.add_argument("--out", type=Path, default=Path("data/eval/cohort_split.json"))
    a = ap.parse_args(argv)

    coverage = live_coverage(a.pairs, a.cache)
    split = classify(coverage, a.census_min)
    census, strat = split["census"], split["stratified"]

    # The guard that would have caught the stale-CSV error: a bought day and a
    # stratified day differ by ~0.68 in coverage, so the two cohorts separate with
    # a wide empty band. If that band closes, something is wrong -- a tranche is
    # mid-flight, or the cache and the pairs file disagree -- and the split should
    # not be trusted. Warn rather than emit a plausible-looking wrong answer.
    mismatch: list[str] = []
    gap_lo = max(_frac(coverage, strat)) if strat else 0.0
    gap_hi = min(_frac(coverage, census)) if census else 1.0
    if census and strat and gap_hi - gap_lo < MIN_SEPARATION:
        mismatch.append(
            f"cohorts are not cleanly separated: stratified max {gap_lo:.4f}, census min "
            f"{gap_hi:.4f}, gap {gap_hi - gap_lo:.4f} < {MIN_SEPARATION}. A judging tranche "
            "may be mid-flight, or the cache predates the pairs file."
        )

    out = {
        "census_min": a.census_min,
        "n_census": len(census),
        "n_stratified": len(strat),
        "census": census,
        "stratified": strat,
        "census_coverage": {
            "min": min(_frac(coverage, census)) if census else None,
            "median": statistics.median(_frac(coverage, census)) if census else None,
        },
        "stratified_coverage": {
            "min": min(_frac(coverage, strat)) if strat else None,
            "max": max(_frac(coverage, strat)) if strat else None,
        },
        "note": "Stratified days were drawn with --stratify month,sector,session. Sector and "
        "session are outcome-correlated and persistence correlates with sector, so their "
        "within-day bucket mix is distorted by the sampling design. Do not pool them with "
        "census days for a within-day contrast (A6 Addendum J9).",
    }
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(
        f"census {len(census)} days (min {out['census_coverage']['min']:.4f}), "
        f"stratified {len(strat)} days (max {out['stratified_coverage']['max']:.4f}) -> {a.out}"
    )
    for m in mismatch:
        print(f"WARNING: {m}", file=sys.stderr)
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
