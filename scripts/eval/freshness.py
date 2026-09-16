"""Is the freshness effect latency, or is it time of day?

I reported that live's chronological "first 5 of the day" beats every learned
ranker by 70-87 bp on the dashboard pool, and the poll interval was cut 180s ->
20s on the strength of it. But "first 5 of the day" is **rank within the day**,
not **detection delay**. The first entries of a day are the 09:01 premarket batch,
which differ from later entries in liquidity, in how the no-chase filter treats
them, and in the auction mechanics around the open. Those are two different
claims and only one of them justifies a latency change.

This separates them:

- `rank`      chronological position within the day (what I actually measured)
- `delay`     publication -> our decision, in minutes (what a poll interval changes)
- `act_min`   minutes since 09:00 at the decision (time of day, the confound)

and reports the reward by each, then `rank` and `delay` *within* time-of-day
strata so the confound cannot carry the result.

Dashboard pool only -- the pool live trades. 154 entries over 7 days is thin for
something a strategy would be built on, and the dispersion columns are there to
keep that visible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(paths: Path, events_dir: Path) -> list[dict]:
    rows = [json.loads(line) for line in paths.read_text().splitlines() if line]
    rows = [r for r in rows if r.get("engine_reward") is not None]
    ev: dict[str, dict] = {}
    for day in sorted({r["day"] for r in rows}):
        p = events_dir / f"{day}.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line:
                    e = json.loads(line)
                    ev[e["key"]] = e
    out = []
    for r in rows:
        e = ev.get(r["key"])
        if e is None:
            continue
        pub = dt.datetime.fromisoformat(e["published_at"])
        dec = dt.datetime.fromisoformat(e["decide_at"])
        batch = dt.datetime.fromisoformat(e["batch_at"]) if e.get("batch_at") else dec
        r["pub"], r["dec"], r["batch"] = pub, dec, batch
        r["delay_min"] = (dec - pub).total_seconds() / 60.0
        r["judge_min"] = (batch - dec).total_seconds() / 60.0
        r["act_min"] = dec.hour * 60 + dec.minute - 9 * 60
        r["fill_min"] = int(r["buy_minute"]) - (dec.hour * 60 + dec.minute)
        out.append(r)
    out.sort(key=lambda r: (r["day"], r["dec"]))
    by_day: dict[str, int] = defaultdict(int)
    for r in out:
        by_day[r["day"]] += 1
        r["rank"] = by_day[r["day"]]
    return out


def show(name: str, groups: list[tuple[str, list[float]]]) -> None:
    print(f"\n{name}")
    print(f"{'bucket':22s} {'n':>5s} {'mean bp':>9s} {'sd bp':>8s} {'se bp':>7s} {'win':>6s}")
    for label, vals in groups:
        if not vals:
            print(f"{label:22s} {0:>5d}")
            continue
        a = np.array(vals)
        se = a.std() / np.sqrt(len(a)) if len(a) > 1 else float("nan")
        print(
            f"{label:22s} {len(a):>5d} {a.mean() * 1e4:>+9.0f} {a.std() * 1e4:>8.0f}"
            f" {se * 1e4:>7.0f} {float((a > 0).mean()):>6.3f}"
        )


def bucketize(
    rows: list[dict], key: str, edges: tuple[float, ...]
) -> list[tuple[str, list[float]]]:
    out: list[tuple[str, list[float]]] = []
    lo = -1e18
    for hi in (*edges, 1e18):
        sel = [r["engine_reward"] for r in rows if lo < r[key] <= hi]
        lab = f"{key} <= {hi:g}" if hi < 1e17 else f"{key} > {edges[-1]:g}"
        out.append((lab, sel))
        lo = hi
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    args = ap.parse_args()

    rows = load(args.paths, args.events)
    y = np.array([r["engine_reward"] for r in rows])
    print(
        f"entries={len(rows)} days={len({r['day'] for r in rows})} pool mean {y.mean() * 1e4:+.0f} bp"
    )

    pre = [r for r in rows if r["act_min"] <= 1]
    intra = [r for r in rows if r["act_min"] > 1]
    print(f"premarket/open-batch (decision <= 09:01): {len(pre)}   intraday: {len(intra)}")

    print("\n=== leg decomposition (minutes)")
    print(f"{'leg':28s} {'median':>8s} {'p25':>8s} {'p75':>8s} {'p90':>8s}")
    for lab, k in (
        ("publication -> decision", "delay_min"),
        ("decision -> batch (judge)", "judge_min"),
        ("decision -> fill", "fill_min"),
    ):
        v = np.array([r[k] for r in rows], dtype=float)
        print(
            f"{lab:28s} {np.median(v):>8.1f} {np.quantile(v, 0.25):>8.1f}"
            f" {np.quantile(v, 0.75):>8.1f} {np.quantile(v, 0.9):>8.1f}"
        )
    vi = np.array([r["delay_min"] for r in intra], dtype=float)
    print(
        f"{'  (intraday only)':28s} {np.median(vi):>8.1f} {np.quantile(vi, 0.25):>8.1f}"
        f" {np.quantile(vi, 0.75):>8.1f} {np.quantile(vi, 0.9):>8.1f}"
    )

    show(
        "=== by chronological RANK within the day (what I originally measured)",
        [
            ("rank 1-5", [r["engine_reward"] for r in rows if r["rank"] <= 5]),
            ("rank 6-10", [r["engine_reward"] for r in rows if 5 < r["rank"] <= 10]),
            ("rank 11+", [r["engine_reward"] for r in rows if r["rank"] > 10]),
        ],
    )

    show(
        "=== by TIME OF DAY (the confound)",
        [
            ("09:00-09:01 (open)", [r["engine_reward"] for r in pre]),
            ("09:02-10:00", [r["engine_reward"] for r in rows if 1 < r["act_min"] <= 60]),
            ("10:00-12:00", [r["engine_reward"] for r in rows if 60 < r["act_min"] <= 180]),
            ("12:00+", [r["engine_reward"] for r in rows if r["act_min"] > 180]),
        ],
    )

    show(
        "=== by DETECTION DELAY, intraday only (what the poll interval changes)",
        bucketize(intra, "delay_min", (5, 15, 60)),
    )

    show(
        "=== RANK within the INTRADAY subset (confound controlled)",
        [
            (
                "intraday rank 1-5",
                [r["engine_reward"] for r in intra[:0] or [x for x in intra if x["rank"] <= 5]],
            ),
            ("intraday rank 6+", [x["engine_reward"] for x in intra if x["rank"] > 5]),
        ],
    )

    # Within-day rank recomputed inside the intraday subset, so "first" means
    # first intraday signal rather than first signal overall.
    by_day: dict[str, int] = defaultdict(int)
    for r in intra:
        by_day[r["day"]] += 1
        r["intra_rank"] = by_day[r["day"]]
    show(
        "=== rank RECOMPUTED within intraday (first intraday signal of the day)",
        [
            ("intraday rank 1-3", [r["engine_reward"] for r in intra if r["intra_rank"] <= 3]),
            ("intraday rank 4-8", [r["engine_reward"] for r in intra if 3 < r["intra_rank"] <= 8]),
            ("intraday rank 9+", [r["engine_reward"] for r in intra if r["intra_rank"] > 8]),
        ],
    )

    x = np.array([r["delay_min"] for r in intra])
    yy = np.array([r["engine_reward"] for r in intra])
    if len(x) > 3:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(yy)).astype(float)
        print(
            f"\nspearman(detection delay, reward) intraday: {np.corrcoef(rx, ry)[0, 1]:+.3f} (n={len(x)})"
        )


if __name__ == "__main__":
    main()
