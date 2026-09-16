"""Does dashboard membership predict what the stock does next? A judgment-free test.

The P&L version of this question is blocked twice over: only ~30% of sweep pairs
were judged (so only a handful of dashboard pairs have v2 features), and the
minute-bar backfill leaves most of the overlap window un-replayable. Both blocks
are about *judgments and replay*, not about prices -- and the economic core of
the question ("are dashboard-selected articles better news?") can be asked with
prices alone.

So: for every (article, ticker) pair in the window, measure the stock's forward
return from the first completed minute after publication, and compare
dashboard-present pairs against sweep-only pairs on the SAME days.

This is a proxy, and the differences from the live reward are large and worth
being explicit about:

- No entry filter: the live policy trades a pair only if the v2 score clears
  0.15, the subject and session gates pass, and the stock has not already run.
  Here every pair counts.
- No exit rule, no costs, no fill model: this is a raw price move, not a trade.
  The live reward is ~23 bp worse than any raw move just from tax and commission.
- No concurrency, no ranking, no caps.

What it can settle is the direction and rough size of any membership effect, on a
sample large enough to say something. What it cannot do is price that effect.

kr_quant read-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from uuid import uuid5

import numpy as np
import psycopg2

from daytrade_it.application.news.universe import TOSS_NEWS_NAMESPACE

OPEN_MINUTE = 9 * 60
CLOSE_MINUTE = 15 * 60
HORIZONS = (30, 60, 120)


def query(sql: str, params: tuple) -> list[dict]:
    with psycopg2.connect(os.environ["KR_QUANT_DB"]) as c, c.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def forward_returns(bars: dict[int, tuple[float, float]], start: int) -> dict[str, float] | None:
    """Return from the open of the first bar at/after `start` to later bar closes.

    `bars` maps minute -> (open, close). A premarket or pre-open publication
    enters at 09:00, which is what the live policy's premarket batch does.
    """
    entry_minute = max(start, OPEN_MINUTE)
    later = sorted(m for m in bars if m >= entry_minute)
    if not later:
        return None
    e = bars[later[0]][0]
    if e <= 0:
        return None
    out = {}
    for h in HORIZONS:
        target = [m for m in later if m <= later[0] + h]
        out[f"r{h}"] = bars[target[-1]][1] / e - 1 if target else 0.0
    to_close = [m for m in later if m <= CLOSE_MINUTE]
    out["r_close"] = bars[to_close[-1]][1] / e - 1 if to_close else 0.0
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days-file", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True, help="sweep pairs jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from bayes_judge import judge

    days = [line.strip() for line in args.days_file.read_text().splitlines() if line.strip()]
    print(f"days: {len(days)} ({days[0]}..{days[-1]})")

    # Dashboard pairs: article -> tickers, from kr_quant.
    dash = query(
        """
        SELECT a.id, (a.published_at AT TIME ZONE 'Asia/Seoul') AS kst_ts, t.ticker
        FROM news_articles a
        JOIN news_article_tickers t ON t.article_id = a.id
        WHERE a.source = 'toss'
          AND (a.published_at AT TIME ZONE 'Asia/Seoul')::date = ANY(%s)
        """,
        ([dt.date.fromisoformat(d) for d in days],),
    )
    dash_pairs: dict[tuple[str, str, str], int] = {}
    dash_article_ids: set[str] = set()
    for r in dash:
        uid = str(uuid5(TOSS_NEWS_NAMESPACE, r["id"]))
        dash_article_ids.add(uid)
        day = str(r["kst_ts"].date())
        minute = r["kst_ts"].hour * 60 + r["kst_ts"].minute
        key = (day, uid, r["ticker"])
        dash_pairs[key] = min(minute, dash_pairs.get(key, 10**9))
    print(f"dashboard pairs in window: {len(dash_pairs)} ({len(dash_article_ids)} articles)")

    # Sweep pairs, same days, excluding any article the dashboard also carried.
    sweep_pairs: dict[tuple[str, str, str], int] = {}
    day_set = set(days)
    for r in load_jsonl(args.pairs):
        if r["day"] not in day_set:
            continue
        uid = r["key"].split("|")[0]
        ts = dt.datetime.fromisoformat(r["published_at"])
        sweep_pairs[(r["day"], uid, r["code"])] = ts.hour * 60 + ts.minute
    sweep_only = {k: v for k, v in sweep_pairs.items() if k[1] not in dash_article_ids}
    both = {k: v for k, v in sweep_pairs.items() if k[1] in dash_article_ids}
    print(
        f"sweep pairs: {len(sweep_pairs)}  (dashboard-present {len(both)}, sweep-only {len(sweep_only)})"
    )

    codes = sorted({k[2] for k in list(dash_pairs) + list(sweep_pairs)})
    rows = query(
        """
        SELECT code, ts::date AS d,
               extract(hour from ts) * 60 + extract(minute from ts) AS minute,
               open, close
        FROM minute_bars
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        """,
        (codes, days[0], (dt.date.fromisoformat(days[-1]) + dt.timedelta(days=1)).isoformat()),
    )
    bars: dict[tuple[str, str], dict[int, tuple[float, float]]] = defaultdict(dict)
    for r in rows:
        bars[(str(r["d"]), r["code"])][int(r["minute"])] = (float(r["open"]), float(r["close"]))
    print(f"minute bars loaded for {len(bars)} (day, code) pairs")

    def collect(pairs: dict) -> dict[str, np.ndarray]:
        acc: dict[str, list] = defaultdict(list)
        acc_day: list[str] = []
        for (day, _uid, code), minute in pairs.items():
            b = bars.get((day, code))
            if not b:
                continue
            fr = forward_returns(b, minute)
            if fr is None:
                continue
            for k, v in fr.items():
                acc[k].append(v)
            acc_day.append(day)
        out = {k: np.array(v) for k, v in acc.items()}
        out["day"] = np.array(acc_day)
        return out

    groups = {
        "dashboard": collect(dash_pairs),
        "dashboard∩sweep": collect(both),
        "sweep-only": collect(sweep_only),
    }
    print()
    hdr = (
        f"{'group':18s} {'n':>7s} "
        + " ".join(f"{f'r{h}':>9s}" for h in HORIZONS)
        + f" {'r_close':>9s} {'win@close':>10s}"
    )
    print(hdr)
    for name, g in groups.items():
        if not len(g.get("r_close", [])):
            print(f"{name:18s} {'0':>7s}  (no pairs)")
            continue
        cells = " ".join(f"{g[f'r{h}'].mean() * 1e4:>+9.0f}" for h in HORIZONS)
        print(
            f"{name:18s} {len(g['r_close']):>7d} {cells}"
            f" {g['r_close'].mean() * 1e4:>+9.0f} {float((g['r_close'] > 0).mean()):>10.3f}"
        )
    for name, g in groups.items():
        if len(g.get("r_close", [])):
            print(f"  {name}: r_close sd {g['r_close'].std() * 1e4:.0f} bp")

    # Two-group comparison at the day level, so `bayes_judge` sees the same
    # shape it does everywhere else: mean forward return per day, per group.
    def day_means(g: dict[str, np.ndarray]) -> dict[str, dict]:
        out = {}
        for day in sorted(set(g["day"].tolist())):
            sel = g["day"] == day
            r = g["r_close"][sel]
            out[day] = {
                "day": day,
                "net": 1e4 * float(r.mean()),
                "trades": int(sel.sum()),
                "losers": int((r < 0).sum()),
                "worst_trade": 1e4 * float(r.min()),
                "errors": 0,
                "leftovers": 0,
            }
        return out

    cand = day_means(groups["dashboard∩sweep"])
    base = day_means(groups["sweep-only"])
    common = sorted(set(cand) & set(base))
    verdict = None
    if len(common) >= 2:
        verdict = judge({d: cand[d] for d in common}, {d: base[d] for d in common}, prior_sd=50.0)
        print(
            f"\nbayes_judge (day-mean r_close in bp, dashboard∩sweep vs sweep-only,"
            f" {len(common)} days):"
        )
        print(
            f"  p(mu>0)={verdict['p_mu_gt_0']:.3f} mu={verdict['mu_mean']:+.1f} bp"
            f" days>=0={verdict['days_ge_0']}/{verdict['n_days']} adopt={verdict['adopt']}"
        )
        for r in verdict["reasons"]:
            print(f"    - {r}")

    # Power: what difference could this n even detect?
    d = groups["dashboard∩sweep"]
    s = groups["sweep-only"]
    if len(d.get("r_close", [])) and len(s.get("r_close", [])):
        se = float(
            np.sqrt(d["r_close"].var() / len(d["r_close"]) + s["r_close"].var() / len(s["r_close"]))
        )
        print(
            f"\npower: se of the group difference = {se * 1e4:.0f} bp;"
            f" 80% power at 5% needs a true gap of about {2.8 * se * 1e4:.0f} bp"
        )
        print(f"  observed gap = {(d['r_close'].mean() - s['r_close'].mean()) * 1e4:+.0f} bp")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "days": days,
                "n": {k: int(len(v.get("r_close", []))) for k, v in groups.items()},
                "mean_bp": {
                    k: {
                        f: float(v[f].mean() * 1e4)
                        for f in ("r30", "r60", "r120", "r_close")
                        if f in v and len(v[f])
                    }
                    for k, v in groups.items()
                },
                "verdict": verdict,
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
