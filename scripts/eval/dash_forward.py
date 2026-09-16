"""Dashboard membership vs the sweep pool, on a news-quality label with a null baseline.

Two corrections drive this file's design.

**MFE is not evidence on its own.** The maximum over a path is mechanically at or
above the entry for any stock and any entry, so "mean MFE +154 bp against −51 bp
realized" says nothing by itself about exits leaving money behind. Every MFE
number here is therefore reported next to a NULL group: entries drawn on the
same days, at the same minutes, on random tickers from that day's tradable
universe. Only the excess over that null is a claim.

**Realized reward is a contaminated label for news quality.** It bakes in the
exit rule, so a genuinely good article whose gain was given back is labelled bad.
For ranking news this file therefore also reports the post-article maximum, over
several horizons so the label is not an artifact of one window, plus the
distribution of minutes-to-peak.

The MFE label is NOT achievable and is never P&L: nobody sells at the maximum.
It exists to rank news quality. Any rule built on it still has to be verified on
realized KRW under an actual exit rule, out-of-sample.

kr_quant read-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from uuid import uuid5

import numpy as np
import psycopg2

from daytrade_it.application.news.universe import TOSS_NEWS_NAMESPACE

OPEN_MINUTE = 9 * 60
CLOSE_MINUTE = 15 * 60
HORIZONS = (30, 60, 120)
NULL_DRAWS = 3


def query(sql: str, params: tuple) -> list[dict]:
    with psycopg2.connect(os.environ["KR_QUANT_DB"]) as c, c.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def forward_stats(bars: dict[int, tuple[float, float, float]], start: int) -> dict | None:
    """Endpoint returns and post-entry maxima from the first bar at/after `start`.

    `bars` maps minute -> (open, high, close). A pre-open publication enters at
    09:00, as the live premarket batch does. `mfe*` is the best unrealized move
    over the window (a maximum, not a tradable outcome); `peak_min` is how many
    minutes after entry that maximum occurred.
    """
    entry_minute = max(start, OPEN_MINUTE)
    later = sorted(m for m in bars if m >= entry_minute)
    if not later:
        return None
    e = bars[later[0]][0]
    if e <= 0:
        return None
    # Pre-entry 1-minute volatility, from bars that ended BEFORE the entry minute.
    # A maximum scales with volatility mechanically, so every MFE below is also
    # reported divided by this: without it, "higher MFE" and "more volatile"
    # are the same statement.
    prior = sorted(m for m in bars if m < entry_minute and bars[m][2] > 0)[-31:]
    if len(prior) >= 3:
        c = np.array([bars[m][2] for m in prior])
        sigma = max(float((c[1:] / c[:-1] - 1).std()), 0.0015)
    else:
        sigma = float("nan")
    out: dict[str, float] = {"sigma": sigma}
    for h in list(HORIZONS) + ["close"]:
        end = CLOSE_MINUTE if h == "close" else later[0] + h
        window = [m for m in later if m <= end]
        if not window:
            out[f"r{h}"] = 0.0
            out[f"mfe{h}"] = 0.0
            continue
        out[f"r{h}"] = bars[window[-1]][2] / e - 1
        out[f"mfe{h}"] = max(bars[m][1] for m in window) / e - 1
    out["mfe_close_z"] = out["mfeclose"] / sigma if sigma == sigma else float("nan")
    out["r_close_z"] = out["rclose"] / sigma if sigma == sigma else float("nan")
    to_close = [m for m in later if m <= CLOSE_MINUTE]
    if to_close:
        peak = max(to_close, key=lambda m: bars[m][1])
        out["peak_min"] = float(peak - later[0])
    else:
        out["peak_min"] = 0.0
    return out


def collect(pairs: dict, bars: dict) -> dict[str, np.ndarray]:
    acc: dict[str, list] = defaultdict(list)
    for (day, _uid, code), minute in pairs.items():
        b = bars.get((day, code))
        if not b:
            continue
        fs = forward_stats(b, minute)
        if fs is None:
            continue
        for k, v in fs.items():
            acc[k].append(v)
        acc["day"].append(day)
    return {k: (np.array(v) if k != "day" else np.array(v)) for k, v in acc.items()}


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days-file", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from bayes_judge import judge

    days = [line.strip() for line in args.days_file.read_text().splitlines() if line.strip()]
    print(f"days: {len(days)} ({days[0]}..{days[-1]})")

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
    dash_ids = {str(uuid5(TOSS_NEWS_NAMESPACE, r["id"])) for r in dash}

    sweep_pairs: dict[tuple[str, str, str], int] = {}
    universe_by_day: dict[str, set[str]] = defaultdict(set)
    day_set = set(days)
    for r in load_jsonl(args.pairs):
        if r["day"] not in day_set:
            continue
        uid = r["key"].split("|")[0]
        ts = dt.datetime.fromisoformat(r["published_at"])
        sweep_pairs[(r["day"], uid, r["code"])] = ts.hour * 60 + ts.minute
        universe_by_day[r["day"]].add(r["code"])
    both = {k: v for k, v in sweep_pairs.items() if k[1] in dash_ids}
    sweep_only = {k: v for k, v in sweep_pairs.items() if k[1] not in dash_ids}
    print(
        f"sweep pairs {len(sweep_pairs)}: dashboard-present {len(both)}, sweep-only {len(sweep_only)}"
    )

    # Null: same days, same minutes, random tickers from that day's tradable universe.
    rng = random.Random(args.seed)
    null_pairs: dict[tuple[str, str, str], int] = {}
    for i, ((day, _uid, _code), minute) in enumerate(sweep_pairs.items()):
        pool = sorted(universe_by_day[day])
        for j in range(NULL_DRAWS):
            null_pairs[(day, f"null{i}_{j}", rng.choice(pool))] = minute
    print(f"null draws: {len(null_pairs)} ({NULL_DRAWS} per sweep pair, same day and minute)")

    codes = sorted({k[2] for k in list(sweep_pairs) + list(null_pairs)})
    rows = query(
        """
        SELECT code, ts::date AS d,
               extract(hour from ts) * 60 + extract(minute from ts) AS minute,
               open, high, close
        FROM minute_bars
        WHERE code = ANY(%s) AND ts >= %s AND ts < %s
        """,
        (codes, days[0], (dt.date.fromisoformat(days[-1]) + dt.timedelta(days=1)).isoformat()),
    )
    bars: dict[tuple[str, str], dict[int, tuple[float, float, float]]] = defaultdict(dict)
    for r in rows:
        bars[(str(r["d"]), r["code"])][int(r["minute"])] = (
            float(r["open"]),
            float(r["high"]),
            float(r["close"]),
        )
    print(f"minute bars for {len(bars)} (day, code) pairs")

    groups = {
        "dashboard∩sweep": collect(both, bars),
        "sweep-only": collect(sweep_only, bars),
        "NULL (random)": collect(null_pairs, bars),
    }

    print("\n=== endpoint return from entry (bp) -- realized-style, no exit rule, no costs")
    print(
        f"{'group':18s} {'n':>7s} {'r30':>8s} {'r60':>8s} {'r120':>8s} {'r_close':>9s} {'win':>7s}"
    )
    for name, g in groups.items():
        if not len(g.get("rclose", [])):
            continue
        print(
            f"{name:18s} {len(g['rclose']):>7d}"
            + "".join(f" {g[f'r{h}'].mean() * 1e4:>+8.0f}" for h in HORIZONS)
            + f" {g['rclose'].mean() * 1e4:>+9.0f} {float((g['rclose'] > 0).mean()):>7.3f}"
        )

    print("\n=== post-entry MAXIMUM (bp) -- a ranking label, NOT achievable P&L")
    print(f"{'group':18s} {'n':>7s} {'mfe30':>8s} {'mfe60':>8s} {'mfe120':>8s} {'mfe_close':>10s}")
    for name, g in groups.items():
        if not len(g.get("mfeclose", [])):
            continue
        print(
            f"{name:18s} {len(g['mfeclose']):>7d}"
            + "".join(f" {g[f'mfe{h}'].mean() * 1e4:>+8.0f}" for h in HORIZONS)
            + f" {g['mfeclose'].mean() * 1e4:>+10.0f}"
        )
    null = groups["NULL (random)"]
    print("\nexcess over the null (bp) -- this, not the raw MFE, is the claim:")
    for name in ("dashboard∩sweep", "sweep-only"):
        g = groups[name]
        if not len(g.get("mfeclose", [])):
            continue
        cells = "".join(
            f" {(g[f'mfe{h}'].mean() - null[f'mfe{h}'].mean()) * 1e4:>+8.0f}" for h in HORIZONS
        )
        se = float(
            np.sqrt(
                g["mfeclose"].var() / len(g["mfeclose"])
                + null["mfeclose"].var() / len(null["mfeclose"])
            )
        )
        print(
            f"{name:18s} {'':>7s}{cells}"
            f" {(g['mfeclose'].mean() - null['mfeclose'].mean()) * 1e4:>+10.0f}"
            f"   (se on mfe_close {se * 1e4:.0f} bp)"
        )

    print("\n=== volatility control: pre-entry 1-min sigma, and MFE measured in sigmas")
    print(
        f"{'group':18s} {'n':>7s} {'sigma bp':>9s} {'sd r_close':>11s}"
        f" {'mfe/sigma':>10s} {'r/sigma':>9s}"
    )
    for name, g in groups.items():
        if "mfe_close_z" not in g:
            continue
        ok = np.isfinite(g["mfe_close_z"])
        print(
            f"{name:18s} {int(ok.sum()):>7d} {np.nanmean(g['sigma']) * 1e4:>9.0f}"
            f" {g['rclose'].std() * 1e4:>11.0f}"
            f" {g['mfe_close_z'][ok].mean():>10.2f} {g['r_close_z'][ok].mean():>9.2f}"
        )
    zd = groups["dashboard∩sweep"]["mfe_close_z"]
    zs = groups["sweep-only"]["mfe_close_z"]
    zn = groups["NULL (random)"]["mfe_close_z"]
    zd, zs, zn = zd[np.isfinite(zd)], zs[np.isfinite(zs)], zn[np.isfinite(zn)]
    se_ds = float(np.sqrt(zd.var() / len(zd) + zs.var() / len(zs)))
    se_dn = float(np.sqrt(zd.var() / len(zd) + zn.var() / len(zn)))
    print(
        f"  dashboard - sweep-only: {zd.mean() - zs.mean():+.3f} sigma (se {se_ds:.3f})"
        f"   dashboard - null: {zd.mean() - zn.mean():+.3f} sigma (se {se_dn:.3f})"
    )
    print(
        f"  volatility-implied MFE for dashboard if it were sweep-only quality:"
        f" {groups['sweep-only']['mfeclose'].mean() * (np.nanmean(groups['dashboard∩sweep']['sigma']) / np.nanmean(groups['sweep-only']['sigma'])) * 1e4:+.0f} bp"
        f"  vs observed {groups['dashboard∩sweep']['mfeclose'].mean() * 1e4:+.0f} bp"
    )

    print("\n=== minutes from entry to the post-entry peak")
    print(
        f"{'group':18s} {'p10':>6s} {'p25':>6s} {'median':>7s} {'p75':>6s} {'p90':>6s} {'mean':>6s}"
    )
    for name, g in groups.items():
        if not len(g.get("peak_min", [])):
            continue
        p = g["peak_min"]
        print(
            f"{name:18s} {np.quantile(p, 0.1):>6.0f} {np.quantile(p, 0.25):>6.0f}"
            f" {np.median(p):>7.0f} {np.quantile(p, 0.75):>6.0f}"
            f" {np.quantile(p, 0.9):>6.0f} {p.mean():>6.0f}"
        )

    def day_means(g: dict[str, np.ndarray], field: str, scale: float = 1e4) -> dict[str, dict]:
        out = {}
        for day in sorted(set(g["day"].tolist())):
            sel = (g["day"] == day) & np.isfinite(g[field])
            if not sel.any():
                continue
            r = g[field][sel]
            out[day] = {
                "day": day,
                "net": scale * float(r.mean()),
                "trades": int(sel.sum()),
                "losers": int((r < 0).sum()),
                "worst_trade": scale * float(r.min()),
                "errors": 0,
                "leftovers": 0,
            }
        return out

    verdicts = {}
    for field, label in (
        ("rclose", "realized endpoint"),
        ("mfeclose", "post-entry maximum"),
        ("mfe_close_z", "post-entry maximum, in sigmas"),
    ):
        scale = 1.0 if field.endswith("_z") else 1e4
        cand = day_means(groups["dashboard∩sweep"], field, scale)
        base = day_means(groups["sweep-only"], field, scale)
        common = sorted(set(cand) & set(base))
        if len(common) < 2:
            continue
        v = judge(
            {d: cand[d] for d in common},
            {d: base[d] for d in common},
            prior_sd=0.5 if field.endswith("_z") else 50.0,
        )
        verdicts[field] = v
        print(
            f"\nbayes_judge, {label} (day means in bp, dashboard∩sweep vs sweep-only,"
            f" {len(common)} days):"
        )
        print(
            f"  p(mu>0)={v['p_mu_gt_0']:.3f} mu={v['mu_mean']:+.1f} bp"
            f" days>=0={v['days_ge_0']}/{v['n_days']} adopt={v['adopt']}"
        )
        d, s = groups["dashboard∩sweep"][field], groups["sweep-only"][field]
        d, s = d[np.isfinite(d)], s[np.isfinite(s)]
        se = float(np.sqrt(d.var() / len(d) + s.var() / len(s)))
        print(
            f"  pair-level gap {(d.mean() - s.mean()) * scale:+.2f}, se {se * scale:.2f};"
            f" 80% power needs about {2.8 * se * scale:.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "days": days,
                "n": {k: int(len(v.get("rclose", []))) for k, v in groups.items()},
                "mean_bp": {
                    k: {f: float(v[f].mean() * 1e4) for f in v if f != "day"}
                    for k, v in groups.items()
                },
                "verdicts": verdicts,
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
