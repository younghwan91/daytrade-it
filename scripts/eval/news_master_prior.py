"""master.jsonl-shaped pairs for the 149 prior trading days that have bars and
news but no LLM judgment (2025-08-01 .. 2026-03-16) -- a strictly-prior holdout.

    uv run python scripts/eval/news_master_prior.py \
        --start 2025-08-01 --end 2026-03-16 \
        --out data/eval/rlnews/master_prior.jsonl        # simnode

Shape and semantics match `data/eval/prompt_v3/master.jsonl` closely enough for
`scripts/eval/prompt_labels.py --master <this>` to consume it unchanged: every
row carries `key`, `code`, `day`, `published_at`, `decide_at`.

Pair construction, live-parity:
  * universe -- `universe_coverage.select_universe` over the PREVIOUS trading
    day's `daily_bars_adjusted.trade_value` (top 300). 069500 is the market
    proxy, not a tradeable pair, so it is excluded from the emitted pairs.
  * articles -- every row of `data/toss_universe_42m/<code>.jsonl` whose
    `createdAt` (naive KST) falls on `day` inside the live sweep window
    [08:00, 15:20) (`application/news/universe._DEFAULT_START` / `_CUTOFF`).
    `stockCodes` is null on every archive row; the article<->code link is the
    file the row lives in.
  * key -- `f"{live_article_id(news_id)}|{code}"`, the live daemon's dedup id
    (`application/news/universe.live_article_id`), so a prior-day pair key is
    the same shape as a judged pair key.

`decide_at` is measured, not invented -- see DECIDE_LATENCY below.

READ-ONLY on kr_quant (two SELECTs total). No API, no Toss network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import edge_db  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

from daytrade_it.application.news.universe import live_article_id  # noqa: E402

ARCHIVE = Path("data/toss_universe_42m")
MARKET = "069500"
SWEEP_START = dt.time(8, 0)
SWEEP_CUTOFF = dt.time(15, 20)
KST = dt.timezone(dt.timedelta(hours=9))

#: Live `decide_at - published_at`, measured on all 135,946 judged pairs of
#: `data/eval/prompt_v3/master.jsonl` (the archive's `createdAt` reproduces
#: `published_at` exactly: 6,381/6,381 matches on a 60-code sample).
#:
#: The distribution is sharply bimodal by discovery path, exactly as suspected:
#:
#:   session=intraday    n=52,176  p10=0.72 p25=1.65 med=4.00 p75=9.00 p90=35.25
#:   session=pre_market  n=83,770  p10=62.2 p25=183  med=904  p75=997  p90=2445
#:
#: The pre_market mode is not a latency at all -- it is an overnight backlog
#: (articles published the evening before, judged in the 09:00 batch), and
#: `prompt_labels.label_day` already collapses it: any decision minute before
#: 09:00 acts at 09:00, at that bar's open. Prior-day pairs are all
#: sweep-discovered by construction, so the intraday mode is the right one.
#:
#: 4 minutes = the intraday median, rounded to the minute. Articles published
#: in [08:00, 09:00) get decide_at in [08:04, 09:04); those before 09:00 are
#: clamped to the 09:00 open by `prompt_labels`, same as live premarket.
DECIDE_LATENCY = dt.timedelta(minutes=4)


def sweep_window_articles(
    path: Path, wanted_days: set[dt.date]
) -> dict[dt.date, list[tuple[str, dt.datetime]]]:
    """{day: [(news_id, published_at)]} for one code file, sweep window only."""
    out: dict[dt.date, list[tuple[str, dt.datetime]]] = defaultdict(list)
    seen: set[tuple[dt.date, str]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            created = row.get("createdAt")
            if not created:
                continue
            # naive KST, "YYYY-MM-DDTHH:MM:SS" -- no tz conversion, ever.
            ts = dt.datetime.fromisoformat(created)
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            day = ts.date()
            if day not in wanted_days:
                continue
            if not (SWEEP_START <= ts.time() < SWEEP_CUTOFF):
                continue
            nid = row["id"]
            if (day, nid) in seen:
                continue
            seen.add((day, nid))
            out[day].append((nid, ts))
    return dict(out)


def build(start: dt.date, end: dt.date, out_path: Path, size: int | None = None) -> dict:
    conn = edge_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT date FROM daily_bars_adjusted "
                "WHERE date >= %s AND date <= %s ORDER BY date",
                (start - dt.timedelta(days=20), end),
            )
            trading_days = [r[0] for r in cur.fetchall()]
            cur.execute(
                "SELECT code, date, trade_value FROM daily_bars_adjusted "
                "WHERE date >= %s AND date <= %s",
                (start - dt.timedelta(days=20), end),
            )
            rows_by_day: dict[dt.date, list[tuple[str, float]]] = defaultdict(list)
            for code, date, tv in cur.fetchall():
                rows_by_day[date].append((code, float(tv or 0)))
    finally:
        conn.close()

    target_days = [d for d in trading_days if start <= d <= end]
    if not target_days or trading_days[0] >= start:
        raise RuntimeError(f"no trading day before {start}; widen the buffer")
    index = {d: i for i, d in enumerate(trading_days)}

    universes: dict[dt.date, list[str]] = {}
    for day in target_days:
        prev = trading_days[index[day] - 1]
        universes[day] = [
            c for c in select_universe(rows_by_day.get(prev, []), size=size) if c != MARKET
        ]

    # Which (code, day) slots we need, so each archive file is read once.
    days_by_code: dict[str, set[dt.date]] = defaultdict(set)
    for day, codes in universes.items():
        for code in codes:
            days_by_code[code].add(day)

    stats = {
        "trading_days": len(target_days),
        "universe_codes": len(days_by_code),
        "codes_missing_archive": [],
        "universe_pairs_slots": sum(len(v) for v in universes.values()),
    }

    articles: dict[str, dict[dt.date, list[tuple[str, dt.datetime]]]] = {}
    for code, days in sorted(days_by_code.items()):
        path = ARCHIVE / f"{code}.jsonl"
        if not path.exists():
            stats["codes_missing_archive"].append(code)
            continue
        articles[code] = sweep_window_articles(path, days)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_day: Counter = Counter()
    per_month: Counter = Counter()
    n = 0
    with out_path.open("w") as f:
        for day in target_days:
            for code in universes[day]:
                for nid, published in articles.get(code, {}).get(day, []):
                    decide = (published + DECIDE_LATENCY).replace(second=0, microsecond=0)
                    rec = {
                        "key": f"{live_article_id(nid)}|{code}",
                        "code": code,
                        "day": day.isoformat(),
                        "month": f"{day.year:04d}-{day.month:02d}",
                        "news_id": nid,
                        "session": "pre_market" if published.time() < dt.time(9, 0) else "intraday",
                        "published_at": published.replace(tzinfo=KST).isoformat(),
                        "decide_at": decide.replace(tzinfo=KST).isoformat(),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    per_day[day] += 1
                    per_month[rec["month"]] += 1
    stats["pairs"] = n
    stats["by_month"] = dict(sorted(per_month.items()))
    stats["days_with_zero_pairs"] = [str(d) for d in target_days if per_day[d] == 0]
    stats["first_day"] = str(target_days[0])
    stats["last_day"] = str(target_days[-1])
    stats["decide_latency_min"] = DECIDE_LATENCY.total_seconds() / 60
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-03-16")
    ap.add_argument("--out", default="data/eval/rlnews/master_prior.jsonl")
    ap.add_argument("--universe-size", type=int, default=None)
    args = ap.parse_args()
    stats = build(
        dt.date.fromisoformat(args.start),
        dt.date.fromisoformat(args.end),
        Path(args.out),
        size=args.universe_size,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
