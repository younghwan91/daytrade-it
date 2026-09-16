"""Why are 80% of dashboard articles absent from the sweep pool?

Containment alone does not say whether the sweep *feed* misses dashboard
articles (which would make a dashboard-vs-sweep split inside the pool
meaningless) or whether the pool's own construction drops them for reasons that
have nothing to do with the feed -- no ticker mapping, a ticker outside the
N=300 universe, or a day the pool does not cover. This splits the gap into those
buckets and counts what is actually left to compare with.

kr_quant read-only.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid5

import psycopg2

from daytrade_it.application.news.universe import TOSS_NEWS_NAMESPACE


def query(sql: str, params: tuple) -> list[dict]:
    with psycopg2.connect(os.environ["KR_QUANT_DB"]) as c, c.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--end", default="2026-09-16")
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arts = query(
        """
        SELECT a.id, (a.published_at AT TIME ZONE 'Asia/Seoul')::date AS kst_day,
               a.title, count(t.ticker) AS n_tickers,
               array_remove(array_agg(t.ticker), NULL) AS tickers
        FROM news_articles a
        LEFT JOIN news_article_tickers t ON t.article_id = a.id
        WHERE a.source = 'toss' AND a.published_at >= %s AND a.published_at < %s
        GROUP BY a.id, a.published_at, a.title
        """,
        (args.start, args.end),
    )
    for a in arts:
        a["uuid"] = str(uuid5(TOSS_NEWS_NAMESPACE, a["id"]))
        a["day"] = str(a["kst_day"])

    pool_rows = load_jsonl(args.pairs)
    pool_days = {r["day"] for r in pool_rows}
    pool_ids: dict[str, set[str]] = defaultdict(set)
    universe_by_day: dict[str, set[str]] = defaultdict(set)
    for r in pool_rows:
        pool_ids[r["day"]].add(r["key"].split("|")[0])
        universe_by_day[r["day"]].add(r["code"])

    judged_ids: dict[str, set[str]] = {}
    judged_pairs: dict[str, set[str]] = {}
    for day in sorted(pool_days):
        p = args.events / f"{day}.jsonl"
        if p.exists():
            ev = load_jsonl(p)
            judged_ids[day] = {e["article_id"] for e in ev}
            judged_pairs[day] = {e["key"] for e in ev}

    buckets: Counter = Counter()
    per_day: dict[str, Counter] = defaultdict(Counter)
    dash_judged_pairs: set[str] = set()
    for a in arts:
        day = a["day"]
        if day not in pool_days:
            buckets["day not in sweep pool (weekend / not swept)"] += 1
            per_day[day]["no_pool_day"] += 1
            continue
        if a["uuid"] in pool_ids[day]:
            buckets["in sweep pool"] += 1
            per_day[day]["in_pool"] += 1
            for t in a["tickers"]:
                k = f"{a['uuid']}|{t}"
                if k in judged_pairs.get(day, set()):
                    dash_judged_pairs.add(k)
            continue
        if not a["n_tickers"]:
            buckets["no ticker mapping in kr_quant"] += 1
            per_day[day]["no_ticker"] += 1
        elif not (set(a["tickers"]) & universe_by_day[day]):
            buckets["tickers all outside the N=300 universe"] += 1
            per_day[day]["outside_universe"] += 1
        else:
            buckets["has a universe ticker but the sweep never saw it"] += 1
            per_day[day]["missed_by_sweep"] += 1

    total = len(arts)
    print(f"dashboard articles {args.start}..{args.end}: {total}\n")
    print(f"{'bucket':48s} {'n':>6s} {'share':>7s}")
    for k, n in buckets.most_common():
        print(f"{k:48s} {n:>6d} {n / total:>7.3f}")

    swept = total - buckets["day not in sweep pool (weekend / not swept)"]
    in_pool = buckets["in sweep pool"]
    missed = buckets["has a universe ticker but the sweep never saw it"]
    print(f"\non days the sweep covers: {swept} dashboard articles")
    print(f"  in the sweep pool: {in_pool} ({in_pool / max(1, swept):.3f})")
    print(
        f"  a universe ticker but absent from the sweep: {missed}"
        f" ({missed / max(1, swept):.3f})  <- genuine feed gap"
    )
    reachable = in_pool + missed
    print(
        f"  of the {reachable} the sweep COULD have carried, it carried"
        f" {in_pool / max(1, reachable):.3f}"
    )
    print(f"\ndashboard pairs that are also judged in the sweep events: {len(dash_judged_pairs)}")
    tot_judged = sum(len(v) for v in judged_pairs.values())
    print(f"  against {tot_judged} judged sweep pairs on those days")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "total": total,
                "buckets": dict(buckets),
                "swept_days_articles": swept,
                "in_pool": in_pool,
                "missed_by_sweep": missed,
                "dash_judged_pairs": sorted(dash_judged_pairs),
                "judged_pairs_total": tot_judged,
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
