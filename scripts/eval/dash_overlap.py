"""Is Toss's dashboard selection itself the edge? Step 1: containment.

Establishes the overlap window and measures what fraction of dashboard articles
also appear in the sweep pool, matched on the live article key. Everything
downstream rests on that containment being high; if it is not, a
"dashboard-present vs sweep-only" split inside the sweep pool is not measuring
dashboard membership at all.

kr_quant is read with `default_transaction_read_only`; nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from uuid import uuid5

import psycopg2

from daytrade_it.application.news.universe import TOSS_NEWS_NAMESPACE


def dashboard_articles(lo: str, hi: str) -> list[dict]:
    """Dashboard-sourced Toss articles published in [lo, hi), as kr_quant holds them."""
    sql = """
        SELECT id, url, title, published_at, (published_at AT TIME ZONE 'Asia/Seoul')::date AS kst_day
        FROM news_articles
        WHERE source = 'toss' AND published_at >= %s AND published_at < %s
        ORDER BY published_at
    """
    with psycopg2.connect(os.environ["KR_QUANT_DB"]) as c, c.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute(sql, (lo, hi))
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def key_of(row: dict) -> tuple[str, bool]:
    """(live article uuid, whether the stored id reproduces from its url).

    `news_articles.id` is already `toss:<md5(url)[:12]>`, so the live key is
    `uuid5(TOSS_NEWS_NAMESPACE, id)`. The second element re-derives the digest
    from the stored url and reports whether it matches, so a silent change in
    how ids are minted cannot pass as a containment failure.
    """
    stored = row["id"]
    digest = hashlib.md5(row["url"].encode()).hexdigest()[:12]  # noqa: S324
    return str(uuid5(TOSS_NEWS_NAMESPACE, stored)), stored == f"toss:{digest}"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--end", default="2026-09-16", help="exclusive")
    ap.add_argument("--events", type=Path, required=True, help="sweep events dir (judged pairs)")
    ap.add_argument("--pairs", type=Path, help="sweep pairs jsonl (all pairs, judged or not)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = dashboard_articles(args.start, args.end)
    mismatched = sum(1 for r in rows if not key_of(r)[1])
    print(f"dashboard articles {args.start}..{args.end}: {len(rows)}")
    print(f"  id not reproducible from url: {mismatched} (expect 0)")

    dash_by_day: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        dash_by_day[str(r["kst_day"])].add(key_of(r)[0])

    # Sweep pool: judged pairs (events) and, if given, all pairs before judging.
    ev_by_day: dict[str, set[str]] = {}
    for day in sorted(dash_by_day):
        p = args.events / f"{day}.jsonl"
        if p.exists():
            ev_by_day[day] = {e["article_id"] for e in load_jsonl(p)}

    # The full sweep pool, before judging. Containment has to be measured against
    # this: matching only against the judged events would conflate "the sweep
    # never saw this article" with "the sweep saw it but it was not in the ~30%
    # sampled for judging".
    pool_by_day: dict[str, set[str]] = defaultdict(set)
    if args.pairs:
        for pr in load_jsonl(args.pairs):
            pool_by_day[pr["day"]].add(pr["key"].split("|")[0])

    print()
    print(
        f"{'day':12s} {'dash':>6s} {'pool':>7s} {'judged':>7s}"
        f" {'in pool':>8s} {'contained':>10s} {'judged∩':>8s} {'of pool':>8s}"
    )
    summary = []
    for day in sorted(dash_by_day):
        d = dash_by_day[day]
        pool = pool_by_day.get(day, set())
        judged = ev_by_day.get(day, set())
        in_pool = len(d & pool)
        in_judged = len(d & judged)
        summary.append(
            {
                "day": day,
                "dashboard": len(d),
                "pool": len(pool),
                "judged": len(judged),
                "in_pool": in_pool,
                "contained": in_pool / len(d) if d else 0.0,
                "in_judged": in_judged,
                "judged_share_of_contained": in_judged / in_pool if in_pool else 0.0,
            }
        )
        print(
            f"{day:12s} {len(d):>6d} {len(pool):>7d} {len(judged):>7d}"
            f" {in_pool:>8d} {in_pool / len(d):>10.3f} {in_judged:>8d}"
            f" {(in_judged / in_pool if in_pool else 0):>8.3f}"
        )

    tot_d = sum(s["dashboard"] for s in summary)
    tot_p = sum(s["in_pool"] for s in summary)
    tot_j = sum(s["in_judged"] for s in summary)
    print(
        f"{'TOTAL':12s} {tot_d:>6d} {'':>7s} {'':>7s} {tot_p:>8d}"
        f" {tot_p / max(1, tot_d):>10.3f} {tot_j:>8d} {tot_j / max(1, tot_p):>8.3f}"
    )
    tot_b = tot_p

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "window": [args.start, args.end],
                "dashboard_articles": len(rows),
                "id_mismatch": mismatched,
                "per_day": summary,
                "containment": tot_b / max(1, tot_d),
                "dashboard_ids_by_day": {k: sorted(v) for k, v in dash_by_day.items()},
            },
            indent=1,
        )
    )
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
