"""Cheap check (≤ 300 requests) that grouping-derived related tickers match the live pairs.

    uv run python scripts/eval/validate_news_universe.py --days 2026-09-14,2026-09-15

1. Archive pairs (article_id, ticker) from data/news_judgments/{day}.jsonl.
2. Per archive ticker, company-news pages back to the first day (page budget), grouped by newsId
   into live article_id -> codes; agreement on articles present in both.
   Only archive tickers are fetched, so "extra" pairs are extras within that ticker set.
3. Detail endpoint for up to --details matched articles: grouping codes vs `companyCodes`
   (both raw and restricted to the fetched ticker set).
Same pacing / block guard / market-hours pause as the backfill (Collector). Writes a JSON summary.

Offline (0 requests) once the backfill exists: `--rows-from data/toss_universe` replaces step 2 and
`--detail-cache data/toss_detail` (the live daemon's per-day detail JSON, keyed by article_id)
replaces step 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from backfill_toss_company_news import (
    URL,
    Collector,
    Stop,
    live_article_id,
    page_rows,
    setup_logging,
)
from news_universe_pairs import compare_sets, group_related, iter_rows

DETAIL_URL = "https://wts-info-api.tossinvest.com/api/v2/news/{news_id}"
CODE6 = re.compile(r"^A?(\d{6})$")


def company_codes(payload: dict[str, Any]) -> set[str]:
    """6-digit listed codes from a detail payload's companyCodes (drops ETF/foreign ids)."""
    out = set()
    for c in payload.get("companyCodes") or []:
        m = CODE6.match(str(c))
        if m:
            out.add(m.group(1))
    return out


def fetch_rows(col: Collector, tickers: list[str], since: str, budget: int) -> tuple[list, int]:
    rows: list[dict[str, Any]] = []
    pending = dict.fromkeys(tickers, 1)  # ticker -> next page
    while pending and budget > 0:
        for t in list(pending):
            if budget <= 0:
                break
            r = col.get(URL.format(code=t), {"number": pending[t], "size": 100})
            budget -= 1
            if r is None or r.status_code != 200:
                pending.pop(t)
                continue
            result = (r.json() or {}).get("result") or {}
            got, more = page_rows(result.get("body") or [], since, t)
            rows.extend(got)
            if more:
                pending[t] += 1
            else:
                pending.pop(t)
    return rows, len(pending)


def load_detail_cache(root: Path, days: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for day in days:
        for p in (root / day).glob("*.json"):
            out[p.stem] = json.loads(p.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2026-09-14,2026-09-15")
    ap.add_argument("--archive", type=Path, default=Path("data/news_judgments"))
    ap.add_argument("--out", type=Path, default=Path("data/toss_universe_validation"))
    ap.add_argument("--tickers", default=None, help="restrict the fetched ticker set")
    ap.add_argument("--rows-from", type=Path, default=None, help="use backfill files, no fetch")
    ap.add_argument("--detail-cache", type=Path, default=None, help="offline companyCodes")
    ap.add_argument("--page-budget", type=int, default=240)
    ap.add_argument("--details", type=int, default=50)
    a = ap.parse_args()
    setup_logging()
    days = sorted(a.days.split(","))
    a.out.mkdir(parents=True, exist_ok=True)

    ref: dict[str, set[str]] = {}
    for day in days:
        for line in (a.archive / f"{day}.jsonl").open():
            r = json.loads(line)
            ref.setdefault(r["article_id"], set()).add(r["ticker"])
    col = Collector(a.out, days[0], 1.2, 1)

    if a.rows_from:
        rows = [r for r in iter_rows(a.rows_from) if r["createdAt"][:10] >= days[0]]
        fetched = {r["code"] for r in rows} | {p.stem for p in a.rows_from.glob("*.done")}
        incomplete = 0
    else:
        tickers = (
            a.tickers.split(",") if a.tickers else sorted({t for s in ref.values() for t in s})
        )
        rows, incomplete = fetch_rows(col, tickers, days[0], a.page_budget)
        fetched = set(tickers)
        (a.out / "rows.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
        )
    grouped = group_related(rows)
    by_live = {live_article_id(nid): (nid, g["codes"]) for nid, g in grouped.items()}
    got = {aid: codes for aid, (_, codes) in by_live.items()}
    ref_in = {k: v & fetched for k, v in ref.items() if v & fetched}
    archive_cmp = compare_sets(got, ref_in)
    archive_cmp |= {
        "tickers_fetched": len(fetched),
        "tickers_incomplete": incomplete,
        "archive_articles_with_fetched_ticker": len(ref_in),
    }

    detail: dict[str, set[str]] = {}
    if a.detail_cache:
        cache = load_detail_cache(a.detail_cache, days)
        detail = {aid: company_codes(d) for aid, d in cache.items() if aid in ref_in}
    else:
        try:
            for aid in sorted(got.keys() & ref_in.keys())[: a.details]:
                r = col.get(DETAIL_URL.format(news_id=by_live[aid][0]))
                if r is not None and r.status_code == 200:
                    payload = ((r.json() or {}).get("result") or {}).get("kr") or {}
                    detail[aid] = company_codes(payload)
        except Stop as exc:
            logging.warning("detail phase stopped: %s", exc)
    detail_in = {k: v & fetched for k, v in detail.items()}
    # Articles live-archived but absent from every fetched list (the pool gap), by source.
    missing = [aid for aid in ref_in if aid not in got]
    summary = {
        "requests": col.requests,
        "errors": col.errors,
        "grouping_vs_archive": archive_cmp,
        "archive_articles_missing_from_lists": len(missing),
        "grouping_vs_companyCodes_within_fetched": compare_sets(got, detail_in),
        "companyCodes_articles_found_in_lists": sum(k in got for k in detail_in),
        "companyCodes_articles": len(detail_in),
        "mismatch_examples": [
            {"news_id": by_live[k][0], "grouping": sorted(got[k]), "companyCodes": sorted(v)}
            for k, v in detail_in.items()
            if k in got and got[k] != v
        ][:15],
    }
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
