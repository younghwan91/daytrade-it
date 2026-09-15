"""Cheap check (≤ 300 requests) that grouping-derived related tickers match the live pairs.

    uv run python scripts/eval/validate_news_universe.py --days 2026-09-14,2026-09-15

1. Archive pairs (article_id, ticker) from data/news_judgments/{day}.jsonl.
2. Per archive ticker, company-news pages back to the first day (page budget), grouped by newsId
   into live article_id -> codes; agreement on articles present in both.
   Only archive tickers are fetched, so "extra" pairs are extras within that ticker set.
3. Detail endpoint for up to --details matched articles: grouping codes vs `companyCodes`
   (both raw and restricted to the fetched ticker set).
Same pacing / block guard / market-hours pause as the backfill (Collector). Writes a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from backfill_toss_company_news import URL, Collector, Stop, live_article_id, page_rows
from news_universe_pairs import compare_sets, group_related

DETAIL_URL = "https://wts-info-api.tossinvest.com/api/v2/news/{news_id}"


def company_codes(payload: dict[str, Any]) -> set[str]:
    """6-digit codes from a detail payload's companyCodes (tolerates 'A005930' / dict items)."""
    out = set()
    for c in payload.get("companyCodes") or []:
        if isinstance(c, dict):
            c = c.get("code") or c.get("stockCode") or ""
        c = str(c)
        if len(c) >= 6 and c[-6:].isdigit():
            out.add(c[-6:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2026-09-14,2026-09-15")
    ap.add_argument("--archive", type=Path, default=Path("data/news_judgments"))
    ap.add_argument("--out", type=Path, default=Path("data/toss_universe_validation"))
    ap.add_argument("--page-budget", type=int, default=240)
    ap.add_argument("--details", type=int, default=50)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days = sorted(a.days.split(","))
    a.out.mkdir(parents=True, exist_ok=True)

    ref: dict[str, set[str]] = {}
    for day in days:
        for line in (a.archive / f"{day}.jsonl").open():
            r = json.loads(line)
            ref.setdefault(r["article_id"], set()).add(r["ticker"])
    tickers = sorted({t for s in ref.values() for t in s})
    col = Collector(a.out, days[0], 1.2, 1)

    rows: list[dict[str, Any]] = []
    pending = dict.fromkeys(tickers, 1)  # ticker -> next page
    budget = a.page_budget
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
            got, more = page_rows(
                result.get("body") or [], days[0], bool(result.get("lastPage")), t
            )
            rows.extend(got)
            if more:
                pending[t] += 1
            else:
                pending.pop(t)
    (a.out / "rows.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
    )
    grouped = group_related(rows)
    by_live = {live_article_id(nid): (nid, g["codes"]) for nid, g in grouped.items()}
    got = {aid: codes for aid, (_, codes) in by_live.items()}
    archive_cmp = compare_sets(got, ref)
    archive_cmp["tickers_fetched"] = len(tickers)
    archive_cmp["tickers_incomplete"] = len(pending)
    archive_cmp["archive_articles"] = len(ref)

    fetched = set(tickers)
    matched = sorted(got.keys() & ref.keys())[: a.details]
    detail_raw: dict[str, set[str]] = {}
    detail_restricted: dict[str, set[str]] = {}
    examples = []
    try:
        for aid in matched:
            nid = by_live[aid][0]
            r = col.get(DETAIL_URL.format(news_id=nid))
            if r is None or r.status_code != 200:
                continue
            payload = ((r.json() or {}).get("result") or {}).get("kr") or {}
            cc = company_codes(payload)
            detail_raw[aid] = cc
            detail_restricted[aid] = cc & fetched
            if got[aid] != detail_restricted[aid]:
                examples.append(
                    {
                        "news_id": nid,
                        "grouping": sorted(got[aid]),
                        "companyCodes": sorted(cc),
                        "archive": sorted(ref[aid]),
                    }
                )
    except Stop as exc:
        logging.warning("detail phase stopped: %s", exc)
    summary = {
        "requests": col.requests,
        "errors": col.errors,
        "grouping_vs_archive": archive_cmp,
        "grouping_vs_companyCodes_raw": compare_sets(got, detail_raw),
        "grouping_vs_companyCodes_within_fetched": compare_sets(got, detail_restricted),
        "archive_vs_companyCodes_raw": compare_sets(ref, detail_raw),
        "mismatch_examples": examples[:15],
    }
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
