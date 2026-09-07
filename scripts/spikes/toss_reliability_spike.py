"""Task 2 spike: repeated-call reliability of krx-news-client's TossScraper.

Polls TossScraper().scrape_news() on an interval and logs, per poll:
  - timestamp
  - article count returned (total + per feed/category)
  - any exception raised out of scrape_news() itself
  - any per-feed error the scraper swallowed internally (it catches exceptions
    per FEED_TYPE and logs+continues rather than raising -- see
    krx_news_client.scrapers.toss.TossScraper.scrape_news) -- captured here by
    attaching a logging handler to that library's logger
  - duplicate-id rate against the running set of article ids seen so far this
    session (NewsArticle.id = f"{source}:{md5(url)[:12]}", stable across polls
    for the same article)

This measures whether repeated calls succeed, return consistent shapes, and
whether the (undocumented, unofficial) Toss dashboard API rate-limits or
blocks a client making many calls in one session. It does NOT measure
multi-day stability -- a single interactive session cannot observe that; see
docs/spikes/002-toss-reliability.md for that caveat.

Usage:
    python scripts/spikes/toss_reliability_spike.py \
        --count 10 --interval 30 \
        --log-out docs/spikes/002-poll-log.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from krx_news_client import TossScraper

LIBRARY_LOGGER_NAME = "krx_news_client.scrapers.toss"


class _CapturingHandler(logging.Handler):
    """Captures log records so we can see per-feed errors TossScraper
    swallows internally (it logs+continues rather than raising)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


async def poll_once(scraper: TossScraper, seen_ids: set[str]) -> dict:
    capture = _CapturingHandler()
    capture.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    lib_logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    lib_logger.addHandler(capture)

    started = time.monotonic()
    ts = datetime.now(timezone.utc).isoformat()
    result: dict = {"timestamp": ts}

    try:
        articles = await scraper.scrape_news()
    except Exception as exc:  # noqa: BLE001 -- we want to record any exception shape
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["article_count"] = 0
        result["per_category"] = {}
        result["new_ids"] = 0
        result["duplicate_ids"] = 0
        result["duplicate_rate"] = None
        result["swallowed_feed_errors"] = list(capture.records)
        result["duration_ms"] = round((time.monotonic() - started) * 1000, 1)
        lib_logger.removeHandler(capture)
        return result

    lib_logger.removeHandler(capture)

    ids = [a.id for a in articles]
    id_counter = Counter(ids)
    duplicate_within_poll = sum(c - 1 for c in id_counter.values() if c > 1)

    new_ids = [i for i in ids if i not in seen_ids]
    dup_against_history = len(ids) - len(new_ids)
    seen_ids.update(ids)

    per_category = Counter(str(a.category) for a in articles)

    result.update(
        {
            "exception": None,
            "article_count": len(articles),
            "unique_article_count": len(set(ids)),
            "duplicate_within_poll": duplicate_within_poll,
            "per_category": dict(per_category),
            "new_ids": len(new_ids),
            "duplicate_ids": dup_against_history,
            "duplicate_rate": round(dup_against_history / len(ids), 4) if ids else None,
            "swallowed_feed_errors": list(capture.records),
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }
    )
    return result


async def run(count: int, interval: float, log_out: Path) -> list[dict]:
    scraper = TossScraper()
    seen_ids: set[str] = set()
    results: list[dict] = []

    log_out.parent.mkdir(parents=True, exist_ok=True)
    with log_out.open("w", encoding="utf-8") as f:
        try:
            for i in range(count):
                poll_result = await poll_once(scraper, seen_ids)
                poll_result["poll_index"] = i
                results.append(poll_result)
                f.write(json.dumps(poll_result, ensure_ascii=False) + "\n")
                f.flush()

                status = "OK" if not poll_result["exception"] else "EXCEPTION"
                print(
                    f"[{i + 1}/{count}] {poll_result['timestamp']} {status} "
                    f"articles={poll_result['article_count']} "
                    f"new={poll_result['new_ids']} dup={poll_result['duplicate_ids']} "
                    f"dur={poll_result['duration_ms']}ms "
                    f"swallowed_errs={len(poll_result['swallowed_feed_errors'])}"
                )

                if i < count - 1:
                    await asyncio.sleep(interval)
        finally:
            await scraper.close()

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    exceptions = [r for r in results if r["exception"]]
    counts = [r["article_count"] for r in results if not r["exception"]]
    swallowed = sum(len(r["swallowed_feed_errors"]) for r in results)
    dup_rates = [r["duplicate_rate"] for r in results if r["duplicate_rate"] is not None]

    return {
        "polls": n,
        "exceptions": len(exceptions),
        "exception_rate": round(len(exceptions) / n, 4) if n else None,
        "swallowed_feed_errors_total": swallowed,
        "article_count_min": min(counts) if counts else None,
        "article_count_max": max(counts) if counts else None,
        "article_count_mean": round(sum(counts) / len(counts), 1) if counts else None,
        "duplicate_rate_min": min(dup_rates) if dup_rates else None,
        "duplicate_rate_max": max(dup_rates) if dup_rates else None,
        "duplicate_rate_mean": round(sum(dup_rates) / len(dup_rates), 4) if dup_rates else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="number of polls")
    parser.add_argument(
        "--interval", type=float, default=30.0, help="seconds to sleep between polls"
    )
    parser.add_argument(
        "--log-out",
        type=Path,
        default=Path("docs/spikes/002-poll-log.jsonl"),
        help="JSONL file to append one record per poll",
    )
    args = parser.parse_args()

    all_results = asyncio.run(run(args.count, args.interval, args.log_out))
    summary = summarize(all_results)
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
