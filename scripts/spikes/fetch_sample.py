"""One-off helper for the sentiment validation spike: pull a sample of Toss
news articles via krx-news-client and dump them to JSON for hand-labeling.

Not part of the spike's "approaches under test" -- just data collection.
Run once, then hand-label the resulting JSON (see label_dataset.py / the
`hand_label` field in the output).

Usage:
    python scripts/spikes/fetch_sample.py --out docs/spikes/001-sample.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from krx_news_client import TossScraper


async def fetch(out_path: Path, limit: int) -> None:
    scraper = TossScraper()
    try:
        articles = await scraper.scrape_news()
    finally:
        await scraper.close()

    # TossScraper.scrape_news() only exposes market/breaking/stock feeds today
    # (see FEED_TYPES in krx_news_client.scrapers.toss) -- there is no
    # standalone "economy" feed, so this spike's sample is drawn from those
    # three categories rather than the four named in the task brief.
    records = [
        {
            "id": a.id,
            "category": str(a.category),
            "title": a.title,
            "url": a.url,
            # `content` comes back empty from the Toss dashboard feed for
            # every article in this sample; `summary` is the substantive
            # text field, so that's what gets scored/labeled.
            "text": a.summary or a.content,
            "tickers": a.tickers,
            "published_at": a.published_at.isoformat() if a.published_at else None,
        }
        for a in articles
        if (a.summary or a.content).strip()
    ]
    records = records[:limit] if limit else records

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} articles to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/spikes/001-sample.json"))
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()
    asyncio.run(fetch(args.out, args.limit))
