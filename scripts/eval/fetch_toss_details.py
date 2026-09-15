"""Cache full Toss news detail JSON for every article in the eval set.

The dashboard feed only carries ~300-char summaries. Toss's own detail
endpoint (`GET wts-info-api.tossinvest.com/api/v2/news/{newsId}`, the call
behind the article modal) returns the full body as typed blocks, Toss's
3-sentence summary, its own sentiment label, all related stock codes, and
the publisher's original URL -- uniformly across all ~25 publishers, so no
per-publisher HTML scraping is needed.

Reads newsIds from kr_quant.news_articles.url, writes one JSON per article
to data/eval/toss_detail/ via krx_news_client's TossScraper (headers,
retries). Sequential with a delay; existing files are skipped, so it resumes.

Usage: uv run python scripts/eval/fetch_toss_details.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import psycopg2
from krx_news_client import TossScraper
from krx_news_client.scrapers.toss import NEWS_DETAIL_URL, news_id_from_url

EVAL_SET = Path("data/eval/news_eval_set.jsonl")
OUT_DIR = Path("data/eval/toss_detail")
DELAY_SECONDS = 0.8


async def fetch_all(urls: dict[str, str], article_ids: list[str]) -> None:
    # The raw payload is cached (not TossScraper.fetch_article_detail's parsed
    # NewsArticle): eval_news_prompts.full_body reads the typed body blocks.
    # Going through the scraper still gives its headers, pacing and 429/5xx
    # retries.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = missing = failed = 0
    async with TossScraper() as scraper:
        scraper.min_delay = scraper.max_delay = DELAY_SECONDS
        for article_id in article_ids:
            news_id = news_id_from_url(urls[article_id])
            path = OUT_DIR / f"{article_id.replace(':', '_')}.json"
            if path.exists():
                ok += 1
                continue
            if news_id is None:
                failed += 1
                print(f"no newsId in url for {article_id}")
                continue
            try:
                resp = await scraper.fetch(NEWS_DETAIL_URL.format(news_id=news_id))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    missing += 1
                else:
                    failed += 1
                    print(f"HTTP {exc.response.status_code} for {news_id}")
                continue
            result = resp.json().get("result") or {}
            payload = result.get("kr")  # v2 nests the article per language
            if not payload:
                missing += 1
                print(f"empty result for {news_id}: {resp.text[:120]}")
            else:
                path.write_text(json.dumps({"news_id": news_id, **payload}, ensure_ascii=False))
                ok += 1
    print(f"articles={len(article_ids)} cached={ok} not_found={missing} failed={failed}")


def main() -> None:
    article_ids = sorted({json.loads(line)["article_id"] for line in EVAL_SET.open()})

    conn = psycopg2.connect(
        host=os.environ["GPTQ_QUANT_AIRFLOW_DB_HOST"],
        port=int(os.environ["GPTQ_QUANT_AIRFLOW_DB_PORT"]),
        user=os.environ["GPTQ_QUANT_AIRFLOW_DB_USER"],
        password=os.environ["GPTQ_QUANT_AIRFLOW_DB_PASSWORD"],
        dbname=os.environ["GPTQ_QUANT_AIRFLOW_DB_DBNAME"],
    )
    conn.set_session(readonly=True)
    with conn.cursor() as cur:
        cur.execute("SELECT id, url FROM news_articles WHERE id = ANY(%s)", (article_ids,))
        urls = dict(cur.fetchall())

    asyncio.run(fetch_all(urls, article_ids))


if __name__ == "__main__":
    main()
