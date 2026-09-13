"""Cache full Toss news detail JSON for every article in the eval set.

The dashboard feed only carries ~300-char summaries. Toss's own detail
endpoint (`GET wts-info-api.tossinvest.com/api/v2/news/{newsId}`, the call
behind the article modal) returns the full body as typed blocks, Toss's
3-sentence summary, its own sentiment label, all related stock codes, and
the publisher's original URL -- uniformly across all ~25 publishers, so no
per-publisher HTML scraping is needed.

Reads newsIds from kr_quant.news_articles.url, writes one JSON per article
to data/eval/toss_detail/. Sequential with a delay; existing files are
skipped, so it resumes.

Usage: uv run python scripts/eval/fetch_toss_details.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from pathlib import Path

import httpx
import psycopg2

EVAL_SET = Path("data/eval/news_eval_set.jsonl")
OUT_DIR = Path("data/eval/toss_detail")
DETAIL_URL = "https://wts-info-api.tossinvest.com/api/v2/news/{news_id}"
DELAY_SECONDS = 0.8
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}


def news_id_from_url(url: str) -> str:
    params = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return json.loads(params["contentParams"][0])["id"]


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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = missing = failed = 0
    with httpx.Client(headers=HEADERS, timeout=15) as client:
        for article_id in article_ids:
            news_id = news_id_from_url(urls[article_id])
            path = OUT_DIR / f"{article_id.replace(':', '_')}.json"
            if path.exists():
                ok += 1
                continue
            resp = client.get(DETAIL_URL.format(news_id=news_id))
            if resp.status_code == 404:
                missing += 1
            elif resp.is_success:
                result = resp.json().get("result") or {}
                payload = result.get("kr")  # v2 nests the article per language
                if not payload:
                    missing += 1
                    print(f"empty result for {news_id}: {resp.text[:120]}")
                else:
                    path.write_text(json.dumps({"news_id": news_id, **payload}, ensure_ascii=False))
                    ok += 1
            else:
                failed += 1
                print(f"HTTP {resp.status_code} for {news_id}")
            time.sleep(DELAY_SECONDS)
    print(f"articles={len(article_ids)} cached={ok} not_found={missing} failed={failed}")


if __name__ == "__main__":
    main()
