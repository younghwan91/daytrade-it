"""Toss 종목별 과거 뉴스 수집 — 분봉이 있는 기간(기본 2025-08-01~)만.

    uv run python scripts/eval/backfill_toss_company_news.py --since 2025-08-01 [--codes 005930,000911]

엔드포인트는 비공식이다(2026-09-15 확인): `GET wts-info-api.tossinvest.com/api/v2/news/companies/{6자리}`
`?number=1..&size=100`, 최신순, `result.body[]`에 id·title·summary·createdAt(KST, 초)·stockCodes 등.
`A005930`·ISIN 형식은 빈 목록을 준다. 종목당 `data/toss_history/{code}.jsonl` 하나로 쓰고,
파일이 있으면 건너뛴다(재개). 요청 사이 `--sleep`초 쉰다 — 대시보드 폴러와 같은 서버라 느리게.
kr_quant `news_articles`에는 쓰지 않는다(쓰기는 quant-airflow 수집 쪽 소관).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import psycopg2

URL = "https://wts-info-api.tossinvest.com/api/v2/news/companies/{code}"
OUT_DIR = Path("data/toss_history")
KEEP = ("id", "title", "summary", "createdAt", "stockCodes", "newsType", "ticsTitles")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"


def minute_bar_codes() -> list[str]:
    with psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5) as c, c.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
        cur.execute(
            "SELECT DISTINCT code FROM minute_bars WHERE ts >= '2025-08-01' AND code ~ '^[0-9]{6}$'"
        )
        return sorted(r[0] for r in cur.fetchall())


def fetch_code(
    client: httpx.Client, code: str, since: str, sleep: float, max_pages: int
) -> tuple[list[dict], str]:
    rows: list[dict] = []
    for number in range(1, max_pages + 1):
        for attempt in range(4):
            try:
                r = client.get(URL.format(code=code), params={"number": number, "size": 100})
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError("retry", request=r.request, response=r)
                r.raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(sleep * 10 * (attempt + 1))
        else:
            return rows, "http_error"
        result = r.json()["result"]
        body = result["body"]
        for a in body:
            if a["createdAt"][:10] >= since:
                src = a.get("source") or {}
                rows.append(
                    {**{k: a.get(k) for k in KEEP}, "source": src.get("code"), "code": code}
                )
        time.sleep(sleep)
        if not body or result.get("lastPage") or body[-1]["createdAt"][:10] < since:
            return rows, "done"
    return rows, "max_pages"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-08-01")
    ap.add_argument("--codes", default=None)
    ap.add_argument("--sleep", type=float, default=0.7)
    ap.add_argument("--max-pages", type=int, default=1500)
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    codes = a.codes.split(",") if a.codes else minute_bar_codes()
    client = httpx.Client(
        headers={"User-Agent": UA, "Referer": "https://tossinvest.com/"}, timeout=20
    )
    total = 0
    for i, code in enumerate(codes, 1):
        path = OUT_DIR / f"{code}.jsonl"
        if path.exists():
            continue
        rows, status = fetch_code(client, code, a.since, a.sleep, a.max_pages)
        if status == "http_error":
            print(
                f"[{i}/{len(codes)}] {code}: HTTP 실패, 건너뜀 (다음 실행에서 재시도)", flush=True
            )
            continue
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.rename(path)
        total += len(rows)
        span = f"{rows[-1]['createdAt'][:10]}~{rows[0]['createdAt'][:10]}" if rows else "-"
        print(
            f"[{i}/{len(codes)}] {code}: {len(rows)}건 {span} ({status}) 누적 {total}", flush=True
        )


if __name__ == "__main__":
    main()
