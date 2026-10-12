"""Build the news→price-reaction evaluation set.

경고(2026-10-12): 이 평가셋으로 한 연구는 기각됐고 코드는 고치지 않는다. 청산가 = daily_bars_adjusted 종가인데
2026-09-14 부터 그 종가에 KRX 애프터마켓(16:00~20:00) 체결이 섞였다 — 그 이후 행의 close·excess 는 정규장
청산이 아니다. 다시 만들 일이 있으면 scripts/eval/krx_session.regular_daily 로 읽는다.

For every (Toss article, KRX ticker) pair in kr_quant's `news_articles` /
`news_article_tickers`, attach what a day trader acting on that article
would actually have got: enter after publication, flat by that session's
close.

Entry price depends on when the article landed (KST):
- before 09:00 on a trading day   -> that day's open
- 09:00-15:20 on a trading day    -> first tick >= publish + 60s (`ticks`,
                                     only for codes scalp-it subscribed that
                                     day; otherwise unmeasurable -> dropped)
- after 15:20, weekend, holiday   -> next trading day's open
Exit is always the trade day's close. `excess` subtracts that day's
cross-sectional median open->close, so a broad market move doesn't count
as the article being "right".

Queries are per-pair and index-backed with parallel workers disabled: this
reads the live primary that scalp-it's collectors write to (a first
set-based version exhausted its shared memory).

Usage: uv run python scripts/eval/build_news_eval_set.py --out data/eval/news_eval_set.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
from pathlib import Path

import psycopg2

INTRADAY_START = dt.time(9, 0)
INTRADAY_CUTOFF = dt.time(15, 20)
ENTRY_DELAY = dt.timedelta(seconds=60)


def connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(
        host=os.environ["GPTQ_QUANT_AIRFLOW_DB_HOST"],
        port=int(os.environ["GPTQ_QUANT_AIRFLOW_DB_PORT"]),
        user=os.environ["GPTQ_QUANT_AIRFLOW_DB_USER"],
        password=os.environ["GPTQ_QUANT_AIRFLOW_DB_PASSWORD"],
        dbname=os.environ["GPTQ_QUANT_AIRFLOW_DB_DBNAME"],
    )
    conn.set_session(readonly=True)
    with conn.cursor() as cur:
        cur.execute("SET max_parallel_workers_per_gather = 0")
    return conn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    conn = connect()
    cur = conn.cursor()

    cur.execute(
        "SELECT date FROM daily_bars_adjusted WHERE code = '005930' AND date >= '2026-08-01' ORDER BY date"
    )
    trading_days = [r[0] for r in cur.fetchall()]
    last_day = trading_days[-1]

    cur.execute("SELECT code, name, market, sector FROM stocks")
    stocks = {r[0]: {"name": r[1], "market": r[2], "sector": r[3]} for r in cur.fetchall()}

    cur.execute(
        """
        SELECT n.id, n.title, n.summary, n.category,
               n.published_at AT TIME ZONE 'Asia/Seoul', t.ticker
        FROM news_articles n JOIN news_article_tickers t ON t.article_id = n.id
        WHERE t.ticker ~ '^[0-9]{6}$'
        ORDER BY n.published_at
        """
    )
    pairs = cur.fetchall()

    market_median: dict[dt.date, float] = {}

    def median_open_to_close(day: dt.date) -> float:
        if day not in market_median:
            cur.execute(
                "SELECT close / open - 1 FROM daily_bars_adjusted "
                "WHERE date = %s AND open > 0 AND trade_value > 0",
                (day,),
            )
            market_median[day] = statistics.median(r[0] for r in cur.fetchall())
        return market_median[day]

    def bar(code: str, day: dt.date) -> tuple[float, float] | None:
        cur.execute(
            "SELECT open, close FROM daily_bars_adjusted WHERE code = %s AND date = %s",
            (code, day),
        )
        row = cur.fetchone()
        return (row[0], row[1]) if row and row[0] and row[0] > 0 else None

    rows_out = []
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for article_id, title, summary, category, published_kst, ticker in pairs:
        pub_date, pub_time = published_kst.date(), published_kst.time()
        is_trading_day = pub_date in trading_days

        if is_trading_day and pub_time < INTRADAY_START:
            session, trade_day = "pre_market", pub_date
        elif is_trading_day and pub_time < INTRADAY_CUTOFF:
            session, trade_day = "intraday", pub_date
        else:
            session = "after_close"
            later = [d for d in trading_days if d > pub_date]
            if not later:
                drop("no_next_trading_day_in_data")
                continue
            trade_day = later[0]

        if trade_day > last_day:
            drop("trade_day_beyond_data")
            continue

        day_bar = bar(ticker, trade_day)
        if day_bar is None:
            drop("no_daily_bar")
            continue
        day_open, day_close = day_bar

        prior = [d for d in trading_days if d < trade_day]
        prev_bar = bar(ticker, prior[-1]) if prior else None
        prev_close = prev_bar[1] if prev_bar else None

        if session == "intraday":
            cur.execute(
                "SELECT price FROM ticks WHERE code = %s AND ts >= %s AND ts < %s "
                "ORDER BY ts, seq LIMIT 1",
                (
                    ticker,
                    published_kst + ENTRY_DELAY,
                    dt.datetime.combine(trade_day, dt.time(15, 30)),
                ),
            )
            tick = cur.fetchone()
            if tick is None:
                drop("intraday_no_ticks")
                continue
            entry = abs(float(tick[0]))
        else:
            entry = day_open

        ret = day_close / entry - 1
        mkt = median_open_to_close(trade_day)
        rows_out.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "name": stocks.get(ticker, {}).get("name"),
                "market": stocks.get(ticker, {}).get("market"),
                "sector": stocks.get(ticker, {}).get("sector"),
                "title": title,
                "summary": summary,
                "category": category,
                "published_kst": published_kst.isoformat(),
                "session": session,
                "trade_day": trade_day.isoformat(),
                "prev_close": prev_close,
                "entry": entry,
                "close": day_close,
                "pre_news_move": (entry / prev_close - 1) if prev_close else None,
                "ret": ret,
                "market_median_oc": mkt,
                "excess": ret - mkt,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_session: dict[str, int] = {}
    for r in rows_out:
        by_session[r["session"]] = by_session.get(r["session"], 0) + 1
    print(f"pairs={len(pairs)} kept={len(rows_out)} by_session={by_session} dropped={dropped}")


if __name__ == "__main__":
    main()
