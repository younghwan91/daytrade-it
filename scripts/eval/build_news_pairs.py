"""뉴스 쌍 전체(09-01~) + v2 판정 → 경로 분석용 표본.

    uv run python scripts/eval/build_news_pairs.py   # trader (Anthropic 키), 결과를 simnode로 복사

`build_news_eval_set.py`와 달리 장중 기사를 틱이 아니라 분봉으로 진입시킬 것이라
틱 없는 장중 쌍도 버리지 않는다. 진입 분만 정하고 가격은 simnode 분석에서 붙인다.
- kr_quant `news_articles`/`news_article_tickers` 쌍: 판정은 eval 캐시(v2, Haiku),
  없으면 `eval_news_prompts.judge_all`로 채운다(같은 캐시에 append).
- 데몬 판정 아카이브 `data/news_judgments/*.jsonl`: 운영이 실제로 쓴 v2 판정을 그대로.
같은 (종목, 제목)은 한 번만 쓴다(DB 쌍 우선).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

import psycopg2

OUT = Path("data/eval/news_pairs.jsonl")
ARCHIVE = Path("data/news_judgments")
MODEL = "claude-haiku-4-5-20251001"
INTRADAY_START = dt.time(9, 0)
INTRADAY_CUTOFF = dt.time(15, 20)

_p = Path(__file__).with_name("eval_news_prompts.py")
_s = importlib.util.spec_from_file_location("eval_news_prompts", _p)
enp = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = enp
_s.loader.exec_module(enp)

_q = Path(__file__).with_name("exit_samples.py")
_t = importlib.util.spec_from_file_location("exit_samples", _q)
es = importlib.util.module_from_spec(_t)
sys.modules[_t.name] = es
_t.loader.exec_module(es)

FEATURES = (
    "subject",
    "material_type",
    "persistence",
    "specificity",
    "freshness",
    "surprise",
    "sentiment_direction",
    "sentiment_strength",
    "scale_vs_size",
    "reports_price_move",
)


def session_of(published: dt.datetime, trading_days: list[dt.date]) -> tuple[str, dt.date] | None:
    d, t = published.date(), published.time()
    if d in trading_days and t < INTRADAY_START:
        return "pre_market", d
    if d in trading_days and t < INTRADAY_CUTOFF:
        return "intraday", d
    later = [x for x in trading_days if x > d]
    return ("after_close", later[0]) if later else None


def score_of(j: dict | None) -> float | None:
    if not j:
        return None
    try:
        return enp.v2_score(j, drop_mention=True, drop_price_move=False, use_surprise=False)
    except (KeyError, ValueError, TypeError):
        return None


def main() -> None:
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True)
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute(
        "SELECT date FROM daily_bars_adjusted WHERE code = '005930' AND date >= '2026-08-25' ORDER BY date"
    )
    trading_days = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT code, name, market, sector FROM stocks")
    stocks = {r[0]: {"name": r[1], "market": r[2], "sector": r[3]} for r in cur.fetchall()}
    cur.execute(
        """
        SELECT n.id, n.title, n.summary, n.category, n.published_at AT TIME ZONE 'Asia/Seoul', t.ticker
        FROM news_articles n JOIN news_article_tickers t ON t.article_id = n.id
        WHERE t.ticker ~ '^[0-9]{6}$' ORDER BY n.published_at
        """
    )
    db_rows = []
    for article_id, title, summary, category, published, ticker in cur.fetchall():
        st = stocks.get(ticker, {})
        db_rows.append(
            {
                "article_id": article_id,
                "ticker": ticker,
                "name": st.get("name"),
                "market": st.get("market"),
                "sector": st.get("sector"),
                "title": title,
                "summary": summary,
                "category": category,
                "published": published,
                "source": "db",
            }
        )

    scored = asyncio.run(enp.judge_all(db_rows, "v2", MODEL))
    judgment = {f"{x.row['article_id']}|{x.row['ticker']}": x.judgment for x in scored}
    for r in db_rows:
        r["v2"] = judgment.get(f"{r['article_id']}|{r['ticker']}")

    seen = {(r["ticker"], r["title"]) for r in db_rows}
    arch_rows = []
    for path in sorted(ARCHIVE.glob("*.jsonl")):
        for line in path.open():
            a = json.loads(line)
            if (a["ticker"], a["title"]) in seen:
                continue
            seen.add((a["ticker"], a["title"]))
            st = stocks.get(a["ticker"], {})
            arch_rows.append(
                {
                    "article_id": a["article_id"],
                    "ticker": a["ticker"],
                    "name": a.get("name") or st.get("name"),
                    "market": st.get("market"),
                    "sector": st.get("sector"),
                    "title": a["title"],
                    "category": a.get("category"),
                    "published": dt.datetime.fromisoformat(a["published_at"]).replace(tzinfo=None),
                    "source": "daemon",
                    "v2": {k: v for k, v in a["v2"].items() if k != "score"},
                }
            )

    n_out = 0
    with OUT.open("w") as f:
        for r in db_rows + arch_rows:
            s = session_of(r["published"], trading_days)
            if s is None:
                continue
            session, trade_day = s
            j = r["v2"] or {}
            rec = {
                "key": f"{r['article_id']}|{r['ticker']}",
                "ticker": r["ticker"],
                "name": r["name"],
                "market": r["market"],
                "sector": r["sector"],
                "title": r["title"],
                "category": r["category"],
                "source": r["source"],
                "published_kst": r["published"].isoformat(timespec="seconds"),
                "session": session,
                "trade_day": trade_day.isoformat(),
                "entry_ts": es.entry_minute(
                    session, r["published"].isoformat(timespec="seconds"), trade_day.isoformat()
                ),
                "score": score_of(r["v2"]),
                **{k: j.get(k) for k in FEATURES},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(
        f"db pairs={len(db_rows)} judged={len(judgment)} daemon extra={len(arch_rows)} written={n_out}"
    )


if __name__ == "__main__":
    main()
