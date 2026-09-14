"""청산 규칙 평가 표본 — 평가셋(09-01~11) + 운영 신호(09-14~).

    uv run python scripts/eval/exit_samples.py > data/eval/exit_backfill_targets.csv

진입 시각은 프롬프트 평가와 같다(docs/research/2026-09-13-news-prompt-eval.md):
장전·장후 기사는 거래일 09:00, 장중은 발표 +60초가 속한 분의 다음 분.
운영 신호는 (종목, 거래일)당 첫 신호만 쓴다.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

import psycopg2

EVAL_SET = Path("data/eval/news_eval_set.jsonl")
V2_CACHE = Path("data/eval/cache/v2__claude-haiku-4-5-20251001.jsonl")
OUT = Path("data/eval/exit_samples.jsonl")
DEV_END = dt.date(2026, 9, 8)
TEST_END = dt.date(2026, 9, 11)
BUY_THRESHOLD = 0.15
MAX_PRE_MOVE = 0.02
KST = dt.timezone(dt.timedelta(hours=9))

_p = Path(__file__).with_name("eval_news_prompts.py")
_s = importlib.util.spec_from_file_location("eval_news_prompts", _p)
_enp = importlib.util.module_from_spec(_s)
sys.modules[_s.name] = _enp  # eval_news_prompts의 dataclass가 모듈 등록을 요구한다
_s.loader.exec_module(_enp)


def entry_minute(session: str, published_kst: str, trade_day: str) -> str:
    if session != "intraday":
        return f"{trade_day}T09:00"
    t = dt.datetime.fromisoformat(published_kst) + dt.timedelta(seconds=60)
    if t.second or t.microsecond:  # 발표+60초 이후 첫 틱이 속한 분 = 올림
        t = t.replace(second=0, microsecond=0) + dt.timedelta(minutes=1)
    return t.strftime("%Y-%m-%dT%H:%M")


def split_of(trade_day: str) -> str:
    d = dt.date.fromisoformat(trade_day)
    return "dev" if d <= DEV_END else "test" if d <= TEST_END else "forward"


def eval_rows() -> list[dict]:
    cache = {}
    for line in V2_CACHE.open():
        rec = json.loads(line)
        cache[rec["key"]] = rec["judgment"]
    out = []
    for line in EVAL_SET.open():
        r = json.loads(line)
        key = f"{r['article_id']}|{r['ticker']}"
        j = cache.get(key)
        # eval_news_prompts.report()와 같게: 스키마 밖 판정은 BUY로 세지 않고, 기준은 초과(>).
        try:
            score = _enp.v2_score(j, drop_mention=True, drop_price_move=False, use_surprise=False)
        except (KeyError, ValueError, TypeError):
            score = 0.0
        wide = r["pre_news_move"] <= MAX_PRE_MOVE
        out.append(
            {
                "key": key,
                "ticker": r["ticker"],
                "trade_day": r["trade_day"],
                "entry_ts": entry_minute(r["session"], r["published_kst"], r["trade_day"]),
                "split": split_of(r["trade_day"]),
                "ops": wide and score > BUY_THRESHOLD,
                "wide": wide,
                "market_median_oc": r["market_median_oc"],
            }
        )
    return out


def forward_rows(dsn: str) -> list[dict]:
    with psycopg2.connect(dsn, connect_timeout=5) as c, c.cursor() as cur:
        cur.execute(
            "select id::text, ticker, created_at from trading_signals "
            "where direction = 'BUY' and created_at >= '2026-09-14' order by created_at"
        )
        rows = cur.fetchall()
    seen: set[tuple[str, str]] = set()
    out = []
    for sid, ticker, created in rows:
        local = created.astimezone(KST)
        day = local.date().isoformat()
        if (ticker, day) in seen:
            continue
        seen.add((ticker, day))
        t = local.replace(second=0, microsecond=0, tzinfo=None) + dt.timedelta(minutes=1)
        out.append(
            {
                "key": f"signal:{sid}",
                "ticker": ticker,
                "trade_day": day,
                "entry_ts": t.strftime("%Y-%m-%dT%H:%M"),
                "split": "forward",
                "ops": True,
                "wide": True,
                "market_median_oc": None,
            }
        )
    return out


def main() -> None:
    dsn = os.environ["GPTQ_DATABASE_URL"].replace("postgresql+asyncpg", "postgresql")
    rows = eval_rows() + forward_rows(dsn)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for code, day in sorted({(r["ticker"], r["trade_day"]) for r in rows if r["wide"]}):
        print(f"{code},{day.replace('-', '')}")


if __name__ == "__main__":
    main()
