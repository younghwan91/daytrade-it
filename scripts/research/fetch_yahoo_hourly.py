#!/usr/bin/env python
"""Yahoo Finance KRX 60분봉을 긁어 JSONL 로 쌓는다 (재실행하면 이어받는다).

키움 ``ka10080`` 은 2025-09-01 이전을 서버가 안 들고 있다(봉 크기를 키워도
같다). Yahoo 는 ``interval=1h`` 로 **730 거래일**을 주는데, 이 창이 거래일
단위로 굴러가므로 하루 미룰 때마다 하루치 과거가 영구히 잘려 나간다.

받은 봉은 ``minute_bars`` 에 **절대 넣지 않는다** — 그 테이블은 1분봉 전용이고
해상도가 섞이면 조용히 틀린 답이 나온다. 여기서는 파일로만 떨군다.

kr_quant 은 유니버스·검증용 일봉 조회에만 쓰고 읽기 전용이다.

접미사 함정
    ``stocks.market`` 이 틀린 종목이 있어서 ``.KS``/``.KQ`` 폴백이 필요한데,
    **틀린 접미사가 404 를 주리라 믿으면 안 된다.** 247540(코스닥)을
    ``247540.KS`` 로 부르면 728봉짜리 *다른 종목*이 조용히 돌아온다. 그래서
    받은 봉을 ``daily_bars_adjusted`` 와 대조해 같은 종목인지 확인하고,
    통과 못 하면 반대 접미사로 다시 받는다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1h&range={rng}"
# 브라우저처럼 생긴 UA 를 쓰면 Yahoo 가 crumb 없는 요청으로 보고 전부 429 를 준다.
# 밋밋한 "Mozilla/5.0" 은 그냥 통과한다 (2026-09-22 simnode 에서 A/B 확인).
UA = "Mozilla/5.0"
SOURCE = "yahoo_1h"
THROTTLE_BACKOFF = (5, 15, 30, 60, 90, 120)
# 730d 가 422 로 튕기는 신규 상장 종목이 있다. 짧은 창으로 내려가며 다시 시도한다.
RANGES = ("730d", "1y", "60d")
# 같은 종목인지 판정: 일봉 고저 범위 안에 들어오는 날의 비율이 이 값 이상이어야 한다.
MATCH_THRESHOLD = 0.8
REF_DAYS = 40  # 대조에 쓸 최근 거래일 수

MARKET_SUFFIX = {"거래소": "KS", "코스닥": "KQ", "코넥스": "KQ"}


def match_score(rows: list[dict], ref: dict) -> tuple[float, int]:
    """받은 봉이 우리가 아는 그 종목인지 — 일봉 고저 범위 포함 비율.

    무상증자 등으로 수정계수가 미세하게 달라 0.5% 여유를 둔다. 겹치는 날이
    없으면 판정 불가라 ``(0.0, 0)`` 을 준다.
    """
    if not rows or not ref:
        return 0.0, 0
    day_hi: dict[str, float] = {}
    day_lo: dict[str, float] = {}
    for r in rows:
        d = r["ts"][:10]
        day_hi[d] = max(day_hi.get(d, r["high"]), r["high"])
        day_lo[d] = min(day_lo.get(d, r["low"]), r["low"])
    hits = checked = 0
    for d, (dh, dl) in ref.items():
        if d not in day_hi:
            continue
        checked += 1
        if day_hi[d] <= dh * 1.005 and day_lo[d] >= dl * 0.995:
            hits += 1
    return (hits / checked if checked else 0.0), checked


class Fetcher:
    def __init__(self, sleep: float, retries: int = 6) -> None:
        self.sleep = sleep
        self.retries = retries

    def _get(self, sym: str, rng: str) -> dict:
        req = urllib.request.Request(CHART_URL.format(sym=sym, rng=rng), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=40) as resp:
            return json.load(resp)

    def _try_suffix(self, code: str, sfx: str) -> tuple[list[dict], str | None]:
        last_err = None
        for rng in RANGES:
            for attempt in range(self.retries):
                try:
                    return self._rows(code, self._get(f"{code}.{sfx}", rng)), None
                except urllib.error.HTTPError as exc:
                    last_err = f"HTTP {exc.code}"
                    if exc.code == 404:
                        return [], last_err  # 이 티커 자체가 없다
                    if exc.code == 422:
                        break  # 이 창이 너무 길다 — 짧은 창으로 내려간다
                    if exc.code in (429, 502, 503):
                        time.sleep(
                            THROTTLE_BACKOFF[min(attempt, len(THROTTLE_BACKOFF) - 1)]
                            + random.random() * 3
                        )
                        continue
                    return [], last_err
                except Exception as exc:  # noqa: BLE001 — 타임아웃·JSON 깨짐 등
                    last_err = f"{type(exc).__name__}: {exc}"
                    time.sleep(1 + attempt)
        return [], last_err

    def fetch(self, code: str, market: str, ref: dict) -> dict:
        """접미사 둘을 필요한 만큼 시도하고, 일봉 대조를 통과한 쪽을 쓴다."""
        first = MARKET_SUFFIX.get(market or "", "KS")
        order = [first, "KQ" if first == "KS" else "KS"]
        tried: list[dict] = []
        for i, sfx in enumerate(order):
            rows, err = self._try_suffix(code, sfx)
            score, days = match_score(rows, ref)
            tried.append(
                {
                    "suffix": sfx,
                    "bars": len(rows),
                    "score": round(score, 3),
                    "checked_days": days,
                    "error": err,
                    "rows": rows,
                }
            )
            if rows and score >= MATCH_THRESHOLD:
                return self._result(code, market, tried, i, suspect=False)
            time.sleep(self.sleep)
        # 어느 쪽도 대조를 통과 못 했다 — 그나마 나은 쪽을 쓰되 의심으로 표시한다
        best = max(range(len(tried)), key=lambda k: (tried[k]["score"], tried[k]["bars"]))
        return self._result(code, market, tried, best, suspect=bool(tried[best]["rows"]))

    @staticmethod
    def _result(code: str, market: str, tried: list[dict], pick: int, *, suspect: bool) -> dict:
        chosen = tried[pick]
        return {
            "code": code,
            "market": market,
            "suffix": chosen["suffix"],
            "suffix_fallback": pick > 0,
            "bars": len(chosen["rows"]),
            "match_score": chosen["score"],
            "checked_days": chosen["checked_days"],
            "suspect": suspect,
            "error": chosen["error"],
            "attempts": [{k: v for k, v in t.items() if k != "rows"} for t in tried],
            "rows": chosen["rows"],
        }

    @staticmethod
    def _rows(code: str, payload: dict) -> list[dict]:
        result = payload["chart"]["result"][0]
        stamps = result.get("timestamp") or []
        q = result["indicators"]["quote"][0]
        rows = []
        for i, epoch in enumerate(stamps):
            o, h, lo, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
            if o is None or h is None or lo is None or c is None:
                continue  # 거래 없는 시간대는 전부 null 로 온다
            ts = datetime.fromtimestamp(epoch, KST)
            rows.append(
                {
                    "code": code,
                    "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": float(o),
                    "high": float(h),
                    "low": float(lo),
                    "close": float(c),
                    "volume": int(q["volume"][i] or 0),
                    "source": SOURCE,
                }
            )
        return rows


def universe(dsn: str) -> list[tuple[str, str]]:
    """``minute_bars`` 가 실제로 담고 있는 종목 = 우리 유니버스."""
    import psycopg2

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "select s.code, coalesce(s.market, '') "
            "from stocks s where s.code in (select distinct code from minute_bars) "
            "order by s.code"
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def reference_bars(dsn: str, codes: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """대조용 일봉 고저 — 최근 ``REF_DAYS`` 거래일."""
    import psycopg2

    ref: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("select max(date) from daily_bars_adjusted")
        last = cur.fetchone()[0]
        cur.execute(
            "select code, date::text, high, low from daily_bars_adjusted "
            "where code = any(%s) and date > %s::date - %s and high > 0 and low > 0",
            (codes, last, REF_DAYS * 2),
        )
        for code, date, high, low in cur.fetchall():
            ref[code][date] = (float(high), float(low))
    return ref


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Yahoo KRX 60분봉 수집")
    ap.add_argument("--out", default="/home/young/git/daytrade-it/data/yahoo_hourly")
    ap.add_argument("--dsn", default=os.environ.get("KR_QUANT_DB"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--codes", default=None, help="쉼표 구분. 생략하면 유니버스 전체")
    args = ap.parse_args(argv)
    if not args.dsn:
        print("KR_QUANT_DB 가 없다.", file=sys.stderr)
        return 2

    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    bars_path = os.path.join(raw_dir, "bars.jsonl")
    done_path = os.path.join(raw_dir, "done.jsonl")

    done: set[str] = set()
    if os.path.exists(done_path):
        with open(done_path) as fh:
            for line in fh:
                # 중단된 줄은 버린다
                with contextlib.suppress(Exception):
                    done.add(json.loads(line)["code"])

    targets = universe(args.dsn)
    if args.codes:
        wanted = {c.strip() for c in args.codes.split(",") if c.strip()}
        targets = [t for t in targets if t[0] in wanted]
    ref = reference_bars(args.dsn, [c for c, _ in targets])
    targets = [t for t in targets if t[0] not in done]
    print(
        f"대상 {len(targets)}종목 (이미 받은 {len(done)}종목 건너뜀) · 워커 {args.workers}",
        file=sys.stderr,
    )
    if not targets:
        return 0

    fetcher = Fetcher(args.sleep)
    lock = threading.Lock()
    c: dict[str, int] = defaultdict(int)
    started = time.time()

    with open(bars_path, "a") as bars_fh, open(done_path, "a") as done_fh:

        def work(item: tuple[str, str]) -> None:
            code, market = item
            rec = fetcher.fetch(code, market, ref.get(code, {}))
            rows = rec.pop("rows")
            rec["first"] = rows[0]["ts"] if rows else None
            rec["last"] = rows[-1]["ts"] if rows else None
            with lock:
                for r in rows:
                    bars_fh.write(json.dumps(r) + "\n")
                done_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                c["rows"] += len(rows)
                c["fallback"] += int(rec["suffix_fallback"])
                c["suspect"] += int(rec["suspect"])
                c["ok" if rows else "fail"] += 1
                n = c["ok"] + c["fail"]
                if n % 100 == 0:
                    bars_fh.flush()
                    done_fh.flush()
                    rate = n / max(1e-9, time.time() - started)
                    print(
                        f"  {n}/{len(targets)} · {c['rows']:,}행 · 실패 {c['fail']} · "
                        f"폴백 {c['fallback']} · 의심 {c['suspect']} · "
                        f"ETA {(len(targets) - n) / max(1e-9, rate) / 60:.0f}분",
                        file=sys.stderr,
                        flush=True,
                    )
            time.sleep(args.sleep)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, targets))

    print(
        f"완료 · {c['rows']:,}행 · 성공 {c['ok']} · 실패 {c['fail']} · "
        f"접미사 폴백 {c['fallback']} · 의심 {c['suspect']} · "
        f"{(time.time() - started) / 60:.1f}분",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
