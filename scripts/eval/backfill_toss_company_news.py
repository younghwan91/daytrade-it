"""Toss 종목별 과거 뉴스를 상장 종목 전체에 대해 수집한다 (news universe, "A-plan").

    uv run python scripts/eval/backfill_toss_company_news.py --since 2026-03-16 [--codes 005930,000911]

엔드포인트는 비공식이다(2026-09-15 확인):
`GET wts-info-api.tossinvest.com/api/v2/news/companies/{6자리}?number=1..&size=100`, 대략 최신순,
`result.body[]`에 id(newsId)·title·summary·createdAt(KST, tz 없음)·stockCodes(항상 null)·
newsType·ticsTitles·source. 정렬이 느슨하고(2026-09-16 확인: 005930 한 페이지에 역전 45회)
최대 100페이지(1만 건)까지만 준다(101페이지는 HTTP 400 → `page_cap` 으로 끝냄; 000660 급 대형주는
6개월을 다 못 채운다). `lastPage` 도 믿을 수 없다(005930 1페이지가 lastPage=true 인데 2페이지에 100건). 그래서 페이지에
`--since` 이후 기사가 하나도 없거나 빈 페이지일 때만 멈춘다. 관련 종목은 같은 newsId 가 어느 종목 목록에 나오는지로 복원한다
(`news_universe_pairs.py`).

**라이브 매매 보호가 최우선** — trader 의 라이브 데몬이 같은 네트워크에서 토스를 폴링한다.
- 단일 스레드, 요청 시작 간격 ≥ `--sleep`(최소 1.0초).
- 429/5xx/타임아웃은 지수 백오프. 403/429 가 연속 3번이면 차단 가능성으로 보고 로그 남기고 종료(코드 2).
- 평일 08:20~15:45 KST 에는 요청하지 않고 15:45 까지 잔다(공휴일도 평일이면 쉰다 — 보수적).
- SIGTERM/SIGINT 는 진행 중인 페이지를 저장하고 깨끗이 끝낸다.

재개: 종목당 `{code}.partial.jsonl`(받은 페이지 누적) + `{code}.state.json`(마지막으로 끝낸 페이지)
를 두고, 끝나면 newsId 중복을 제거해 `{code}.jsonl` 로 원자적으로 옮긴 뒤 `{code}.done` 을 쓴다.
`.done` 이 있는 종목은 건너뛴다. 재개 사이에 새 기사가 끼면 페이지 경계가 뒤로 밀려 중복이 생기고
(중복은 제거), 느슨한 정렬 탓에 페이지 경계의 기사가 드물게 빠질 수 있다.

kr_quant 는 `stocks` 읽기만 한다(read-only 세션). `data/toss_history` 는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid5

import httpx
from krx_news_client.scrapers.base import make_article_id
from krx_news_client.scrapers.toss import build_article_url

from daytrade_it.infrastructure.external.toss_news_client import TOSS_NEWS_NAMESPACE

URL = "https://wts-info-api.tossinvest.com/api/v2/news/companies/{code}"
OUT_DIR = Path("data/toss_universe")
KEEP = ("id", "title", "summary", "createdAt", "stockCodes", "newsType", "ticsTitles")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128 Safari/537.36"
KST = dt.timezone(dt.timedelta(hours=9))
PAUSE_START = dt.time(8, 20)
PAUSE_END = dt.time(15, 45)
BLOCK_STATUSES = frozenset({403, 429})
MIN_SLEEP = 1.0

log = logging.getLogger("toss_universe")


# --- pure helpers -------------------------------------------------------------------


def live_article_id(news_id: str) -> str:
    """The live daemon's domain id for a Toss newsId (uuid5 over `toss:<md5(url)[:12]>`)."""
    return str(uuid5(TOSS_NEWS_NAMESPACE, make_article_id("toss", build_article_url(news_id))))


def market_pause_seconds(now: dt.datetime) -> float:
    """Seconds to sleep before the next request: >0 only on weekdays 08:20 ≤ t < 15:45 KST."""
    kst = now.astimezone(KST)
    if kst.weekday() >= 5 or not (PAUSE_START <= kst.time() < PAUSE_END):
        return 0
    end = kst.replace(hour=PAUSE_END.hour, minute=PAUSE_END.minute, second=0, microsecond=0)
    return (end - kst).total_seconds()


class BlockGuard:
    """Counts consecutive 403/429 responses; any other status breaks the streak."""

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.streak = 0

    def record(self, status: int) -> bool:
        self.streak = self.streak + 1 if status in BLOCK_STATUSES else 0
        return self.blocked

    @property
    def blocked(self) -> bool:
        return self.streak >= self.limit


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    return float(min(cap, base * 2**attempt))


def page_rows(
    body: list[dict[str, Any]], since: str, code: str
) -> tuple[list[dict[str, Any]], bool]:
    """Rows on this page created on/after `since`, and whether another page is needed.

    Toss lists are only roughly newest-first and `lastPage` is unreliable, so paging continues
    while any row on the page is still >= since; it stops on a fully older or empty page.
    """
    rows = []
    for a in body:
        if a["createdAt"][:10] >= since:
            src = a.get("source") or {}
            rows.append({**{k: a.get(k) for k in KEEP}, "source": src.get("code"), "code": code})
    more = bool(rows)
    return rows, more


def page_error_outcome(status_code: int, page_number: int) -> str:
    """Outcome of a non-200 page: 404 = no list; 400 past page 1 = Toss's 100-page cap."""
    if status_code == 404:
        return "done"
    if status_code == 400 and page_number > 1:
        return "page_cap"
    return "failed"


def progress_line(
    done: int,
    total: int,
    articles: int,
    requests: int,
    errors: int,
    elapsed_active_s: float,
    done_this_run: int,
) -> str:
    left = total - done
    if done_this_run:
        eta_s = int(elapsed_active_s / done_this_run * left)
        eta = f"{eta_s // 3600}h{eta_s % 3600 // 60:02d}m"
    else:
        eta = "?"
    return (
        f"progress codes {done}/{total} articles {articles} requests {requests} "
        f"errors {errors} eta {eta} (active time, excl. market-hour pauses)"
    )


# --- IO -----------------------------------------------------------------------------------


class Stop(Exception):
    """Raised to unwind cleanly on SIGTERM or a suspected block."""


class Collector:
    def __init__(self, out_dir: Path, since: str, sleep: float, max_pages: int) -> None:
        self.out_dir = out_dir
        self.since = since
        self.sleep = max(MIN_SLEEP, sleep)
        self.max_pages = max_pages
        self.stop_event = threading.Event()
        self.guard = BlockGuard(3)
        self.requests = 0
        self.errors = 0
        self.paused_s = 0.0
        self._last_request = 0.0
        self.client = httpx.Client(
            headers={"User-Agent": UA, "Referer": "https://tossinvest.com/"}, timeout=20
        )

    def _wait(self, seconds: float) -> None:
        if seconds > 0 and self.stop_event.wait(seconds):
            raise Stop("signal")

    def _gate(self) -> None:
        while (pause := market_pause_seconds(dt.datetime.now(KST))) > 0:
            log.info("market hours: pausing %.0f s until 15:45 KST", pause)
            t0 = time.monotonic()
            self._wait(pause + 1)
            self.paused_s += time.monotonic() - t0
        self._wait(self._last_request + self.sleep - time.monotonic())
        if self.stop_event.is_set():
            raise Stop("signal")

    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response | None:
        """GET with pacing/backoff. None = gave up (transient); raises Stop on block/signal."""
        for attempt in range(5):
            self._gate()
            self._last_request = time.monotonic()
            self.requests += 1
            try:
                r = self.client.get(url, params=params)
            except httpx.HTTPError as exc:
                self.errors += 1
                log.warning("%s %s: %s (attempt %d)", url, params, type(exc).__name__, attempt)
                self._wait(backoff_delay(attempt, 5, 120))
                continue
            if self.guard.record(r.status_code):
                log.error(
                    "STOP: %d consecutive 403/429 (last %d) — possible IP block",
                    self.guard.streak,
                    r.status_code,
                )
                raise Stop("blocked")
            if r.status_code in BLOCK_STATUSES:
                self.errors += 1
                log.warning("%s %s: HTTP %d, backing off", url, params, r.status_code)
                self._wait(backoff_delay(attempt, 60, 600))
                continue
            if r.status_code >= 500:
                self.errors += 1
                log.warning("%s %s: HTTP %d (attempt %d)", url, params, r.status_code, attempt)
                self._wait(backoff_delay(attempt, 5, 120))
                continue
            return r
        return None

    def fetch_code(self, code: str) -> tuple[str, int]:
        """Fetch one code to completion. Returns (status, rows). status in done|page_cap|max_pages|failed."""
        partial = self.out_dir / f"{code}.partial.jsonl"
        state_path = self.out_dir / f"{code}.state.json"
        page = json.loads(state_path.read_text())["page"] if state_path.exists() else 0
        status = "max_pages"
        while page < self.max_pages:
            r = self.get(URL.format(code=code), {"number": page + 1, "size": 100})
            if r is None:
                return "failed", 0
            if r.status_code != 200:
                status = page_error_outcome(r.status_code, page + 1)
                if status == "failed":
                    self.errors += 1
                    log.warning("%s page %d: HTTP %d, skipping code", code, page + 1, r.status_code)
                    return "failed", 0
                break
            result = (r.json() or {}).get("result") or {}
            rows, more = page_rows(result.get("body") or [], self.since, code)
            with partial.open("a") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            page += 1
            state_path.write_text(json.dumps({"page": page}))
            if not more:
                status = "done"
                break
        return status, self._finalize(code, partial, state_path, status)

    def _finalize(self, code: str, partial: Path, state_path: Path, status: str) -> int:
        seen: dict[str, str] = {}
        if partial.exists():
            for line in partial.open():
                if line.strip():
                    seen.setdefault(json.loads(line)["id"], line)
        final = self.out_dir / f"{code}.jsonl"
        tmp = final.with_suffix(".tmp")
        tmp.write_text("".join(seen.values()))
        tmp.rename(final)
        (self.out_dir / f"{code}.done").write_text(
            json.dumps(
                {
                    "status": status,
                    "rows": len(seen),
                    "since": self.since,
                    "finished_at": dt.datetime.now(KST).isoformat(),
                }
            )
        )
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        return len(seen)


def setup_logging() -> None:
    """Plain line-buffered log lines (the app import installs a JSON root handler on stdout)."""
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def universe_codes() -> list[str]:
    import psycopg2

    with psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5) as c:
        c.set_session(readonly=True)
        with c.cursor() as cur:
            cur.execute("SELECT DISTINCT code FROM stocks WHERE code ~ '^[0-9]{6}$'")
            return sorted(r[0] for r in cur.fetchall())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-16")
    ap.add_argument("--codes", default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--max-pages", type=int, default=300)
    a = ap.parse_args()
    setup_logging()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    codes = a.codes.split(",") if a.codes else universe_codes()
    col = Collector(a.out_dir, a.since, a.sleep, a.max_pages)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, _f: (log.info("signal %d: stopping", s), col.stop_event.set()))

    done = sum((a.out_dir / f"{c}.done").exists() for c in codes)
    log.info(
        "start: %d codes, %d already done, since %s, sleep %.1fs",
        len(codes),
        done,
        a.since,
        col.sleep,
    )
    articles = done_run = fail_streak = 0
    t0 = time.monotonic()
    exit_code = 0
    try:
        for code in codes:
            if (a.out_dir / f"{code}.done").exists():
                continue
            status, n = col.fetch_code(code)
            if status == "failed":
                fail_streak += 1
                log.warning("%s: failed, left for next run", code)
                if fail_streak >= 20:
                    log.error("STOP: 20 consecutive codes failed — network or endpoint problem")
                    raise Stop("failing")
                continue
            fail_streak = 0
            done += 1
            done_run += 1
            articles += n
            if status != "done":
                log.warning("%s: %s (%d rows)", code, status, n)
            if done_run % 50 == 0:
                active = time.monotonic() - t0 - col.paused_s
                log.info(
                    progress_line(
                        done, len(codes), articles, col.requests, col.errors, active, done_run
                    )
                )
    except Stop as exc:
        log.warning("stopped (%s); partial pages kept, rerun the same command to resume", exc)
        exit_code = 0 if str(exc) == "signal" else 2
    active = time.monotonic() - t0 - col.paused_s
    log.info(progress_line(done, len(codes), articles, col.requests, col.errors, active, done_run))
    log.info("exit %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
