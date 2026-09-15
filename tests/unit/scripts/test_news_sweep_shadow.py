"""Universe sweep shadow runner (plan C5): schedule, dedupe, stale, block stop, archive format.

No network, no DB -- everything below runs on pure helpers or fakes.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import news_sweep_shadow as shadow  # noqa: E402
from krx_quant_core.market.session import KST  # noqa: E402

from daytrade_it.application.news.judging import JudgmentLog, judgment_record  # noqa: E402
from daytrade_it.application.news.universe import (  # noqa: E402
    PAGE_SIZE,
    SweepRequest,
    judge_queue,
    live_article_id,
    page_items,
    simulate_sweep,
)
from daytrade_it.domain.value_objects import Ticker  # noqa: E402
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures  # noqa: E402
from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo  # noqa: E402

DAY = dt.date(2026, 9, 17)
UNIVERSE = ["005930", "000660", "035420"]


def at(h: int, m: int, s: int = 0) -> dt.datetime:
    return dt.datetime.combine(DAY, dt.time(h, m, s), tzinfo=KST)


# --- schedule ---------------------------------------------------------------


class TestSchedule:
    def test_first_slot_at_or_before_start_is_zero(self) -> None:
        sched = shadow.SweepSchedule(UNIVERSE, DAY)
        assert sched.first_slot(at(7, 30)) == 0
        assert sched.first_slot(at(8, 0)) == 0

    def test_first_slot_rounds_up_mid_sweep(self) -> None:
        sched = shadow.SweepSchedule(UNIVERSE, DAY)
        assert sched.first_slot(at(8, 0, 4)) == 4
        assert sched.first_slot(at(8, 0, 4) + dt.timedelta(milliseconds=1)) == 5

    def test_page_one_plan_repeats_universe(self) -> None:
        sched = shadow.SweepSchedule(UNIVERSE, DAY)
        codes = [sched.code_at(m) for m in range(7)]
        assert codes == UNIVERSE * 2 + ["005930"]
        assert [sched.planned_at(m) for m in range(7)] == [at(8, 0, m) for m in range(7)]

    def test_slots_stop_at_1520(self) -> None:
        sched = shadow.SweepSchedule(UNIVERSE, DAY)
        last = sched.planned_at(sched.end_slot() - 1)
        assert last < at(15, 20) <= sched.planned_at(sched.end_slot())

    def test_dry_run_lines_do_not_touch_network(self) -> None:
        sched = shadow.SweepSchedule(UNIVERSE, DAY)
        lines = shadow.dry_run_lines(sched, at(8, 0, 1), 4)
        assert [line.split()[1] for line in lines] == ["000660", "035420", "005930", "000660"]
        assert "companies/000660?number=1&size=20" in lines[0]


# --- dedupe -----------------------------------------------------------------


def _write_dashboard(path: Path, news_ids: list[str], ticker: str = "005930") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for nid in news_ids:
            aid = live_article_id(nid)
            row = {
                "key": f"{aid}|{ticker}",
                "article_id": aid,
                "ticker": ticker,
                "category": "breaking",
                "v2": {**dataclasses.asdict(_features()), "score": _features().score()},
                "v4_shadow": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestDedupe:
    def test_dashboard_file_is_reread_as_it_grows(self, tmp_path: Path) -> None:
        dash_path = tmp_path / "news_judgments" / f"{DAY}.jsonl"
        keys = shadow.DashboardKeys(dash_path)
        assert live_article_id("yna_1") not in keys  # no file yet
        _write_dashboard(dash_path, ["yna_1"])
        assert live_article_id("yna_1") in keys
        _write_dashboard(dash_path, ["yna_3"])
        assert live_article_id("yna_3") in keys

    def test_dashboard_rows_by_pair_key(self, tmp_path: Path) -> None:
        dash_path = tmp_path / "news_judgments" / f"{DAY}.jsonl"
        _write_dashboard(dash_path, ["yna_1"])
        keys = shadow.DashboardKeys(dash_path)
        aid = live_article_id("yna_1")
        assert keys.row(f"{aid}|005930")["category"] == "breaking"
        assert keys.row(f"{aid}|000660") is None


# --- stale ------------------------------------------------------------------


class TestStale:
    def test_stale_before_previous_session_close(self) -> None:
        stale_before = dt.datetime(2026, 9, 16, 15, 20, tzinfo=KST)
        assert shadow.is_stale({"createdAt": "2026-09-16T15:19:59"}, stale_before)
        assert not shadow.is_stale({"createdAt": "2026-09-16T15:20:00"}, stale_before)
        assert not shadow.is_stale({"createdAt": "2026-09-17T08:10:00"}, stale_before)

    def test_unparseable_created_at_is_stale(self) -> None:
        stale_before = dt.datetime(2026, 9, 16, 15, 20, tzinfo=KST)
        assert shadow.is_stale({}, stale_before)
        assert shadow.is_stale({"createdAt": "garbage"}, stale_before)


# --- block stop -------------------------------------------------------------


class TestBlockGuard:
    def test_three_consecutive_blocks_trip(self) -> None:
        guard = shadow.BlockGuard()
        assert not guard.record(429)
        assert not guard.record(403)
        assert guard.record(429)
        assert guard.tripped

    def test_success_resets_counter(self) -> None:
        guard = shadow.BlockGuard()
        guard.record(429)
        guard.record(429)
        assert not guard.record(200)
        assert not guard.record(429)
        assert not guard.record(403)
        assert not guard.tripped

    def test_other_errors_neither_count_nor_reset(self) -> None:
        guard = shadow.BlockGuard()
        guard.record(429)
        guard.record(None)
        guard.record(500)
        guard.record(429)
        assert guard.record(403)


class TestRateLimiter:
    async def test_spaces_requests_by_interval(self) -> None:
        clock = [100.0]
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock[0] += seconds

        limiter = shadow.RateLimiter(1.0, clock=lambda: clock[0], sleep=fake_sleep)
        await limiter.wait()
        clock[0] += 0.25
        await limiter.wait()
        clock[0] += 3.0
        await limiter.wait()
        assert slept == [0.75]


# --- article + archive row --------------------------------------------------


def _features() -> ArticleFeatures:
    return ArticleFeatures(
        subject="main",
        reports_price_move=False,
        material_type="수주",
        persistence="multi_quarter",
        specificity=2,
        freshness="new",
        surprise=1,
        sentiment_direction=1,
        sentiment_strength=2,
        scale_vs_size="medium",
    )


ITEM = {
    "id": "yna_AKR1",
    "title": "삼성전자 대규모 수주",
    "summary": "요약",
    "createdAt": "2026-09-17T08:31:02",
    "source": {"name": "연합뉴스"},
    "stockCodes": ["005930"],
}


def test_build_article_uses_live_key_and_detail_company_codes() -> None:
    # companyCodes arrives in Toss API order; tickers come out sorted ascending
    # by code so C3's `related_codes` (`tuple(sorted(codes))`) judges an
    # article's pairs in the same order the live runner does (judge_queue's
    # "an article's pairs judged one at a time, in order" refers to one order).
    article = shadow.build_article(ITEM, {"companyCodes": ["005930", "000660"]})
    assert str(article.id) == live_article_id("yna_AKR1")
    assert [t.value for t in article.tickers] == ["000660", "005930"]
    assert article.published_at == at(8, 31, 2)
    assert article.category == ""
    assert article.summary == "요약"


def test_build_article_falls_back_to_page_stock_codes_without_detail() -> None:
    article = shadow.build_article(ITEM, None)
    assert [t.value for t in article.tickers] == ["005930"]


def test_universe_row_equals_live_row_plus_pool(tmp_path: Path) -> None:
    article = shadow.build_article(ITEM, {"companyCodes": ["005930"]})
    ticker = Ticker(value="005930")
    stock = StockInfo(code="005930", name="삼성전자", market="KOSPI", sector="전기전자")
    judged = at(8, 32)
    record = judgment_record(article, ticker, stock, _features(), None, None, judged_at=judged)
    key = f"{article.id}|005930"

    live = JudgmentLog(tmp_path / "live", DAY)
    live.append(key, record)
    uni = shadow.UniverseJudgmentLog(tmp_path / "uni", DAY)
    uni.set_context(article.id, at(8, 31, 40))
    uni.append(key, record)

    live_row = json.loads(live.path.read_text())
    uni_row = json.loads(uni.path.read_text())
    assert uni_row.pop("pool") == "universe"
    assert uni_row.pop("sweep_seen_at") == at(8, 31, 40).isoformat()
    assert uni_row == live_row
    assert list(uni_row) == list(live_row)  # same key order too
    assert uni.seen(key)
    assert shadow.UniverseJudgmentLog(tmp_path / "uni", DAY).seen(key)


# --- company page fetch -----------------------------------------------------


async def test_fetch_company_page_reports_status_and_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v2/news/companies/005930"
        assert request.url.params["number"] == "1"
        assert request.url.params["size"] == "20"
        return httpx.Response(200, json={"result": {"body": [ITEM], "lastPage": True}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        status, items = await shadow.fetch_company_page(http, "005930")
    assert status == 200 and items == [ITEM]


async def test_fetch_company_page_blocked_and_network_error() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(429))
    ) as http:
        assert await shadow.fetch_company_page(http, "005930") == (429, [])

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as http:
        assert await shadow.fetch_company_page(http, "005930") == (None, [])


# --- no order path ----------------------------------------------------------

FORBIDDEN = {
    "AutoTrader",
    "auto_trader",
    "execute_trade",
    "place_order",
    "cancel_order",
    "enter_batch",
    "rank_batch",
    "news_signal_poller",
    "UnitOfWork",
}


def test_module_source_references_no_order_or_trader_symbol() -> None:
    tree = ast.parse((SCRIPTS / "news_sweep_shadow.py").read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.update(a.name.split("."))
    assert not (names & FORBIDDEN)


def test_importing_module_does_not_load_the_trader_or_live_daemon() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import news_sweep_shadow; "
        "bad = [m for m in sys.modules if m.endswith('auto_trader') or m == 'news_signal_poller']; "
        "print(bad); assert not bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_parser_has_safety_flags() -> None:
    args = shadow.build_parser().parse_args(
        ["--dry-run-http", "--max-requests", "5", "--proxy", "http://p:1"]
    )
    assert args.dry_run_http and not args.with_broker
    assert args.max_requests == 5 and args.proxy == "http://p:1"


def test_broker_is_opt_in() -> None:
    # F6: default is judge-only; the live Kiwoom token is used only with --with-broker.
    parser = shadow.build_parser()
    assert parser.parse_args([]).with_broker is False
    assert parser.parse_args(["--with-broker"]).with_broker is True
    assert "live Kiwoom" in parser.format_help()


def test_refuses_to_sweep_from_trader_ip_without_proxy() -> None:
    parser = shadow.build_parser()
    with pytest.raises(SystemExit, match="--proxy"):
        shadow.check_network_args(parser.parse_args([]))
    shadow.check_network_args(parser.parse_args(["--proxy", "http://p:1"]))
    shadow.check_network_args(parser.parse_args(["--allow-trader-ip"]))
    shadow.check_network_args(parser.parse_args(["--dry-run-http"]))  # no HTTP at all


async def test_main_refuses_before_any_db_or_http_without_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a, **_k):
        raise AssertionError("must refuse before touching DB/HTTP")

    monkeypatch.setattr(shadow, "load_universe_file", boom)
    monkeypatch.setattr(shadow, "prior_day_trade_values", boom)
    args = shadow.build_parser().parse_args(["--universe-file", "x.txt"])
    with pytest.raises(SystemExit, match="--proxy"):
        await shadow.main_async(args)


@pytest.mark.parametrize("status", [403, 429])
def test_block_statuses(status: int) -> None:
    assert status in shadow.BLOCK_STATUSES


# --- runner (fake HTTP, fake Claude, fixed clock) -----------------------------


class FakeClaude:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.categories: list[object] = []

    async def score_article(
        self, *, code: str, category: object = None, full_body: bool = False, **_: object
    ):
        self.calls.append((code, full_body))
        self.categories.append(category)
        return _features()


class NoWait:
    async def wait(self) -> None:
        return None


def _runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, handler, **kw) -> shadow.ShadowRunner:
    monkeypatch.setattr(shadow, "DASHBOARD_JUDGMENT_DIR", tmp_path / "news_judgments")
    monkeypatch.setattr(shadow, "UNIVERSE_JUDGMENT_DIR", tmp_path / "news_judgments_universe")
    monkeypatch.setattr(shadow, "SWEEP_LOG_DIR", tmp_path / "news_sweep")
    monkeypatch.setattr(shadow, "CANDIDATE_DIR", tmp_path / "news_sweep_candidates")
    stocks = {
        c: StockInfo(code=c, name=c, market="KOSPI", sector="x") for c in ["005930", "000660"]
    }
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runner = shadow.ShadowRunner(
        day=DAY,
        schedule=shadow.SweepSchedule(UNIVERSE, DAY),
        stocks=stocks,
        stale_before=dt.datetime(2026, 9, 16, 15, 20, tzinfo=KST),
        http=http,
        claude=kw.pop("claude", FakeClaude()),
        candidate_ctx=None,
        max_requests=kw.pop("max_requests", None),
    )
    runner.limiter = NoWait()  # type: ignore[assignment]
    runner.now = lambda: at(10, 0)
    return runner


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def test_runner_judges_new_fresh_articles_once_and_logs_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = {**ITEM, "id": "old_1", "createdAt": "2026-09-16T10:00:00"}
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if "/companies/" in request.url.path:
            return httpx.Response(200, json={"result": {"body": [ITEM, stale]}})
        return httpx.Response(
            200, json={"result": {"kr": {"id": "yna_AKR1", "companyCodes": ["005930", "999999"]}}}
        )

    claude = FakeClaude()
    runner = _runner(tmp_path, monkeypatch, handler, claude=claude, max_requests=4)
    await runner.run()

    # 3 pages would fit, but the detail call for the one new article used a slot.
    assert requests == [
        "/api/v2/news/companies/005930",
        "/api/v2/news/yna_AKR1",
        "/api/v2/news/companies/000660",
        "/api/v2/news/companies/035420",
    ]
    assert runner.stop_reason == "max_requests 4 reached"
    assert claude.calls == [("005930", False)]  # 999999 unknown; v4 shadow off
    rows = _rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")
    assert len(rows) == 1
    assert rows[0]["key"] == f"{live_article_id('yna_AKR1')}|005930"
    assert rows[0]["pool"] == "universe" and rows[0]["v4_shadow"] is None
    log = _rows(tmp_path / "news_sweep" / f"{DAY}.jsonl")
    assert [r["code"] for r in log] == ["005930", "005930", "000660", "035420"]
    assert log[0]["new_ids"] == ["yna_AKR1"] and log[1]["detail_id"] == "yna_AKR1"
    assert log[2]["new_ids"] == []
    assert not (tmp_path / "news_sweep_candidates").exists()


async def test_dashboard_article_copies_dashboard_pairs_and_judges_missing_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dashboard judged (yna_AKR1, 005930) only; detail says 000660 is related too.
    _write_dashboard(tmp_path / "news_judgments" / f"{DAY}.jsonl", ["yna_AKR1"])
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if "/companies/" in request.url.path:
            return httpx.Response(200, json={"result": {"body": [ITEM]}})
        kr = {"id": "yna_AKR1", "companyCodes": ["005930", "000660"]}
        return httpx.Response(200, json={"result": {"kr": kr}})

    claude = FakeClaude()
    runner = _runner(tmp_path, monkeypatch, handler, claude=claude, max_requests=2)
    await runner.run()

    assert requests == ["/api/v2/news/companies/005930", "/api/v2/news/yna_AKR1"]
    assert claude.calls == [("000660", False)]  # 005930 not re-judged
    assert claude.categories == [""]
    rows = {r["ticker"]: r for r in _rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")}
    log = _rows(tmp_path / "news_sweep" / f"{DAY}.jsonl")
    page_ts = log[0]["ts"]
    aid = live_article_id("yna_AKR1")

    dash = rows["005930"]
    assert dash["key"] == f"{aid}|005930"
    assert dash["pool"] == "dashboard" and dash["sweep_seen_at"] == page_ts
    assert dash["category"] == "breaking"  # dashboard row copied as is
    dashboard_row = _rows(tmp_path / "news_judgments" / f"{DAY}.jsonl")[0]
    assert {k: v for k, v in dash.items() if k not in ("pool", "sweep_seen_at")} == dashboard_row

    uni = rows["000660"]
    assert uni["pool"] == "universe" and uni["category"] == "" and uni["sweep_seen_at"] == page_ts
    assert log[0]["dashboard_ids"] == ["yna_AKR1"] and log[0]["new_ids"] == ["yna_AKR1"]


async def test_dashboard_copy_is_not_duplicated_on_revisit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dashboard(tmp_path / "news_judgments" / f"{DAY}.jsonl", ["yna_AKR1"])

    def handler(request: httpx.Request) -> httpx.Response:
        if "/companies/" in request.url.path:
            return httpx.Response(200, json={"result": {"body": [ITEM]}})
        return httpx.Response(200, json={"result": {"kr": {"companyCodes": ["005930"]}}})

    runner = _runner(tmp_path, monkeypatch, handler, max_requests=2)
    await runner.run()
    runner2 = _runner(tmp_path, monkeypatch, handler, max_requests=2)  # restart
    await runner2.run()
    assert len(_rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")) == 1


async def test_no_detail_call_after_1520(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [at(15, 19, 59)]
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        clock[0] = at(15, 20)  # the page response lands at the cutoff
        return httpx.Response(200, json={"result": {"body": [ITEM]}})

    runner = _runner(tmp_path, monkeypatch, handler)
    runner.now = lambda: clock[0]
    await runner.run()
    assert len(requests) == 1 and "/companies/" in requests[0]  # new article, but no detail
    assert runner.stop_reason == "15:20 KST"


def test_judging_body_mirrors_live_mapping() -> None:
    from daytrade_it.infrastructure.external.toss_news_client import map_toss_article_to_domain

    long = "본문" * 400
    cases = [
        ({"summary": "요약", "contentText": long}, "요약"),
        ({"summary": "", "contentText": long}, long[:500]),
        ({"summary": None}, ITEM["title"]),
    ]
    for extra, expected in cases:
        article = shadow.build_article({**ITEM, **extra}, None)
        assert (article.summary or article.content) == expected

        class Lib:  # krx_news_client NewsArticle stand-in with the same fields
            id = "toss:x"
            title = ITEM["title"]
            content = extra.get("contentText") or ""
            summary = extra.get("summary") or ""
            tickers: list[str] = []
            url = "https://tossinvest.com/news?x=1"
            published_at = at(8, 0)

            class source:  # noqa: N801
                value = "toss"

            class category:  # noqa: N801
                value = "stock"

        live = map_toss_article_to_domain(Lib)  # type: ignore[arg-type]
        assert (live.summary or live.content) == expected


async def test_runner_stops_after_three_consecutive_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(429 if len(calls) != 2 else 403)

    runner = _runner(tmp_path, monkeypatch, handler, max_requests=100)
    await runner.run()
    assert len(calls) == 3
    assert "429/403" in runner.stop_reason


async def test_runner_stops_at_1520(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request after 15:20")

    runner = _runner(tmp_path, monkeypatch, handler)
    runner.now = lambda: at(15, 20)
    await runner.run()
    assert runner.stop_reason == "15:20 KST"


async def test_halt_signal_stops_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner: shadow.ShadowRunner

    def handler(request: httpx.Request) -> httpx.Response:
        runner._halt("signal SIGTERM")
        return httpx.Response(200, json={"result": {"body": []}})

    runner = _runner(tmp_path, monkeypatch, handler)
    await runner.run()
    assert runner.requests == 1 and runner.stop_reason == "signal SIGTERM"


async def test_universe_pairs_judged_with_empty_category_and_detail_completion_seen_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    second = {**ITEM, "id": "yna_AKR2", "title": "SK하이닉스 공급 계약"}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/companies/" in request.url.path:
            return httpx.Response(200, json={"result": {"body": [ITEM, second]}})
        news_id = request.url.path.rsplit("/", 1)[-1]
        # Even if detail carries something category-like, universe pairs use "".
        kr = {"id": news_id, "companyCodes": ["005930"], "category": "HOT", "newsType": "x"}
        return httpx.Response(200, json={"result": {"kr": kr}})

    claude = FakeClaude()
    times = iter(at(10, 0, s) for s in range(60))  # strictly increasing clock
    runner = _runner(tmp_path, monkeypatch, handler, claude=claude, max_requests=3)
    runner.now = lambda: next(times, at(10, 0, 9))
    await runner.run()

    assert claude.categories == ["", ""]
    rows = _rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")
    assert [r["category"] for r in rows] == ["", ""]
    # F3: an article is discovered when its own detail call completes (simulate_sweep).
    log = _rows(tmp_path / "news_sweep" / f"{DAY}.jsonl")
    detail_ts = {r["detail_id"]: r["ts"] for r in log if "detail_id" in r}
    seen = {r["article_id"]: r["sweep_seen_at"] for r in rows}
    for nid in ("yna_AKR1", "yna_AKR2"):
        assert seen[live_article_id(nid)] > detail_ts[nid] > log[0]["ts"]
    assert seen[live_article_id("yna_AKR2")] > seen[live_article_id("yna_AKR1")]


async def test_judge_keeps_only_related_codes_in_todays_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # F1 (S13): 000660 is a known stock and in the detail's companyCodes, but not in the universe.
    def handler(request: httpx.Request) -> httpx.Response:
        if "/companies/" in request.url.path:
            return httpx.Response(200, json={"result": {"body": [ITEM]}})
        kr = {"id": "yna_AKR1", "companyCodes": ["005930", "000660"]}
        return httpx.Response(200, json={"result": {"kr": kr}})

    claude = FakeClaude()
    runner = _runner(tmp_path, monkeypatch, handler, claude=claude, max_requests=2)
    runner.schedule = shadow.SweepSchedule(["005930", "035420"], DAY)
    await runner.run()
    assert claude.calls == [("005930", False)]
    rows = _rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")
    assert [r["ticker"] for r in rows] == ["005930"]


# --- F3/F4: live request order == simulate_sweep ------------------------------


async def test_live_request_sequence_equals_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe = ["005930", "000660", "035420"]
    stale_before = dt.datetime(2026, 9, 16, 15, 20, tzinfo=KST)
    arrivals = {
        # 23 fresh overnight + 2 stale on 005930: page 1 full of new -> page 2
        "005930": [(f"s{i:02d}", at(7, 0, i)) for i in range(23)]
        + [(f"old{i}", dt.datetime(2026, 9, 16, 10, 0, i, tzinfo=KST)) for i in range(2)],
        # shared with 005930 (seen already: free) + one new
        "000660": [("s05", at(7, 0, 5)), ("h1", at(7, 30))],
        # published mid-sweep: visible only from 08:00:40
        "035420": [("n1", at(8, 0, 40))],
    }
    related = {"s05": ["005930", "000660"]}
    listings = {c: sorted((t, n) for n, t in rows) for c, rows in arrivals.items()}
    clock = [at(8, 0)]
    live: list[tuple] = []
    detail_done: dict[str, dt.datetime] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/companies/" in path:
            code = path.rsplit("/", 1)[-1]
            page = int(request.url.params["number"])
            assert int(request.url.params["size"]) == PAGE_SIZE
            live.append(("page", code, page, None))
            items = page_items(listings.get(code, []), clock[0], page, PAGE_SIZE)
            body = [
                {
                    "id": n,
                    "title": f"t {n}",
                    "summary": "s",
                    "createdAt": t.replace(tzinfo=None).isoformat(),
                }
                for n, t in items
            ]
            clock[0] += dt.timedelta(seconds=1)  # response lands one interval later
            return httpx.Response(200, json={"result": {"body": body}})
        news_id = path.rsplit("/", 1)[-1]
        live.append(("detail", None, None, news_id))
        clock[0] += dt.timedelta(seconds=1)
        detail_done[news_id] = clock[0]
        codes = related.get(news_id, ["005930"])
        return httpx.Response(200, json={"result": {"kr": {"id": news_id, "companyCodes": codes}}})

    n_requests = 70
    runner = _runner(tmp_path, monkeypatch, handler, max_requests=n_requests)
    runner.schedule = shadow.SweepSchedule(universe, DAY)
    runner.cursor = shadow.SweepCursor(universe, stale_before, PAGE_SIZE)
    # NoWait limiter: each response lands 1 s after its request, so request k starts at 08:00 + k s
    runner.now = lambda: clock[0]
    await runner.run()

    trace: list[SweepRequest] = []
    found = simulate_sweep(
        universe,
        DAY,
        arrivals,
        stale_before,
        end=(at(8, 0) + dt.timedelta(seconds=n_requests)).time(),
        trace=trace,
    )
    expected = [
        (r.kind, r.code, r.page, None) if r.kind == "page" else ("detail", None, None, r.news_id)
        for r in trace
    ]
    assert live == expected
    assert ("page", "005930", 2, None) in live  # F4: paged past page 1
    assert {"h1", "n1", "s22", "s00"} <= set(found)
    assert detail_done == found  # same discovery times
    rows = _rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")
    seen = {r["article_id"]: r["sweep_seen_at"] for r in rows}
    assert seen[live_article_id("h1")] == found["h1"].isoformat()


# --- A-plan fix: judge_queue models the single judge worker -----------------


async def test_judged_order_matches_judge_queue_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner has exactly one `judge_worker` task consuming one `asyncio.Queue`
    (`ShadowRunner.queue`/`judge_worker` in news_sweep_shadow.py): articles are
    dequeued FIFO in discovery order, and `_judge` judges every one of an
    article's known pairs, one at a time, before the worker asks for the next
    article -- exactly the batching `judge_queue` (application/news/universe.py)
    assumes ("an article's pairs are all enqueued together ... and judged in
    that order"). `companyCodes` arrives in Toss API order (005930 before
    000660 here), but `build_article` sorts tickers ascending by code, the
    same order C3's `related_codes` uses (`tuple(sorted(codes))`) -- without
    that, C3 and the live runner would judge_queue two different orderings of
    the same article's pairs. This asserts the resulting per-pair order *and*
    that it lines up index-for-index with `judge_queue`'s per-pair finish
    times.

    It does not assert the archived `judged_at` timestamps equal those finish
    times: `judged_at` comes from `dt.datetime.now(KST)` inside
    `judgment_record` (application/news/judging.py), a real wall-clock call
    the runner's injectable `self.now` does not reach (`self.now` only drives
    scheduling/HTTP timing here), so a fixed virtual per-pair delay can't be
    observed in the archive without changing judging.py, which is out of
    scope for this fix.
    """
    # Two articles, both first seen on 005930's page (page order fixes discovery
    # order), each related to both universe codes -> 2 pairs apiece, judged
    # one article fully before the next (judge_queue's batching).
    first = {**ITEM, "id": "burst_1"}
    second = {**ITEM, "id": "burst_2", "createdAt": "2026-09-17T08:31:03"}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/companies/" in request.url.path:
            return httpx.Response(200, json={"result": {"body": [first, second]}})
        news_id = request.url.path.rsplit("/", 1)[-1]
        # API order is 005930 first; build_article must still judge 000660 first.
        kr = {"id": news_id, "companyCodes": ["005930", "000660"]}
        return httpx.Response(200, json={"result": {"kr": kr}})

    claude = FakeClaude()
    runner = _runner(tmp_path, monkeypatch, handler, claude=claude, max_requests=4)
    await runner.run()

    rows = _rows(tmp_path / "news_judgments_universe" / f"{DAY}.jsonl")
    got = [(r["article_id"], r["ticker"]) for r in rows]
    aid1, aid2 = live_article_id("burst_1"), live_article_id("burst_2")
    # burst_1's two pairs are contiguous (discovered/queued first) and code-sorted
    # (000660 before 005930, not API order), then burst_2's, same way.
    assert got == [(aid1, "000660"), (aid1, "005930"), (aid2, "000660"), (aid2, "005930")]

    # judge_queue, given the same discovery order and pair counts, agrees on
    # both the article order and each article's per-pair finish times; zipped
    # against `got`'s per-article pair sequence, pair k of an article (its
    # k-th code ascending) gets judge_queue's k-th finish time for it.
    per_pair = dt.timedelta(seconds=3)
    discovered = {aid1: at(10, 0), aid2: at(10, 0, 1)}
    finishes = judge_queue([(aid1, discovered[aid1], 2), (aid2, discovered[aid2], 2)], per_pair)
    assert list(finishes) == [aid1, aid2]  # article order unchanged
    assert finishes[aid1] == [at(10, 0) + per_pair, at(10, 0) + 2 * per_pair]
    # aid2 queues behind aid1 (discovered before aid1's worker is free) -> its
    # finishes carry aid1's backlog forward instead of starting from its own
    # discovery time.
    assert finishes[aid2] == [at(10, 0) + 3 * per_pair, at(10, 0) + 4 * per_pair]

    by_article: dict[str, list[str]] = {}
    for aid, ticker in got:
        by_article.setdefault(aid, []).append(ticker)
    for aid in (aid1, aid2):
        assert by_article[aid] == sorted(by_article[aid]) == ["000660", "005930"]
        assert len(by_article[aid]) == len(finishes[aid]) == 2
