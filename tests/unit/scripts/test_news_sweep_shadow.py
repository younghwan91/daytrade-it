"""Universe sweep shadow runner (plan C5): schedule, dedupe, stale, block stop, archive format.

No network, no DB -- everything below runs on pure helpers or fakes.
"""

from __future__ import annotations

import ast
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
from daytrade_it.application.news.universe import live_article_id, next_visit  # noqa: E402
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

    def test_visit_order_repeats_universe_and_matches_next_visit(self) -> None:
        sched = shadow.SweepSchedule(UNIVERSE, DAY)
        codes = [sched.code_at(m) for m in range(7)]
        assert codes == UNIVERSE * 2 + ["005930"]
        for m in range(7):
            planned = sched.planned_at(m)
            assert planned == at(8, 0, m)
            assert next_visit(sched.code_at(m), UNIVERSE, DAY, planned) == planned

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


def _write_dashboard(path: Path, news_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for nid in news_ids:
            aid = live_article_id(nid)
            f.write(json.dumps({"key": f"{aid}|005930", "article_id": aid}) + "\n")


class TestDedupe:
    def test_new_ids_dedupes_across_codes(self, tmp_path: Path) -> None:
        state = shadow.SweepState(shadow.DashboardKeys(tmp_path / "none.jsonl"))
        new1, dash1 = state.new_items([{"id": "a"}, {"id": "b"}])
        new2, dash2 = state.new_items([{"id": "b"}, {"id": "c"}, {"id": "c"}])
        assert [i["id"] for i in new1] == ["a", "b"]
        assert [i["id"] for i in new2] == ["c"]
        assert dash1 == dash2 == []

    def test_dashboard_archive_keys_are_skipped_and_reloaded(self, tmp_path: Path) -> None:
        dash_path = tmp_path / "news_judgments" / f"{DAY}.jsonl"
        _write_dashboard(dash_path, ["yna_1"])
        state = shadow.SweepState(shadow.DashboardKeys(dash_path))
        new, dash = state.new_items([{"id": "yna_1"}, {"id": "yna_2"}])
        assert [i["id"] for i in new] == ["yna_2"]
        assert dash == ["yna_1"]

        # The live daemon keeps appending during the day.
        _write_dashboard(dash_path, ["yna_3"])
        new, dash = state.new_items([{"id": "yna_3"}])
        assert new == [] and dash == ["yna_3"]

    def test_items_without_id_are_ignored(self, tmp_path: Path) -> None:
        state = shadow.SweepState(shadow.DashboardKeys(tmp_path / "x.jsonl"))
        assert state.new_items([{"title": "no id"}]) == ([], [])


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
    article = shadow.build_article(ITEM, {"companyCodes": ["005930", "000660"]})
    assert str(article.id) == live_article_id("yna_AKR1")
    assert [t.value for t in article.tickers] == ["005930", "000660"]
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
        ["--dry-run-http", "--max-requests", "5", "--no-broker", "--proxy", "http://p:1"]
    )
    assert args.dry_run_http and args.no_broker
    assert args.max_requests == 5 and args.proxy == "http://p:1"


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


async def test_runner_skips_articles_the_dashboard_already_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dashboard(tmp_path / "news_judgments" / f"{DAY}.jsonl", ["yna_AKR1"])

    def handler(request: httpx.Request) -> httpx.Response:
        assert "/companies/" in request.url.path, "no detail call for a dashboard article"
        return httpx.Response(200, json={"result": {"body": [ITEM]}})

    claude = FakeClaude()
    runner = _runner(tmp_path, monkeypatch, handler, claude=claude, max_requests=2)
    await runner.run()
    assert claude.calls == []
    log = _rows(tmp_path / "news_sweep" / f"{DAY}.jsonl")
    assert log[0]["dashboard_ids"] == ["yna_AKR1"] and log[0]["new_ids"] == []


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


async def test_universe_pairs_judged_with_empty_category_and_shared_visit_seen_at(
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
    # Both articles came from one visit: they share that visit's timestamp.
    page_ts = _rows(tmp_path / "news_sweep" / f"{DAY}.jsonl")[0]["ts"]
    assert [r["sweep_seen_at"] for r in rows] == [page_ts, page_ts]
