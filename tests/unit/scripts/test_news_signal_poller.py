"""JudgmentLog: every judged pair recorded once per day, restart-safe."""

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from news_signal_poller import JudgmentLog  # noqa: E402


def test_judgment_log_appends_and_dedups_across_restart(tmp_path: Path) -> None:
    day = dt.date(2026, 9, 14)
    log = JudgmentLog(tmp_path, day)
    assert not log.seen("a|005930")
    log.append("a|005930", {"v2": {"score": 0.3}, "v4_shadow": None})
    assert log.seen("a|005930")

    restarted = JudgmentLog(tmp_path, day)
    assert restarted.seen("a|005930")
    assert not restarted.seen("a|000660")
    assert len((tmp_path / "2026-09-14.jsonl").read_text().splitlines()) == 1


async def test_enter_batch_enters_by_surprise_and_survives_a_failure() -> None:
    import datetime as dt

    from news_signal_poller import enter_batch

    from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
    from daytrade_it.domain.value_objects import Confidence, Ticker

    def sig(code: str) -> TradingSignal:
        return TradingSignal(
            ticker=Ticker(value=code),
            direction=SignalDirection.BUY,
            confidence=Confidence(value=0.8),
            source=SignalSource.COMPOSITE,
            expires_at=dt.datetime.now(dt.UTC) + dt.timedelta(hours=1),
        )

    class Trader:
        def __init__(self) -> None:
            self.entered: list[str] = []

        async def enter(self, signal: TradingSignal) -> None:
            self.entered.append(signal.ticker.value)
            if signal.ticker.value == "000002":
                raise RuntimeError("broker down")

    trader = Trader()
    await enter_batch(
        trader, [(sig("000001"), 0, 0.9), (sig("000002"), 2, 0.3), (sig("000003"), 1, 0.4)]
    )  # type: ignore[arg-type]
    assert trader.entered == ["000002", "000003", "000001"]


def test_second_instance_returns_before_touching_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    import news_signal_poller
    from krx_quant_core.execution.oms import InstanceLock

    lock_path = tmp_path / "auto_trader" / "poller.lock"
    monkeypatch.setattr(news_signal_poller, "POLLER_LOCK_PATH", lock_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a second instance must not build clients")

    for name in ("DatabaseManager", "KiwoomAPI", "KiwoomBroker"):
        monkeypatch.setattr(news_signal_poller, name, boom)

    with InstanceLock(lock_path):  # the first (running) poller
        asyncio.run(
            news_signal_poller.run(
                once=True, stop_at=dt.time(15, 20), execute=True, confirm_real=True
            )
        )


def test_paper_and_confirm_real_are_mutually_exclusive() -> None:
    from news_signal_poller import build_parser, parse_args

    with pytest.raises(SystemExit):
        parse_args(build_parser(), ["--execute", "--paper", "--confirm-real"])
    with pytest.raises(SystemExit):
        parse_args(build_parser(), ["--paper"])  # paper trading needs --execute

    args = parse_args(build_parser(), ["--execute", "--paper"])
    assert args.paper and not args.confirm_real
    live = parse_args(build_parser(), ["--execute", "--confirm-real"])
    assert live.confirm_real and not live.paper


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):  # noqa: ANN204
        def log(event: str, **kw: object) -> None:
            self.records.append((level, event, kw))

        return log


def test_lock_path_is_anchored_to_the_repo_root() -> None:
    import news_signal_poller

    repo = Path(news_signal_poller.__file__).resolve().parents[1]
    assert repo / "data" / "auto_trader" / "poller.lock" == news_signal_poller.POLLER_LOCK_PATH


def test_second_instance_logs_error_with_holder_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import os

    import news_signal_poller
    from krx_quant_core.execution.oms import InstanceLock

    lock_path = tmp_path / "poller.lock"
    monkeypatch.setattr(news_signal_poller, "POLLER_LOCK_PATH", lock_path)
    log = _RecordingLogger()
    monkeypatch.setattr(news_signal_poller, "logger", log)

    with InstanceLock(lock_path):
        asyncio.run(
            news_signal_poller.run(
                once=True, stop_at=dt.time(15, 20), execute=True, confirm_real=True
            )
        )

    [(level, _event, kw)] = log.records
    assert level == "error"
    assert kw["holder_pid"] == str(os.getpid())
    assert kw["lock"] == str(lock_path)


def test_live_poller_does_not_import_the_paper_broker() -> None:
    # Only daytrade's adapter is checked: krx_quant_core.execution/__init__
    # re-exports PaperBroker, so importing InstanceLock loads core's paper
    # module regardless (import only -- nothing is constructed).
    import subprocess

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    code = (
        f"import sys; sys.path.insert(0, {str(scripts)!r}); import news_signal_poller; "
        "bad = [m for m in sys.modules if m.endswith('core_paper_broker')]; "
        "print(bad); sys.exit(bool(bad))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


async def test_paper_pair_skips_the_signal_db_and_dedups_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import news_signal_poller
    from news_signal_poller import process_pair

    from daytrade_it.domain.entities.news import NewsArticle
    from daytrade_it.domain.entities.signal import SignalDirection, SignalSource, TradingSignal
    from daytrade_it.domain.services.signal_generator import SignalConfig, SignalGenerator
    from daytrade_it.domain.value_objects import Confidence, Ticker
    from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures
    from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

    def no_db(*args: object, **kwargs: object) -> None:
        raise AssertionError("paper mode must not touch the signal DB")

    monkeypatch.setattr(news_signal_poller, "UnitOfWork", no_db)
    ticker = Ticker(value="005930")
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)

    def entry(tk, broker, db_client, expires_at):  # noqa: ANN001, ANN202
        return TradingSignal(
            ticker=tk,
            direction=SignalDirection.BUY,
            confidence=Confidence(value=0.6),
            source=SignalSource.TECHNICAL,
            expires_at=expires_at,
            metadata={"pre_entry_move": 0.01},
        )

    monkeypatch.setattr(news_signal_poller, "generate_no_chase_confirmation", entry)

    class Claude:
        calls = 0

        async def score_article(self, **kw: object) -> ArticleFeatures:
            Claude.calls += 1
            return ArticleFeatures(
                subject="main",
                reports_price_move=False,
                material_type="contract",
                persistence="structural",
                specificity=3,
                freshness="new",
                surprise=2,
                sentiment_direction=1,
                sentiment_strength=3,
                scale_vs_size="large",
            )

    article = NewsArticle(
        title="삼성전자, 5조원 규모 공급계약 체결",
        content="본문",
        source="toss",
        url="https://example.com/a",
        published_at=dt.datetime.now(dt.UTC),
        tickers=[ticker],
    )
    stock = StockInfo(code="005930", name="삼성전자", market="KOSPI")
    paper_seen: set[str] = set()
    pending: list = []
    kw = dict(  # noqa: C408 -- keyword names mirror process_pair's signature
        claude=Claude(),
        broker=object(),
        db_client=object(),
        gen=SignalGenerator(SignalConfig(min_confidence=0.0)),
        db=object(),
        expires_at=expires,
        trader=object(),
        full_body_text=None,
        pending=pending,
        paper_seen=paper_seen,
    )

    await process_pair(
        article, ticker, stock, judgments=JudgmentLog(tmp_path, dt.date.today()), **kw
    )
    assert len(pending) == 1
    assert paper_seen == {f"{article.id}|005930"}

    # A restart that lost the judgment log still doesn't re-enter the pair.
    await process_pair(
        article, ticker, stock, judgments=JudgmentLog(tmp_path / "x", dt.date.today()), **kw
    )
    assert len(pending) == 1


@pytest.mark.parametrize("paper", [False, True])
async def test_scoring_failure_stop_file_is_live_only_in_live_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paper: bool
) -> None:
    """e02949e's consecutive-failure STOP: live writes data/live_STOP as before;
    paper writes data/paper_STOP and never the live kill file."""
    from uuid import uuid4

    import news_signal_poller

    from daytrade_it.domain.entities.news import NewsArticle
    from daytrade_it.domain.value_objects import Ticker
    from daytrade_it.infrastructure.external.quant_airflow_db_client import StockInfo

    live_stop = tmp_path / "live_STOP"
    paper_stop = tmp_path / "paper_STOP"
    monkeypatch.setattr(news_signal_poller, "KILL_SWITCH_PATH", live_stop)
    monkeypatch.setattr(news_signal_poller, "PAPER_KILL_SWITCH_PATH", paper_stop)
    monkeypatch.setattr(news_signal_poller, "_scoring_failures", 0)

    class _Rows:
        def first(self) -> None:
            return None

    class _Session:
        async def execute(self, *_: object) -> _Rows:
            return _Rows()

    class _Uow:
        session = _Session()

        def __init__(self, db: object) -> None: ...

        async def __aenter__(self) -> "_Uow":
            return self

        async def __aexit__(self, *_: object) -> None: ...

    monkeypatch.setattr(news_signal_poller, "UnitOfWork", _Uow)

    class _Boom:
        async def score_article(self, **_: object) -> None:
            raise RuntimeError("credit balance is too low")

    ticker = Ticker(value="005930")
    judgments = JudgmentLog(tmp_path / "j", dt.date(2026, 9, 17))
    for _ in range(news_signal_poller.SCORING_FAILURE_LIMIT):
        article = NewsArticle(
            id=uuid4(),
            title="삼성전자 공급계약",
            content="본문",
            source="toss",
            url="https://example.invalid/1",
            published_at=dt.datetime.now(dt.UTC),
            tickers=[ticker],
        )
        await news_signal_poller.process_pair(
            article,
            ticker,
            StockInfo(code="005930", name="삼성전자"),
            claude=_Boom(),
            broker=object(),
            db_client=object(),
            gen=object(),
            db=object(),
            expires_at=dt.datetime.now(dt.UTC),
            trader=None,
            judgments=judgments,
            full_body_text=None,
            paper_seen=set() if paper else None,
        )

    assert live_stop.exists() is (not paper)
    assert paper_stop.exists() is paper


def test_paper_lock_is_separate_and_repo_anchored() -> None:
    import news_signal_poller

    repo = Path(news_signal_poller.__file__).resolve().parents[1]
    paper_lock = news_signal_poller.PAPER_POLLER_LOCK_PATH
    live_lock = news_signal_poller.POLLER_LOCK_PATH
    assert paper_lock == repo / "data" / "auto_trader_paper" / "poller.lock"
    assert live_lock == repo / "data" / "auto_trader" / "poller.lock"


@pytest.mark.parametrize("held, paper", [("live", True), ("paper", False)])
def test_live_and_paper_locks_do_not_block_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: str, paper: bool
) -> None:
    """A paper run never idles the live daemon, and vice versa: holding one
    mode's lock, the other mode gets past the lock (to building its clients)."""
    import asyncio

    import news_signal_poller
    from krx_quant_core.execution.oms import InstanceLock

    live_lock = tmp_path / "auto_trader" / "poller.lock"
    paper_lock = tmp_path / "auto_trader_paper" / "poller.lock"
    monkeypatch.setattr(news_signal_poller, "POLLER_LOCK_PATH", live_lock)
    monkeypatch.setattr(news_signal_poller, "PAPER_POLLER_LOCK_PATH", paper_lock)

    class _PastTheLock(Exception):
        pass

    def reached(*args: object, **kwargs: object) -> None:
        raise _PastTheLock

    monkeypatch.setenv("GPTQ_DATABASE_URL", "postgresql+asyncpg://unused/unused")
    monkeypatch.setattr(news_signal_poller, "DatabaseManager", reached)

    with InstanceLock(live_lock if held == "live" else paper_lock):
        with pytest.raises(_PastTheLock):
            asyncio.run(
                news_signal_poller.run(
                    once=True,
                    stop_at=dt.time(15, 20),
                    execute=True,
                    confirm_real=not paper,
                    paper=paper,
                )
            )
        # ...and the same mode is still refused while its own lock is held.
        asyncio.run(
            news_signal_poller.run(
                once=True,
                stop_at=dt.time(15, 20),
                execute=True,
                confirm_real=held == "live",
                paper=held == "paper",
            )
        )
