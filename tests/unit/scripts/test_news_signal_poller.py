"""JudgmentLog: every judged pair recorded once per day, restart-safe."""

import datetime as dt
import sys
from pathlib import Path

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
