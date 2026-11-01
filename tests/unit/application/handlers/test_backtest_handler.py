"""BacktestHandler over krx-quant-core run_replay (Bar events, through-close fills).

Fill model (core PaperBroker, conservative): a signal decided on day t's bar
becomes a limit order at day t's close; it fills at that limit only when a
LATER bar's close trades through it (buy: close < limit, sell: close > limit).
Costs (commission + 증권거래세) come from core PositionBook with the handler's
cost model.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.events import Fill

from daytrade_it.application.handlers.backtest import BacktestCommand, BacktestHandler
from daytrade_it.domain.services.cost_model import CostModelConfig, KoreanCostModel
from daytrade_it.infrastructure.brokers.base import KST
from daytrade_it.infrastructure.external.quant_airflow_db_client import AdjustedDailyBar


def _bar(day: int, close: float, volume: int = 0) -> AdjustedDailyBar:
    return AdjustedDailyBar(
        code="005930",
        date=dt.date(2026, 9, day),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        source="test",
    )


class _FakeMarketData:
    def __init__(self, bars: list[AdjustedDailyBar]) -> None:
        self._bars = bars

    def get_adjusted_daily_bars(self, ticker, *, start, end):  # noqa: ANN001, ANN201
        return self._bars


def _command() -> BacktestCommand:
    return BacktestCommand(
        tickers=["005930"],
        start_date=dt.datetime(2026, 9, 1, tzinfo=dt.UTC),
        end_date=dt.datetime(2026, 9, 30, tzinfo=dt.UTC),
        initial_capital=Decimal("10000000"),
        strategy="news_momentum",
    )


def _expected_pnl(
    qty: int, buy: int, sell: int, sell_day: int, cost: KoreanCostModel | None = None
) -> float:
    book = PositionBook(cost_model=cost or KoreanCostModel())
    ts = dt.datetime(2026, 9, 1, 15, 30, tzinfo=KST)
    book.apply(Fill(ord_no="b", code="005930", side="buy", qty=qty, price=buy, ts=ts))
    sell_ts = dt.datetime(2026, 9, sell_day, 15, 30, tzinfo=KST)
    return book.apply(Fill(ord_no="s", code="005930", side="sell", qty=qty, price=sell, ts=sell_ts))


async def test_round_trip_fills_through_the_next_close_at_the_limit() -> None:
    bars = [
        _bar(1, 10_000, volume=2_000_000),  # high volume -> BUY, limit 10,000
        _bar(2, 9_900),  # closes through 10,000 -> bought at 10,000
        _bar(3, 10_600),  # +6% -> SELL, limit 10,600
        _bar(4, 10_700),  # closes through 10,600 -> sold at 10,600
    ]
    handler = BacktestHandler(market_data_client=_FakeMarketData(bars))  # type: ignore[arg-type]

    result = await handler.handle(_command())

    [trade] = result.trades
    qty = int(10_000_000 * 0.1 / 10_000)
    assert trade.entry_price == Decimal("10000")
    assert trade.exit_price == Decimal("10600")
    assert trade.quantity == Decimal(qty)
    assert trade.entry_date == dt.datetime(2026, 9, 2, 15, 30, tzinfo=KST)
    assert trade.exit_date == dt.datetime(2026, 9, 4, 15, 30, tzinfo=KST)
    assert float(trade.pnl) == round(_expected_pnl(qty, 10_000, 10_600, 4), 2)
    assert result.metrics.total_trades == 1
    assert result.metrics.winning_trades == 1
    assert [p["date"][:10] for p in result.equity_curve] == [
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
        "2026-09-04",
    ]
    assert result.final_capital == Decimal(str(result.equity_curve[-1]["equity"]))


async def test_limit_that_is_never_traded_through_does_not_fill() -> None:
    bars = [
        _bar(1, 10_000, volume=2_000_000),
        _bar(2, 10_000),  # touch, not through -> no fill
        _bar(3, 10_100),
    ]
    handler = BacktestHandler(market_data_client=_FakeMarketData(bars))  # type: ignore[arg-type]

    result = await handler.handle(_command())

    assert result.trades == []
    assert result.metrics.total_trades == 0
    assert result.final_capital == Decimal("10000000")


async def test_open_position_is_liquidated_at_the_last_close_with_costs() -> None:
    bars = [
        _bar(1, 10_000, volume=2_000_000),
        _bar(2, 9_900),  # bought at 10,000
        _bar(3, 10_100),  # +1%: no exit signal
    ]
    handler = BacktestHandler(market_data_client=_FakeMarketData(bars))  # type: ignore[arg-type]

    result = await handler.handle(_command())

    [trade] = result.trades
    qty = int(10_000_000 * 0.1 / 10_000)
    assert trade.exit_price == Decimal("10100")
    assert float(trade.pnl) == round(_expected_pnl(qty, 10_000, 10_100, 3), 2)
    gross = (trade.exit_price - trade.entry_price) * trade.quantity
    assert trade.pnl != gross


def _costly() -> KoreanCostModel:
    # 수수료 1% — 코어 기본(0.015%)과 크게 달라 어느 비용 모델로 셌는지 수량·자산에서 드러난다.
    return KoreanCostModel(CostModelConfig(commission_rate=Decimal("0.01")))


async def test_equity_and_sizing_use_the_handlers_cost_model() -> None:
    # 수량을 정하는 현금·자산곡선·거래 손익이 같은 비용 모델(핸들러의 것)이어야 한다. 전엔 run_replay
    # 안쪽 장부(코어 기본 비용)가 현금·자산곡선을 정하고 거래만 핸들러 비용 모델로 다시 적었다.
    bars = [
        _bar(1, 10_000, volume=2_000_000),  # BUY @10,000
        _bar(2, 9_900),  # bought
        _bar(3, 10_600),  # SELL @10,600
        _bar(4, 10_700),  # sold
        _bar(5, 10_050, volume=2_000_000),  # BUY again — qty from post-cost cash
        _bar(6, 10_000),  # bought @10,050
    ]
    cost = _costly()
    handler = BacktestHandler(market_data_client=_FakeMarketData(bars), cost_model=cost)  # type: ignore[arg-type]

    result = await handler.handle(_command())

    first, second = result.trades
    day4 = next(p for p in result.equity_curve if p["date"].startswith("2026-09-04"))
    assert Decimal(str(round(day4["equity"], 2))) == Decimal("10000000") + first.pnl
    cash = 10_000_000 + _expected_pnl(100, 10_000, 10_600, 4, cost)
    assert int(cash * 0.1 / 10_050) == 99  # 코어 기본 비용이면 100 이 된다
    assert second.quantity == Decimal(99)


async def test_final_capital_includes_end_liquidation_costs() -> None:
    bars = [
        _bar(1, 10_000, volume=2_000_000),
        _bar(2, 9_900),  # bought at 10,000, still open at the end
        _bar(3, 10_100),
    ]
    handler = BacktestHandler(market_data_client=_FakeMarketData(bars), cost_model=_costly())  # type: ignore[arg-type]

    result = await handler.handle(_command())

    [trade] = result.trades
    assert result.final_capital == Decimal(str(10_000_000 + float(trade.pnl)))
    last = result.equity_curve[-1]
    assert last["cash"] == last["equity"]
    assert last["drawdown"] > 0  # 청산 비용만큼 고점(초기자본) 아래
