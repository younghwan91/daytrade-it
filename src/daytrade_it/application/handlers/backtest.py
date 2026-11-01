"""
Backtest Handler.

Handles backtesting of trading strategies against historical data.

As of Task 13 (Phase 5), this handler:
  - Sources historical daily bars from Task 6's read-only
    ``QuantAirflowDBClient`` (``daily_bars_adjusted``, split-adjusted KRW
    OHLCV) instead of the previously-wired-but-dead ``PolygonClient``
    (flagged as never actually connected in Task 8's report).
  - Replays the bars through krx-quant-core's ``run_replay`` (2026-09-16):
    orders go through core ``OrderManager``/``PaperBroker`` with conservative
    through-close limit fills, and every closed trade is costed by core
    ``PositionBook`` with the shared ``KoreanCostModel`` (증권거래세 +
    위탁수수료) -- see ``_simulate``.

``QuantAirflowDBClient``'s query methods are blocking (``psycopg2``, no
async driver) -- each per-ticker fetch is run via ``asyncio.to_thread`` so
this handler's ``handle()`` stays a well-behaved coroutine without
requiring an async Postgres driver just for a LAN batch/EOD query.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
import structlog
from krx_quant_core.backtest.ledger import performance_metrics
from krx_quant_core.backtest.replay import run_replay
from krx_quant_core.execution.book import PositionBook
from krx_quant_core.execution.engine import Bar, Event, StrategyContext
from krx_quant_core.execution.guards import OrderGuardConfig
from krx_quant_core.execution.orders import OrderIntent
from krx_quant_core.market.session import KST

from daytrade_it.domain.entities.stock import Exchange
from daytrade_it.domain.services.cost_model import KoreanCostModel
from daytrade_it.domain.value_objects import Ticker

if TYPE_CHECKING:
    from daytrade_it.infrastructure.external.quant_airflow_db_client import (
        AdjustedDailyBar,
        QuantAirflowDBClient,
    )

logger = structlog.get_logger(__name__)

#: Daily bars are known at the close; core ``Bar.ts`` is when a bar closed.
_BAR_CLOSE = time(15, 30)
#: The replay sizes positions at 10% of capital, so the guard's per-order
#: quantity cap (live default 1 share) is lifted; band/count limits are off.
_REPLAY_GUARD = OrderGuardConfig(
    max_qty=10**9,
    max_qty_hard_cap=10**9,
    price_band_pct=0.0,
    max_orders_per_code=0,
    max_orders_total=0,
)


@dataclass
class BacktestCommand:
    """Command to run a backtest."""

    tickers: list[str]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("100000")
    strategy: str = "news_momentum"  # Strategy identifier
    params: dict = field(default_factory=dict)
    correlation_id: str | None = None


@dataclass
class BacktestTrade:
    """A trade executed during backtest."""

    ticker: str
    direction: str
    entry_date: datetime
    entry_price: Decimal
    exit_date: datetime | None = None
    exit_price: Decimal | None = None
    quantity: Decimal = Decimal("0")
    pnl: Decimal = Decimal("0")
    pnl_pct: Decimal = Decimal("0")


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""

    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    avg_trade_duration_hours: Decimal
    best_trade: Decimal
    worst_trade: Decimal


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: list[dict]  # List of {date, equity, drawdown}


class BacktestHandler:
    """
    Handles backtesting of trading strategies.

    Features:
    - Multiple strategy support
    - Transaction cost modeling
    - Position sizing
    - Slippage simulation
    - Performance metrics calculation
    """

    def __init__(
        self,
        market_data_client: QuantAirflowDBClient,
        cost_model: KoreanCostModel | None = None,
        market: Exchange = Exchange.KOSPI,
    ) -> None:
        """Initialize backtest handler.

        Args:
            market_data_client: Task 6's read-only client over
                quant-airflow's TimescaleDB, used for adjusted daily bars.
            cost_model: Shared Korean transaction-cost model (증권거래세 +
                위탁수수료 + slippage). Defaults to
                ``KoreanCostModel()`` (its own default rates) when not
                given -- pass the same instance/config used elsewhere
                (e.g. a ``PaperBroker``'s cost model) to keep backtest and
                paper-trading costs consistent.
            market: Exchange tier for tax/slippage rate selection. This
                handler has no per-ticker KOSPI/KOSDAQ mapping wired (see
                ``cost_model.py``'s module docstring), so one tier applies
                to every ticker in a given backtest run. Defaults to
                KOSPI.
        """
        self._market_data = market_data_client
        self._cost_model = cost_model or KoreanCostModel()
        self._market = market

    async def handle(self, command: BacktestCommand) -> BacktestResult:
        """
        Run a backtest for the given parameters.

        Args:
            command: Backtest configuration.

        Returns:
            BacktestResult with metrics and trade history.
        """
        log = logger.bind(
            tickers=command.tickers,
            start_date=command.start_date.isoformat(),
            end_date=command.end_date.isoformat(),
            strategy=command.strategy,
            correlation_id=command.correlation_id,
        )
        log.info("Starting backtest")

        # Fetch historical data for all tickers from Task 6's read-only
        # quant-airflow TimescaleDB client (split-adjusted daily bars).
        # QuantAirflowDBClient's query methods are blocking (psycopg2), so
        # each fetch runs off the event loop via asyncio.to_thread.
        price_data: dict[str, list[AdjustedDailyBar]] = {}
        for ticker in command.tickers:
            bars = await asyncio.to_thread(
                self._market_data.get_adjusted_daily_bars,
                Ticker(value=ticker),
                start=command.start_date.date(),
                end=command.end_date.date(),
            )
            # Rows with no close price are unusable for signal/trade
            # simulation (e.g. halted-day placeholder rows); drop them.
            price_data[ticker] = [b for b in bars if b.close is not None]
            log.debug(f"Loaded {len(price_data[ticker])} bars for {ticker}")

        # Run backtest simulation
        trades, equity_curve = await self._simulate(
            price_data=price_data,
            initial_capital=command.initial_capital,
            strategy=command.strategy,
            params=command.params,
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=command.initial_capital,
            start_date=command.start_date,
            end_date=command.end_date,
        )

        final_capital = equity_curve[-1]["equity"] if equity_curve else command.initial_capital

        log.info(
            "Backtest complete",
            total_return_pct=float(metrics.total_return_pct),
            sharpe_ratio=float(metrics.sharpe_ratio),
            total_trades=metrics.total_trades,
            win_rate=float(metrics.win_rate),
        )

        return BacktestResult(
            start_date=command.start_date,
            end_date=command.end_date,
            initial_capital=command.initial_capital,
            final_capital=Decimal(str(final_capital)),
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
        )

    async def _simulate(
        self,
        price_data: dict[str, list[AdjustedDailyBar]],
        initial_capital: Decimal,
        strategy: str,
        params: dict,
    ) -> tuple[list[BacktestTrade], list[dict]]:
        """Replay the daily bars through krx-quant-core's ``run_replay``.

        Each daily bar is a core ``Bar`` event stamped at that day's 15:30 KST
        close. ``_generate_signal`` still makes every decision (wrapped in
        ``_SignalStrategy``); orders go through core ``OrderManager`` to its
        ``PaperBroker``, and costs through ``PositionBook`` with this handler's
        cost model -- the same order/fill/book code the live engines use.

        Fill model (changed from the old next-open fill): a signal decided on
        day t becomes a limit order at day t's close and fills at that limit
        only when a later close trades *through* it (buy: close < limit, sell:
        close > limit). An unfilled order is cancelled and re-decided on the
        next bar. Positions still open at the end are liquidated at their last
        close (commission + tax deducted), as before.

        Returns:
            Tuple of (trades, equity_curve).
        """
        bars = [
            Bar(
                ts=datetime.combine(bar.date, _BAR_CLOSE, tzinfo=KST),
                code=str(ticker),
                open=float(bar.open if bar.open is not None else bar.close),
                high=float(bar.high if bar.high is not None else bar.close),
                low=float(bar.low if bar.low is not None else bar.close),
                close=float(bar.close),
                volume=float(bar.volume or 0),
            )
            for ticker, ticker_bars in price_data.items()
            for bar in ticker_bars
        ]
        bars.sort(key=lambda b: (b.ts, b.code))
        runner = _SignalStrategy(self, float(initial_capital), strategy, params)
        replay = await asyncio.to_thread(
            run_replay,
            runner,
            bars,
            guard_config=_REPLAY_GUARD,
            market_of=lambda _code: self._market.value,
            cost_model=self._cost_model,
            liquidate_at_end=True,
        )
        if not replay.fills.empty and (replay.fills["ord_no"] == "END").any():
            runner.close_out(replay.book)

        trades = [
            BacktestTrade(
                ticker=row["code"],
                direction="SELL",
                entry_date=pd.Timestamp(row["entry_ts"]).to_pydatetime(),
                entry_price=Decimal(str(round(row["entry_price"], 2))),
                exit_date=pd.Timestamp(row["exit_ts"]).to_pydatetime(),
                exit_price=Decimal(str(row["exit_price"])),
                quantity=Decimal(str(row["qty"])),
                pnl=Decimal(str(round(row["pnl"], 2))),
                pnl_pct=Decimal(str(round(row["exit_price"] / row["entry_price"] - 1, 4))),
            )
            for row in replay.trades.to_dict("records")
        ]
        return trades, runner.equity_curve

    def _generate_signal(
        self,
        ticker: str,
        prices: dict,
        strategy: str,
        params: dict,
        position: dict | None,
    ) -> dict | None:
        """
        Generate a trading signal based on strategy.

        Simple momentum strategy for demonstration.
        """
        if strategy == "news_momentum":
            # Simple strategy: buy on high volume, sell on low volume
            # In reality, this would use news sentiment
            volume = prices.get("volume", 0)
            avg_volume = params.get("avg_volume", 1000000)

            if position is None:
                # Look for entry
                if volume > avg_volume * 1.5:  # High volume = entry
                    return {"action": "BUY", "confidence": 0.7}
            else:
                # Look for exit
                entry_price = position["entry_price"]
                current_price = prices["close"]
                pnl_pct = (current_price - entry_price) / entry_price

                # Exit on profit target or stop loss
                if pnl_pct > 0.05:  # 5% profit target
                    return {"action": "SELL", "confidence": 0.8}
                elif pnl_pct < -0.02:  # 2% stop loss
                    return {"action": "SELL", "confidence": 0.9}

        return None

    def _calculate_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[dict],
        initial_capital: Decimal,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestMetrics:
        """Calculate backtest performance metrics.

        The math lives in ``krx_quant_core.backtest.ledger.performance_metrics``
        (moved there verbatim, same fields/rounding); this maps its result onto
        ``BacktestMetrics`` so consumers keep the type they already read.
        """
        m = performance_metrics(
            trades,
            equity_curve,
            initial_capital,
            start_date=start_date,
            end_date=end_date,
        )
        return BacktestMetrics(
            **{f.name: getattr(m, f.name) for f in dataclass_fields(BacktestMetrics)}
        )


class _SignalStrategy:
    """Core replay ``Strategy`` that asks ``BacktestHandler._generate_signal``.

    Per bar: cancel this ticker's unfilled order, ask for a signal given the
    current core-book position, and place a limit at the bar's close (BUY:
    10% of cash; SELL: the whole position). Records one equity point per day.
    """

    def __init__(
        self, handler: BacktestHandler, initial_capital: float, strategy: str, params: dict
    ) -> None:
        self._handler = handler
        self._initial = initial_capital
        self._strategy = strategy
        self._params = params
        self._peak = initial_capital
        self._by_date: dict[str, dict] = {}

    @property
    def equity_curve(self) -> list[dict]:
        return list(self._by_date.values())

    def on_event(self, ev: Event, ctx: StrategyContext) -> None:
        assert isinstance(ev, Bar)
        oms = ctx.oms
        for order in oms.broker.open_orders():
            if order.code == ev.code:
                intent = OrderIntent(order.side, order.code, order.qty, order.price)
                oms.cancel(order.ord_no, intent)

        holding = oms.book.position(ev.code)
        prices = {
            "open": ev.open,
            "high": ev.high,
            "low": ev.low,
            "close": ev.close,
            "volume": ev.volume,
        }
        position = {"entry_price": holding.avg_price} if holding is not None else None
        signal = self._handler._generate_signal(
            ticker=ev.code,
            prices=prices,
            strategy=self._strategy,
            params=self._params,
            position=position,
        )
        cash = (
            self._initial
            + oms.book.realized_krw
            - sum(h.qty * h.avg_price for h in oms.book.positions().values())
        )
        price = int(ev.close)
        action = signal["action"] if signal else None
        if action == "BUY" and holding is None and price > 0:
            qty = int(cash * 0.1 / price)
            if qty > 0:
                oms.buy(ev.code, qty, price, ref_price=ev.close)
        elif action == "SELL" and holding is not None:
            oms.sell(ev.code, holding.qty, price)

        self._record_equity(ev, ctx, cash)

    def close_out(self, book: PositionBook) -> None:
        """기간 끝 청산(END) 뒤 장부로 마지막 자산 점을 다시 적는다.

        봉마다 적는 자산은 보유를 종가로 평가해 청산 비용(수수료·세금)이 빠져 있다. 끝에 남은
        보유를 장부에서 청산하고 나면 ``final_capital`` 이 초기자본 + 거래 손익 합과 같아야 한다.
        """
        last_key = next(reversed(self._by_date))
        equity = self._initial + book.realized_krw
        # 덮어쓰는 점의 평가액이 고점이었을 수 있으니 고점은 그 앞 점들로 다시 잡는다.
        earlier = [p["equity"] for k, p in self._by_date.items() if k != last_key]
        self._peak = max([self._initial, *earlier, equity])
        drawdown = (self._peak - equity) / self._peak if self._peak > 0 else 0
        self._by_date[last_key] = {
            **self._by_date[last_key],
            "equity": equity,
            "drawdown": drawdown,
            "cash": equity,
        }

    def _record_equity(self, ev: Bar, ctx: StrategyContext, cash: float) -> None:
        book = ctx.oms.book
        equity = self._initial + book.realized_krw
        for code, h in book.positions().items():
            last = ctx.last.get(code)
            close = last.close if isinstance(last, Bar) else h.avg_price
            equity += h.qty * (close - h.avg_price)
        self._peak = max(self._peak, equity)
        drawdown = (self._peak - equity) / self._peak if self._peak > 0 else 0
        day = datetime.combine(ev.ts.date(), time.min, tzinfo=UTC)
        self._by_date[day.isoformat()] = {
            "date": day.isoformat(),
            "equity": equity,
            "drawdown": drawdown,
            "cash": cash,
        }
