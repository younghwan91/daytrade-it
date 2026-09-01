"""
Backtest Handler.

Handles backtesting of trading strategies against historical data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from gpt_quant_v2.infrastructure.external.polygon_client import PolygonClient

logger = structlog.get_logger(__name__)


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
        polygon_client: PolygonClient,
        transaction_cost_pct: Decimal = Decimal("0.001"),  # 0.1% per trade
        slippage_pct: Decimal = Decimal("0.0005"),  # 0.05% slippage
    ) -> None:
        """Initialize backtest handler."""
        self._polygon = polygon_client
        self._transaction_cost_pct = transaction_cost_pct
        self._slippage_pct = slippage_pct

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

        # Fetch historical data for all tickers
        price_data = {}
        for ticker in command.tickers:
            bars = await self._polygon.get_bars(
                ticker=ticker,
                timespan="day",
                start_date=command.start_date,
                end_date=command.end_date,
            )
            price_data[ticker] = bars
            log.debug(f"Loaded {len(bars)} bars for {ticker}")

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
        price_data: dict[str, list],
        initial_capital: Decimal,
        strategy: str,
        params: dict,
    ) -> tuple[list[BacktestTrade], list[dict]]:
        """
        Run the backtest simulation.

        Returns:
            Tuple of (trades, equity_curve).
        """
        trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []

        capital = float(initial_capital)
        positions: dict[str, dict] = {}  # ticker -> {quantity, entry_price, entry_date}

        # Get all dates from price data
        all_dates = set()
        for bars in price_data.values():
            for bar in bars:
                if isinstance(bar.get("timestamp"), str):
                    all_dates.add(bar["timestamp"][:10])
                elif "date" in bar:
                    all_dates.add(str(bar["date"])[:10])

        sorted_dates = sorted(all_dates)

        for date_str in sorted_dates:
            date = datetime.fromisoformat(date_str).replace(tzinfo=UTC)

            # Get prices for this date
            day_prices = {}
            for ticker, bars in price_data.items():
                for bar in bars:
                    bar_date = bar.get("timestamp", bar.get("date", ""))[:10]
                    if bar_date == date_str:
                        day_prices[ticker] = {
                            "open": float(bar.get("open", bar.get("o", 0))),
                            "high": float(bar.get("high", bar.get("h", 0))),
                            "low": float(bar.get("low", bar.get("l", 0))),
                            "close": float(bar.get("close", bar.get("c", 0))),
                            "volume": float(bar.get("volume", bar.get("v", 0))),
                        }
                        break

            if not day_prices:
                continue

            # Generate signals for each ticker
            for ticker, prices in day_prices.items():
                signal = self._generate_signal(
                    ticker=ticker,
                    prices=prices,
                    strategy=strategy,
                    params=params,
                    position=positions.get(ticker),
                )

                if signal and signal["action"]:
                    trade = self._execute_backtest_trade(
                        ticker=ticker,
                        signal=signal,
                        prices=prices,
                        date=date,
                        capital=capital,
                        positions=positions,
                    )

                    if trade and trade.pnl != 0:  # Closing trade
                        capital += float(trade.pnl)
                        trades.append(trade)

            # Calculate portfolio value
            portfolio_value = capital
            for ticker, pos in positions.items():
                if ticker in day_prices:
                    portfolio_value += pos["quantity"] * day_prices[ticker]["close"]

            # Calculate drawdown
            peak = max([e["equity"] for e in equity_curve] + [float(initial_capital)])
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0

            equity_curve.append(
                {
                    "date": date.isoformat(),
                    "equity": portfolio_value,
                    "drawdown": drawdown,
                    "cash": capital,
                }
            )

        # Close any remaining positions
        for ticker, pos in list(positions.items()):
            if price_data.get(ticker):
                last_bar = price_data[ticker][-1]
                close_price = float(last_bar.get("close", last_bar.get("c", pos["entry_price"])))
                close_date = datetime.now(UTC)

                pnl = (close_price - pos["entry_price"]) * pos["quantity"]
                pnl_pct = (close_price - pos["entry_price"]) / pos["entry_price"]

                trades.append(
                    BacktestTrade(
                        ticker=ticker,
                        direction="BUY",
                        entry_date=pos["entry_date"],
                        entry_price=Decimal(str(pos["entry_price"])),
                        exit_date=close_date,
                        exit_price=Decimal(str(close_price)),
                        quantity=Decimal(str(pos["quantity"])),
                        pnl=Decimal(str(round(pnl, 2))),
                        pnl_pct=Decimal(str(round(pnl_pct, 4))),
                    )
                )

        return trades, equity_curve

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

    def _execute_backtest_trade(
        self,
        ticker: str,
        signal: dict,
        prices: dict,
        date: datetime,
        capital: float,
        positions: dict,
    ) -> BacktestTrade | None:
        """Execute a trade in the backtest simulation."""
        action = signal["action"]

        if action == "BUY" and ticker not in positions:
            # Calculate position size (10% of capital)
            position_value = capital * 0.1
            price = prices["close"] * (1 + float(self._slippage_pct))  # Slippage
            quantity = int(position_value / price)

            if quantity > 0:
                # Apply transaction cost
                cost = position_value * float(self._transaction_cost_pct)

                positions[ticker] = {
                    "quantity": quantity,
                    "entry_price": price,
                    "entry_date": date,
                }

                return BacktestTrade(
                    ticker=ticker,
                    direction="BUY",
                    entry_date=date,
                    entry_price=Decimal(str(round(price, 2))),
                    quantity=Decimal(str(quantity)),
                )

        elif action == "SELL" and ticker in positions:
            pos = positions.pop(ticker)
            price = prices["close"] * (1 - float(self._slippage_pct))  # Slippage

            # Calculate P&L
            gross_pnl = (price - pos["entry_price"]) * pos["quantity"]
            cost = abs(gross_pnl) * float(self._transaction_cost_pct)
            net_pnl = gross_pnl - cost
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]

            return BacktestTrade(
                ticker=ticker,
                direction="SELL",
                entry_date=pos["entry_date"],
                entry_price=Decimal(str(round(pos["entry_price"], 2))),
                exit_date=date,
                exit_price=Decimal(str(round(price, 2))),
                quantity=Decimal(str(pos["quantity"])),
                pnl=Decimal(str(round(net_pnl, 2))),
                pnl_pct=Decimal(str(round(pnl_pct, 4))),
            )

        return None

    def _calculate_metrics(
        self,
        trades: list[BacktestTrade],
        equity_curve: list[dict],
        initial_capital: Decimal,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestMetrics:
        """Calculate backtest performance metrics."""
        if not trades:
            return BacktestMetrics(
                total_return=Decimal("0"),
                total_return_pct=Decimal("0"),
                annualized_return=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                sortino_ratio=Decimal("0"),
                max_drawdown=Decimal("0"),
                max_drawdown_pct=Decimal("0"),
                win_rate=Decimal("0"),
                profit_factor=Decimal("0"),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=Decimal("0"),
                avg_loss=Decimal("0"),
                avg_trade_duration_hours=Decimal("0"),
                best_trade=Decimal("0"),
                worst_trade=Decimal("0"),
            )

        # Basic metrics
        closed_trades = [t for t in trades if t.exit_date is not None]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]

        total_pnl = sum(t.pnl for t in closed_trades)
        total_return_pct = total_pnl / initial_capital

        # Calculate duration
        days = (end_date - start_date).days
        years = days / 365.25

        # Annualized return
        if years > 0 and total_return_pct > -1:
            annualized = (1 + float(total_return_pct)) ** (1 / years) - 1
        else:
            annualized = 0

        # Win rate
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("inf")

        # Average win/loss
        avg_win = gross_profit / len(winning) if winning else Decimal("0")
        avg_loss = gross_loss / len(losing) if losing else Decimal("0")

        # Max drawdown
        max_dd = max(e["drawdown"] for e in equity_curve) if equity_curve else 0
        max_dd_value = max_dd * float(initial_capital)

        # Trade duration
        durations = []
        for t in closed_trades:
            if t.entry_date and t.exit_date:
                hours = (t.exit_date - t.entry_date).total_seconds() / 3600
                durations.append(hours)
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Best/worst trades
        pnls = [float(t.pnl) for t in closed_trades]
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0

        # Sharpe ratio (simplified - using daily returns)
        if len(equity_curve) > 1:
            daily_returns = []
            for i in range(1, len(equity_curve)):
                prev = equity_curve[i - 1]["equity"]
                curr = equity_curve[i]["equity"]
                if prev > 0:
                    daily_returns.append((curr - prev) / prev)

            if daily_returns:
                import statistics

                avg_return = statistics.mean(daily_returns)
                std_return = statistics.stdev(daily_returns) if len(daily_returns) > 1 else 1
                sharpe = (avg_return * 252) / (std_return * (252**0.5)) if std_return > 0 else 0

                # Sortino (downside deviation)
                negative_returns = [r for r in daily_returns if r < 0]
                downside_std = (
                    statistics.stdev(negative_returns) if len(negative_returns) > 1 else std_return
                )
                sortino = (
                    (avg_return * 252) / (downside_std * (252**0.5)) if downside_std > 0 else 0
                )
            else:
                sharpe = 0
                sortino = 0
        else:
            sharpe = 0
            sortino = 0

        return BacktestMetrics(
            total_return=total_pnl,
            total_return_pct=Decimal(str(round(float(total_return_pct), 4))),
            annualized_return=Decimal(str(round(annualized, 4))),
            sharpe_ratio=Decimal(str(round(sharpe, 2))),
            sortino_ratio=Decimal(str(round(sortino, 2))),
            max_drawdown=Decimal(str(round(max_dd_value, 2))),
            max_drawdown_pct=Decimal(str(round(max_dd, 4))),
            win_rate=Decimal(str(round(win_rate, 4))),
            profit_factor=(
                Decimal(str(round(float(profit_factor), 2)))
                if profit_factor != Decimal("inf")
                else Decimal("999")
            ),
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win=Decimal(str(round(float(avg_win), 2))),
            avg_loss=Decimal(str(round(float(avg_loss), 2))),
            avg_trade_duration_hours=Decimal(str(round(avg_duration, 1))),
            best_trade=Decimal(str(round(best, 2))),
            worst_trade=Decimal(str(round(worst, 2))),
        )
