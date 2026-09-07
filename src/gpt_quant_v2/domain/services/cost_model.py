"""Korean equity transaction-cost model.

This is the single source of truth for what a KRX trade actually costs,
shared by both the backtest engine (``application/handlers/backtest.py``)
and paper trading (``infrastructure/brokers/paper_broker.py``) so the two
never drift into two hand-maintained sets of numbers (Task 13 / Phase 5).

Three cost components, per the task brief:

1. **증권거래세 (securities transaction tax) -- sell-side only, researched,
   cited below.**
2. **위탁수수료 (brokerage commission) -- a labeled representative default,
   not a real broker fee schedule** (``KiwoomBroker`` has no live fee
   schedule to read from -- see Task 10's report -- so this is a
   placeholder, configurable rather than hardcoded deep in logic).
3. **Slippage -- a simple two-tier (KOSPI vs KOSDAQ) placeholder, not a
   real market-cap/liquidity-rank model.** ``domain/entities/stock.py``'s
   ``is_large_cap``/``is_mid_cap``/``is_small_cap`` already documents (Task
   9 research) that KRX's actual large/mid/small-cap buckets are a
   *market-wide rank*, recomputed semiannually against the full listed
   universe -- not something derivable from a single ticker or its
   ``Exchange`` alone. Building a real rank-based liquidity tier would need
   a cross-sectional ranking service, which is out of scope here (a stretch
   goal per the brief, not a requirement). What's implemented instead is
   the coarser, honestly-labeled KOSPI-vs-KOSDAQ split (KOSDAQ small-caps
   as a class trade with wider spreads than KOSPI large-caps), flagged as
   needing calibration against real historical bid/ask spreads later.

Tax rate research (Task 13, researched 2026-09-08 via web search -- same
research approach as Tasks 9/11's tick-size/disclosure-taxonomy research):

    2026-01-01 amendment (금융투자소득세 폐지에 따른 조치) currently in
    effect at the time of this research:
      - KOSPI:  0.05% 증권거래세 + 0.15% 농어촌특별세 = 0.20% total
      - KOSDAQ: 0.20% 증권거래세 (농특세 미부과)         = 0.20% total
    Both markets net to the same 0.20% total sell-side rate under the
    current law, which is why ``DEFAULT_TAX_RATE`` below is a single
    number -- but this has changed twice in the last three years (0.23%
    in 2021 -> 0.20% in 2022 -> 0.18% in 2023 -> 0.00%/0.15% split in
    2025 -> 0.20%/0.20% in 2026, per 나무위키's "증권거래세" history
    table), so `market` is still accepted as an explicit parameter for
    forward compatibility, the same pattern `tick_size.py` uses for its
    KOSPI/KOSDAQ/KONEX-uniform table.
    Sources: 머니투데이 "손절해도 '세금'...증권거래세 부활에 두 번 우는
    단타 개미들" (2025-12-30); 나무위키 "증권거래세" (rate-history table);
    키움투자자산운용 "2026년 달라지는 투자 정책 총정리"; 네이트뉴스
    "[2026 달라지는 것] 증권거래세율↑·배당소득 분리과세 최고세율 '30%'".

Brokerage commission: NO confirmed live fee schedule exists in this repo
(``KiwoomBroker`` doesn't expose one -- Task 10's report). ``DEFAULT_COMMISSION_RATE``
below (0.015%, i.e. 0.00015) is a representative placeholder in line with
typical Korean online-brokerage promotional rates (most retail brokers,
including Kiwoom's own 영웅문 online channel, advertise commission around
this order of magnitude for online equity trades), NOT a verified number --
treat it as needing replacement once a real fee schedule is available.
It is configurable via ``CostModelConfig``, not hardcoded inside the cost
calculation, per the brief's explicit instruction.

Slippage: ``DEFAULT_KOSPI_SLIPPAGE_PCT``/``DEFAULT_KOSDAQ_SLIPPAGE_PCT``
are flat placeholder rates, NOT calibrated against real historical
bid/ask spreads -- flagged honestly rather than presented as calibrated,
per the brief's explicit allowance that a full liquidity-tier model is a
stretch goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from gpt_quant_v2.domain.entities.stock import Exchange

#: Accepted trade-direction strings. This module deliberately does not
#: import ``infrastructure.brokers.base.OrderSide`` -- a domain service
#: should not depend on the infrastructure layer -- so callers pass a plain
#: ``"BUY"``/``"SELL"`` string (case-insensitive), matching the convention
#: already used by ``BacktestTrade.direction`` in
#: ``application/handlers/backtest.py``. ``PaperBroker`` converts its own
#: ``OrderSide`` enum to this string at the call site.
_SELL = "SELL"

# -- Researched (see module docstring for sources / research date) --------
#: 2026-01-01 amendment: KOSPI 0.05% 증권거래세 + 0.15% 농특세 = 0.20% total.
DEFAULT_KOSPI_TAX_RATE = Decimal("0.0020")
#: 2026-01-01 amendment: KOSDAQ 0.20% 증권거래세, no 농특세 = 0.20% total.
DEFAULT_KOSDAQ_TAX_RATE = Decimal("0.0020")

# -- Honestly-labeled placeholders (NOT researched/confirmed) --------------
#: Representative online-brokerage commission rate. Not read from any real
#: fee schedule -- KiwoomBroker exposes none (Task 10). Applied to BOTH
#: buy and sell notional, unlike the sell-only transaction tax.
DEFAULT_COMMISSION_RATE = Decimal("0.00015")
#: Flat placeholder slippage for KOSPI-tier (large-cap-like) liquidity.
#: NOT calibrated against real historical spreads.
DEFAULT_KOSPI_SLIPPAGE_PCT = Decimal("0.0005")
#: Flat placeholder slippage for KOSDAQ-tier (smaller-cap-like) liquidity,
#: deliberately set higher than the KOSPI rate (KOSDAQ small-caps slip more
#: than KOSPI large-caps) but still a flat, uncalibrated placeholder within
#: that tier -- not a real market-cap/liquidity-rank model. See module
#: docstring for why a real rank-based tier is out of scope here.
DEFAULT_KOSDAQ_SLIPPAGE_PCT = Decimal("0.0015")


@dataclass(frozen=True)
class CostModelConfig:
    """Configurable rates for :class:`KoreanCostModel`.

    All rates are fractions (``0.001`` == 0.1%), not percentages.
    """

    kospi_tax_rate: Decimal = DEFAULT_KOSPI_TAX_RATE
    kosdaq_tax_rate: Decimal = DEFAULT_KOSDAQ_TAX_RATE
    commission_rate: Decimal = DEFAULT_COMMISSION_RATE
    min_commission: Decimal = field(default_factory=lambda: Decimal("0"))
    kospi_slippage_pct: Decimal = DEFAULT_KOSPI_SLIPPAGE_PCT
    kosdaq_slippage_pct: Decimal = DEFAULT_KOSDAQ_SLIPPAGE_PCT


@dataclass(frozen=True)
class TradeCost:
    """Breakdown of costs for a single simulated fill.

    ``fill_price`` already includes slippage applied in the adverse
    direction (higher for buys, lower for sells); ``commission`` and
    ``tax`` are computed on the *slipped* notional (``fill_price *
    quantity``), matching how a real broker would charge on the actual
    executed price, not the pre-slippage quote.
    """

    fill_price: Decimal
    commission: Decimal
    tax: Decimal

    @property
    def total_cost(self) -> Decimal:
        """Commission + tax (slippage itself is embedded in ``fill_price``)."""
        return self.commission + self.tax


class KoreanCostModel:
    """Korean-market transaction-cost model (tax + commission + slippage).

    Shared by :class:`~gpt_quant_v2.application.handlers.backtest.BacktestHandler`
    and :class:`~gpt_quant_v2.infrastructure.brokers.paper_broker.PaperBroker`
    so both draw from one configuration rather than two hand-maintained sets
    of constants (Task 13).
    """

    def __init__(self, config: CostModelConfig | None = None) -> None:
        self._config = config or CostModelConfig()

    @property
    def config(self) -> CostModelConfig:
        return self._config

    def _slippage_pct(self, market: Exchange) -> Decimal:
        if market == Exchange.KOSDAQ:
            return self._config.kosdaq_slippage_pct
        # KOSPI and KONEX (no dedicated data) fall back to the KOSPI rate --
        # KONEX is a much smaller/thinner market in practice, so this is a
        # conservative-but-unconfirmed default, not a researched KONEX rate.
        return self._config.kospi_slippage_pct

    def _tax_rate(self, market: Exchange) -> Decimal:
        if market == Exchange.KOSDAQ:
            return self._config.kosdaq_tax_rate
        return self._config.kospi_tax_rate

    def apply_slippage(
        self,
        price: Decimal,
        side: str,
        market: Exchange = Exchange.KOSPI,
    ) -> Decimal:
        """Return `price` adjusted for slippage in the adverse direction.

        Buys fill higher, sells fill lower -- worse for the trader either
        way, which is the conservative assumption for a cost model.

        Args:
            price: Reference (quote/close) price before slippage.
            side: ``"BUY"`` or ``"SELL"`` (case-insensitive).
            market: Exchange tier used to pick the slippage rate. Callers
                that don't yet have a ticker-to-exchange mapping wired
                (see module docstring) should pass an explicit best-guess
                rather than silently relying on the KOSPI default.
        """
        slip = price * self._slippage_pct(market)
        if side.upper() == _SELL:
            return price - slip
        return price + slip

    def commission(self, notional: Decimal) -> Decimal:
        """Brokerage commission (위탁수수료) on `notional`, both sides."""
        return max(notional * self._config.commission_rate, self._config.min_commission)

    def transaction_tax(
        self,
        notional: Decimal,
        side: str,
        market: Exchange = Exchange.KOSPI,
    ) -> Decimal:
        """증권거래세: sell-side only, zero on buys."""
        if side.upper() != _SELL:
            return Decimal("0")
        return notional * self._tax_rate(market)

    def cost_of_trade(
        self,
        price: Decimal,
        quantity: Decimal,
        side: str,
        market: Exchange = Exchange.KOSPI,
    ) -> TradeCost:
        """Full cost breakdown for one fill: slippage + commission + tax.

        Args:
            price: Reference price before slippage (e.g. last quote/close).
            quantity: Number of shares (must be non-negative).
            side: Order direction.
            market: Exchange tier for tax/slippage rate selection.

        Returns:
            :class:`TradeCost` with the slipped fill price and the
            commission/tax charged on that fill's notional.
        """
        fill_price = self.apply_slippage(price, side, market)
        notional = fill_price * quantity
        return TradeCost(
            fill_price=fill_price,
            commission=self.commission(notional),
            tax=self.transaction_tax(notional, side, market),
        )


__all__ = [
    "DEFAULT_COMMISSION_RATE",
    "DEFAULT_KOSDAQ_SLIPPAGE_PCT",
    "DEFAULT_KOSDAQ_TAX_RATE",
    "DEFAULT_KOSPI_SLIPPAGE_PCT",
    "DEFAULT_KOSPI_TAX_RATE",
    "CostModelConfig",
    "KoreanCostModel",
    "TradeCost",
]
