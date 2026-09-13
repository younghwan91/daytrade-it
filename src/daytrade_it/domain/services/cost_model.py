"""Korean equity transaction-cost model -- re-exported from krx-quant-core.

The model (증권거래세 + 위탁수수료 + slippage, tick-rounded fills) and its
default rates now live in :mod:`krx_quant_core.costs.model`, shared with
scalp-it, and are still the single source of truth for both the backtest
engine (``application/handlers/backtest.py``) and paper trading
(``infrastructure/brokers/paper_broker.py``). This module keeps the import
path callers here already use.

Compatibility with this package:

- ``market`` arguments take anything ``krx_quant_core.market.Market.parse``
  accepts. ``Exchange`` is a ``StrEnum`` with the same ``KOSPI``/``KOSDAQ``/
  ``KONEX`` values, so existing callers pass it unchanged.
- The default ``CostModelConfig`` tax rates are the same explicit 0.20%
  as before; core's date-based tax schedule is only used when a rate is set
  to ``None``, which nothing here does.

Rate provenance and confidence (all secondary-source or placeholder, NOT
primary-source-confirmed) are documented in ``krx_quant_core.costs.tax``
and ``krx_quant_core.costs.model``.
"""

from __future__ import annotations

from krx_quant_core.costs.model import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_KOSDAQ_SLIPPAGE_PCT,
    DEFAULT_KOSDAQ_TAX_RATE,
    DEFAULT_KOSPI_SLIPPAGE_PCT,
    DEFAULT_KOSPI_TAX_RATE,
    CostModelConfig,
    KoreanCostModel,
    TradeCost,
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
