"""Factory for creating broker instances.

This module provides a factory function to create broker instances
based on configuration settings.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_quant_v2.domain.services.cost_model import CostModelConfig
from gpt_quant_v2.infrastructure.brokers.base import BrokerInterface
from gpt_quant_v2.infrastructure.brokers.kiwoom_broker import KiwoomBroker
from gpt_quant_v2.infrastructure.brokers.paper_broker import (
    PaperBroker,
    PaperBrokerConfig,
)

if TYPE_CHECKING:
    from gpt_quant_v2.shared.config.settings import Settings


class BrokerFactoryError(Exception):
    """Error raised when broker creation fails."""

    pass


def create_broker(
    broker_type: str,
    config: Settings | None = None,
    **kwargs,
) -> BrokerInterface:
    """Create a broker instance based on the specified type.

    Args:
        broker_type: Type of broker to create ("paper", "alpaca", "ibkr", "kiwoom").
        config: Application settings for broker configuration.
        **kwargs: Additional keyword arguments passed to the broker.

    Returns:
        BrokerInterface implementation.

    Raises:
        BrokerFactoryError: If broker type is not supported.

    Example:
        >>> from gpt_quant_v2.shared.config import get_settings
        >>> settings = get_settings()
        >>> broker = create_broker("paper", settings)
        >>> broker.connect()
        True
    """
    broker_type = broker_type.lower()

    if broker_type == "paper":
        return _create_paper_broker(config, **kwargs)
    elif broker_type == "alpaca":
        raise BrokerFactoryError("Alpaca broker not yet implemented. Use 'paper' for testing.")
    elif broker_type == "ibkr":
        raise BrokerFactoryError("IBKR broker not yet implemented. Use 'paper' for testing.")
    elif broker_type == "kiwoom":
        return _create_kiwoom_broker(config, **kwargs)
    else:
        raise BrokerFactoryError(
            f"Unknown broker type: {broker_type}. Supported types: paper, alpaca, ibkr, kiwoom"
        )


def _create_paper_broker(
    settings: Settings | None = None,
    **kwargs,
) -> PaperBroker:
    """Create a paper trading broker.

    Args:
        settings: Application settings.
        **kwargs: Override configuration values.

    Returns:
        Configured PaperBroker instance.
    """
    # Extract paper broker settings from app settings if available.
    #
    # slippage_pct/commission_per_share/min_commission are accepted here as
    # legacy override kwargs for backward compatibility, but as of Task 13
    # they no longer map onto PaperBrokerConfig fields directly -- fills are
    # priced by the shared `KoreanCostModel` (domain/services/cost_model.py)
    # instead, so an override here builds a `CostModelConfig` for that
    # model rather than the two separate hand-maintained numbers this used
    # to set. See cost_model.py for why "commission_per_share" (a per-share
    # US-style fee) doesn't fit the Korean percentage-of-notional model --
    # it is treated as a percentage override, not a per-share one.
    initial_cash = kwargs.pop("initial_cash", None)
    slippage_pct = kwargs.pop("slippage_pct", None)
    commission_pct = kwargs.pop("commission_per_share", None)
    min_commission = kwargs.pop("min_commission", None)

    # Try to get values from settings if not explicitly provided
    if settings is not None and hasattr(settings, "broker"):
        broker_settings = settings.broker
        if initial_cash is None and hasattr(broker_settings, "paper_initial_cash"):
            initial_cash = broker_settings.paper_initial_cash
        if slippage_pct is None and hasattr(broker_settings, "paper_slippage_pct"):
            slippage_pct = broker_settings.paper_slippage_pct
        if commission_pct is None and hasattr(broker_settings, "paper_commission"):
            commission_pct = broker_settings.paper_commission

    # Build config with provided or default values
    config_kwargs: dict[str, object] = {}
    if initial_cash is not None:
        config_kwargs["initial_cash"] = Decimal(str(initial_cash))

    cost_overrides: dict[str, Decimal] = {}
    if slippage_pct is not None:
        # No ticker-level KOSPI/KOSDAQ distinction available at this call
        # site, so an explicit override applies to both tiers uniformly.
        cost_overrides["kospi_slippage_pct"] = Decimal(str(slippage_pct))
        cost_overrides["kosdaq_slippage_pct"] = Decimal(str(slippage_pct))
    if commission_pct is not None:
        cost_overrides["commission_rate"] = Decimal(str(commission_pct))
    if min_commission is not None:
        cost_overrides["min_commission"] = Decimal(str(min_commission))
    if cost_overrides:
        config_kwargs["cost_model_config"] = CostModelConfig(**cost_overrides)

    paper_config = PaperBrokerConfig(**config_kwargs)
    return PaperBroker(config=paper_config, settings=settings)


def _create_kiwoom_broker(
    settings: Settings | None = None,
    **kwargs,
) -> KiwoomBroker:
    """Create a Kiwoom Securities broker.

    Credentials are sourced from `settings.broker` (env vars
    `GPTQ_KIWOOM_APP_KEY` / `GPTQ_KIWOOM_APP_SECRET` /
    `GPTQ_KIWOOM_ACCOUNT_NO` / `GPTQ_KIWOOM_IS_MOCK` -- `BrokerSettings`
    uses a flat `GPTQ_` env prefix, not a nested `GPTQ_BROKER__` one, same
    as the existing `GPTQ_ALPACA_API_KEY` etc.), never hardcoded -- see
    `BrokerSettings.validate_kiwoom_credentials` for the required-field
    check.

    Args:
        settings: Application settings.
        **kwargs: Override values -- `app_key`, `app_secret`, `account_no`,
            `is_mock`, or a pre-built `api` (a `KiwoomAPI`-shaped object,
            mainly for tests to inject a mock without constructing a real
            `KiwoomAPI`).

    Returns:
        Configured KiwoomBroker instance.

    Raises:
        BrokerFactoryError: If credentials are missing from both `kwargs`
            and `settings`.
    """
    api = kwargs.pop("api", None)
    account_no = kwargs.pop("account_no", None)

    if api is None:
        app_key = kwargs.pop("app_key", None)
        app_secret = kwargs.pop("app_secret", None)
        is_mock = kwargs.pop("is_mock", None)

        if settings is not None and hasattr(settings, "broker"):
            broker_settings = settings.broker
            if app_key is None and broker_settings.kiwoom_app_key is not None:
                app_key = broker_settings.kiwoom_app_key.get_secret_value()
            if app_secret is None and broker_settings.kiwoom_app_secret is not None:
                app_secret = broker_settings.kiwoom_app_secret.get_secret_value()
            if account_no is None:
                account_no = broker_settings.kiwoom_account_no
            if is_mock is None:
                is_mock = broker_settings.kiwoom_is_mock

        if not app_key or not app_secret:
            raise BrokerFactoryError(
                "Kiwoom broker requires 'kiwoom_app_key'/'kiwoom_app_secret' "
                "(via settings.broker or explicit kwargs)."
            )

        # Imported lazily so environments without kiwoom-client installed
        # can still import this module for non-Kiwoom broker types.
        from kiwoom_client import KiwoomAPI

        api = KiwoomAPI(app_key=app_key, app_secret=app_secret, is_mock=bool(is_mock))

    return KiwoomBroker(api=api, account_no=account_no)
