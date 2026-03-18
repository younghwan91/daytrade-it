"""Factory for creating broker instances.

This module provides a factory function to create broker instances
based on configuration settings.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_quant_v2.infrastructure.brokers.base import BrokerInterface
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
        broker_type: Type of broker to create ("paper", "alpaca", "ibkr").
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
        raise BrokerFactoryError(
            "Alpaca broker not yet implemented. Use 'paper' for testing."
        )
    elif broker_type == "ibkr":
        raise BrokerFactoryError(
            "IBKR broker not yet implemented. Use 'paper' for testing."
        )
    else:
        raise BrokerFactoryError(
            f"Unknown broker type: {broker_type}. "
            f"Supported types: paper, alpaca, ibkr"
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
    # Extract paper broker settings from app settings if available
    initial_cash = kwargs.pop("initial_cash", None)
    slippage_pct = kwargs.pop("slippage_pct", None)
    commission_per_share = kwargs.pop("commission_per_share", None)
    min_commission = kwargs.pop("min_commission", None)

    # Try to get values from settings if not explicitly provided
    if settings is not None and hasattr(settings, "broker"):
        broker_settings = settings.broker
        if initial_cash is None and hasattr(broker_settings, "paper_initial_cash"):
            initial_cash = broker_settings.paper_initial_cash
        if slippage_pct is None and hasattr(broker_settings, "paper_slippage_pct"):
            slippage_pct = broker_settings.paper_slippage_pct
        if commission_per_share is None and hasattr(broker_settings, "paper_commission"):
            commission_per_share = broker_settings.paper_commission

    # Build config with provided or default values
    config_kwargs = {}
    if initial_cash is not None:
        config_kwargs["initial_cash"] = Decimal(str(initial_cash))
    if slippage_pct is not None:
        config_kwargs["slippage_pct"] = Decimal(str(slippage_pct))
    if commission_per_share is not None:
        config_kwargs["commission_per_share"] = Decimal(str(commission_per_share))
    if min_commission is not None:
        config_kwargs["min_commission"] = Decimal(str(min_commission))

    paper_config = PaperBrokerConfig(**config_kwargs)
    return PaperBroker(config=paper_config, settings=settings)
