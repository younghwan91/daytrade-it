"""API routers registration.

This module provides router registration for the FastAPI application.
Add new routers here and register them in the main application.

Example:
    from gpt_quant_v2.interfaces.api.routers import signals, portfolio, orders

    # In main.py create_app():
    app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
    app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
    app.include_router(orders.router, prefix="/api/v1/orders", tags=["orders"])
"""

# Router imports will be added here as they are created
# from . import signals
# from . import portfolio
# from . import orders

__all__: list[str] = []
