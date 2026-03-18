"""API routers registration.

This module provides router registration for the FastAPI application.
Add new routers here and register them in the main application.

Example:
    from gpt_quant_v2.interfaces.api.routers import signals, portfolio, trades, news, models

    # In main.py create_app():
    app.include_router(signals.router, prefix="/api/v1/signals", tags=["signals"])
    app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
    app.include_router(trades.router, prefix="/api/v1/trades", tags=["trades"])
    app.include_router(news.router, prefix="/api/v1/news", tags=["news"])
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
"""

from gpt_quant_v2.interfaces.api.routers import models, news, portfolio, signals, trades

__all__ = ["signals", "trades", "portfolio", "news", "models"]
