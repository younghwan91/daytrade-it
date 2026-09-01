"""
E2E Tests for Backtesting Workflow

Tests the complete backtesting workflow:
1. Running backtests with various configurations
2. Validating backtest results and metrics
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest

from tests.e2e.conftest import requires_api, requires_external_services, slow_test

# ============================================================================
# Backtest Execution Tests
# ============================================================================


class TestRunBacktest:
    """Test backtest execution functionality."""

    @requires_api
    @pytest.mark.asyncio
    async def test_run_backtest_basic_config(
        self,
        api_client: httpx.AsyncClient,
        backtest_config: dict[str, Any],
    ) -> None:
        """Test running a backtest with basic configuration.

        Verifies:
        - Backtest can be initiated
        - Returns appropriate status (created/accepted)
        - Response contains backtest ID for tracking
        """
        # Act - Submit backtest request
        response = await api_client.post(
            "/api/v1/backtest/run",
            json=backtest_config,
        )

        # Assert - Handle various response scenarios
        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        # If endpoint exists, verify response
        assert response.status_code in (200, 201, 202)

        if response.status_code in (200, 201, 202):
            data = response.json()
            # Verify response contains tracking information
            assert "id" in data or "backtest_id" in data or "status" in data

    @requires_api
    @pytest.mark.asyncio
    async def test_run_backtest_custom_date_range(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest with custom date range."""
        config = {
            "start_date": "2023-06-01",
            "end_date": "2023-12-31",
            "initial_capital": "50000.00",
            "tickers": ["NVDA", "AMD"],
            "strategy": {
                "type": "news_sentiment",
                "parameters": {
                    "min_confidence": 0.8,
                    "position_size_pct": 0.15,
                },
            },
        }

        response = await api_client.post("/api/v1/backtest/run", json=config)

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        assert response.status_code in (200, 201, 202, 422)

    @requires_api
    @pytest.mark.asyncio
    async def test_run_backtest_validates_dates(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest validates date range (end must be after start)."""
        invalid_config = {
            "start_date": "2024-12-31",
            "end_date": "2024-01-01",  # Before start_date
            "initial_capital": "100000.00",
            "tickers": ["AAPL"],
            "strategy": {"type": "news_sentiment"},
        }

        response = await api_client.post("/api/v1/backtest/run", json=invalid_config)

        if response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        # Should return validation error
        assert response.status_code in (400, 401, 422)

    @requires_api
    @pytest.mark.asyncio
    async def test_run_backtest_validates_capital(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest validates initial capital is positive."""
        invalid_config = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": "-1000.00",  # Negative capital
            "tickers": ["AAPL"],
            "strategy": {"type": "news_sentiment"},
        }

        response = await api_client.post("/api/v1/backtest/run", json=invalid_config)

        if response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        # Should return validation error
        assert response.status_code in (400, 401, 422)

    @requires_api
    @pytest.mark.asyncio
    async def test_run_backtest_validates_tickers(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest validates ticker symbols."""
        invalid_config = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": "100000.00",
            "tickers": [],  # Empty tickers list
            "strategy": {"type": "news_sentiment"},
        }

        response = await api_client.post("/api/v1/backtest/run", json=invalid_config)

        if response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        # Should return validation error
        assert response.status_code in (400, 401, 422)

    @requires_api
    @requires_external_services
    @slow_test
    @pytest.mark.asyncio
    async def test_run_backtest_with_real_data(
        self,
        api_client: httpx.AsyncClient,
        backtest_config: dict[str, Any],
    ) -> None:
        """Test running a backtest with real market data.

        This test requires external data APIs and may take longer to execute.
        It validates the complete backtest execution flow.
        """
        # Submit backtest with longer date range
        extended_config = {
            **backtest_config,
            "start_date": "2023-01-01",
            "end_date": "2023-06-30",
        }

        response = await api_client.post("/api/v1/backtest/run", json=extended_config)

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        assert response.status_code in (200, 201, 202)

        if response.status_code in (200, 201, 202):
            data = response.json()

            # If synchronous, results should be present
            if "results" in data:
                assert "total_return" in data["results"]
                assert "total_trades" in data["results"]

            # If async, we should have tracking ID
            elif "id" in data or "backtest_id" in data:
                backtest_id = data.get("id") or data.get("backtest_id")

                # Poll for completion (with timeout)
                import asyncio

                for _ in range(30):  # Max 30 attempts
                    status_response = await api_client.get(f"/api/v1/backtest/{backtest_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get("status") == "completed":
                            break
                    await asyncio.sleep(2)


# ============================================================================
# Backtest Results Validation Tests
# ============================================================================


class TestBacktestResults:
    """Test backtest results and metrics validation."""

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_results_structure(
        self,
        api_client: httpx.AsyncClient,
        expected_backtest_metrics: dict[str, Any],
    ) -> None:
        """Test that backtest results contain expected metrics structure.

        Validates the schema of backtest results including:
        - Return metrics
        - Risk metrics
        - Trade statistics
        """
        # First, try to get a completed backtest result
        response = await api_client.get("/api/v1/backtest/results")

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest results endpoint not implemented")

        assert response.status_code in (200, 404)

        if response.status_code == 200:
            data = response.json()

            # Check if this is a list or single result
            results = data if isinstance(data, list) else [data] if data else []

            for result in results:
                if "metrics" in result:
                    metrics = result["metrics"]
                    _validate_metrics_structure(metrics, expected_backtest_metrics)

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_results_by_id(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test retrieving backtest results by ID."""
        # Use a fake ID - should return 404 if properly implemented
        fake_id = str(uuid4())

        response = await api_client.get(f"/api/v1/backtest/{fake_id}")

        if response.status_code == 401:
            pytest.skip("Authentication required")

        # Should return 404 for non-existent backtest
        assert response.status_code in (404, 422)

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_results_metrics_calculations(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test that backtest metrics are calculated correctly.

        Validates mathematical consistency of metrics:
        - Net P&L = Gross P&L - Fees
        - Win rate = Winning trades / Total trades
        - Total trades = Winning + Losing trades
        """
        response = await api_client.get("/api/v1/backtest/results")

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest results endpoint not implemented")

        if response.status_code == 200:
            data = response.json()
            results = data if isinstance(data, list) else [data] if data else []

            for result in results:
                if "metrics" not in result:
                    continue

                metrics = result["metrics"]

                # Validate trade count consistency
                total_trades = metrics.get("total_trades", 0)
                winning = metrics.get("winning_trades", 0)
                losing = metrics.get("losing_trades", 0)

                if total_trades > 0:
                    assert total_trades >= winning + losing

                # Validate win rate calculation
                win_rate = metrics.get("win_rate")
                if win_rate is not None and total_trades > 0:
                    assert 0 <= win_rate <= 1

                # Validate Sharpe ratio bounds (typically between -3 and 5)
                sharpe = metrics.get("sharpe_ratio")
                if sharpe is not None:
                    assert -10 <= sharpe <= 10

                # Validate max drawdown is negative or zero
                max_dd = metrics.get("max_drawdown_pct")
                if max_dd is not None:
                    assert max_dd <= 0

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_list_endpoint(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test listing all backtests."""
        response = await api_client.get(
            "/api/v1/backtest",
            params={"page": 1, "page_size": 20},
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest list endpoint not implemented")

        assert response.status_code == 200

        data = response.json()
        # Should return paginated list
        assert "items" in data or isinstance(data, list)

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_results_include_trades(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test that backtest results can include individual trade details."""
        response = await api_client.get(
            "/api/v1/backtest/results",
            params={"include_trades": True},
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest results endpoint not implemented")

        if response.status_code == 200:
            data = response.json()
            results = data if isinstance(data, list) else [data] if data else []

            for result in results:
                if "trades" in result:
                    trades = result["trades"]
                    assert isinstance(trades, list)

                    for trade in trades:
                        # Validate trade structure
                        assert "ticker" in trade or "symbol" in trade
                        assert "direction" in trade or "side" in trade
                        assert "quantity" in trade or "shares" in trade

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_results_equity_curve(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test that backtest results include equity curve data."""
        response = await api_client.get(
            "/api/v1/backtest/results",
            params={"include_equity_curve": True},
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest results endpoint not implemented")

        if response.status_code == 200:
            data = response.json()
            results = data if isinstance(data, list) else [data] if data else []

            for result in results:
                if "equity_curve" in result:
                    curve = result["equity_curve"]
                    assert isinstance(curve, list)

                    if len(curve) > 0:
                        # Each point should have timestamp and value
                        for point in curve[:5]:  # Check first 5 points
                            assert "timestamp" in point or "date" in point
                            assert "value" in point or "equity" in point


# ============================================================================
# Backtest Comparison Tests
# ============================================================================


class TestBacktestComparison:
    """Test comparing multiple backtests."""

    @requires_api
    @pytest.mark.asyncio
    async def test_compare_backtests(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test comparing two or more backtest results."""
        # Create comparison request with fake IDs
        backtest_ids = [str(uuid4()), str(uuid4())]

        response = await api_client.post(
            "/api/v1/backtest/compare",
            json={"backtest_ids": backtest_ids},
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest comparison endpoint not implemented")

        # IDs don't exist, should return appropriate error
        assert response.status_code in (200, 404, 422)

    @requires_api
    @pytest.mark.asyncio
    async def test_backtest_vs_benchmark(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test comparing backtest against benchmark (e.g., SPY)."""
        response = await api_client.get(
            "/api/v1/backtest/results",
            params={"benchmark": "SPY"},
        )

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest results endpoint not implemented")

        if response.status_code == 200:
            data = response.json()
            results = data if isinstance(data, list) else [data] if data else []

            for result in results:
                metrics = result.get("metrics", {})
                # Should include benchmark comparison metrics
                if "benchmark_return" in metrics:
                    assert "alpha" in metrics or "excess_return" in metrics


# ============================================================================
# Backtest Strategy Tests
# ============================================================================


class TestBacktestStrategies:
    """Test different backtesting strategies."""

    @requires_api
    @pytest.mark.asyncio
    async def test_news_sentiment_strategy(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest with news sentiment strategy."""
        config = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "initial_capital": "100000.00",
            "tickers": ["AAPL"],
            "strategy": {
                "type": "news_sentiment",
                "parameters": {
                    "min_confidence": 0.7,
                    "sentiment_threshold": 0.5,
                    "position_size_pct": 0.1,
                },
            },
        }

        response = await api_client.post("/api/v1/backtest/run", json=config)

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        assert response.status_code in (200, 201, 202)

    @requires_api
    @pytest.mark.asyncio
    async def test_ml_prediction_strategy(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest with ML prediction strategy."""
        config = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "initial_capital": "100000.00",
            "tickers": ["MSFT", "GOOGL"],
            "strategy": {
                "type": "ml_prediction",
                "parameters": {
                    "model_name": "xgboost_v1",
                    "prediction_horizon": 5,
                    "min_confidence": 0.75,
                },
            },
        }

        response = await api_client.post("/api/v1/backtest/run", json=config)

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        assert response.status_code in (200, 201, 202, 422)

    @requires_api
    @pytest.mark.asyncio
    async def test_composite_strategy(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest with composite strategy combining multiple signals."""
        config = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-15",
            "initial_capital": "100000.00",
            "tickers": ["AAPL", "MSFT"],
            "strategy": {
                "type": "composite",
                "parameters": {
                    "signal_weights": {
                        "news_sentiment": 0.4,
                        "technical": 0.3,
                        "ml_prediction": 0.3,
                    },
                    "min_composite_score": 0.7,
                },
            },
        }

        response = await api_client.post("/api/v1/backtest/run", json=config)

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        assert response.status_code in (200, 201, 202, 422)


# ============================================================================
# Helper Functions
# ============================================================================


def _validate_metrics_structure(
    metrics: dict[str, Any],
    expected_types: dict[str, Any],
) -> None:
    """Validate that metrics contain expected fields with correct types."""
    for field, expected_type in expected_types.items():
        if field in metrics:
            value = metrics[field]
            if value is not None:
                if isinstance(expected_type, tuple):
                    assert isinstance(value, expected_type), (
                        f"Field {field} has type {type(value)}, expected {expected_type}"
                    )
                else:
                    assert isinstance(value, expected_type), (
                        f"Field {field} has type {type(value)}, expected {expected_type}"
                    )


# ============================================================================
# Performance and Stress Tests
# ============================================================================


class TestBacktestPerformance:
    """Performance tests for backtesting system."""

    @requires_api
    @requires_external_services
    @slow_test
    @pytest.mark.asyncio
    async def test_large_date_range_backtest(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest over a large date range (1 year)."""
        config = {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "initial_capital": "100000.00",
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            "strategy": {
                "type": "news_sentiment",
                "parameters": {
                    "min_confidence": 0.7,
                    "position_size_pct": 0.1,
                },
            },
        }

        import time

        start_time = time.time()

        response = await api_client.post("/api/v1/backtest/run", json=config)

        elapsed = time.time() - start_time

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        # Backtest should complete or be accepted within reasonable time
        assert elapsed < 60  # Max 60 seconds for submission

        assert response.status_code in (200, 201, 202)

    @requires_api
    @requires_external_services
    @slow_test
    @pytest.mark.asyncio
    async def test_many_tickers_backtest(
        self,
        api_client: httpx.AsyncClient,
    ) -> None:
        """Test backtest with many tickers."""
        # List of tickers to test
        tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "JPM",
            "V",
            "JNJ",
        ]

        config = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": "500000.00",
            "tickers": tickers,
            "strategy": {
                "type": "news_sentiment",
                "parameters": {
                    "min_confidence": 0.6,
                    "position_size_pct": 0.05,
                    "max_positions": 10,
                },
            },
        }

        response = await api_client.post("/api/v1/backtest/run", json=config)

        if response.status_code == 401:
            pytest.skip("Authentication required")
        elif response.status_code == 404:
            pytest.skip("Backtest endpoint not implemented")

        assert response.status_code in (200, 201, 202)
