"""Unit tests for TradingSettings' KRX trading-hours defaults (Task 9)."""

from datetime import time

from gpt_quant_v2.shared.config.settings import TradingSettings


class TestTradingWindowDefaults:
    def test_window_defaults_to_krx_regular_session(self) -> None:
        """KRX's regular session runs 09:00-15:30 KST, no US-style pre/post-market."""
        settings = TradingSettings()
        assert settings.window_start == time(9, 0)
        assert settings.window_end == time(15, 30)

    def test_window_end_after_start(self) -> None:
        settings = TradingSettings()
        assert settings.window_end > settings.window_start

    def test_window_parses_hh_mm_strings(self) -> None:
        settings = TradingSettings(window_start="09:00", window_end="15:30")
        assert settings.window_start == time(9, 0)
        assert settings.window_end == time(15, 30)
