"""Unit tests for TradingSettings' KRX trading-hours defaults (Task 9)."""

from datetime import time

from daytrade_it.shared.config.settings import BrokerSettings, BrokerType, TradingSettings


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


class TestBrokerSettingsTypeEnvVar:
    """Regression tests for Task 10's `alias="broker_type"` bug (Task 14 fix).

    pydantic-settings treats an explicit `alias` as the *complete* env var
    name and ignores `env_prefix` entirely for that field. With the old
    `alias="broker_type"`, `BrokerSettings.type` could only be set via the
    unprefixed `BROKER_TYPE`, so `GPTQ_BROKER_TYPE` -- the only spelling
    consistent with every other setting in this app -- was silently
    ignored. The fix spells the alias out as the fully prefixed
    `GPTQ_BROKER_TYPE` instead.
    """

    def test_defaults_to_paper(self, monkeypatch) -> None:
        monkeypatch.delenv("GPTQ_BROKER_TYPE", raising=False)
        monkeypatch.delenv("BROKER_TYPE", raising=False)
        settings = BrokerSettings()
        assert settings.type == BrokerType.PAPER

    def test_gptq_prefixed_env_var_selects_kiwoom(self, monkeypatch) -> None:
        """GPTQ_BROKER_TYPE (the app's consistent env prefix) must work."""
        monkeypatch.setenv("GPTQ_BROKER_TYPE", "kiwoom")
        monkeypatch.setenv("GPTQ_KIWOOM_APP_KEY", "test-app-key")
        monkeypatch.setenv("GPTQ_KIWOOM_APP_SECRET", "test-app-secret")
        settings = BrokerSettings()
        assert settings.type == BrokerType.KIWOOM

    def test_unprefixed_env_var_no_longer_selects_broker(self, monkeypatch) -> None:
        """The old (buggy) unprefixed `BROKER_TYPE` must no longer be honored."""
        monkeypatch.delenv("GPTQ_BROKER_TYPE", raising=False)
        monkeypatch.setenv("BROKER_TYPE", "kiwoom")
        settings = BrokerSettings()
        assert settings.type == BrokerType.PAPER

    def test_gptq_broker_type_selects_kiwoom_broker_via_factory(self, monkeypatch) -> None:
        """End-to-end: GPTQ_BROKER_TYPE=kiwoom actually produces a KiwoomBroker."""
        from daytrade_it.infrastructure.brokers.factory import create_broker
        from daytrade_it.infrastructure.brokers.kiwoom_broker import KiwoomBroker

        monkeypatch.setenv("GPTQ_BROKER_TYPE", "kiwoom")
        monkeypatch.setenv("GPTQ_KIWOOM_APP_KEY", "test-app-key")
        monkeypatch.setenv("GPTQ_KIWOOM_APP_SECRET", "test-app-secret")
        broker_settings = BrokerSettings()
        assert broker_settings.type == BrokerType.KIWOOM

        class _FakeSettings:
            broker = broker_settings

        broker = create_broker(broker_settings.type.value, _FakeSettings())
        assert isinstance(broker, KiwoomBroker)
