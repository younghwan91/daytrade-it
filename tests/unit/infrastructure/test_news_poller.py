"""Unit tests for NewsPoller's Toss/DART circuit-breaker + fallback behavior.

No network access — `TossNewsClient`/`DartClient` are replaced by fakes that
raise on demand, so these tests exercise the actual state transitions
(CLOSED -> OPEN -> HALF_OPEN -> CLOSED), not just the described behavior.
"""

from __future__ import annotations

import datetime as dt

from krx_news_client import DartQuotaExceededError

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Ticker
from daytrade_it.infrastructure.external.news_poller import CircuitState, NewsPoller
from daytrade_it.shared.errors import error_codes
from daytrade_it.shared.errors.exceptions import DartAPIError, TossNewsAPIError


def make_article(title: str = "Test") -> NewsArticle:
    return NewsArticle(
        title=title,
        content="content",
        source="toss",
        url="https://tossinvest.com/news?id=1",
        published_at=dt.datetime(2026, 9, 7, tzinfo=dt.UTC),
        tickers=[Ticker(value="005930")],
    )


class FakeTossClient:
    """Stands in for TossNewsClient — each call pops the next scripted result."""

    def __init__(self, responses: list[list[NewsArticle] | Exception]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.closed = False

    async def poll_once(self) -> list[NewsArticle]:
        self.calls += 1
        response = self._responses.pop(0) if self._responses else []
        if isinstance(response, Exception):
            raise TossNewsAPIError(message=str(response)) from response
        return response

    async def close(self) -> None:
        self.closed = True


class FakeDartClient:
    """Stands in for DartClient — always returns a fixed list (or raises)."""

    def __init__(self, responses: list[list[NewsArticle] | Exception] | None = None) -> None:
        self._responses = list(responses) if responses is not None else None
        self.calls = 0
        self.closed = False
        # NewsPoller._poll_dart skips the DART call entirely when no API
        # key is configured (whole-branch review finding #6) -- these
        # tests exercise the DART call path itself, so the fake must look
        # "configured" the way a real DartClient with a key would.
        self.api_key = "fake-dart-key-for-tests"

    async def poll_once(self) -> list[NewsArticle]:
        self.calls += 1
        if self._responses is None:
            return []
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True


class TestNewsPollerHappyPath:
    async def test_merges_toss_and_dart_articles(self) -> None:
        toss = FakeTossClient(responses=[[make_article("toss-1")]])
        dart = FakeDartClient(responses=[[make_article("dart-1")]])
        poller = NewsPoller(toss_client=toss, dart_client=dart, failure_threshold=3)

        result = await poller.poll_once()

        titles = {a.title for a in result}
        assert titles == {"toss-1", "dart-1"}
        assert poller.circuit_state == CircuitState.CLOSED


class TestNewsPollerCircuitBreaker:
    async def test_circuit_stays_closed_below_failure_threshold(self) -> None:
        toss = FakeTossClient(
            responses=[RuntimeError("fail-1"), RuntimeError("fail-2")]
        )
        dart = FakeDartClient()
        poller = NewsPoller(toss_client=toss, dart_client=dart, failure_threshold=3)

        await poller.poll_once()
        await poller.poll_once()

        assert poller.circuit_state == CircuitState.CLOSED
        assert poller.consecutive_toss_failures == 2

    async def test_circuit_opens_after_threshold_consecutive_failures(self) -> None:
        toss = FakeTossClient(
            responses=[RuntimeError("f1"), RuntimeError("f2"), RuntimeError("f3")]
        )
        dart = FakeDartClient()
        poller = NewsPoller(toss_client=toss, dart_client=dart, failure_threshold=3)

        await poller.poll_once()
        await poller.poll_once()
        await poller.poll_once()

        assert poller.circuit_state == CircuitState.OPEN

    async def test_open_circuit_skips_toss_and_falls_back_to_dart_only(self) -> None:
        """This is the actual fallback behavior: once OPEN, Toss must not be
        called again until the reset window elapses — DART keeps flowing."""
        toss = FakeTossClient(
            responses=[
                RuntimeError("f1"),
                RuntimeError("f2"),
                RuntimeError("f3"),
                # If the circuit didn't skip Toss, this 4th call would raise
                # KeyError from popping an empty list / or be consumed —
                # either way it proves whether Toss was actually called again.
            ]
        )
        dart = FakeDartClient(responses=[[], [], [], [make_article("dart-only")]])
        poller = NewsPoller(
            toss_client=toss,
            dart_client=dart,
            failure_threshold=3,
            circuit_reset_seconds=999999.0,  # never elapses within the test
        )

        for _ in range(3):
            await poller.poll_once()
        assert poller.circuit_state == CircuitState.OPEN
        assert toss.calls == 3

        result = await poller.poll_once()

        assert toss.calls == 3  # Toss was NOT called a 4th time — circuit open
        assert dart.calls == 4  # DART still polled every cycle
        assert [a.title for a in result] == ["dart-only"]

    async def test_half_open_probe_success_closes_circuit(self) -> None:
        toss = FakeTossClient(
            responses=[
                RuntimeError("f1"),
                RuntimeError("f2"),
                RuntimeError("f3"),
                [make_article("recovered")],  # succeeds on the probe
            ]
        )
        dart = FakeDartClient()
        poller = NewsPoller(
            toss_client=toss,
            dart_client=dart,
            failure_threshold=3,
            circuit_reset_seconds=0.0,  # elapses immediately
        )

        for _ in range(3):
            await poller.poll_once()
        assert poller.circuit_state == CircuitState.OPEN

        result = await poller.poll_once()

        assert toss.calls == 4  # probed Toss again after reset window
        assert poller.circuit_state == CircuitState.CLOSED
        assert poller.consecutive_toss_failures == 0
        assert [a.title for a in result] == ["recovered"]

    async def test_half_open_probe_failure_reopens_circuit(self) -> None:
        toss = FakeTossClient(
            responses=[
                RuntimeError("f1"),
                RuntimeError("f2"),
                RuntimeError("f3"),
                RuntimeError("f4-probe-fails-too"),
            ]
        )
        dart = FakeDartClient()
        poller = NewsPoller(
            toss_client=toss,
            dart_client=dart,
            failure_threshold=3,
            circuit_reset_seconds=0.0,
        )

        for _ in range(3):
            await poller.poll_once()
        await poller.poll_once()  # the failing probe

        assert poller.circuit_state == CircuitState.OPEN
        assert toss.calls == 4


class TestNewsPollerDartQuotaHandling:
    async def test_dart_quota_exceeded_swallowed_not_propagated(self) -> None:
        toss = FakeTossClient(responses=[[]])
        dart = FakeDartClient(responses=[DartQuotaExceededError("quota gone")])
        poller = NewsPoller(toss_client=toss, dart_client=dart)

        # Must not raise -- the poller degrades gracefully, doesn't hard-fail.
        result = await poller.poll_once()
        assert result == []


class TestNewsPollerDartAPIErrorHandling:
    """Whole-branch review finding #6: a general DartAPIError (not just
    quota exhaustion) used to propagate uncaught and block the entire
    poll, even with a healthy Toss. Both must degrade gracefully now."""

    async def test_dart_api_error_swallowed_not_propagated(self) -> None:
        toss = FakeTossClient(responses=[[make_article("toss-1")]])
        dart = FakeDartClient(
            responses=[
                DartAPIError(message="network blew up", error_code=error_codes.DART_API_ERROR)
            ]
        )
        poller = NewsPoller(toss_client=toss, dart_client=dart)

        # Must not raise -- Toss's articles still come back, DART's
        # failure this cycle is skipped, not fatal to the whole poll.
        result = await poller.poll_once()
        assert [a.title for a in result] == ["toss-1"]

    async def test_no_dart_api_key_skips_dart_call_entirely(self) -> None:
        toss = FakeTossClient(responses=[[make_article("toss-1")]])
        dart = FakeDartClient(responses=[[make_article("should-not-be-returned")]])
        dart.api_key = ""  # unconfigured, same as this environment
        poller = NewsPoller(toss_client=toss, dart_client=dart)

        result = await poller.poll_once()

        assert dart.calls == 0  # DART was never called at all
        assert [a.title for a in result] == ["toss-1"]

    async def test_toss_polled_before_dart(self) -> None:
        """Toss is the key-free, always-available primary source and must
        not be starved by DART trouble -- see NewsPoller.poll_once's
        docstring."""
        call_order: list[str] = []

        class OrderTrackingToss(FakeTossClient):
            async def poll_once(self) -> list[NewsArticle]:
                call_order.append("toss")
                return await super().poll_once()

        class OrderTrackingDart(FakeDartClient):
            async def poll_once(self) -> list[NewsArticle]:
                call_order.append("dart")
                return await super().poll_once()

        toss = OrderTrackingToss(responses=[[]])
        dart = OrderTrackingDart(responses=[[]])
        poller = NewsPoller(toss_client=toss, dart_client=dart)

        await poller.poll_once()

        assert call_order == ["toss", "dart"]
