"""Combined Toss/DART news poller with a circuit breaker for Toss.

Per the plan's original Phase 1 note, if Toss polling fails repeatedly the
poller should degrade to DART-only rather than hard-failing entirely. This
module owns that orchestration -- ``toss_news_client.TossNewsClient`` and
``dart_client.DartClient`` each stay focused on their own source, and this
class combines them with a simple consecutive-failure circuit breaker.

Persistence: this poller's output is returned to the caller in-memory
(``list[NewsArticle]`` per poll) and is *not* written anywhere by this
module. Per spike 004 / the task brief, this poller must not write into
``quant-airflow``'s ``news_articles``/``news_judgments`` tables or reuse its
upsert keys, and inventing a new TimescaleDB schema is out of scope for this
task -- wiring this into a store (if any) is a follow-up for whoever
consumes ``NewsPoller.poll_once()``/``poll_forever()``.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import TYPE_CHECKING

from krx_news_client import DartQuotaExceededError

from daytrade_it.infrastructure.external.dart_client import DartClient
from daytrade_it.infrastructure.external.toss_news_client import TossNewsClient
from daytrade_it.shared.config import get_settings
from daytrade_it.shared.errors.exceptions import DartAPIError
from daytrade_it.shared.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from daytrade_it.domain.entities.news import NewsArticle

logger = get_logger(__name__)


class CircuitState(StrEnum):
    """Toss circuit breaker state."""

    CLOSED = "closed"  # Toss polling active (normal)
    OPEN = "open"  # Toss polling skipped, DART-only fallback
    HALF_OPEN = "half_open"  # circuit_reset_seconds elapsed; probing Toss again


class NewsPoller:
    """Polls Toss + DART each cycle, degrading to DART-only on repeated Toss failures.

    The circuit breaker is a simple consecutive-failure counter, not a
    time-windowed rate: ``toss_failure_threshold`` consecutive
    ``poll_once()`` failures open the circuit. While open, Toss is not
    called at all (DART-only) until ``circuit_reset_seconds`` has elapsed,
    at which point the next cycle probes Toss again (half-open): success
    closes the circuit, failure re-opens it and resets the reset-timer.

    Example:
        >>> async with NewsPoller() as poller:
        ...     articles = await poller.poll_once()
        ...     print(poller.circuit_state)
    """

    def __init__(
        self,
        toss_client: TossNewsClient | None = None,
        dart_client: DartClient | None = None,
        failure_threshold: int | None = None,
        circuit_reset_seconds: float | None = None,
        poll_interval: float | None = None,
    ) -> None:
        settings = get_settings().krx_news
        self._toss = toss_client or TossNewsClient()
        self._dart = dart_client or DartClient()
        self.failure_threshold = (
            failure_threshold if failure_threshold is not None else settings.toss_failure_threshold
        )
        self.circuit_reset_seconds = (
            circuit_reset_seconds
            if circuit_reset_seconds is not None
            else settings.circuit_reset_seconds
        )
        self.poll_interval = (
            poll_interval if poll_interval is not None else settings.toss_poll_interval_seconds
        )

        self._consecutive_toss_failures = 0
        self._circuit_state = CircuitState.CLOSED
        self._circuit_opened_at: float | None = None

        logger.info(
            "NewsPoller initialized",
            failure_threshold=self.failure_threshold,
            circuit_reset_seconds=self.circuit_reset_seconds,
        )

    async def __aenter__(self) -> NewsPoller:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close both underlying clients' HTTP connections."""
        await self._toss.close()
        await self._dart.close()

    @property
    def circuit_state(self) -> CircuitState:
        """Current Toss circuit breaker state."""
        return self._circuit_state

    @property
    def consecutive_toss_failures(self) -> int:
        """Consecutive Toss poll failures since the last success."""
        return self._consecutive_toss_failures

    def _circuit_should_probe(self) -> bool:
        """Whether an OPEN circuit has waited long enough to try Toss again."""
        if self._circuit_state != CircuitState.OPEN:
            return False
        if self._circuit_opened_at is None:
            return True
        elapsed = asyncio.get_event_loop().time() - self._circuit_opened_at
        return elapsed >= self.circuit_reset_seconds

    async def poll_once(self) -> list[NewsArticle]:
        """Poll Toss and DART, merging new articles.

        Toss is polled first: it's the key-free, always-available primary
        source (per spike 004 / Task 7), so it should never be blocked by
        DART trouble. DART is polled second and degrades gracefully on
        failure -- see ``_poll_dart`` (whole-branch review finding #6: this
        used to poll DART first and only catch ``DartQuotaExceededError``,
        so in an environment with no DART key configured (this one), a
        DART failure would raise ``DartAPIError`` uncaught and block the
        *entire* poll even though Toss was healthy).
        """
        articles: list[NewsArticle] = []

        articles.extend(await self._poll_toss())
        articles.extend(await self._poll_dart())

        return articles

    async def _poll_toss(self) -> list[NewsArticle]:
        if self._circuit_state == CircuitState.OPEN:
            if not self._circuit_should_probe():
                logger.debug("Toss circuit open, skipping poll (DART-only)")
                return []
            logger.info("Toss circuit reset window elapsed, probing Toss again")
            self._circuit_state = CircuitState.HALF_OPEN

        try:
            new_articles = await self._toss.poll_once()
        except Exception:  # noqa: BLE001 -- any failure counts toward the breaker
            self._consecutive_toss_failures += 1
            logger.warning(
                "Toss poll failed",
                consecutive_failures=self._consecutive_toss_failures,
                threshold=self.failure_threshold,
                circuit_state=self._circuit_state.value,
            )
            if (
                self._circuit_state == CircuitState.HALF_OPEN
                or self._consecutive_toss_failures >= self.failure_threshold
            ):
                self._circuit_state = CircuitState.OPEN
                self._circuit_opened_at = asyncio.get_event_loop().time()
                logger.warning(
                    "Toss circuit breaker opened, degrading to DART-only",
                    consecutive_failures=self._consecutive_toss_failures,
                )
            return []
        else:
            if self._consecutive_toss_failures or self._circuit_state != CircuitState.CLOSED:
                logger.info("Toss poll succeeded, circuit breaker closed")
            self._consecutive_toss_failures = 0
            self._circuit_state = CircuitState.CLOSED
            self._circuit_opened_at = None
            return new_articles

    async def _poll_dart(self) -> list[NewsArticle]:
        """Poll DART, degrading gracefully on any failure -- never blocks the poll.

        DART is optional and secondary to Toss (see ``poll_once``): if no
        API key is configured (``settings.krx_news.dart_api_key`` empty --
        the case in this environment), DART is skipped entirely without
        even attempting the call. Otherwise, both the quota-specific
        ``DartQuotaExceededError`` and the general ``DartAPIError``
        (network errors, unexpected response shapes, etc. -- see
        ``dart_client.DartClient.poll_once``'s docstring) are caught the
        same way: logged, this cycle's disclosures skipped, Toss unaffected.
        """
        if not self._dart.api_key:
            logger.debug("DART API key not configured, skipping DART poll (Toss remains primary)")
            return []
        try:
            return await self._dart.poll_once()
        except DartQuotaExceededError as exc:
            logger.warning("DART quota exhausted, skipping this cycle's disclosures", error=str(exc))
            return []
        except DartAPIError as exc:
            logger.warning(
                "DART poll failed, degrading gracefully (Toss remains primary)", error=str(exc)
            )
            return []

    async def poll_forever(
        self,
        max_iterations: int | None = None,
    ) -> AsyncIterator[list[NewsArticle]]:
        """Poll on ``self.poll_interval``, yielding merged new articles each cycle."""
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            yield await self.poll_once()
            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                await asyncio.sleep(self.poll_interval)
