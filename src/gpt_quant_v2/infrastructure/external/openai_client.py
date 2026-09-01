"""OpenAI client for sentiment analysis with async httpx support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gpt_quant_v2.domain.value_objects import Sentiment, SentimentLabel
from gpt_quant_v2.infrastructure.external.prompts import (
    SENTIMENT_ANALYSIS_SCHEMA,
    SENTIMENT_SYSTEM_PROMPT,
    format_sentiment_user_prompt,
)
from gpt_quant_v2.shared.errors import OpenAIAPIError, error_codes
from gpt_quant_v2.shared.logging import get_logger

if TYPE_CHECKING:
    from gpt_quant_v2.domain.entities import NewsArticle
    from gpt_quant_v2.shared.config import Settings

logger = get_logger(__name__)

# Token pricing per 1M tokens (as of 2024)
TOKEN_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

# Average characters per token (rough estimate)
CHARS_PER_TOKEN = 4


class SentimentAnalysis(BaseModel):
    """Response model for sentiment analysis results."""

    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: list[str]
    reasoning: str

    @field_validator("sentiment_label")
    @classmethod
    def validate_sentiment_label(cls, v: str) -> str:
        """Validate sentiment label is valid."""
        valid_labels = {"BULLISH", "BEARISH", "NEUTRAL"}
        upper_v = v.upper()
        if upper_v not in valid_labels:
            raise ValueError(f"Invalid sentiment label: {v}. Must be one of: {valid_labels}")
        return upper_v

    def to_sentiment(self) -> Sentiment:
        """Convert to domain Sentiment value object."""
        return Sentiment(
            score=self.sentiment_score,
            label=SentimentLabel(self.sentiment_label),
            confidence=self.confidence,
        )


@dataclass
class UsageStats:
    """Token usage and cost tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class OpenAIClientConfig:
    """Configuration for OpenAI client."""

    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3
    base_url: str = "https://api.openai.com/v1"


@dataclass
class RateLimitState:
    """Track rate limit state from response headers."""

    requests_remaining: int | None = None
    tokens_remaining: int | None = None
    reset_requests_at: datetime | None = None
    reset_tokens_at: datetime | None = None

    def is_rate_limited(self) -> bool:
        """Check if we're likely rate limited."""
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            return True
        return bool(self.tokens_remaining is not None and self.tokens_remaining <= 0)


class OpenAIClient:
    """Async OpenAI client for sentiment analysis using httpx.

    This client provides:
    - Async HTTP requests with proper timeout handling
    - Structured output using function calling
    - Retry logic with exponential backoff
    - Rate limiting awareness
    - Token counting and cost estimation
    - Comprehensive error handling and logging
    """

    def __init__(
        self,
        config: OpenAIClientConfig | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the OpenAI client.

        Args:
            config: Direct configuration object
            settings: Application settings (used if config not provided)

        Raises:
            ConfigurationError: If API key is not configured
        """
        if config:
            self._config = config
        elif settings:
            self._config = OpenAIClientConfig(
                api_key=settings.openai.api_key.get_secret_value(),
                model=settings.openai.model,
                temperature=settings.openai.temperature,
                max_tokens=settings.openai.max_tokens,
                timeout=settings.openai.timeout,
            )
        else:
            from gpt_quant_v2.shared.config import get_settings

            s = get_settings()
            self._config = OpenAIClientConfig(
                api_key=s.openai.api_key.get_secret_value(),
                model=s.openai.model,
                temperature=s.openai.temperature,
                max_tokens=s.openai.max_tokens,
                timeout=s.openai.timeout,
            )

        if not self._config.api_key:
            from gpt_quant_v2.shared.errors import ConfigurationError

            raise ConfigurationError(
                "OpenAI API key is not configured",
                details={"setting": "GPTQ_OPENAI_API_KEY"},
            )

        self._client: httpx.AsyncClient | None = None
        self._rate_limit_state = RateLimitState()
        self._total_usage = UsageStats()

        logger.info(
            "OpenAI client initialized",
            model=self._config.model,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

    @property
    def config(self) -> OpenAIClientConfig:
        """Get client configuration."""
        return self._config

    @property
    def rate_limit_state(self) -> RateLimitState:
        """Get current rate limit state."""
        return self._rate_limit_state

    @property
    def total_usage(self) -> UsageStats:
        """Get total usage statistics."""
        return self._total_usage

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout),
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.debug("OpenAI client closed")

    async def __aenter__(self) -> OpenAIClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is a rough estimate based on character count.
        For more accurate counting, use tiktoken library.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        return max(1, len(text) // CHARS_PER_TOKEN)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD for token usage.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        model = self._config.model
        pricing = TOKEN_PRICING.get(model, TOKEN_PRICING.get("gpt-4o-mini", {}))

        input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0.15)
        output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0.60)

        return input_cost + output_cost

    def _update_rate_limit_state(self, headers: httpx.Headers) -> None:
        """Update rate limit state from response headers."""
        if "x-ratelimit-remaining-requests" in headers:
            self._rate_limit_state.requests_remaining = int(
                headers["x-ratelimit-remaining-requests"]
            )
        if "x-ratelimit-remaining-tokens" in headers:
            self._rate_limit_state.tokens_remaining = int(headers["x-ratelimit-remaining-tokens"])
        if "x-ratelimit-reset-requests" in headers:
            reset_str = headers["x-ratelimit-reset-requests"]
            # Parse reset time (format varies)
            try:
                if reset_str.endswith("s"):
                    seconds = float(reset_str[:-1])
                    self._rate_limit_state.reset_requests_at = datetime.now(UTC) + __import__(
                        "datetime"
                    ).timedelta(seconds=seconds)
            except (ValueError, TypeError):
                pass

    def _update_usage_stats(self, usage: dict[str, int]) -> UsageStats:
        """Update and return usage statistics."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Update totals
        self._total_usage.prompt_tokens += prompt_tokens
        self._total_usage.completion_tokens += completion_tokens
        self._total_usage.total_tokens += total_tokens
        self._total_usage.estimated_cost_usd += self.estimate_cost(prompt_tokens, completion_tokens)

        return UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self.estimate_cost(prompt_tokens, completion_tokens),
        )

    def _should_retry(self, status_code: int) -> bool:
        """Determine if request should be retried based on status code."""
        # Retry on rate limits and server errors
        return status_code in {429, 500, 502, 503, 504}

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _make_request(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Make a chat completion request with retry logic.

        Args:
            messages: Chat messages
            tools: Optional function definitions

        Returns:
            API response data

        Raises:
            OpenAIAPIError: On API errors
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

        # Use structured outputs via tools
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tools[0]["function"]["name"]},
            }

        logger.debug(
            "Making OpenAI request",
            model=self._config.model,
            message_count=len(messages),
            has_tools=bool(tools),
        )

        try:
            response = await client.post("/chat/completions", json=payload)
            self._update_rate_limit_state(response.headers)

            if response.status_code == 429:
                retry_after = response.headers.get("retry-after", "60")
                logger.warning(
                    "Rate limited by OpenAI",
                    retry_after=retry_after,
                    requests_remaining=self._rate_limit_state.requests_remaining,
                )
                raise OpenAIAPIError(
                    "OpenAI rate limit exceeded",
                    error_code=error_codes.OPENAI_RATE_LIMIT,
                    status_code=429,
                    model=self._config.model,
                    details={"retry_after": retry_after},
                )

            if response.status_code != 200:
                error_body = response.text
                logger.error(
                    "OpenAI API error",
                    status_code=response.status_code,
                    response_body=error_body[:500],
                )
                raise OpenAIAPIError(
                    f"OpenAI API request failed with status {response.status_code}",
                    status_code=response.status_code,
                    response_body=error_body,
                    model=self._config.model,
                )

            data: dict[str, Any] = response.json()
            return data

        except httpx.TimeoutException as e:
            logger.error("OpenAI request timeout", timeout=self._config.timeout)
            raise OpenAIAPIError(
                "OpenAI request timed out",
                model=self._config.model,
                details={"timeout": self._config.timeout},
            ) from e
        except httpx.ConnectError as e:
            logger.error("OpenAI connection error", error=str(e))
            raise OpenAIAPIError(
                "Failed to connect to OpenAI API",
                model=self._config.model,
            ) from e

    async def analyze_sentiment(
        self,
        article: NewsArticle,
    ) -> SentimentAnalysis:
        """Analyze sentiment of a news article.

        Args:
            article: NewsArticle entity to analyze

        Returns:
            SentimentAnalysis with score, label, confidence, key points, and reasoning

        Raises:
            OpenAIAPIError: On API or parsing errors
        """
        # Format prompts
        tickers = [str(t) for t in article.tickers] if article.tickers else None
        user_prompt = format_sentiment_user_prompt(
            title=article.title,
            content=article.content,
            source=article.source,
            published_at=article.published_at.isoformat(),
            tickers=tickers,
        )

        messages = [
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Define tool for structured output
        tools = [
            {
                "type": "function",
                "function": SENTIMENT_ANALYSIS_SCHEMA,
            }
        ]

        # Estimate input tokens
        input_text = SENTIMENT_SYSTEM_PROMPT + user_prompt
        estimated_input_tokens = self.estimate_tokens(input_text)

        logger.info(
            "Analyzing article sentiment",
            article_id=str(article.id),
            title=article.title[:50],
            estimated_input_tokens=estimated_input_tokens,
        )

        try:
            response_data = await self._make_request(messages, tools)
        except RetryError as e:
            logger.error(
                "OpenAI request failed after retries",
                article_id=str(article.id),
                error=str(e),
            )
            raise OpenAIAPIError(
                "OpenAI request failed after multiple retries",
                model=self._config.model,
            ) from e

        # Update usage stats
        usage = response_data.get("usage", {})
        stats = self._update_usage_stats(usage)

        logger.debug(
            "OpenAI response received",
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            estimated_cost_usd=f"${stats.estimated_cost_usd:.6f}",
        )

        # Parse response
        try:
            choice = response_data["choices"][0]
            message = choice["message"]

            # Extract function call result
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                arguments = json.loads(tool_call["function"]["arguments"])
            elif "content" in message and message["content"]:
                # Fallback to parsing content as JSON
                arguments = json.loads(message["content"])
            else:
                raise OpenAIAPIError(
                    "No valid response content from OpenAI",
                    error_code=error_codes.OPENAI_INVALID_RESPONSE,
                    model=self._config.model,
                )

            # Validate and create response model
            analysis = SentimentAnalysis(**arguments)

            logger.info(
                "Sentiment analysis complete",
                article_id=str(article.id),
                sentiment_label=analysis.sentiment_label,
                sentiment_score=analysis.sentiment_score,
                confidence=analysis.confidence,
            )

            return analysis

        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse OpenAI response JSON",
                article_id=str(article.id),
                error=str(e),
            )
            raise OpenAIAPIError(
                "Failed to parse OpenAI response as JSON",
                error_code=error_codes.OPENAI_INVALID_RESPONSE,
                model=self._config.model,
            ) from e
        except (KeyError, IndexError) as e:
            logger.error(
                "Unexpected OpenAI response structure",
                article_id=str(article.id),
                error=str(e),
            )
            raise OpenAIAPIError(
                "Unexpected response structure from OpenAI",
                error_code=error_codes.OPENAI_INVALID_RESPONSE,
                model=self._config.model,
            ) from e
        except ValueError as e:
            logger.error(
                "Invalid sentiment analysis values",
                article_id=str(article.id),
                error=str(e),
            )
            raise OpenAIAPIError(
                f"Invalid sentiment analysis response: {e}",
                error_code=error_codes.OPENAI_INVALID_RESPONSE,
                model=self._config.model,
            ) from e


__all__ = [
    "OpenAIClient",
    "OpenAIClientConfig",
    "SentimentAnalysis",
    "UsageStats",
    "RateLimitState",
]
