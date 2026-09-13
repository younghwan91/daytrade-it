"""Anthropic (Claude) news-material scorer.

Ports the fixed-schema extraction design from the sibling repo `scalp-it`
(`src/scalp_it/score/{schema,prompt,provider}.py`) rather than reusing
`openai_client.py`'s "ask the LLM to output BULLISH/BEARISH directly"
design. scalp-it's approach is deliberately more disciplined: the LLM only
*extracts facts* about the article (material type, how persistent/specific/
fresh/large it is, and its direction+strength) under a prompt that
explicitly forbids trading-judgment language ("수익률", "매수", "목표주가",
etc. -- see scalp-it's `FORBIDDEN_PROMPT_TERMS`); a separate, deterministic
formula turns those facts into a score. That keeps the LLM call auditable
and re-scoreable without re-prompting, and avoids asking it to reason
about tradeability directly (a harder, less reliable ask than "what does
this specific fact mean").

Deliberate deviation from scalp-it: scalp-it feeds only the article's first
200 chars (`body200`) into the prompt, matching the 200-char BigKinds
snippets it originally trained scoring consistency against. daytrade-it has
no such legacy-training constraint, so this module accepts whatever text is
available (Toss currently gives ~300-char summaries only -- see
`toss_news_client.py` and 2026-09-13's session notes; full original-article
fetching was investigated and deferred, see that session's summary) under
the same field name (`body200`) the prompt template expects.

No Anthropic SDK dependency, mirroring scalp-it's `score/provider.py`: the
only thing used is "system + user -> JSON text", which is a ~20-line raw
httpx call, and pulling in a whole SDK for that isn't worth the install
weight for a public-facing repo.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any

import httpx

from daytrade_it.domain.entities.news import NewsArticle
from daytrade_it.domain.value_objects import Sentiment, SentimentLabel
from daytrade_it.shared.errors.exceptions import ExternalAPIError
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

# --- Fixed schema (ported from scalp-it's score/schema.py) -----------------

MATERIAL_TYPES: tuple[str, ...] = (
    "수주",
    "실적",
    "신약",
    "증설",
    "정책",
    "테마",
    "루머",
    "지분",
    "기타",
)
PERSISTENCE_LEVELS: tuple[str, ...] = ("one_off", "multi_quarter", "structural")
FRESHNESS_LEVELS: tuple[str, ...] = ("repeat", "follow_up", "new")
SCALE_LEVELS: tuple[str, ...] = ("unknown", "small", "medium", "large")

_CATEGORICAL = {
    "material_type": MATERIAL_TYPES,
    "persistence": PERSISTENCE_LEVELS,
    "freshness": FRESHNESS_LEVELS,
    "scale_vs_size": SCALE_LEVELS,
}
_NUMERIC = {
    "specificity": (0, 1, 2, 3),
    "sentiment_direction": (-1, 0, 1),
    "sentiment_strength": (0, 1, 2, 3),
}

#: Normalization divisors -- ported from scalp-it's `llm_material_score`
#: weights (persistence 0.5, scale 0.3, specificity 0.2) but re-derived
#: here as a [0, 1] magnitude (scalp-it feeds its weighted sum straight
#: into an ML model as a raw feature and never needs it bounded; this
#: module needs a bounded magnitude to build a domain `Sentiment.score`
#: in [-1, 1]). Divisor is each level's max encoded value.
_PERSISTENCE_SCORE = {"one_off": 0.0, "multi_quarter": 1.0, "structural": 2.0}
_FRESHNESS_SCORE = {"repeat": 0.0, "follow_up": 1.0, "new": 2.0}
_SCALE_SCORE = {"unknown": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
_MAGNITUDE_WEIGHTS = {"persistence": 0.5, "scale": 0.3, "specificity": 0.2}
_MAGNITUDE_MAX = (
    _MAGNITUDE_WEIGHTS["persistence"] * max(_PERSISTENCE_SCORE.values())
    + _MAGNITUDE_WEIGHTS["scale"] * max(_SCALE_SCORE.values())
    + _MAGNITUDE_WEIGHTS["specificity"] * max(_NUMERIC["specificity"])
)


class SchemaError(ValueError):
    """LLM output didn't match the fixed schema -- caller's cue to retry."""


@dataclass(frozen=True)
class ArticleFeatures:
    """The 7 fixed fields extracted from one article (scalp-it's schema)."""

    material_type: str
    persistence: str
    specificity: int
    freshness: str
    sentiment_direction: int
    sentiment_strength: int
    scale_vs_size: str

    def magnitude(self) -> float:
        """How large/durable/specific this material is, in [0, 1].

        Deliberately excludes direction -- this is "how much does it
        matter", combined with `sentiment_direction` separately in
        `to_sentiment()` to get a signed score.
        """
        raw = (
            _MAGNITUDE_WEIGHTS["persistence"] * _PERSISTENCE_SCORE[self.persistence]
            + _MAGNITUDE_WEIGHTS["scale"] * _SCALE_SCORE[self.scale_vs_size]
            + _MAGNITUDE_WEIGHTS["specificity"] * self.specificity
        )
        return raw / _MAGNITUDE_MAX

    def to_sentiment(self) -> Sentiment:
        """Map the fixed schema onto the domain `Sentiment` value object.

        - score: direction * magnitude, in [-1, 1] (direction from the
          article's own stated sentiment_direction/strength; magnitude
          from persistence/scale/specificity -- "how big and durable",
          not "how positive-sounding").
        - confidence: how concretely sourced the claim is (specificity)
          and how new it is (freshness) -- NOT how strong the sentiment
          is. A large but vague rumor should score low confidence even if
          `sentiment_strength` is high.
        """
        direction_score = self.sentiment_direction * (self.sentiment_strength / 3.0)
        score = max(-1.0, min(1.0, direction_score * (0.5 + 0.5 * self.magnitude())))

        confidence = max(
            0.0,
            min(
                1.0, 0.5 * (self.specificity / 3.0) + 0.5 * (_FRESHNESS_SCORE[self.freshness] / 2.0)
            ),
        )

        if score > 0.15:
            label = SentimentLabel.BULLISH
        elif score < -0.15:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL

        return Sentiment(score=score, label=label, confidence=confidence)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of `text` (tolerates fences/prose)."""
    start = text.find("{")
    if start < 0:
        raise SchemaError(f"No JSON object in response: {text[:120]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    result: dict[str, Any] = json.loads(text[start : i + 1])
                    return result
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"JSON parse failed: {exc}") from exc
    raise SchemaError(f"Unterminated JSON object: {text[:120]!r}")


def _as_int(name: str, value: object) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} is not an int: {value!r}") from exc


def parse_features(raw: str | dict[str, Any]) -> ArticleFeatures:
    payload = raw if isinstance(raw, dict) else _extract_json_object(raw)

    for field in fields(ArticleFeatures):
        if field.name not in payload:
            raise SchemaError(f"Missing field: {field.name}")

    values: dict[str, Any] = {}
    for name, allowed_strings in _CATEGORICAL.items():
        value = str(payload[name]).strip()
        if value not in allowed_strings:
            raise SchemaError(f"{name} outside allowed set: {value!r}")
        values[name] = value
    for name, allowed_numbers in _NUMERIC.items():
        number = _as_int(name, payload[name])
        if number not in allowed_numbers:
            raise SchemaError(f"{name} outside allowed range: {number}")
        values[name] = number

    return ArticleFeatures(**values)


# --- Prompt (ported from scalp-it's score/prompt.py) ------------------------

_SYSTEM = """너는 한국 상장사 기사에서 고정 스키마 필드를 채우는 추출기다.

규칙:
- 주어진 텍스트에 적힌 내용만 쓴다. 배경지식으로 보충하지 않는다.
- 이 작업에 시세 판단은 포함되지 않는다. 스키마에 없는 것은 답하지 않는다.
- 근거가 없으면 가장 보수적인 값(unknown 또는 0)을 고른다.
- 출력은 JSON 객체 하나뿐이다. 코드펜스·설명·주석을 붙이지 않는다."""

_USER_TEMPLATE = """다음은 종목코드 {code} 관련 기사다.

제목: {title}
본문: {body200}
키워드: {keywords}

아래 7개 필드를 채운 JSON 객체 하나만 출력하라.

material_type: 재료 유형. 다음 중 하나 — {material_types}
persistence: 실적에 반영되는 지속성. one_off(일회성 사건) / multi_quarter(여러 분기에 걸쳐 반영) / structural(사업 구조가 바뀌는 수준)
specificity: 구체성. 0(방향만 언급) / 1(대상과 시기 중 하나만 명시) / 2(둘 다 있으나 금액 없음) / 3(금액·기간·상대방이 모두 명시)
freshness: 정보 신선도. new(처음 알려지는 내용) / follow_up(기존 건의 진행 경과) / repeat(이미 알려진 내용의 재보도)
sentiment_direction: 이 사안이 회사 사업에 미치는 방향. -1(부정) / 0(중립) / 1(긍정)
sentiment_strength: 그 방향의 강도. 0(거의 없음) / 1(작음) / 2(보통) / 3(큼)
scale_vs_size: 회사 규모(시가총액·매출) 대비 사안의 크기. unknown(판단 근거가 본문에 없음) / small / medium / large

출력 형식 예시:
{{"material_type":"수주","persistence":"multi_quarter","specificity":3,"freshness":"new","sentiment_direction":1,"sentiment_strength":2,"scale_vs_size":"medium"}}"""


def build_prompt(*, code: str, title: str, body200: str, keywords: str = "") -> tuple[str, str]:
    user = _USER_TEMPLATE.format(
        code=code.strip(),
        title=title.strip(),
        body200=body200.strip(),
        keywords=keywords.strip(),
        material_types=" / ".join(MATERIAL_TYPES),
    )
    return _SYSTEM, user


# --- HTTP call ---------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_OUTPUT_TOKENS = 300
_MAX_RETRIES = 2


class AnthropicClient:
    """Minimal Anthropic Messages API client for article material-scoring.

    Raw httpx, no SDK (see module docstring). API key read from the
    `ANTHROPIC_API_KEY` env var if not passed explicitly -- never logged.
    """

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ExternalAPIError("ANTHROPIC_API_KEY not set")
        self._api_key = key
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AnthropicClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _complete(self, system: str, user: str) -> str:
        response = await self._client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self._model,
                "system": system,
                "temperature": 0,
                "max_tokens": _MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": user}],
            },
        )
        response.raise_for_status()
        payload = response.json()
        blocks = payload.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text

    async def score_article(
        self, *, code: str, title: str, body200: str, keywords: str = ""
    ) -> ArticleFeatures:
        """Score one article, retrying on a schema violation (not a network error)."""
        system, user = build_prompt(code=code, title=title, body200=body200, keywords=keywords)

        last_error: SchemaError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            text = await self._complete(system, user)
            try:
                return parse_features(text)
            except SchemaError as exc:
                last_error = exc
                logger.warning(
                    "anthropic_client: schema violation, retrying",
                    attempt=attempt,
                    error=str(exc),
                )
        raise ExternalAPIError(
            f"Article scoring failed schema validation after {_MAX_RETRIES + 1} attempts: {last_error}"
        )

    async def analyze_sentiment(self, article: NewsArticle) -> Sentiment:
        """Score `article` and map the result onto a domain `Sentiment`.

        Uses `article.summary or article.content` as the scored text --
        Toss currently supplies only a ~300-char summary (see module
        docstring), not a full article body.
        """
        body = article.summary or article.content
        ticker = article.tickers[0].value if article.tickers else ""
        features = await self.score_article(
            code=ticker,
            title=article.title,
            body200=body,
            keywords=article.category or "",
        )
        return features.to_sentiment()
