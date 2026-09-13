"""Anthropic (Claude) news-material scorer.

Fixed-schema fact extraction, ported from scalp-it's `score/` package: the
LLM only fills fields about the article, under a prompt with no trading-
judgment language; a deterministic formula turns the fields into a score.

Prompt `v2` (2026-09-13) is the version that was measured against real
outcomes -- see `docs/research/2026-09-13-news-prompt-eval.md` and
`scripts/eval/eval_news_prompts.py`, which holds the identical prompt text.
Changes from the scalp-it port (`v1`), each backed by that eval:

- The prompt names the company (name, market, sector), not just the
  6-digit code. The model cannot reliably tell that 088350 is 한화생명.
- `subject` (main/partial/mention): Toss tags every company an article
  touches. On the dev split, `mention` pairs had ~0 excess return vs +2.2%
  for `main`; they score 0.
- `reports_price_move` and `surprise` are recorded but NOT used in the
  score -- dropping price-move reports hurt on dev, and surprise changed
  nothing. Kept for later analysis.
- A direct "how will it trade" prompt (v3) scored worse than the
  all-articles baseline, so judgment stays in code, not in the prompt.

Toss gives only ~300-char summaries; full article text isn't fetched.
No SDK dependency: a raw httpx call, same as scalp-it's provider.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any

import httpx

from daytrade_it.domain.value_objects import Sentiment, SentimentLabel
from daytrade_it.shared.errors.exceptions import ExternalAPIError
from daytrade_it.shared.logging import get_logger

logger = get_logger(__name__)

PROMPT_VERSION = "v2"

# --- Fixed schema -----------------------------------------------------------

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
SUBJECT_LEVELS: tuple[str, ...] = ("main", "partial", "mention")
PERSISTENCE_LEVELS: tuple[str, ...] = ("one_off", "multi_quarter", "structural")
FRESHNESS_LEVELS: tuple[str, ...] = ("repeat", "follow_up", "new")
SCALE_LEVELS: tuple[str, ...] = ("unknown", "small", "medium", "large")

_CATEGORICAL = {
    "subject": SUBJECT_LEVELS,
    "material_type": MATERIAL_TYPES,
    "persistence": PERSISTENCE_LEVELS,
    "freshness": FRESHNESS_LEVELS,
    "scale_vs_size": SCALE_LEVELS,
}
_NUMERIC = {
    "specificity": (0, 1, 2, 3),
    "surprise": (0, 1, 2),
    "sentiment_direction": (-1, 0, 1),
    "sentiment_strength": (0, 1, 2, 3),
}
_BOOLEAN = ("reports_price_move",)

_PERSISTENCE_SCORE = {"one_off": 0.0, "multi_quarter": 1.0, "structural": 2.0}
_FRESHNESS_SCORE = {"repeat": 0.0, "follow_up": 1.0, "new": 2.0}
_SCALE_SCORE = {"unknown": 0.0, "small": 1.0, "medium": 2.0, "large": 3.0}
_MAGNITUDE_WEIGHTS = {"persistence": 0.5, "scale": 0.3, "specificity": 0.2}
_MAGNITUDE_MAX = (
    _MAGNITUDE_WEIGHTS["persistence"] * max(_PERSISTENCE_SCORE.values())
    + _MAGNITUDE_WEIGHTS["scale"] * max(_SCALE_SCORE.values())
    + _MAGNITUDE_WEIGHTS["specificity"] * max(_NUMERIC["specificity"])
)

#: |score| above this is BULLISH/BEARISH. It is also the BUY threshold the
#: eval used, which works because `SignalGenerator.generate_from_news()`
#: returns nothing for a NEUTRAL sentiment.
SIGNAL_THRESHOLD = 0.15


class SchemaError(ValueError):
    """LLM output didn't match the fixed schema -- caller's cue to retry."""


@dataclass(frozen=True)
class ArticleFeatures:
    """Fields extracted from one (article, company) pair."""

    subject: str
    reports_price_move: bool
    material_type: str
    persistence: str
    specificity: int
    freshness: str
    surprise: int
    sentiment_direction: int
    sentiment_strength: int
    scale_vs_size: str

    def magnitude(self) -> float:
        """How large/durable/specific the material is, in [0, 1]."""
        raw = (
            _MAGNITUDE_WEIGHTS["persistence"] * _PERSISTENCE_SCORE[self.persistence]
            + _MAGNITUDE_WEIGHTS["scale"] * _SCALE_SCORE[self.scale_vs_size]
            + _MAGNITUDE_WEIGHTS["specificity"] * self.specificity
        )
        return raw / _MAGNITUDE_MAX

    def score(self) -> float:
        """Signed material score in [-1, 1] -- the evaluated v2 rule."""
        if self.subject == "mention":
            return 0.0
        s = self.sentiment_direction * (self.sentiment_strength / 3.0)
        s *= 0.5 + 0.5 * self.magnitude()
        if self.freshness == "repeat":
            s *= 0.5
        return max(-1.0, min(1.0, s))

    def to_sentiment(self) -> Sentiment:
        """Map onto the domain `Sentiment`.

        `confidence` is informational only (specificity + freshness). The
        poller runs `SignalGenerator` with `min_confidence=0` so that the
        BUY set is exactly `score() > SIGNAL_THRESHOLD`, which is what was
        measured -- a second confidence gate was never evaluated.
        """
        score = self.score()
        confidence = 0.5 * (self.specificity / 3.0) + 0.5 * (_FRESHNESS_SCORE[self.freshness] / 2.0)
        if score > SIGNAL_THRESHOLD:
            label = SentimentLabel.BULLISH
        elif score < -SIGNAL_THRESHOLD:
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


def _as_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "false"):
        return text == "true"
    raise SchemaError(f"{name} is not a bool: {value!r}")


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
    for name in _BOOLEAN:
        values[name] = _as_bool(name, payload[name])

    return ArticleFeatures(**values)


# --- Prompt (identical to USER_V2 in scripts/eval/eval_news_prompts.py) -----

_SYSTEM = """너는 한국 상장사 기사에서 고정 스키마 필드를 채우는 추출기다.

규칙:
- 주어진 텍스트에 적힌 내용만 쓴다. 배경지식으로 보충하지 않는다.
- 이 작업에 시세 판단은 포함되지 않는다. 스키마에 없는 것은 답하지 않는다.
- 근거가 없으면 가장 보수적인 값(unknown 또는 0)을 고른다.
- 출력은 JSON 객체 하나뿐이다. 코드펜스·설명·주석을 붙이지 않는다."""

_USER_TEMPLATE = """다음은 {name}({code}, {market}, 업종: {sector}) 관련으로 분류된 기사다.

제목: {title}
본문(요약): {body}
분류: {category}

아래 필드를 채운 JSON 객체 하나만 출력하라. 모든 판단은 **{name}** 기준이다.

subject: 이 기사에서 {name}의 비중. main(기사의 주인공) / partial(여러 회사 중 하나로 비중 있게 다룸) / mention(업계 나열·지나가는 언급·다른 회사가 주인공)
reports_price_move: 기사가 {name}의 **이미 일어난** 주가 움직임(급등·급락·상한가·하한가·신고가·장 마감 시세)을 주요 내용으로 보도하는가. true / false
material_type: 재료 유형. 다음 중 하나 — {material_types}
persistence: 실적에 반영되는 지속성. one_off / multi_quarter / structural
specificity: 구체성. 0(방향만) / 1(대상·시기 중 하나) / 2(둘 다, 금액 없음) / 3(금액·기간·상대방 모두)
freshness: new(처음 알려짐) / follow_up(기존 건 진행 경과) / repeat(이미 알려진 내용의 재보도·해설·칼럼)
surprise: 이 내용이 기존 기대와 얼마나 다른가. 0(예상된 일·일정대로) / 1(다소 뜻밖) / 2(크게 뜻밖)
sentiment_direction: {name} 사업에 미치는 방향. -1 / 0 / 1
sentiment_strength: 그 방향의 강도. 0 / 1 / 2 / 3
scale_vs_size: {name} 규모 대비 사안의 크기. unknown / small / medium / large

규칙: 본문에 없는 내용은 추정하지 않는다. 인터뷰·칼럼·업계 동향 기사는 대체로 repeat 이고 강도가 낮다.

출력 형식 예시:
{{"subject":"main","reports_price_move":false,"material_type":"수주","persistence":"multi_quarter","specificity":3,"freshness":"new","surprise":1,"sentiment_direction":1,"sentiment_strength":2,"scale_vs_size":"medium"}}"""


#: Shadow variant v4: identical questions, full article body instead of the
#: dashboard summary. Scored and recorded alongside v2 but never used for
#: signals -- on the 2026-09-13 eval the dev and test splits disagreed on
#: whether full text helps (docs/research/2026-09-13-news-prompt-eval.md), so
#: it is being compared on forward data first.
_USER_TEMPLATE_FULL_BODY = _USER_TEMPLATE.replace("본문(요약): {body}", "본문: {body}")


def build_prompt(
    *,
    code: str,
    name: str,
    market: str | None,
    sector: str | None,
    title: str,
    body: str,
    category: str | None,
    full_body: bool = False,
) -> tuple[str, str]:
    template = _USER_TEMPLATE_FULL_BODY if full_body else _USER_TEMPLATE
    user = template.format(
        code=code.strip(),
        name=name.strip(),
        market=market or "-",
        sector=sector or "-",
        title=title.strip(),
        body=body.strip(),
        category=category or "",
        material_types=" / ".join(MATERIAL_TYPES),
    )
    return _SYSTEM, user


# --- HTTP call ---------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_OUTPUT_TOKENS = 400
_MAX_RETRIES = 2


class AnthropicClient:
    """Minimal Anthropic Messages API client for article material-scoring.

    API key read from `ANTHROPIC_API_KEY` if not passed -- never logged.
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
        blocks = response.json().get("content") or []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    async def score_article(
        self,
        *,
        code: str,
        name: str,
        market: str | None,
        sector: str | None,
        title: str,
        body: str,
        category: str | None,
        full_body: bool = False,
    ) -> ArticleFeatures:
        """Score one (article, company) pair, retrying on schema violations."""
        system, user = build_prompt(
            code=code,
            name=name,
            market=market,
            sector=sector,
            title=title,
            body=body,
            category=category,
            full_body=full_body,
        )
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
