"""Unit tests for the Anthropic material-score schema/prompt/scoring logic
(the pure, network-free parts of `anthropic_client.py`).
"""

import pytest

from daytrade_it.infrastructure.external.anthropic_client import (
    ArticleFeatures,
    SchemaError,
    build_prompt,
    parse_features,
)


def test_parse_features_happy_path() -> None:
    raw = (
        '{"material_type":"수주","persistence":"multi_quarter","specificity":3,'
        '"freshness":"new","sentiment_direction":1,"sentiment_strength":2,'
        '"scale_vs_size":"medium"}'
    )
    features = parse_features(raw)
    assert features.material_type == "수주"
    assert features.persistence == "multi_quarter"
    assert features.specificity == 3
    assert features.sentiment_direction == 1


def test_parse_features_tolerates_code_fence_and_prose() -> None:
    raw = (
        "Here is the result:\n```json\n"
        '{"material_type":"기타","persistence":"one_off","specificity":0,'
        '"freshness":"repeat","sentiment_direction":0,"sentiment_strength":0,'
        '"scale_vs_size":"unknown"}\n```\ndone.'
    )
    features = parse_features(raw)
    assert features.material_type == "기타"


def test_parse_features_missing_field_raises() -> None:
    raw = '{"material_type":"기타","persistence":"one_off"}'
    with pytest.raises(SchemaError, match="Missing field"):
        parse_features(raw)


def test_parse_features_rejects_value_outside_allowed_set() -> None:
    raw = (
        '{"material_type":"NOT_A_TYPE","persistence":"one_off","specificity":0,'
        '"freshness":"repeat","sentiment_direction":0,"sentiment_strength":0,'
        '"scale_vs_size":"unknown"}'
    )
    with pytest.raises(SchemaError, match="outside allowed set"):
        parse_features(raw)


def test_parse_features_rejects_number_outside_range() -> None:
    raw = (
        '{"material_type":"기타","persistence":"one_off","specificity":9,'
        '"freshness":"repeat","sentiment_direction":0,"sentiment_strength":0,'
        '"scale_vs_size":"unknown"}'
    )
    with pytest.raises(SchemaError, match="outside allowed range"):
        parse_features(raw)


def test_build_prompt_forbids_trading_judgment_terms() -> None:
    """Same discipline as scalp-it's FORBIDDEN_PROMPT_TERMS test -- the
    prompt must never ask the LLM to judge tradeability directly."""
    system, user = build_prompt(code="005930", title="t", body200="b")
    forbidden = ("수익률", "목표주가", "매수", "매도", "투자의견", "오를", "상승률", "유망", "추천")
    for term in forbidden:
        assert term not in system
        assert term not in user


def test_magnitude_zero_for_weakest_features() -> None:
    weakest = ArticleFeatures(
        material_type="기타",
        persistence="one_off",
        specificity=0,
        freshness="repeat",
        sentiment_direction=0,
        sentiment_strength=0,
        scale_vs_size="unknown",
    )
    assert weakest.magnitude() == 0.0


def test_magnitude_one_for_strongest_features() -> None:
    strongest = ArticleFeatures(
        material_type="실적",
        persistence="structural",
        specificity=3,
        freshness="new",
        sentiment_direction=1,
        sentiment_strength=3,
        scale_vs_size="large",
    )
    assert strongest.magnitude() == pytest.approx(1.0)


def test_to_sentiment_strong_positive_material_is_bullish_high_confidence() -> None:
    features = ArticleFeatures(
        material_type="수주",
        persistence="structural",
        specificity=3,
        freshness="new",
        sentiment_direction=1,
        sentiment_strength=3,
        scale_vs_size="large",
    )
    sentiment = features.to_sentiment()
    assert sentiment.score > 0.5
    assert sentiment.is_bullish
    assert sentiment.confidence > 0.8


def test_to_sentiment_vague_rumor_has_low_confidence_despite_strong_sentiment() -> None:
    """A large-sounding but unspecific, stale rumor must not produce a
    confident signal -- confidence tracks specificity/freshness, not the
    raw strength of the claimed sentiment."""
    features = ArticleFeatures(
        material_type="루머",
        persistence="one_off",
        specificity=0,
        freshness="repeat",
        sentiment_direction=1,
        sentiment_strength=3,
        scale_vs_size="unknown",
    )
    sentiment = features.to_sentiment()
    assert sentiment.confidence < 0.2


def test_to_sentiment_neutral_direction_is_neutral_label() -> None:
    features = ArticleFeatures(
        material_type="기타",
        persistence="one_off",
        specificity=1,
        freshness="follow_up",
        sentiment_direction=0,
        sentiment_strength=0,
        scale_vs_size="small",
    )
    sentiment = features.to_sentiment()
    assert sentiment.score == 0.0
    assert sentiment.is_neutral


def test_to_sentiment_negative_direction_is_bearish() -> None:
    features = ArticleFeatures(
        material_type="정책",
        persistence="multi_quarter",
        specificity=3,
        freshness="new",
        sentiment_direction=-1,
        sentiment_strength=2,
        scale_vs_size="large",
    )
    sentiment = features.to_sentiment()
    assert sentiment.score < 0
    assert sentiment.is_bearish
