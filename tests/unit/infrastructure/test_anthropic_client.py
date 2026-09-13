"""Unit tests for the network-free parts of `anthropic_client.py` (prompt v2)."""

import json

import pytest

from daytrade_it.infrastructure.external.anthropic_client import (
    SIGNAL_THRESHOLD,
    ArticleFeatures,
    SchemaError,
    build_prompt,
    parse_features,
)

_VALID = {
    "subject": "main",
    "reports_price_move": False,
    "material_type": "수주",
    "persistence": "multi_quarter",
    "specificity": 3,
    "freshness": "new",
    "surprise": 1,
    "sentiment_direction": 1,
    "sentiment_strength": 2,
    "scale_vs_size": "medium",
}


def _features(**overrides: object) -> ArticleFeatures:
    return parse_features({**_VALID, **overrides})


def test_parse_features_happy_path() -> None:
    features = parse_features(json.dumps(_VALID, ensure_ascii=False))
    assert features.subject == "main"
    assert features.reports_price_move is False
    assert features.specificity == 3


def test_parse_features_tolerates_code_fence_and_string_values() -> None:
    raw = (
        "result:\n```json\n"
        + json.dumps(
            {**_VALID, "reports_price_move": "true", "specificity": "2"}, ensure_ascii=False
        )
        + "\n```"
    )
    features = parse_features(raw)
    assert features.reports_price_move is True
    assert features.specificity == 2


def test_parse_features_missing_field_raises() -> None:
    raw = {k: v for k, v in _VALID.items() if k != "subject"}
    with pytest.raises(SchemaError, match="Missing field: subject"):
        parse_features(raw)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("subject", "headline", "outside allowed set"),
        ("persistence", "unknown", "outside allowed set"),  # seen live from Haiku
        ("specificity", 9, "outside allowed range"),
        ("surprise", 3, "outside allowed range"),
        ("reports_price_move", "maybe", "not a bool"),
    ],
)
def test_parse_features_rejects_out_of_schema(field: str, value: object, match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        parse_features({**_VALID, field: value})


def test_build_prompt_names_the_company_and_forbids_trading_terms() -> None:
    system, user = build_prompt(
        code="088350",
        name="한화생명",
        market="KOSPI",
        sector="보험",
        title="t",
        body="b",
        category="breaking",
    )
    assert "한화생명(088350, KOSPI, 업종: 보험)" in user
    for term in (
        "수익률",
        "목표주가",
        "매수",
        "매도",
        "투자의견",
        "오를",
        "상승률",
        "유망",
        "추천",
    ):
        assert term not in system
        assert term not in user


def test_mention_scores_zero_regardless_of_sentiment() -> None:
    """Toss tags every company an article touches; on the eval dev split
    `mention` pairs had ~0 excess return, so they must never signal."""
    features = _features(subject="mention", sentiment_strength=3, persistence="structural")
    assert features.score() == 0.0
    assert features.to_sentiment().is_neutral


def test_repeat_freshness_halves_the_score() -> None:
    fresh = _features(freshness="new").score()
    repeat = _features(freshness="repeat").score()
    assert repeat == pytest.approx(fresh * 0.5)


def test_price_move_report_and_surprise_do_not_change_score() -> None:
    """Both were measured and not adopted (dev: dropping price-move reports
    hurt; surprise changed nothing). Guard against silently re-adding them."""
    base = _features().score()
    assert _features(reports_price_move=True).score() == base
    assert _features(surprise=2).score() == base


def test_strong_main_subject_material_is_bullish() -> None:
    features = _features(persistence="structural", scale_vs_size="large", sentiment_strength=3)
    sentiment = features.to_sentiment()
    assert sentiment.score > SIGNAL_THRESHOLD
    assert sentiment.is_bullish


def test_weak_positive_repeat_material_stays_neutral() -> None:
    features = _features(
        sentiment_strength=1,
        persistence="one_off",
        scale_vs_size="unknown",
        specificity=0,
        freshness="repeat",
    )
    assert features.score() <= SIGNAL_THRESHOLD
    assert features.to_sentiment().is_neutral


def test_fresh_strength_one_clears_threshold_as_evaluated() -> None:
    """1/3 * 0.5 = 0.167 > 0.15: the evaluated v2 rule lets a fresh,
    main-subject, strength-1 positive through. Pinned so a threshold change
    is a deliberate re-evaluation, not an accident."""
    features = _features(
        sentiment_strength=1, persistence="one_off", scale_vs_size="unknown", specificity=0
    )
    assert features.score() == pytest.approx(1 / 6)
    assert features.to_sentiment().is_bullish


def test_negative_direction_is_bearish() -> None:
    features = _features(sentiment_direction=-1, sentiment_strength=3)
    assert features.to_sentiment().is_bearish


def test_magnitude_bounds() -> None:
    weakest = _features(persistence="one_off", scale_vs_size="unknown", specificity=0)
    strongest = _features(persistence="structural", scale_vs_size="large", specificity=3)
    assert weakest.magnitude() == 0.0
    assert strongest.magnitude() == pytest.approx(1.0)


def test_production_prompt_is_the_evaluated_prompt() -> None:
    """The v2 prompt was measured via scripts/eval/eval_news_prompts.py.
    If production drifts from it, the eval no longer describes production."""
    import importlib.util
    import sys
    from pathlib import Path

    from daytrade_it.infrastructure.external import anthropic_client

    path = Path(__file__).resolve().parents[3] / "scripts" / "eval" / "eval_news_prompts.py"
    spec = importlib.util.spec_from_file_location("eval_news_prompts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_news_prompts"] = module
    spec.loader.exec_module(module)

    assert anthropic_client._SYSTEM == module.SYSTEM_V2
    assert anthropic_client._USER_TEMPLATE == module.USER_V2
    assert anthropic_client._USER_TEMPLATE_FULL_BODY == module.USER_V4
