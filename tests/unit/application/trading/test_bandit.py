"""Bayesian contextual bandit: features, NIG posterior, decision."""

import datetime as dt
import json
from decimal import Decimal

import numpy as np
import pytest

from daytrade_it.application.trading import bandit
from daytrade_it.application.trading.bandit import (
    FEATURES,
    BanditModel,
    decide,
    features_for,
    fit,
    student_t_cdf,
    update,
)
from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures

FEAT = ArticleFeatures(
    subject="main",
    reports_price_move=False,
    material_type="수주",
    persistence="structural",
    specificity=3,
    freshness="new",
    surprise=2,
    sentiment_direction=1,
    sentiment_strength=3,
    scale_vs_size="large",
)
DAY = dt.date(2026, 9, 15)
EXPIRES = dt.datetime(2026, 9, 15, 15, 20, tzinfo=KST)


def _bar(h: int, m: int, close: str) -> MinuteBar:
    c = Decimal(close)
    return MinuteBar(dt.datetime(2026, 9, 15, h, m), c, c, c, c, 100)


def test_features_list_is_the_design() -> None:
    assert FEATURES == (
        "score",
        "surprise",
        "specificity",
        "sentiment_strength",
        "persistence_one_off",
        "persistence_multi_quarter",
        "persistence_structural",
        "subject_main",
        "pre_entry_move",
        "premarket",
        "minutes_since_open",
        "ret_5m",
        "mkt_move",
    )


def test_features_for_values_and_no_look_ahead() -> None:
    decided = dt.datetime(2026, 9, 15, 9, 10, 30, tzinfo=KST)
    bars = [_bar(8, 59, "50")]  # pre-open bar: ignored
    bars += [_bar(9, m, str(100 + m)) for m in range(10)]  # 09:00..09:09 closes 100..109
    bars.append(_bar(9, 10, "999"))  # current minute: not complete, must be ignored
    x = features_for(
        FEAT,
        0.02,
        dt.datetime(2026, 9, 15, 8, 30, tzinfo=KST),
        EXPIRES,
        decided,
        bars,
        Decimal("101"),
        Decimal("100"),
    )
    assert set(x) == set(FEATURES)
    assert x["score"] == pytest.approx(FEAT.score())
    assert (x["surprise"], x["specificity"], x["sentiment_strength"]) == (2.0, 3.0, 3.0)
    assert (
        x["persistence_one_off"],
        x["persistence_multi_quarter"],
        x["persistence_structural"],
    ) == (0.0, 0.0, 1.0)
    assert x["subject_main"] == 1.0
    assert x["pre_entry_move"] == pytest.approx(0.02)
    assert x["premarket"] == 1.0
    assert x["minutes_since_open"] == pytest.approx(10.5)
    # last 5 completed closes: 105..109
    assert x["ret_5m"] == pytest.approx(109 / 105 - 1)
    assert x["mkt_move"] == pytest.approx(0.01)


def test_features_for_fallbacks() -> None:
    decided = dt.datetime(2026, 9, 15, 8, 50, tzinfo=KST)
    x = features_for(
        FEAT,
        0.0,
        dt.datetime(2026, 9, 15, 9, 5, tzinfo=KST),
        EXPIRES,
        decided,
        [_bar(9, m, "100") for m in range(3)],
        None,
        Decimal("100"),
    )
    assert x["minutes_since_open"] == 0.0
    assert x["ret_5m"] == 0.0
    assert x["mkt_move"] == 0.0
    assert x["premarket"] == 0.0
    # fewer than 5 completed regular-session bars
    x = features_for(
        FEAT,
        0.0,
        EXPIRES,
        EXPIRES,
        dt.datetime(2026, 9, 15, 9, 4, tzinfo=KST),
        [_bar(9, m, str(100 + m)) for m in range(5)],
        Decimal("100"),
        None,
    )
    assert x["ret_5m"] == 0.0
    assert x["mkt_move"] == 0.0


def test_student_t_cdf_matches_known_values() -> None:
    assert student_t_cdf(0.0, 3.0) == pytest.approx(0.5)
    assert student_t_cdf(1.0, 1.0) == pytest.approx(0.75)  # Cauchy
    assert student_t_cdf(2.0, 2.0) == pytest.approx(0.5 + 2 / (2 * np.sqrt(6)))
    assert student_t_cdf(-2.0, 2.0) == pytest.approx(0.5 - 2 / (2 * np.sqrt(6)))
    assert student_t_cdf(1.959964, 1e7) == pytest.approx(0.975, abs=1e-5)


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    y = 0.01 * X[:, 0] + rng.normal(scale=0.003, size=n)
    return X, y


def _row(v: float) -> dict[str, float]:
    return {name: (v if i == 0 else 0.0) for i, name in enumerate(FEATURES)}


def test_fit_predict_learns_the_driving_feature() -> None:
    X, y = _synthetic(300, 0)
    model = fit(X, y, lam=1.0)
    m_pos, p_pos = model.predict(_row(1.0))
    m_neg, p_neg = model.predict(_row(-1.0))
    assert m_pos > 0.007 and p_pos > 0.9
    assert m_neg < -0.007 and p_neg < 0.1
    assert model.a == pytest.approx(1.0 + 150)


def test_update_equals_fit_on_concatenated_data() -> None:
    X1, y1 = _synthetic(80, 1)
    X2, y2 = _synthetic(40, 2)
    m1 = fit(X1, y1, lam=3.0)
    seq = update(m1, X2, y2)
    full = fit(np.vstack([X1, X2]), np.concatenate([y1, y2]), lam=3.0, means=m1.means, stds=m1.stds)
    np.testing.assert_allclose(seq.w, full.w, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(seq.V, full.V, rtol=1e-8, atol=1e-12)
    assert seq.a == pytest.approx(full.a)
    assert seq.b == pytest.approx(full.b, rel=1e-8)


def test_json_round_trip() -> None:
    X, y = _synthetic(50, 3)
    model = fit(X, y, lam=10.0, min_prob=0.6, min_edge=0.002, trained_days=("2026-09-14",))
    back = BanditModel.from_json(model.to_json())
    assert back.feature_names == model.feature_names
    np.testing.assert_array_equal(back.w, model.w)
    np.testing.assert_array_equal(back.V, model.V)
    assert (back.a, back.b, back.lam, back.min_prob, back.min_edge, back.trained_days) == (
        model.a,
        model.b,
        model.lam,
        model.min_prob,
        model.min_edge,
        model.trained_days,
    )
    assert back.predict(_row(0.5)) == model.predict(_row(0.5))
    json.loads(model.to_json())


def test_decide_thresholds() -> None:
    X, y = _synthetic(300, 4)
    model = fit(X, y, lam=1.0, min_prob=0.6, min_edge=0.005)
    enter, m, p = decide(model, _row(1.0))
    assert enter and m >= 0.005 and p >= 0.6
    enter, m, _ = decide(model, _row(0.3))  # p high but m ~0.003 < min_edge
    assert not enter and 0 < m < 0.005
    assert not decide(model, _row(-1.0))[0]


def test_load_model_caches_by_path_and_mtime(tmp_path) -> None:  # noqa: ANN001
    X, y = _synthetic(50, 5)
    path = tmp_path / "m.json"
    path.write_text(fit(X, y, lam=1.0).to_json())
    a = bandit.load_model(str(path))
    assert bandit.load_model(str(path)) is a
    import os

    path.write_text(fit(X, y, lam=30.0).to_json())
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    b = bandit.load_model(str(path))
    assert b is not a and b.lam == 30.0
