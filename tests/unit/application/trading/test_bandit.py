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
    # 4 completed bars (09:00..09:03; the 09:04 bar is the current minute)
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
    assert x["ret_5m"] == pytest.approx(103 / 100 - 1)
    assert x["mkt_move"] == 0.0


def _ret(bars: list[MinuteBar], decided: dt.datetime) -> float:
    return features_for(FEAT, 0.0, EXPIRES, EXPIRES, decided, bars, None, None)["ret_5m"]


def test_ret_5m_uses_available_completed_bars() -> None:
    def bar(m: int, o: str, c: str) -> MinuteBar:
        return MinuteBar(
            dt.datetime(2026, 9, 15, 9, m), Decimal(o), Decimal(c), Decimal(c), Decimal(c), 1
        )

    at = lambda m: dt.datetime(2026, 9, 15, 9, m, 20, tzinfo=KST)  # noqa: E731
    bars = [bar(m, str(99 + m), str(100 + m)) for m in range(10)]  # close 100+m, open 99+m
    # 0 completed bars (only the current 09:00 bar)
    assert _ret(bars[:1], at(0)) == 0.0
    # 1 completed bar: close / open
    assert _ret(bars[:2], at(1)) == pytest.approx(100 / 99 - 1)
    # 3 completed bars: last close / first close
    assert _ret(bars[:4], at(3)) == pytest.approx(102 / 100 - 1)
    # 5+ completed: last 5 closes only; current-minute bar (09:09, close 109) ignored
    assert _ret(bars, at(9)) == pytest.approx(108 / 104 - 1)


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


def test_constant_training_column_is_masked_not_blown_up(tmp_path) -> None:  # noqa: ANN001
    X, y = _synthetic(300, 6)
    X[:, FEATURES.index("mkt_move")] = 0.0
    model = fit(X, y, lam=1.0)
    base = _row(1.0)
    live = {**base, "mkt_move": 0.004}
    assert model.predict(live) == pytest.approx(model.predict(base))
    assert model.predict(base)[1] > 0.9
    raw = json.loads(model.to_json())
    assert raw["stds"][FEATURES.index("mkt_move")] is None
    back = BanditModel.from_json(model.to_json())
    assert back.predict(live) == pytest.approx(model.predict(base))
    # sequential update keeps the mask
    assert update(back, X[:10], y[:10]).predict(live) == pytest.approx(
        update(back, X[:10], y[:10]).predict(base)
    )


def test_completed_bars_excludes_current_minute_and_preopen() -> None:
    decided = dt.datetime(2026, 9, 15, 9, 3, 10, tzinfo=KST)
    bars = [_bar(8, 59, "1")] + [_bar(9, m, "1") for m in range(4)]
    assert [b.ts.minute for b in bandit.completed_bars(bars, decided)] == [0, 1, 2]


def test_load_model_cache_key_includes_size(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    X, y = _synthetic(50, 7)
    path = tmp_path / "m.json"
    path.write_text(fit(X, y, lam=1.0).to_json())
    a = bandit.load_model(str(path))
    mtime = path.stat().st_mtime_ns
    import os

    path.write_text(fit(X, y, lam=100.0).to_json() + " " * 7)
    os.utime(path, ns=(mtime, mtime))  # same mtime, different size
    assert bandit.load_model(str(path)) is not a


def test_weights_none_reproduces_the_unweighted_fit_exactly() -> None:
    """The default path must be untouched -- the live model is fitted through it."""
    X, y = _synthetic(400, 11)
    a = fit(X, y, lam=1.0)
    b = fit(X, y, lam=1.0, weights=np.ones(len(y)))
    assert np.allclose(a.w, b.w, atol=0, rtol=0) or np.allclose(a.w, b.w, atol=1e-12)
    assert np.isclose(a.b, b.b, rtol=1e-12)


def test_weights_recover_coefficients_under_heteroscedastic_noise() -> None:
    """Reward dispersion scales with the name's own volatility; unweighted, the
    posterior treats a jumpy name and a quiet one as equally informative."""
    rng = np.random.default_rng(3)
    n, k = 6000, 5
    X = rng.normal(size=(n, k))
    truth = np.array([0.5, -0.3, 0.0, 0.2, 0.4])
    scale = np.exp(rng.normal(scale=0.8, size=n))  # per-row noise sd
    y = X @ truth + rng.normal(size=n) * scale
    names = tuple(f"f{i}" for i in range(k))

    plain = fit(X, y, lam=1e-6, feature_names=names)
    weighted = fit(X, y, lam=1e-6, feature_names=names, weights=1.0 / scale**2)
    err_plain = np.abs(plain.w[1:] - truth).sum()
    err_weighted = np.abs(weighted.w[1:] - truth).sum()
    assert err_weighted < err_plain


def test_weights_must_be_positive_and_one_per_row() -> None:
    X, y = _synthetic(50, 7)
    with pytest.raises(ValueError, match="weights"):
        fit(X, y, lam=1.0, weights=np.ones(len(y) - 1))
    with pytest.raises(ValueError, match="weights"):
        fit(X, y, lam=1.0, weights=np.zeros(len(y)))


def test_standardization_is_decision_invariant_under_a_global_affine_map() -> None:
    """Pooled standardization rescales units; it cannot stand in for, nor undo,
    normalizing a feature against the stock's own history."""
    rng = np.random.default_rng(0)
    n, k = 2000, 5
    X = rng.normal(size=(n, k))
    y = X @ np.array([0.3, -0.2, 0.0, 0.1, 0.5]) + rng.normal(scale=0.5, size=n)
    names = tuple(f"f{i}" for i in range(k))
    scale = np.array([10.0, 0.1, 3.0, 1.0, 100.0])
    shift = np.array([5.0, -2.0, 0.0, 7.0, 1e4])

    plain = fit(X, y, lam=1.0, feature_names=names)
    mapped = fit(X * scale + shift, y, lam=1.0, feature_names=names)
    for row in X[:200]:
        d1 = decide(plain, dict(zip(names, row, strict=True)))
        d2 = decide(mapped, dict(zip(names, row * scale + shift, strict=True)))
        assert d1[0] == d2[0]
        assert np.isclose(d1[1], d2[1], atol=1e-10)
