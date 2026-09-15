"""Bayesian contextual bandit for news entries: enter/skip and ranking.

Spec: docs/superpowers/specs/2026-09-16-bandit-entry-design.md.

r = xᵀw + ε, ε ~ N(0, σ²); w ~ N(0, σ²·Λ₀⁻¹), σ² ~ InvGamma(a₀, b₀), where x is
the standardized feature row with a leading intercept and Λ₀ = diag(λ·1e-3,
λ, …, λ). The conjugate posterior is Normal-Inverse-Gamma (w, V = Λₙ⁻¹, a, b)
and the posterior predictive is Student-t with df = 2a, location xᵀw and
scale² = (b/a)(1 + xᵀVx). `decide` enters iff P(r > 0) ≥ min_prob and
E[r] ≥ min_edge.

Constant training columns (std < 1e-6) are masked: their std is stored as
+inf (JSON null), so their standardized value is always 0 and a live value
the model never saw varies nothing.

Live wiring fails closed: when the bandit is on, `news_entry` skips a
candidate if the 069500 quote or its prior close is unavailable
(`features_for`'s `mkt_move` 0.0 fallback is for offline use only). Few or no
completed bars are not a skip: `ret_5m` uses the bars that exist, so the 09:01
open batch is judged too. Live and replay run this same code path.

The live selection (`news_entry.build_entry_candidate`) and the replay engine
call the same `features_for` / `decide`, so a replayed model runs live as is.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from daytrade_it.infrastructure.brokers.base import KST, MinuteBar
from daytrade_it.infrastructure.external.anthropic_client import ArticleFeatures

FEATURES: tuple[str, ...] = (
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

_OPEN = dt.time(9, 0)
_MASK_STD = 1e-6
_INTERCEPT_PRIOR = 1e-3
_RET_BARS = 5


def completed_bars(bars: list[MinuteBar], decided_at: dt.datetime) -> list[MinuteBar]:
    """Regular-session bars (≥ 09:00) whose minute ended before `decided_at` (naive = KST)."""
    if decided_at.tzinfo is None:
        decided_at = decided_at.replace(tzinfo=KST)
    minute = decided_at.astimezone(KST).replace(tzinfo=None, second=0, microsecond=0)
    return [b for b in bars if b.ts.time() >= _OPEN and b.ts < minute]


def features_for(
    features: ArticleFeatures,
    pre_entry_move: float,
    published_at: dt.datetime,
    expires_at: dt.datetime,
    decided_at: dt.datetime,
    bars: list[MinuteBar],
    mkt_price: Decimal | None,
    mkt_prev_close: Decimal | None,
) -> dict[str, float]:
    """The decision-time feature row. Pure; only data available at `decided_at`.

    - `minutes_since_open`: from 09:00 KST of `decided_at`'s day, clipped at 0.
    - `ret_5m`: over the last (up to) 5 completed regular-session bars
      (ts < floor(decided_at) and ≥ 09:00): last close / first close − 1 with
      2..5 bars; last close / its open − 1 with exactly 1; 0.0 with none.
    - `mkt_move`: mkt_price / mkt_prev_close − 1; 0.0 if either is unavailable.
    - A naive `decided_at` is taken to be KST.
    """
    from daytrade_it.application.signals.news_entry import article_session

    if decided_at.tzinfo is None:
        decided_at = decided_at.replace(tzinfo=KST)
    decided_kst = decided_at.astimezone(KST)
    open_at = dt.datetime.combine(decided_kst.date(), _OPEN, tzinfo=KST)
    minutes = max(0.0, (decided_kst - open_at).total_seconds() / 60.0)

    done = completed_bars(bars, decided_kst)
    window = done[-_RET_BARS:]
    ret_5m = 0.0
    if window:
        base = window[0].open if len(window) == 1 else window[0].close
        if base and base > 0:
            ret_5m = float(window[-1].close / base - 1)

    mkt_move = 0.0
    if mkt_price is not None and mkt_prev_close is not None and mkt_prev_close > 0:
        mkt_move = float(mkt_price / mkt_prev_close - 1)

    return {
        "score": float(features.score()),
        "surprise": float(features.surprise),
        "specificity": float(features.specificity),
        "sentiment_strength": float(features.sentiment_strength),
        "persistence_one_off": float(features.persistence == "one_off"),
        "persistence_multi_quarter": float(features.persistence == "multi_quarter"),
        "persistence_structural": float(features.persistence == "structural"),
        "subject_main": float(features.subject == "main"),
        "pre_entry_move": float(pre_entry_move),
        "premarket": float(article_session(published_at, expires_at) == "premarket"),
        "minutes_since_open": minutes,
        "ret_5m": ret_5m,
        "mkt_move": mkt_move,
    }


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10_000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """P(T ≤ t) for Student-t with `df` degrees of freedom."""
    if t == 0.0:
        return 0.5
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


@dataclass(frozen=True, eq=False)
class BanditModel:
    """NIG posterior over standardized features (+ leading intercept)."""

    feature_names: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    w: np.ndarray
    V: np.ndarray
    a: float
    b: float
    lam: float
    min_prob: float = 0.5
    min_edge: float = 0.0
    trained_days: tuple[str, ...] = ()

    def design_row(self, x: dict[str, float]) -> np.ndarray:
        raw = np.array([float(x[name]) for name in self.feature_names])
        return np.concatenate([[1.0], (raw - self.means) / self.stds])

    def predict(self, x: dict[str, float]) -> tuple[float, float]:
        """(m = E[r], p = P(r > 0)) under the Student-t posterior predictive."""
        z = self.design_row(x)
        m = float(z @ self.w)
        scale = math.sqrt((self.b / self.a) * (1.0 + float(z @ self.V @ z)))
        p = student_t_cdf(m / scale, 2.0 * self.a) if scale > 0 else float(m > 0)
        return m, p

    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_names": list(self.feature_names),
                "means": self.means.tolist(),
                "stds": [None if not math.isfinite(v) else v for v in self.stds.tolist()],
                "w": self.w.tolist(),
                "V": self.V.tolist(),
                "a": self.a,
                "b": self.b,
                "lam": self.lam,
                "min_prob": self.min_prob,
                "min_edge": self.min_edge,
                "trained_days": list(self.trained_days),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> BanditModel:
        raw = json.loads(text)
        names = tuple(raw["feature_names"])
        k = len(names)
        model = cls(
            feature_names=names,
            means=np.asarray(raw["means"], dtype=float),
            stds=np.asarray([math.inf if v is None else v for v in raw["stds"]], dtype=float),
            w=np.asarray(raw["w"], dtype=float),
            V=np.asarray(raw["V"], dtype=float),
            a=float(raw["a"]),
            b=float(raw["b"]),
            lam=float(raw["lam"]),
            min_prob=float(raw.get("min_prob", 0.5)),
            min_edge=float(raw.get("min_edge", 0.0)),
            trained_days=tuple(raw.get("trained_days", ())),
        )
        shapes = (model.means.shape, model.stds.shape, model.w.shape, model.V.shape)
        if (
            shapes != ((k,), (k,), (k + 1,), (k + 1, k + 1))
            or model.a <= 0
            or model.b <= 0
            or not np.all(model.stds > 0)
        ):
            raise ValueError("bandit model: inconsistent shapes or non-positive a/b")
        return model


def _design(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    Z = (np.asarray(X, dtype=float) - means) / stds
    return np.hstack([np.ones((Z.shape[0], 1)), Z])


def fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    a0: float = 1.0,
    b0: float = 1e-4,
    feature_names: tuple[str, ...] = FEATURES,
    min_prob: float = 0.5,
    min_edge: float = 0.0,
    trained_days: tuple[str, ...] = (),
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> BanditModel:
    """Conjugate NIG posterior from rows `X` (columns = `feature_names`) and rewards `y`.

    Columns are standardized with the train mean/std unless `means`/`stds`
    are given (e.g. to reproduce a sequential `update`). A column with train
    std < 1e-6 is masked (std = +inf → always 0).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(feature_names) or y.shape != (X.shape[0],):
        raise ValueError("fit: X must be (n, len(feature_names)) and y (n,)")
    means = X.mean(axis=0) if means is None else np.asarray(means, dtype=float)
    if stds is None:
        stds = X.std(axis=0)
        stds = np.where(stds < _MASK_STD, math.inf, stds)
    else:
        stds = np.asarray(stds, dtype=float)
    Z = _design(X, means, stds)
    k = Z.shape[1]
    prec0 = np.diag([lam * _INTERCEPT_PRIOR] + [lam] * (k - 1))
    prec = prec0 + Z.T @ Z
    V = np.linalg.inv(prec)
    w = V @ (Z.T @ y)
    resid = y - Z @ w
    b = b0 + 0.5 * (float(resid @ resid) + float(w @ prec0 @ w))
    return BanditModel(
        feature_names=tuple(feature_names),
        means=means,
        stds=stds,
        w=w,
        V=V,
        a=a0 + X.shape[0] / 2.0,
        b=b,
        lam=lam,
        min_prob=min_prob,
        min_edge=min_edge,
        trained_days=tuple(trained_days),
    )


def update(model: BanditModel, X_new: np.ndarray, y_new: np.ndarray) -> BanditModel:
    """Sequential conjugate update; same standardization as `model`."""
    y_new = np.asarray(y_new, dtype=float)
    Z = _design(X_new, model.means, model.stds)
    prec_old = np.linalg.inv(model.V)
    prec = prec_old + Z.T @ Z
    V = np.linalg.inv(prec)
    w = V @ (prec_old @ model.w + Z.T @ y_new)
    resid = y_new - Z @ w
    dw = w - model.w
    b = model.b + 0.5 * (float(resid @ resid) + float(dw @ prec_old @ dw))
    return BanditModel(
        feature_names=model.feature_names,
        means=model.means,
        stds=model.stds,
        w=w,
        V=V,
        a=model.a + Z.shape[0] / 2.0,
        b=b,
        lam=model.lam,
        min_prob=model.min_prob,
        min_edge=model.min_edge,
        trained_days=model.trained_days,
    )


def decide(model: BanditModel, x: dict[str, float]) -> tuple[bool, float, float]:
    """(enter, m, p): enter iff p ≥ min_prob and m ≥ min_edge."""
    m, p = model.predict(x)
    return (p >= model.min_prob and m >= model.min_edge), m, p


_CACHE: dict[str, tuple[tuple[int, int], BanditModel]] = {}


def load_model(path: str) -> BanditModel:
    """The model file at `path`, parsed once per (path, mtime_ns, size)."""
    st = os.stat(path)
    stamp = (st.st_mtime_ns, st.st_size)
    cached = _CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    with open(path, encoding="utf-8") as f:
        model = BanditModel.from_json(f.read())
    _CACHE[path] = (stamp, model)
    return model
