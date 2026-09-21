"""Diagonal-Gaussian HMM: EM, forward FILTER, forward-backward smoother.

Written by hand (hmmlearn is not installed) precisely so the causality is
under our control. The two routines that matter:

  `filter_states(params, X)` -- posterior over the state at t given x_1..x_t
    ONLY. Admissible.
  `smooth_states(params, X)` -- posterior given the whole sequence. Look-ahead.
    Reported only as a non-admissible upper bound.

Scaled (linear-space) forward/backward with a per-row max removed from the
emission log-density, so nothing underflows and the loop stays cheap.
"""

from __future__ import annotations

import numpy as np

VAR_FLOOR = 1e-6


def _log_gauss(X, mu, var):
    """(T, K) log N(x_t; mu_k, diag(var_k))."""
    K = mu.shape[0]
    out = np.empty((X.shape[0], K))
    for k in range(K):
        v = var[k]
        z = (X - mu[k]) / np.sqrt(v)
        out[:, k] = -0.5 * (np.einsum("ij,ij->i", z, z) + np.sum(np.log(2 * np.pi * v)))
    return out


def _scaled(logB):
    m = logB.max(axis=1)
    return np.exp(logB - m[:, None]), m


def _forward(B, m, pi, A):
    """Scaled forward. alpha[t] = P(z_t | x_1..x_t) (FILTERED).
    pred[t] = P(z_t | x_1..x_{t-1}).  c[t] = p(x_t | x_1..x_{t-1}) / exp(m[t]).
    """
    T, K = B.shape
    alpha = np.empty((T, K))
    pred = np.empty((T, K))
    c = np.empty(T)
    prev = pi
    for t in range(T):
        pred[t] = prev
        a = prev * B[t]
        s = a.sum()
        if s <= 0:
            s = 1e-300
        c[t] = s
        alpha[t] = a / s
        if t + 1 < T:
            prev = alpha[t] @ A
    ll = float(np.sum(np.log(c) + m))
    return alpha, ll, pred, np.log(c) + m


def _backward(B, A):
    T, K = B.shape
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        b = A @ (B[t + 1] * beta[t + 1])
        s = b.sum()
        beta[t] = b / (s if s > 0 else 1e-300)
    return beta


def fit_em(X, K, seed=0, n_iter=80, tol=1e-6, n_restarts=2):
    """EM on the whole of `X`. Returns a params dict."""
    rng = np.random.default_rng(seed)
    T, D = X.shape
    best = None
    for _ in range(n_restarts):
        idx = rng.permutation(T)
        mu = np.array([X[c].mean(axis=0) for c in np.array_split(idx, K)])
        mu = mu + rng.normal(0, 0.3, mu.shape)
        var = np.tile(X.var(axis=0) + VAR_FLOOR, (K, 1))
        A = np.full((K, K), 0.3 / K) + np.eye(K) * 0.7
        A /= A.sum(axis=1, keepdims=True)
        pi = np.full(K, 1.0 / K)
        prev_ll = -np.inf
        ll = -np.inf
        for _it in range(n_iter):
            B, m = _scaled(_log_gauss(X, mu, var))
            alpha, ll, _, _ = _forward(B, m, pi, A)
            beta = _backward(B, A)
            g = alpha * beta
            g /= g.sum(axis=1, keepdims=True)
            # xi, vectorised over t
            L = B[1:] * beta[1:]  # (T-1, K)
            xi_t = alpha[:-1, :, None] * A[None] * L[None].transpose(1, 0, 2)
            xi_t /= xi_t.sum(axis=(1, 2), keepdims=True) + 1e-300
            xi = xi_t.sum(axis=0)
            pi = g[0] + 1e-8
            pi /= pi.sum()
            A = xi + 1e-6
            A /= A.sum(axis=1, keepdims=True)
            w = g.sum(axis=0) + 1e-8
            mu = (g.T @ X) / w[:, None]
            for k in range(K):
                d = X - mu[k]
                var[k] = (g[:, k] @ (d * d)) / w[k] + VAR_FLOOR
            if ll - prev_ll < tol * abs(prev_ll + 1):
                break
            prev_ll = ll
        if best is None or ll > best[0]:
            best = (ll, pi.copy(), A.copy(), mu.copy(), var.copy())
    ll, pi, A, mu, var = best
    return {"pi": pi, "A": A, "mu": mu, "var": var, "loglik": float(ll), "K": K, "D": D}


def relabel(p, by=0):
    """Sort states ascending on fitted mean of observable column `by`, so a
    label means the same thing across refits. Uses fitted params only."""
    order = np.argsort(p["mu"][:, by])
    q = dict(p)
    q["mu"] = p["mu"][order]
    q["var"] = p["var"][order]
    q["pi"] = p["pi"][order]
    q["A"] = p["A"][np.ix_(order, order)]
    return q


def filter_states(p, X):
    """FILTERED posterior (admissible) + per-day one-step predictive loglik."""
    B, m = _scaled(_log_gauss(X, p["mu"], p["var"]))
    alpha, ll, pred, pll = _forward(B, m, p["pi"], p["A"])
    return alpha, ll, pll


def smooth_states(p, X):
    """SMOOTHED posterior -- uses the WHOLE sequence. NOT admissible."""
    B, m = _scaled(_log_gauss(X, p["mu"], p["var"]))
    alpha, _, _, _ = _forward(B, m, p["pi"], p["A"])
    beta = _backward(B, p["A"])
    g = alpha * beta
    g /= g.sum(axis=1, keepdims=True)
    return g


def visit_counts(labels):
    """(days per state, distinct visits per state) for a hard label sequence."""
    labels = np.asarray(labels)
    K = int(labels.max()) + 1 if labels.size else 0
    days = {k: int((labels == k).sum()) for k in range(K)}
    visits = dict.fromkeys(range(K), 0)
    prev = None
    for v in labels:
        if v != prev:
            visits[int(v)] += 1
        prev = v
    return days, visits


def n_params(K, D):
    return (K - 1) + K * (K - 1) + K * D * 2
