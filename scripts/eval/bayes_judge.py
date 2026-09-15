"""Grid-posterior Bayesian judge for a candidate vs. the baseline policy.

Implements the "베이즈 방법" adoption rules from
`docs/superpowers/specs/2026-09-15-bayes-pipeline-optimization-design.md`:

1. P(mu > 0) >= 0.90, where mu is the posterior mean daily P&L delta
   (candidate - baseline), mu ~ Normal(0, prior_sd) shrinking the effect
   size toward zero on small samples.
2. Loss prevention: the candidate's worst day and losing-trade ratio must
   not be worse than the baseline's ("if you can't make money, don't lose
   it").
3. d_t >= 0 on at least ceil(5/7 * n_days) of the days.
4. Forward 5+ trading days must agree in direction -- not evaluated here,
   always reported as `forward_pending: true`.

The likelihood is Student-t(nu, mu, sigma) per day (heavier tails than
Normal), with sigma integrated out over a log-uniform grid (a scale-
invariant, uniform-in-log prior on sigma -- the grid *is* the integration
measure, so no extra Jacobian term is needed). `posterior_mu` returns the
discrete marginal over a mu grid, normalized to sum to 1.

Usage:
    uv run python scripts/eval/bayes_judge.py --candidate cand.json --baseline base.json

Both files are JSON lists of day records shaped like `replay_common.day_net`
output (keys: day, net, trades, losers, worst_trade, errors, leftovers).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_PRIOR_SD = 5000.0


def _student_t_log_norm_const(nu: float) -> float:
    """log of the Student-t normalizing constant (the part not depending on the data)."""
    return math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)


def posterior_mu(
    d: list[float],
    prior_sd: float,
    nu: float = 4.0,
    grid: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal posterior of mu (mean of `d`) over a grid, sigma integrated out.

    Likelihood: d_t ~ Student-t(nu, mu, sigma), independent across days.
    Prior: mu ~ Normal(0, prior_sd). sigma is integrated over a log-uniform
    grid (equivalent to a scale-invariant prior on sigma).

    Returns (mu_grid, posterior) where `posterior` is a discrete probability
    mass function over `mu_grid` (sums to 1); it is not a density.
    """
    if grid < 2:
        raise ValueError("grid must be >= 2")
    if prior_sd <= 0:
        raise ValueError("prior_sd must be positive")
    arr = np.asarray(d, dtype=float)
    if arr.size == 0:
        raise ValueError("d must not be empty")

    # mu grid: nonuniform (sinh-warped), dense near 0 and sparse in the tails,
    # so a small-but-real edge (tens of KRW) isn't quantized away by a grid
    # spaced to cover +/-4 prior_sd (thousands of KRW). Span still covers the
    # prior's support plus the raw data scale, so an outlier candidate isn't
    # clipped before the prior can shrink it.
    max_abs_d = float(np.max(np.abs(arr)))
    mu_span = max(4.0 * prior_sd + 2.0 * max_abs_d, 1.0)
    t = np.linspace(-1.0, 1.0, grid)
    alpha = 5.0
    mu_grid = mu_span * np.sinh(alpha * t) / math.sinh(alpha)

    # Trapezoidal cell widths for the nonuniform mu grid, so summing
    # density * width approximates the continuous posterior (a naive
    # unweighted sum would over-count the wide tail cells).
    mu_weights = np.empty(grid)
    mu_weights[1:-1] = (mu_grid[2:] - mu_grid[:-2]) / 2.0
    mu_weights[0] = (mu_grid[1] - mu_grid[0]) / 2.0
    mu_weights[-1] = (mu_grid[-1] - mu_grid[-2]) / 2.0

    # sigma grid: log-uniform, scaled to the *data's* own scale (not the mu
    # grid -- a grid-tied floor swamps a small real signal regardless of
    # sample size). A small absolute floor keeps near-zero-variance data
    # (e.g. an identical d_t every day) from producing an unbounded
    # likelihood spike as sigma -> 0.
    median_d = float(np.median(arr))
    data_scale = max(float(np.max(np.abs(arr - median_d))), max_abs_d, prior_sd, 1.0)
    n_sigma = 121
    sigma_lo = 1.0
    sigma_hi = max(10.0 * data_scale, sigma_lo * 10.0)
    log_sigma = np.linspace(math.log(sigma_lo), math.log(sigma_hi), n_sigma)
    sigma_grid = np.exp(log_sigma)
    dlog = (log_sigma[-1] - log_sigma[0]) / (n_sigma - 1)

    log_const = _student_t_log_norm_const(nu)

    # (mu, sigma, day) residual grid.
    resid = (arr[None, None, :] - mu_grid[:, None, None]) / sigma_grid[None, :, None]
    log_lik_per_day = (
        log_const - np.log(sigma_grid)[None, :, None] - (nu + 1) / 2 * np.log1p(resid**2 / nu)
    )
    log_lik = log_lik_per_day.sum(axis=2)  # sum over days -> (grid, n_sigma)

    # Integrate out sigma (log-sum-exp over the log-uniform grid, weight = dlog).
    row_max = log_lik.max(axis=1, keepdims=True)
    integrated = row_max[:, 0] + np.log(np.sum(np.exp(log_lik - row_max), axis=1) * dlog)

    log_prior_mu = -0.5 * np.log(2 * math.pi * prior_sd**2) - mu_grid**2 / (2 * prior_sd**2)

    log_density = integrated + log_prior_mu
    log_density -= log_density.max()
    density = np.exp(log_density)
    post = density * mu_weights
    post /= post.sum()
    return mu_grid, post


def prob_mu_gt_0(mu_grid: np.ndarray, post: np.ndarray) -> float:
    """P(mu > 0) from a `posterior_mu` grid.

    The mu grid is symmetric and (for an odd `grid`) always has a point at
    exactly 0, so for near-zero-variance data virtually all posterior mass
    can land on that single cell. Splitting it half-and-half avoids that
    discretization artifact swinging the result to ~0 or ~1 instead of ~0.5.
    """
    pos = float(post[mu_grid > 0].sum())
    zero_mass = float(post[mu_grid == 0].sum())
    return pos + 0.5 * zero_mass


def _loser_ratio(days: dict[str, dict]) -> float:
    """Sum(losers) / Sum(trades) across days; 0.0 if there were no trades at all."""
    total_trades = sum(int(v["trades"]) for v in days.values())
    if total_trades == 0:
        return 0.0
    total_losers = sum(int(v["losers"]) for v in days.values())
    return total_losers / total_trades


def judge(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    prior_sd: float = DEFAULT_PRIOR_SD,
) -> dict:
    """Judge `candidate` against `baseline` (both keyed by day, values shaped like day_net).

    Applies adoption conditions 1-3 from the spec; condition 4 (forward
    5-day agreement) isn't evaluated here and is reported as
    `forward_pending: true`.
    """
    if set(candidate) != set(baseline):
        raise ValueError(
            "candidate and baseline day sets must match: "
            f"only in candidate={sorted(set(candidate) - set(baseline))}, "
            f"only in baseline={sorted(set(baseline) - set(candidate))}"
        )

    days = sorted(candidate)
    n_days = len(days)
    if n_days == 0:
        raise ValueError("no days to judge")

    deltas = [float(candidate[day]["net"]) - float(baseline[day]["net"]) for day in days]
    days_ge_0 = sum(1 for delta in deltas if delta >= 0)

    mu_grid, post = posterior_mu(deltas, prior_sd=prior_sd)
    p_mu_gt_0 = prob_mu_gt_0(mu_grid, post)
    mu_mean = float(np.sum(mu_grid * post))

    worst_day_candidate = min(float(v["net"]) for v in candidate.values())
    worst_day_baseline = min(float(v["net"]) for v in baseline.values())
    loser_ratio_candidate = _loser_ratio(candidate)
    loser_ratio_baseline = _loser_ratio(baseline)

    threshold = math.ceil(5 / 7 * n_days)

    reasons: list[str] = []
    if p_mu_gt_0 < 0.90:
        reasons.append(f"p_mu_gt_0={p_mu_gt_0:.4f} < 0.90")
    if worst_day_candidate < worst_day_baseline:
        reasons.append(
            f"worst_day_candidate={worst_day_candidate!r} < worst_day_baseline={worst_day_baseline!r}"
        )
    if loser_ratio_candidate > loser_ratio_baseline:
        reasons.append(
            f"loser_ratio_candidate={loser_ratio_candidate:.4f} > "
            f"loser_ratio_baseline={loser_ratio_baseline:.4f}"
        )
    if days_ge_0 < threshold:
        reasons.append(f"days_ge_0={days_ge_0} < ceil(5/7*{n_days})={threshold}")

    adopt = not reasons

    return {
        "p_mu_gt_0": p_mu_gt_0,
        "mu_mean": mu_mean,
        "days_ge_0": days_ge_0,
        "n_days": n_days,
        "worst_day_candidate": worst_day_candidate,
        "worst_day_baseline": worst_day_baseline,
        "loser_ratio_candidate": loser_ratio_candidate,
        "loser_ratio_baseline": loser_ratio_baseline,
        "adopt": adopt,
        "reasons": reasons,
        "forward_pending": True,
    }


def _load_days(path: Path) -> dict[str, dict]:
    records = json.loads(path.read_text())
    return {str(record["day"]): record for record in records}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="JSON list of day records")
    parser.add_argument("--baseline", type=Path, required=True, help="JSON list of day records")
    parser.add_argument("--prior-sd", type=float, default=DEFAULT_PRIOR_SD)
    args = parser.parse_args(argv)

    candidate = _load_days(args.candidate)
    baseline = _load_days(args.baseline)
    verdict = judge(candidate, baseline, prior_sd=args.prior_sd)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
