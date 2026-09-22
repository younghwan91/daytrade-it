"""Audit: where does pooled-vs-per-stock actually bite, and does it break the model?

Two claims to settle before re-running anything.

(a) `fit` standardizes pooled -- but it receives X with NO stock identity, so the
    fix cannot live there. The question is whether a global affine map applied
    on top of already-per-stock-normalized columns reintroduces pooling.

(b) The NIG posterior assumes homoscedastic noise. Feature scaling does not
    touch the noise term, so per-stock FEATURE normalization cannot violate it.
    The target is a different matter: net return's variance should scale with
    the stock's own volatility, and that IS a violation, present in every run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path.home() / "git/daytrade-it-rlnews/data/eval/rlnews"
rows = [json.loads(line) for line in (R / "feats3_prior.jsonl").open()]
print(f"rows {len(rows):,}")
k0 = rows[0]
print("available keys:", sorted(k0)[:40])

f = [r.get("features") or r for r in rows]
sig = np.array([(x.get("sigma") if x.get("sigma") is not None else np.nan) for x in f])
rew = np.array([r.get("reward", r.get("net", np.nan)) for r in rows], dtype=float)
ok = np.isfinite(sig) & np.isfinite(rew)
sig, rew = sig[ok], rew[ok]
print(f"usable {ok.sum():,}   sigma med {np.median(sig):.5f}   reward med {np.median(rew):.5f}")

# (b) heteroscedasticity: reward dispersion by the stock's own trailing vol
q = np.quantile(sig, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
print(
    f"\n{'sigma quintile':16s} {'n':>7s} {'median sigma':>13s} {'sd(reward) bp':>14s} {'ratio to Q1':>12s}"
)
base = None
for i in range(5):
    m = (sig >= q[i]) & (sig <= q[i + 1] if i == 4 else sig < q[i + 1])
    sd = rew[m].std() * 1e4
    base = sd if base is None else base
    print(f"Q{i + 1:<15d} {m.sum():>7d} {np.median(sig[m]):>13.5f} {sd:>14.1f} {sd / base:>12.2f}x")

# does dividing the target by the stock's own sigma stabilise it?
rn = rew / sig
print(f"\n{'sigma quintile':16s} {'sd(reward/sigma)':>18s} {'ratio to Q1':>12s}")
base = None
for i in range(5):
    m = (sig >= q[i]) & (sig <= q[i + 1] if i == 4 else sig < q[i + 1])
    sd = rn[m].std()
    base = sd if base is None else base
    print(f"Q{i + 1:<15d} {sd:>18.3f} {sd / base:>12.2f}x")
