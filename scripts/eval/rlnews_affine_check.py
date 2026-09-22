"""Does fit()'s pooled standardization undo upstream per-stock normalization?

If a column already means "this stock against its own history", a single GLOBAL
affine map applied to that column preserves that meaning -- it rescales units,
which is all standardization is there for. Test it: fit the same data twice,
once raw and once after arbitrary global shifts and scales per column, and
compare the decisions.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "git/daytrade-it-rlnews/src"))
from daytrade_it.application.trading import bandit  # noqa: E402

rng = np.random.default_rng(0)
n, k = 4000, 5
Xn = rng.normal(size=(n, k))  # already per-stock normalized
y = Xn @ np.array([0.3, -0.2, 0.0, 0.1, 0.5]) + rng.normal(scale=0.5, size=n)
names = tuple(f"f{i}" for i in range(k))

m1 = bandit.fit(Xn, y, lam=1.0, feature_names=names)
scale = np.array([10.0, 0.1, 3.0, 1.0, 100.0])
shift = np.array([5.0, -2.0, 0.0, 7.0, 1e4])
m2 = bandit.fit(Xn * scale + shift, y, lam=1.0, feature_names=names)

d1 = [bandit.decide(m1, dict(zip(names, r, strict=True))) for r in Xn[:1000]]
d2 = [bandit.decide(m2, dict(zip(names, r, strict=True))) for r in (Xn[:1000] * scale + shift)]
mu1 = np.array([x[1] for x in d1])
mu2 = np.array([x[1] for x in d2])
p1 = np.array([x[2] for x in d1])
p2 = np.array([x[2] for x in d2])
a1 = np.array([x[0] for x in d1])
a2 = np.array([x[0] for x in d2])
print(f"max |E[reward] diff| = {np.abs(mu1 - mu2).max():.3e}")
print(f"max |p(reward>0) diff| = {np.abs(p1 - p2).max():.3e}")
print(f"identical enter/skip decisions: {int((a1 == a2).sum())}/1000")

# and the thing that is NOT invariant: a per-ROW rescale (what WLS would do)
w = rng.uniform(0.5, 2.0, size=n)
m3 = bandit.fit(Xn * w[:, None], y * w, lam=1.0, feature_names=names)
print(
    f"\nper-row weighting changes the fit (as it must): "
    f"max |w diff| = {np.abs(m1.w - m3.w).max():.4f}"
)
