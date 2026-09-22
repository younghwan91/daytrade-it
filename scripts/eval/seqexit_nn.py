"""A neural-network sequential exit policy, trained by policy gradient.

    .venv/bin/python scripts/eval/seqexit_nn.py \
        --paths data/eval/rlnews/paths_fresh.jsonl \
        --ma data/eval/rlnews/ma_daily_fresh.json \
        --out data/eval/rlnews/seqexit_nn.json                    # simnode

The table policies (`seqexit_search.py`) discretised the state into 15-135
cells. This replaces the table with a function: a small MLP reads the
continuous state at every minute and emits the probability of acting. Below the
entry "act" means cut; at or above it, add. The legality masks live in
`Sim.run_steps`, not here, so no network weight can cut a winner or add to a
loser -- the owner's asymmetry is structural for the network exactly as it was
for the table.

**Training.** REINFORCE on the exact simulator. Each minibatch samples a
Bernoulli action at every legal step, rolls every path out with the same
first-hitting-time walk `seqexit_gate` verified, and scores each path by its
realised KRW net of `costs.round_trip` per leg. The gradient of a Bernoulli
log-probability with respect to its logit is `a - p`, weighted by that path's
advantage and backpropagated by hand. There is no fitted value function, so
there is no bootstrap to diverge and nothing to tune but the step size.

**Selection statistic.** Mean realised KRW per trade, and nothing else --
`seqexit_core.objective`. REINFORCE maximises the expectation of that same
quantity. Worst day and loser ratio are reported as diagnostics only.

**Gates, before anything is trained.**
  * `grad_check`: the hand-written backward pass must agree with a central
    finite difference on every parameter. A policy-gradient trainer with a
    sign error in its backward pass still "trains" -- it just climbs the wrong
    hill -- so this is the network's equivalent of the twin gate.
  * the roll-out is `Sim.run_steps`, i.e. the already-gated walk.

**Normalisation -- no pool statistic is estimated anywhere.** Every input is a
dimensionless per-stock ratio built in `seqexit_core.reduce_row` /
`seqexit_ma.py`, and the only rescaling applied here is a FIXED constant chosen
before training (`INPUT_SCALE`), identical for every stock and never estimated
from the sample. See `FEATURES` for the per-feature method.

**Holdout.** `FRESH_CONFIRM_WINDOW.json`, frozen at commit d55dd90 before any
feature on this book was built. `track_a_prior` is spent and is not read.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, "scripts/eval")

from bayes_judge import judge  # noqa: E402
from exit_worst_day import real_cost, stock_markets  # noqa: E402
from seqexit_core import (  # noqa: E402
    FLATTEN_MINUTE,
    LATTICES,
    NOTIONAL,
    Book,
    Sim,
    build,
    day_records,
)

# ---------------------------------------------------------------- the state
#
# (name, how it is built, why that normalisation). Every entry is either a
# dimensionless ratio that is cross-stock comparable by construction, or such a
# ratio divided by THIS stock's own pre-entry minute volatility `sigma`. None
# is a KRW level and none is standardised with a pool-wide mean or variance.
FEATURES = (
    ("pnl", "close/entry - 1", "a return; comparable in units across any price level"),
    ("pnl_n", "pnl / sigma", "the same return in this stock's own volatility units"),
    ("dd_n", "(close/peak_since_entry - 1) / sigma", "give-back from own peak, own-vol units"),
    ("slope_n", "(close/close[-3] - 1) / sigma", "3-minute momentum, own-vol units"),
    ("lvsurge", "log(vol3 / session_mean_vol_per_bar)", "volume vs this stock's own day so far"),
    ("lvtrend", "log(vol3 / previous vol3)", "volume acceleration vs its own recent past"),
    ("ma_frac_i", "share of available intraday MAs price is above", "a fraction, [0,1]"),
    ("ma_avail_i", "available intraday MA periods / 7", "how many MAs exist yet, [0,1]"),
    ("ma_dist_i_n", "mean((close-MA_k)/MA_k) / sigma", "intraday MA distance, own-vol units"),
    ("ma_frac_d", "share of 7 daily MAs last close is above", "a fraction, [0,1]"),
    ("ma_dist_d_n", "mean((close-MA_k)/MA_k) / own 60d daily-return sd", "daily MA distance"),
    ("elapsed", "minutes held / 330", "the clock"),
    ("to_close", "minutes to 15:00 / 360", "the clock"),
)
N_FEAT = len(FEATURES)
#: FIXED divisors, chosen once before any training and identical for every stock.
#: They only bring inputs to roughly unit scale for the optimiser; they are not
#: estimated from the sample, so they cannot leak one stock's statistics into
#: another's state. Anything beyond +/-CLIP after scaling is clipped.
INPUT_SCALE = np.array([0.02, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 4.0, 1.0, 4.0, 1.0, 1.0])
CLIP = 5.0
#: Features whose alignment with the price path the shuffled null destroys.
#: `pnl`, `pnl_n` decide action legality and are tied mechanically to the
#: reward; `elapsed`, `to_close` are the clock. Everything else is information
#: the policy might use, and is scrambled.
KEEP = {"pnl", "pnl_n", "elapsed", "to_close"}
SCRAMBLE_IDX = [i for i, (n, _, _) in enumerate(FEATURES) if n not in KEEP]
VOLUME_IDX = [i for i, (n, _, _) in enumerate(FEATURES) if n in ("lvsurge", "lvtrend")]
MA_IDX = [i for i, (n, _, _) in enumerate(FEATURES) if n.startswith("ma_")]


def feature_matrix(book: Book, ma_daily: dict[str, dict]) -> np.ndarray:
    """(n_steps, N_FEAT) float32, in `FEATURES` order."""
    lens = book.stop - book.starts
    sig = np.repeat(book.sigma, lens)
    frac_d = np.full(book.n_paths, 0.5)
    dist_d = np.zeros(book.n_paths)
    for i, (d, t) in enumerate(zip(book.day, book.ticker, strict=True)):
        m = ma_daily.get(f"{d}|{t}")
        if m is not None:
            frac_d[i] = m["frac"]
            dist_d[i] = m["dist_norm"]
    cols = [
        book.pnl,
        book.pnl / sig,
        book.dd / sig,
        book.slope / sig,
        np.log(np.clip(book.vsurge, 0.05, 20.0)),
        np.log(np.clip(book.vtrend, 0.05, 20.0)),
        book.ma_frac_intra,
        book.ma_avail_intra / 7.0,
        book.ma_dist_intra / sig,
        np.repeat(frac_d, lens),
        np.repeat(dist_d, lens),
        book.elapsed / 330.0,
        (FLATTEN_MINUTE - book.minute) / 360.0,
    ]
    x = np.stack(cols, axis=1) / INPUT_SCALE
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP).astype(
        np.float32
    )


# ---------------------------------------------------------------- the network


class MLP:
    """`N_FEAT -> H -> H -> 1` with tanh, or a logistic model when `hidden=0`.

    The output is the logit of P(act). Parameters are a flat list so Adam and
    the gradient check treat every architecture the same way.
    """

    def __init__(self, hidden: int, rng: np.random.Generator) -> None:
        self.hidden = hidden
        if hidden == 0:
            self.params = [rng.normal(0, 0.01, (N_FEAT, 1)), np.zeros(1)]
        else:
            s1 = 1.0 / np.sqrt(N_FEAT)
            s2 = 1.0 / np.sqrt(hidden)
            self.params = [
                rng.normal(0, s1, (N_FEAT, hidden)),
                np.zeros(hidden),
                rng.normal(0, s2, (hidden, hidden)),
                np.zeros(hidden),
                rng.normal(0, s2, (hidden, 1)) * 0.1,
                np.zeros(1),
            ]
        # Start near "never act" (hold to the close) so the search leaves the
        # honest null only when the data pushes it off.
        self.params[-1][:] = -2.0

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        if self.hidden == 0:
            w, b = self.params
            return (x @ w)[:, 0] + b[0], (x,)
        w1, b1, w2, b2, w3, b3 = self.params
        h1 = np.tanh(x @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        return (h2 @ w3)[:, 0] + b3[0], (x, h1, h2)

    def backward(self, g: np.ndarray, cache: tuple) -> list[np.ndarray]:
        """Gradients of `sum(g * logit)` with respect to every parameter."""
        if self.hidden == 0:
            (x,) = cache
            return [x.T @ g[:, None], np.array([g.sum()])]
        x, h1, h2 = cache
        _, _, w2, _, w3, _ = self.params
        gw3 = h2.T @ g[:, None]
        gb3 = np.array([g.sum()])
        gh2 = g[:, None] @ w3.T * (1.0 - h2**2)
        gw2 = h1.T @ gh2
        gb2 = gh2.sum(axis=0)
        gh1 = gh2 @ w2.T * (1.0 - h1**2)
        gw1 = x.T @ gh1
        gb1 = gh1.sum(axis=0)
        return [gw1, gb1, gw2, gb2, gw3, gb3]


def grad_check(hidden: int, seed: int = 0) -> float:
    """Max relative error of the hand-written backward pass vs central differences.

    Checked on the exact objective the trainer differentiates: sum over steps
    of `adv * log Bernoulli(a | sigmoid(logit))`.
    """
    rng = np.random.default_rng(seed)
    net = MLP(hidden, rng)
    for p in net.params:
        p[...] = rng.normal(0, 0.5, p.shape)
    x = rng.normal(0, 1, (40, N_FEAT))
    a = (rng.random(40) < 0.5).astype(float)
    adv = rng.normal(0, 1, 40)

    def loss() -> float:
        z, _ = net.forward(x)
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.sum(adv * (a * np.log(p) + (1 - a) * np.log(1 - p))))

    z, cache = net.forward(x)
    p = 1.0 / (1.0 + np.exp(-z))
    grads = net.backward(adv * (a - p), cache)
    worst, eps = 0.0, 1e-6
    for param, grad in zip(net.params, grads, strict=True):
        flat, gflat = param.reshape(-1), grad.reshape(-1)
        for k in range(flat.size):
            old = flat[k]
            flat[k] = old + eps
            up = loss()
            flat[k] = old - eps
            dn = loss()
            flat[k] = old
            num = (up - dn) / (2 * eps)
            worst = max(worst, abs(num - gflat[k]) / max(1e-8, abs(num) + abs(gflat[k])))
    return worst


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float) -> None:
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        """Gradient ASCENT -- the objective is a reward, not a loss."""
        self.t += 1
        b1, b2 = 0.9, 0.999
        for p, g, m, v in zip(params, grads, self.m, self.v, strict=True):
            m[...] = b1 * m + (1 - b1) * g
            v[...] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1**self.t)
            vh = v / (1 - b2**self.t)
            p += self.lr * mh / (np.sqrt(vh) + 1e-8)


# ---------------------------------------------------------------- training


def act_probs(net: MLP, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """P(act) at the given step indices, computed in chunks to bound memory."""
    out = np.empty(rows.size, dtype=np.float64)
    for s in range(0, rows.size, 400_000):
        z, _ = net.forward(x[rows[s : s + 400_000]])
        out[s : s + 400_000] = 1.0 / (1.0 + np.exp(-z))
    return out


def greedy(net: MLP, x: np.ndarray, sim: Sim) -> np.ndarray:
    """Deterministic policy: act wherever P(act) > 1/2, at legal steps only."""
    legal = np.flatnonzero(sim.cut_legal | sim.add_legal)
    fires = np.zeros(sim.book.n_steps, dtype=bool)
    fires[legal] = act_probs(net, x, legal) > 0.5
    return fires


def train(
    x: np.ndarray,
    sim: Sim,
    train_paths: np.ndarray,
    hidden: int,
    seed: int,
    iters: int,
    batch: int,
    lr: float,
    mask_cols: list[int] | None = None,
) -> tuple[MLP, list[float]]:
    """REINFORCE on the paths in `train_paths`. Returns (net, per-iter mean reward).

    `mask_cols` zeroes those input columns for this network -- the ablation
    arms, which blind the policy to one channel while leaving everything else,
    the optimiser included, identical.
    """
    rng = np.random.default_rng(seed)
    net = MLP(hidden, rng)
    opt = Adam(net.params, lr)
    book = sim.book
    legal_all = sim.cut_legal | sim.add_legal
    step_path = np.repeat(np.arange(book.n_paths), book.stop - book.starts)
    xin = x if not mask_cols else _masked(x, mask_cols)
    history = []
    for _ in range(iters):
        chosen = rng.choice(train_paths, size=min(batch, train_paths.size), replace=False)
        in_batch = np.zeros(book.n_paths, dtype=bool)
        in_batch[chosen] = True
        rows = np.flatnonzero(legal_all & in_batch[step_path])
        z, cache = net.forward(xin[rows])
        p = 1.0 / (1.0 + np.exp(-z))
        a = rng.random(rows.size) < p
        fires = np.zeros(book.n_steps, dtype=bool)
        fires[rows] = a
        net_krw, exit_at, add_at = sim.run_steps(fires)
        r = net_krw[chosen]
        history.append(float(r.mean()))
        # Baseline = the batch mean; scale by its spread so the step size does
        # not depend on how large a KRW number the book happens to produce.
        adv_path = np.zeros(book.n_paths)
        adv_path[chosen] = (r - r.mean()) / (r.std() + 1e-9)
        # A step is part of the trajectory only if the policy was actually
        # asked there: at or before the exit, and -- for an add decision -- at
        # or before the add, since an add happens at most once.
        t_path = step_path[rows]
        taken = rows <= exit_at[t_path]
        is_add = sim.add_legal[rows]
        added = add_at[t_path]
        taken &= ~is_add | (added < 0) | (rows <= added)
        g = np.where(taken, adv_path[t_path] * (a.astype(float) - p), 0.0)
        grads = net.backward(g / chosen.size, cache)
        opt.step(net.params, grads)
    return net, history


def pooled_z(x: np.ndarray, sim: Sim, train_paths: np.ndarray) -> np.ndarray:
    """`x` standardised with the pool-wide mean and sd of the training fold.

    Exactly `bandit.fit`'s `means = X.mean(axis=0)`, `stds = X.std(axis=0)`,
    estimated on the legal steps of the training paths only (so the fold's own
    test month does not leak into its scaling). Used ONLY by the `mlp8_pooled`
    arm, which exists to measure this, not to be adopted.
    """
    book = sim.book
    in_tr = np.zeros(book.n_paths, dtype=bool)
    in_tr[train_paths] = True
    step_path = np.repeat(np.arange(book.n_paths), book.stop - book.starts)
    rows = np.flatnonzero((sim.cut_legal | sim.add_legal) & in_tr[step_path])
    mu = x[rows].mean(axis=0)
    sd = x[rows].std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _masked(x: np.ndarray, cols: list[int]) -> np.ndarray:
    y = x.copy()
    y[:, cols] = 0.0
    return y


def scramble_features(x: np.ndarray, book: Book, rng: np.random.Generator) -> np.ndarray:
    """Every path's non-price channels replaced by another path's, tiled to length.

    The same null `seqexit_core.scramble` builds for the table policy, on the
    continuous columns: price path, legality and reward untouched; only the
    alignment of the information channels with THIS path's prices destroyed.
    """
    y = x.copy()
    lens = book.stop - book.starts
    donor = rng.permutation(book.n_paths)
    idx = np.empty(book.n_steps, dtype=np.int64)
    for i in range(book.n_paths):
        d = donor[i]
        src = np.arange(book.starts[d], book.stop[d])
        need = int(lens[i])
        reps = -(-need // src.size)
        idx[book.starts[i] : book.stop[i]] = np.tile(src, reps)[:need]
    y[:, SCRAMBLE_IDX] = x[idx][:, SCRAMBLE_IDX]
    return y


# ---------------------------------------------------------------- reporting


def summary(net_krw: np.ndarray, days: np.ndarray) -> dict:
    recs = day_records(days, net_krw)
    nets = np.array([r["net"] for r in recs])
    return {
        "trades": int(net_krw.size),
        "mean_bp": float(net_krw.mean() / NOTIONAL * 1e4),
        "worst_day_krw": float(nets.min()),
        "loser_ratio": float((net_krw < 0).mean()),
    }


def load_or_build(
    paths: FsPath, ma: FsPath, cache: FsPath | None, limit: int
) -> tuple[Book, np.ndarray, np.ndarray]:
    """(book, per-path round-trip cost, feature matrix), cached after the first build.

    Parsing 490 MB of path JSON costs ~5 GB of Python objects per process;
    several arms doing it at once is what exhausts the machine. The reduced
    book and the feature matrix are ~1.5 GB of plain arrays, so one process
    builds them and every arm loads the result. The raw rows are dropped the
    moment the per-path cost has been read from them.
    """
    if cache is not None and cache.exists() and not limit:
        with cache.open("rb") as f:
            return pickle.load(f)
    rows = [json.loads(line) for line in paths.read_text().splitlines() if line]
    if limit:
        rows = rows[:limit]
    book, keep_idx = build(rows)
    cost = real_cost([rows[i] for i in keep_idx], stock_markets())
    del rows
    x = feature_matrix(book, json.loads(ma.read_text()))
    if cache is not None and not limit:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump((book, cost, x), f, protocol=pickle.HIGHEST_PROTOCOL)
    return book, cost, x


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=FsPath, required=True)
    ap.add_argument("--ma", type=FsPath, required=True)
    ap.add_argument(
        "--windows", type=FsPath, default=FsPath("data/eval/rlnews/FRESH_CONFIRM_WINDOW.json")
    )
    ap.add_argument("--out", type=FsPath, required=True)
    ap.add_argument("--arms", default="lin,mlp8,mlp8_novol,mlp8_noma")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", type=FsPath, default=None, help="reduced book + features pickle")
    ap.add_argument("--build-only", action="store_true", help="build the cache and exit")
    args = ap.parse_args()

    for h in (0, 8):
        err = grad_check(h)
        print(f"grad_check hidden={h}: max relative error {err:.2e}")
        if err > 1e-5:
            raise SystemExit("hand-written backward pass disagrees with finite differences")

    book, cost, x = load_or_build(args.paths, args.ma, args.cache, args.limit)
    if args.build_only:
        print(f"cache built: {book.n_paths} paths, {book.n_steps:,} steps -> {args.cache}")
        return
    sim = Sim(book, LATTICES["novol"], cost)
    print(
        f"paths {book.n_paths}, steps {book.n_steps:,}, features {N_FEAT},"
        f" round trip {cost.mean() * 1e4:.1f} bp"
    )

    win = json.loads(args.windows.read_text())["track_d_fresh"]
    confirm_days = set(win["confirm"])
    lo, hi = win["search"]
    is_confirm = np.isin(book.day, sorted(confirm_days))
    is_search = np.array([lo <= d <= hi for d in book.day.tolist()])
    if (is_search & is_confirm).any() or not (is_search | is_confirm).all():
        raise SystemExit("frozen windows do not partition the book")
    months = sorted({d[:7] for d in book.day[is_search].tolist()})
    folds = months[1:]
    print(
        f"search {int(is_search.sum())} trades, confirm {int(is_confirm.sum())};"
        f" months {months}; folds {folds}"
    )

    arms = {
        "lin": (0, None),
        "mlp8": (8, None),
        "mlp8_novol": (8, VOLUME_IDX),
        "mlp8_noma": (8, MA_IDX),
        # The `application/trading/bandit.py::fit` defect, reproduced on purpose:
        # every column z-scored with the TRAINING POOL's mean and sd. Run beside
        # `mlp8` (fixed constants, nothing estimated) it measures what pooled
        # standardisation does to a state that is already per-stock ratios.
        "mlp8_pooled": (8, None),
    }
    result: dict = {
        "features": [{"name": n, "built_as": b, "normalisation": w} for n, b, w in FEATURES],
        "input_scale": INPUT_SCALE.tolist(),
        "arms": {},
    }
    hold = np.zeros(book.n_steps, dtype=bool)
    hold_net = sim.run_steps(hold)[0]

    for arm in args.arms.split(","):
        hidden, mask = arms[arm]
        t0 = time.time()
        n_params = MLP(hidden, np.random.default_rng(0)).n_params
        print(
            f"\n######## {arm}: hidden={hidden}, {n_params} parameters,"
            f" {int(is_search.sum()) / n_params:.0f} search trades per parameter,"
            f" masked={[FEATURES[i][0] for i in mask] if mask else None}"
        )
        xin = x if not mask else _masked(x, mask)

        fold_rows, oos_net, oos_days = [], [], []
        for m in folds:
            tr = np.flatnonzero(is_search & np.array([d[:7] < m for d in book.day.tolist()]))
            ev = is_search & np.array([d[:7] == m for d in book.day.tolist()])
            xf = pooled_z(x, sim, tr) if arm == "mlp8_pooled" else x
            xf_in = xf if not mask else _masked(xf, mask)
            net, hist = train(xf, sim, tr, hidden, 0, args.iters, args.batch, args.lr, mask)
            fires = greedy(net, xf_in, sim)
            nk, ex, ad = sim.run_steps(fires)
            cut = (ex < book.last_idx)[ev]
            fold_rows.append(
                {
                    "fold": m,
                    "n": int(ev.sum()),
                    "train_last_bp": float(np.mean(hist[-20:]) / NOTIONAL * 1e4),
                    "seq_bp": float(nk[ev].mean() / NOTIONAL * 1e4),
                    "hold_bp": float(hold_net[ev].mean() / NOTIONAL * 1e4),
                    "cut_rate": float(cut.mean()),
                    "add_rate": float((ad >= 0)[ev].mean()),
                }
            )
            oos_net.append(nk[ev])
            oos_days.append(book.day[ev])
            r = fold_rows[-1]
            print(
                f"  {m} n={r['n']:5d}  nn {r['seq_bp']:+7.1f}  hold {r['hold_bp']:+7.1f}"
                f"  cut {r['cut_rate']:.3f} add {r['add_rate']:.3f}",
                flush=True,
            )
        pooled = summary(np.concatenate(oos_net), np.concatenate(oos_days))
        print(f"  pooled search-OOS: nn {pooled['mean_bp']:+.1f} bp over {pooled['trades']}")

        # ---- shuffled-feature null on the inner holdout ------------------
        inner = months[-1]
        itr = np.flatnonzero(is_search & np.array([d[:7] < inner for d in book.day.tolist()]))
        iev = is_search & np.array([d[:7] == inner for d in book.day.tolist()])
        net, hist = train(x, sim, itr, hidden, 0, args.iters, args.batch, args.lr, mask)
        real_test = float(sim.run_steps(greedy(net, xin, sim))[0][iev].mean() / NOTIONAL * 1e4)
        real_train = float(np.mean(hist[-20:]) / NOTIONAL * 1e4)
        sh_test, sh_train = [], []
        for rep in range(args.shuffles):
            xs = scramble_features(x, book, np.random.default_rng(1000 + rep))
            net_s, hist_s = train(xs, sim, itr, hidden, 0, args.iters, args.batch, args.lr, mask)
            xs_in = xs if not mask else _masked(xs, mask)
            sh_test.append(
                float(sim.run_steps(greedy(net_s, xs_in, sim))[0][iev].mean() / NOTIONAL * 1e4)
            )
            sh_train.append(float(np.mean(hist_s[-20:]) / NOTIONAL * 1e4))
            if (rep + 1) % 5 == 0:
                print(
                    f"    scramble {rep + 1}/{args.shuffles}  {time.time() - t0:.0f}s", flush=True
                )
        shuffle = {
            "real_train_bp": real_train,
            "real_test_bp": real_test,
            "noise_test": {
                "mean": float(np.mean(sh_test)) if sh_test else None,
                "p90": float(np.quantile(sh_test, 0.9)) if sh_test else None,
                "max": float(np.max(sh_test)) if sh_test else None,
            },
            "noise_train": {
                "mean": float(np.mean(sh_train)) if sh_train else None,
                "max": float(np.max(sh_train)) if sh_train else None,
            },
        }
        if sh_test:
            print(
                f"  shuffle: real test {real_test:+.2f} vs noise mean/p90/max"
                f" {shuffle['noise_test']['mean']:+.2f}/{shuffle['noise_test']['p90']:+.2f}/"
                f"{shuffle['noise_test']['max']:+.2f}",
                flush=True,
            )

        block = {
            "hidden": hidden,
            "n_params": n_params,
            "masked": [FEATURES[i][0] for i in mask] if mask else [],
            "folds": fold_rows,
            "pooled": pooled,
            "shuffle": shuffle,
        }

        # ---- the frozen confirm slice, one fit per seed, touched once -----
        if args.confirm:
            sp = np.flatnonzero(is_search)
            cdays = book.day[is_confirm]
            seeds = []
            xc = pooled_z(x, sim, sp) if arm == "mlp8_pooled" else x
            xc_in = xc if not mask else _masked(xc, mask)
            for s in range(args.seeds):
                net, hist = train(xc, sim, sp, hidden, s, args.iters, args.batch, args.lr, mask)
                nk, ex, ad = sim.run_steps(greedy(net, xc_in, sim))
                seeds.append(
                    {
                        "seed": s,
                        "confirm_bp": float(nk[is_confirm].mean() / NOTIONAL * 1e4),
                        "cut_rate": float((ex < book.last_idx)[is_confirm].mean()),
                        "add_rate": float((ad >= 0)[is_confirm].mean()),
                        "net": nk[is_confirm],
                    }
                )
                print(
                    f"  CONFIRM seed {s}: {seeds[-1]['confirm_bp']:+.1f} bp"
                    f"  cut {seeds[-1]['cut_rate']:.3f} add {seeds[-1]['add_rate']:.3f}"
                )
            # Judge the MEDIAN seed, not the best: picking the best of three
            # seeds on the confirm slice is a selection on the confirm slice.
            order = np.argsort([s["confirm_bp"] for s in seeds])
            med = seeds[int(order[len(order) // 2])]
            cand = {d["day"]: d for d in day_records(cdays, med["net"])}
            base_hold = {d["day"]: d for d in day_records(cdays, hold_net[is_confirm])}
            block["confirm"] = {
                "seeds": [{k: v for k, v in s.items() if k != "net"} for s in seeds],
                "median_seed": med["seed"],
                "median_bp": med["confirm_bp"],
                "hold_bp": float(hold_net[is_confirm].mean() / NOTIONAL * 1e4),
                "vs_hold": judge(cand, base_hold),
                "daily": sorted(cand.values(), key=lambda d: d["day"]),
            }
            v = block["confirm"]["vs_hold"]
            print(f"  median seed vs hold: p(mu>0)={v['p_mu_gt_0']:.3f} adopt={v['adopt']}")

        block["seconds"] = time.time() - t0
        result["arms"][arm] = block
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"  -> {args.out} ({block['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
