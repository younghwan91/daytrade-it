"""Shared machinery for pipeline stages 2 (entry timing) and 3 (exit timing).

Both stages are the same problem seen from two ends: an optimal-stopping
decision taken once per completed minute. Stage 2 stops a WAIT (buy now or
keep watching), stage 3 stops a HOLD (sell now or keep holding). So both share
one book layout, one trainer and one evaluation, and differ only in which
steps exist, what a step pays, and which features the state carries.

**Live parity, by construction.**
  * A decision at minute `m` reads only bars whose minute is strictly before
    `m` (`exit_rule_hit`'s `b.ts < minute`), and transacts at bar `m`'s open --
    the SimBroker quote rule `prompt_labels.entry_price` and
    `exit_fast.reduce_path` both use. The immediate entry is the recorded
    `buy_price` at `buy_minute`, exactly the label convention.
  * Exits are `exit_fast.reduce_path`'s decision minutes and fills, verbatim:
    the first print at or after `flatten_at` closes the position, or the last
    close when the day ends before it.
  * `entry_end`, `flatten_at` and the live VWAP-exit parameters are read from
    `TradingLimits()` -- never re-typed here.
  * Every simulated exit is held against `exit_fast.first_exit` on EVERY
    (path, entry) pair the stage evaluates (`gate_*`), not a sample.

**Normalisation.** Every return-like input is divided by THIS stock's own
running one-minute volatility (last <= 30 completed returns), or -- for the
two inputs anchored on the previous close -- by its own trailing 60-day daily
volatility. No pool-wide mean/sd is estimated anywhere, and no volatility
level is an input. The only rescaling is the fixed `_squash`.

**Reward weighting.** Net return variance grows with the stock's volatility,
so training is weighted least-squares-style: each path's weight is the inverse
of the reward variance of its daily-volatility quintile, estimated on the
TRAINING fold only. Reported numbers are unweighted means.

**Holdout.** `config.assert_no_holdout_labels` runs on the day set of every
path file right after the lines are read; holdout lines are dropped by their
`day` field before their bars are ever parsed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
REPO = EVAL.parents[1]
for p in (str(EVAL), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import config  # noqa: E402
from costs import round_trip  # noqa: E402

from daytrade_it.application.trading.auto_trader import TradingLimits  # noqa: E402

LIVE = TradingLimits()
OPEN_MINUTE = 9 * 60
FLATTEN_MINUTE = LIVE.flatten_at.hour * 60 + LIVE.flatten_at.minute
ENTRY_END_MINUTE = LIVE.entry_end.hour * 60 + LIVE.entry_end.minute
#: `exit_worst_day.OPEN_WINDOW_MIN`: the first five minutes pay the open spread.
OPEN_WINDOW_MIN = 5
#: Running per-stock minute volatility: window and floor, the values
#: `seqexit_core` uses so the two studies scale states identically.
SIGMA_BARS = 30
SIGMA_FLOOR = 0.0015
DVOL_WINDOW = 60
DVOL_FLOOR = 0.005

DEFAULT_PATHS = (
    Path.home() / "git/daytrade-it-exitseq/data/eval/exitseq/paths_prior.jsonl",
    Path.home() / "git/daytrade-it-exitseq/data/eval/exitseq/paths_fresh.jsonl",
)
CACHE = REPO / "data/eval/pipeline/timing_cache"

_DAY = re.compile(r'"day":\s*"(\d{4}-\d{2}-\d{2})"')


def log(*a) -> None:
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ bars


@dataclass
class Bars:
    """Every path's regular-session bars, concatenated; per-path scalars alongside."""

    day: np.ndarray
    ticker: np.ndarray
    key: np.ndarray
    buy_minute: np.ndarray
    buy_price: np.ndarray
    bstart: np.ndarray
    bstop: np.ndarray
    minute: np.ndarray
    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    v: np.ndarray

    @property
    def n(self) -> int:
        return self.day.size

    def path(self, i: int) -> tuple[np.ndarray, ...]:
        s = slice(self.bstart[i], self.bstop[i])
        return self.minute[s], self.o[s], self.h[s], self.lo[s], self.c[s], self.v[s]

    def row(self, i: int) -> dict:
        """The original jsonl row shape, for handing a path to `exit_fast`."""
        mi, o, h, lo, c, v = self.path(i)
        return {
            "day": str(self.day[i]),
            "ticker": str(self.ticker[i]),
            "key": str(self.key[i]),
            "buy_minute": int(self.buy_minute[i]),
            "buy_price": float(self.buy_price[i]),
            "bars": [
                [int(a), float(b), float(cc), float(d), float(e), int(f)]
                for a, b, cc, d, e, f in zip(mi, o, h, lo, c, v, strict=True)
            ],
            "features": {},
        }


def _fingerprint(files) -> str:
    h = hashlib.md5()
    for f in files:
        st = Path(f).stat()
        h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}".encode())
    h.update(",".join(sorted(config.holdout_days())).encode())
    return h.hexdigest()[:12]


def load_bars(files, stage: int) -> Bars:
    """Stream the path files once, dropping holdout days before parsing bars.

    Cached as one npz keyed by the source files and the holdout set, so a
    changed holdout invalidates the cache instead of silently reusing it.
    """
    files = [Path(f) for f in files]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"bars_{_fingerprint(files)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        b = Bars(**{k: z[k] for k in z.files})
        config.assert_no_holdout_labels(set(b.day.tolist()), stage)
        log(f"bars cache {cache.name}: {b.n} paths")
        return b

    held = config.holdout_days()
    cols = defaultdict(list)
    bars_parts = [[] for _ in range(6)]
    n_skip = 0
    off = 0
    for f in files:
        with f.open() as fh:
            for line in fh:
                m = _DAY.search(line[:200])
                if m is None:
                    continue
                if m.group(1) in held:
                    n_skip += 1
                    continue
                r = json.loads(line)
                b = np.asarray(r["bars"], dtype=float)
                if b.size == 0:
                    continue
                b = b[np.argsort(b[:, 0], kind="stable")]
                cols["day"].append(r["day"])
                cols["ticker"].append(r["ticker"])
                cols["key"].append(r["key"])
                cols["buy_minute"].append(int(r["buy_minute"]))
                cols["buy_price"].append(float(r["buy_price"]))
                cols["bstart"].append(off)
                off += b.shape[0]
                cols["bstop"].append(off)
                for k in range(6):
                    bars_parts[k].append(b[:, k])
        log(f"read {f.name}: {len(cols['day'])} kept so far, {n_skip} holdout lines skipped")
    config.assert_no_holdout_labels(set(cols["day"]), stage)
    cat = [np.concatenate(p) for p in bars_parts]
    b = Bars(
        day=np.asarray(cols["day"]),
        ticker=np.asarray(cols["ticker"]),
        key=np.asarray(cols["key"]),
        buy_minute=np.asarray(cols["buy_minute"], dtype=np.int64),
        buy_price=np.asarray(cols["buy_price"]),
        bstart=np.asarray(cols["bstart"], dtype=np.int64),
        bstop=np.asarray(cols["bstop"], dtype=np.int64),
        minute=cat[0].astype(np.int64),
        o=cat[1],
        h=cat[2],
        lo=cat[3],
        c=cat[4],
        v=cat[5],
    )
    np.savez(cache, **b.__dict__)
    log(f"wrote {cache.name}: {b.n} paths")
    return b


# ------------------------------------------------------------------ aux data


def aux(bars: Bars) -> dict:
    """Market per code; raw previous close and trailing daily volatility per (day, code).

    Both daily quantities are computed from sessions STRICTLY BEFORE the
    decision day, so neither can carry that day's outcome. Volatility is taken
    on the adjusted series (splits would otherwise read as crashes); the
    previous close on the raw one, because the minute bars are raw.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = sorted(
        {f"{d}|{t}" for d, t in zip(bars.day.tolist(), bars.ticker.tolist(), strict=True)}
    )
    fp = hashlib.md5("\n".join(keys).encode()).hexdigest()[:12]
    cache = CACHE / f"aux_{fp}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    import edge_db  # noqa: PLC0415

    codes = sorted({k.split("|")[1] for k in keys})
    lo = dt.date.fromisoformat(min(bars.day.tolist())) - dt.timedelta(days=120)
    hi = dt.date.fromisoformat(max(bars.day.tolist()))
    conn = edge_db.connect()
    cur = conn.cursor()
    cur.execute("SELECT code, market FROM stocks")
    markets = {c: m for c, m in cur.fetchall() if m}
    series = {}
    for table in ("daily_bars", "daily_bars_adjusted"):
        cur.execute(
            f"SELECT code, date, close FROM {table} WHERE code = ANY(%s) AND date >= %s AND date < %s "  # noqa: S608
            "ORDER BY code, date",
            (codes, lo, hi),
        )
        s = defaultdict(list)
        for code, d, c in cur.fetchall():
            if c:
                s[code].append((str(d), float(c)))
        series[table] = s
    conn.close()

    prev_close, dvol = {}, {}
    by_code = defaultdict(list)
    for k in keys:
        d, c = k.split("|")
        by_code[c].append(d)
    for code, days in by_code.items():
        raw = series["daily_bars"].get(code, [])
        adj = series["daily_bars_adjusted"].get(code, [])
        rd = [x[0] for x in raw]
        ad = [x[0] for x in adj]
        ac = np.asarray([x[1] for x in adj])
        for d in days:
            i = int(np.searchsorted(rd, d, side="left"))
            if i > 0:
                prev_close[f"{d}|{code}"] = raw[i - 1][1]
            j = int(np.searchsorted(ad, d, side="left"))
            w = ac[max(0, j - DVOL_WINDOW - 1) : j]
            if w.size >= 11:
                r = w[1:] / w[:-1] - 1.0
                dvol[f"{d}|{code}"] = max(float(r.std()), DVOL_FLOOR)
    out = {"markets": markets, "prev_close": prev_close, "dvol": dvol}
    cache.write_text(json.dumps(out))
    log(f"aux: {len(markets)} markets, {len(prev_close)} prev closes, {len(dvol)} daily vols")
    return out


_COST: dict = {}


def cost_of(day: str, market: str, minute: int) -> float:
    k = (day, market, (minute - OPEN_MINUTE) < OPEN_WINDOW_MIN)
    if k not in _COST:
        _COST[k] = round_trip(day, market, at_open=k[2])
    return _COST[k]


# ------------------------------------------------------------------ per-path reduction


def running_sigma(c: np.ndarray) -> np.ndarray:
    """sigma[j]: std of the last <= SIGMA_BARS one-minute returns ending at bar j.

    Fewer than two returns -> the floor, as `seqexit_core.reduce_row` does.
    """
    r = np.zeros(c.size)
    r[1:] = c[1:] / c[:-1] - 1.0
    cs = np.concatenate([[0.0], np.cumsum(r)])
    cs2 = np.concatenate([[0.0], np.cumsum(r * r)])
    j = np.arange(c.size)
    lo = np.maximum(1, j + 1 - SIGMA_BARS)  # returns r[lo..j]
    n = j + 1 - lo
    s1 = cs[j + 1] - cs[lo]
    s2 = cs2[j + 1] - cs2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.where(n >= 2, s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2, 0.0)
    return np.maximum(np.sqrt(np.maximum(var, 0.0)), SIGMA_FLOOR)


def trailing_mean(x: np.ndarray, j: np.ndarray, w: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.maximum(j + 1 - w, 0)
    return (cs[j + 1] - cs[lo]) / np.maximum(j + 1 - lo, 1)


def exit_grid(mi: np.ndarray, o: np.ndarray, c: np.ndarray, after: int):
    """`exit_fast.reduce_path`'s decision minutes for a position filled at `after`.

    Returns (k: bar index of each decision minute, j: last completed bar index
    before it, fill) or None when no decision minute exists.
    """
    later = np.flatnonzero(mi > after)
    if later.size == 0:
        return None
    at_or = later[mi[later] >= FLATTEN_MINUTE]
    close_at = int(at_or[0]) if at_or.size else int(later[-1])
    real = bool(at_or.size)
    k = later[mi[later] <= mi[close_at]]
    j = np.searchsorted(mi, mi[k], side="left") - 1
    ok = j >= 0
    k, j = k[ok], j[ok]
    if k.size == 0:
        return None
    fill = o[k].copy()
    if not real:
        fill[-1] = c[k[-1]]
    return k, j, fill


def vwap_series(c: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vwap[j] over bars 0..j, valid[j]) with `reduce_path`'s exact float ops."""
    cum_vol = np.cumsum(v)
    cum_pv = np.cumsum(c * v)
    with np.errstate(divide="ignore", invalid="ignore"):
        vw = np.where(cum_vol > 0, cum_pv / np.where(cum_vol > 0, cum_vol, 1.0), np.nan)
    return vw, np.isfinite(vw)


def _squash(x: np.ndarray, scale: float = 5.0) -> np.ndarray:
    """Fixed, sample-independent rescale: clip at +-4 scales, then divide."""
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) / scale, -4.0, 4.0)


# ------------------------------------------------------------------ stopping book


@dataclass
class Book:
    """Stopping steps of every path, concatenated. Path `i` owns `starts[i]:stops[i]`.

    The last step of every path is FORCED (buy at the horizon / sell at the
    flatten minute); `legal` marks the steps where the policy may choose to stop.
    """

    starts: np.ndarray
    stops: np.ndarray
    X: np.ndarray  # float32 (steps, d)
    R: np.ndarray  # float64 net return if the path stops here
    legal: np.ndarray  # bool
    minute: np.ndarray  # int, the minute a stop here transacts at
    fill: np.ndarray  # price a stop here transacts at
    path: np.ndarray  # per step: path index into the owning table
    names: tuple

    @property
    def n_paths(self) -> int:
        return self.starts.size

    @property
    def n_steps(self) -> int:
        return self.R.size

    def lens(self) -> np.ndarray:
        return self.stops - self.starts

    def subset(self, sel: np.ndarray) -> Book:
        """The book restricted to the paths `sel` (indices), steps kept contiguous."""
        idx = seg_index(self.starts, self.lens(), sel)
        lens = self.lens()[sel]
        stops = np.cumsum(lens)
        return Book(
            starts=stops - lens,
            stops=stops,
            X=self.X[idx],
            R=self.R[idx],
            legal=self.legal[idx],
            minute=self.minute[idx],
            fill=self.fill[idx],
            path=self.path[idx],
            names=self.names,
        )


def seg_index(starts: np.ndarray, lens: np.ndarray, sel: np.ndarray) -> np.ndarray:
    ls = lens[sel]
    if ls.sum() == 0:
        return np.zeros(0, dtype=np.int64)
    base = np.repeat(starts[sel] - np.concatenate([[0], np.cumsum(ls)[:-1]]), ls)
    return base + np.arange(ls.sum())


def first_stop(book: Book, fire: np.ndarray) -> np.ndarray:
    """Step index where each path stops: first legal firing step, else its last step."""
    big = np.iinfo(np.int64).max // 4
    idx = np.arange(book.n_steps, dtype=np.int64)
    cand = np.where(fire & book.legal, idx, big)
    first = np.minimum.reduceat(cand, book.starts) if book.n_steps else np.zeros(0, np.int64)
    return np.minimum(first, book.stops - 1)


# ------------------------------------------------------------------ trainer


class Policy:
    """Linear or one-hidden-layer tanh MLP logit; `fit` maximises the EXACT
    expected reward of the stochastic stopping policy.

    With stop probability p_t at step t and the last step forced, the path
    stops at t with probability P_t = p_t * prod_{k<t}(1 - p_k), and
    E[reward] = sum_t P_t R_t is a smooth function of the logits. Its gradient
    is analytic:

        dJ/dz_t = R_t E_t p_t (1 - p_t) - p_t * sum_{j>t} P_j R_j,
        E_t = prod_{k<t}(1 - p_k),

    so there is no sampling noise (unlike REINFORCE) and no value function to
    bootstrap. Evaluation is DETERMINISTIC -- stop at the first legal step with
    logit > 0 -- which is what a live daemon would run.
    """

    def __init__(self, d: int, hidden: int, seed: int, b0: float) -> None:
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        if hidden:
            self.p = {
                "W1": rng.normal(0, 1 / np.sqrt(d), (d, hidden)),
                "b1": np.zeros(hidden),
                "w2": rng.normal(0, 0.1, hidden),
                "b2": np.array([b0]),
            }
        else:
            self.p = {"w": np.zeros(d), "b": np.array([b0])}

    def logits(self, X: np.ndarray, keep: bool = False):
        if self.hidden:
            H = np.tanh(X @ self.p["W1"].astype(np.float32) + self.p["b1"].astype(np.float32))
            z = H @ self.p["w2"].astype(np.float32) + self.p["b2"][0]
            return (z.astype(np.float64), H) if keep else z.astype(np.float64)
        z = (X @ self.p["w"].astype(np.float32)).astype(np.float64) + self.p["b"][0]
        return (z, None) if keep else z

    def grads(self, X, H, gz) -> dict:
        if self.hidden:
            g = {"w2": H.T.astype(np.float64) @ gz, "b2": np.array([gz.sum()])}
            gH = (gz[:, None] * self.p["w2"][None, :]) * (1.0 - H.astype(np.float64) ** 2)
            g["W1"] = X.T.astype(np.float64) @ gH
            g["b1"] = gH.sum(0)
            return g
        return {"w": X.T.astype(np.float64) @ gz, "b": np.array([gz.sum()])}

    def fit(
        self,
        book: Book,
        w_path: np.ndarray,
        *,
        iters: int,
        lr: float,
        batch: int,
        l2: float,
        seed: int,
        pool: np.ndarray | None = None,
    ) -> list[float]:
        """Adam ascent on minibatches of whole paths drawn from `pool` (default all).

        `w_path` is indexed like the book's paths."""
        rng = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        s = {k: np.zeros_like(v) for k, v in self.p.items()}
        lens = book.lens()
        hist = []
        for it in range(1, iters + 1):
            pool_ = np.arange(book.n_paths) if pool is None else pool
            sel = pool_[rng.choice(pool_.size, size=min(batch, pool_.size), replace=False)]
            idx = seg_index(book.starts, lens, sel)
            ls = lens[sel]
            st = np.concatenate([[0], np.cumsum(ls)[:-1]])
            X, R, legal = book.X[idx], book.R[idx], book.legal[idx]
            last = np.zeros(idx.size, bool)
            last[st + ls - 1] = True
            wstep = np.repeat(w_path[sel], ls)
            z, H = self.logits(X, keep=True)
            J, gz = expected_reward(z, R, legal, last, st, ls, wstep)
            g = self.grads(X, H, gz)
            for k in self.p:
                gk = g[k] / w_path[sel].sum()
                if k in ("w", "W1", "w2"):
                    gk = gk - l2 * self.p[k]
                m[k] = 0.9 * m[k] + 0.1 * gk
                s[k] = 0.999 * s[k] + 0.001 * gk * gk
                mh = m[k] / (1 - 0.9**it)
                sh = s[k] / (1 - 0.999**it)
                self.p[k] = self.p[k] + lr * mh / (np.sqrt(sh) + 1e-8)  # ascent
            if it % max(1, iters // 5) == 0:
                hist.append(float(J))
        return hist


def expected_reward(z, R, legal, last, st, ls, wstep):
    """(weighted mean expected reward, dJ/dz per step) for contiguous segments."""
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    p = np.where(legal, p, 0.0)
    p = np.where(last, 1.0, p)
    q = np.log(np.clip(1.0 - p, 1e-12, None))
    q = np.where(last, 0.0, q)
    cq = np.cumsum(q)
    base = np.repeat(cq[st] - q[st], ls)
    S = cq - q - base  # exclusive within segment
    E = np.exp(S)
    P = p * E
    A = P * R
    cA = np.cumsum(A)
    tot = np.repeat(cA[st + ls - 1], ls)
    after = tot - cA  # sum_{j>t} A_j within segment
    dp = np.where(legal & ~last, p * (1 - p), 0.0)
    gz = wstep * (R * E * dp - np.where(legal & ~last, p, 0.0) * after)
    end = cA[st + ls - 1]
    Jpath = end - np.concatenate([[0.0], end[:-1]])
    w_path = wstep[st]
    J = float((w_path * Jpath).sum() / w_path.sum())
    return J, gz


def wls_weights(dvol: np.ndarray, reward: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Per-path weight = 1 / reward variance of its daily-vol quintile (training fold only)."""
    qs = np.quantile(dvol[train], [0.2, 0.4, 0.6, 0.8])
    b = np.searchsorted(qs, dvol, side="right")
    var = np.array(
        [
            reward[train & (b == k)].var() if (train & (b == k)).sum() > 10 else np.nan
            for k in range(5)
        ]
    )
    var = np.where(np.isfinite(var) & (var > 0), var, np.nanmean(var))
    w = 1.0 / var[b]
    return w / w[train].mean()


# ------------------------------------------------------------------ evaluation


def day_cluster(days: np.ndarray, x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of `x` per trade and its day-clustered bootstrap SE (bp)."""
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=x, minlength=ud.size)
    n = np.bincount(inv, minlength=ud.size).astype(float)
    rng = np.random.default_rng(seed)
    b = rng.integers(0, ud.size, size=(n_boot, ud.size))
    boots = s[b].sum(1) / n[b].sum(1)
    return {
        "bp": 1e4 * float(x.mean()),
        "se_bp": 1e4 * float(boots.std()),
        "n": int(x.size),
        "days": int(ud.size),
    }


def month_of(day: str) -> str:
    return day[:7]


def within_day_donors(days: np.ndarray, seed: int) -> np.ndarray:
    """A fixed within-day permutation of path indices (the shuffle-noise donor map)."""
    rng = np.random.default_rng(seed)
    donor = np.arange(days.size)
    order = np.argsort(days, kind="stable")
    sd = days[order]
    cuts = np.flatnonzero(sd[1:] != sd[:-1]) + 1
    for grp in np.split(order, cuts):
        donor[grp] = grp[rng.permutation(grp.size)]
    return donor


def scramble_X(book: Book, donor_of_path: np.ndarray, cols: list[int]) -> np.ndarray:
    """Book features with columns `cols` of each path replaced by its donor's
    series (tiled/truncated to length). Prices, rewards, legality, clock
    columns untouched -- only the alignment of state with its own path is cut."""
    X = book.X.copy()
    lens = book.lens()
    for i in range(book.n_paths):
        d = donor_of_path[i]
        if d == i:
            continue
        need = lens[i]
        src = book.X[book.starts[d] : book.stops[d]][:, cols]
        reps = -(-need // src.shape[0])
        X[book.starts[i] : book.stops[i], cols] = (np.tile(src, (reps, 1)) if reps > 1 else src)[
            :need
        ]
    return X


# ------------------------------------------------------------------ folds + noise


def run_folds(
    book: Book,
    X: np.ndarray,
    R: np.ndarray,
    weights,
    months: np.ndarray,
    fold_months: list[str],
    spec: dict,
    seed: int,
) -> np.ndarray:
    """Per path: (out-of-fold deterministic stop step, -1 outside `fold_months`;
    out-of-fold EXPECTED net of the stochastic policy, nan outside).

    Expanding window: month `m` is decided by a policy fitted only on paths
    whose month is strictly earlier. `weights(train_mask)` returns per-path
    training weights computed from the training fold alone.
    """
    choice = np.full(book.n_paths, -1, dtype=np.int64)
    expected = np.full(book.n_paths, np.nan)
    view = Book(
        book.starts, book.stops, X, R, book.legal, book.minute, book.fill, book.path, book.names
    )
    b0 = -float(np.log(np.median(book.lens())))
    for m in fold_months:
        train = months < m
        test = np.flatnonzero(months == m)
        if not train.any() or test.size == 0:
            continue
        w = weights(train)
        pol = Policy(X.shape[1], spec["hidden"], seed, b0=b0)
        pol.fit(
            view,
            w,
            iters=spec["iters"],
            lr=spec["lr"],
            batch=spec["batch"],
            l2=spec["l2"],
            seed=seed,
            pool=np.flatnonzero(train),
        )
        tb = view.subset(test)
        z = pol.logits(tb.X)
        st = first_stop(tb, z > 0)
        choice[test] = book.starts[test] + (st - tb.starts)
        expected[test] = path_expected(tb, z)
    return choice, expected


def stop_probs(book: Book, z: np.ndarray) -> np.ndarray:
    """P(the stochastic policy stops at each step), last step forced."""
    last = np.zeros(book.n_steps, bool)
    last[book.stops - 1] = True
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    p = np.where(book.legal, p, 0.0)
    p = np.where(last, 1.0, p)
    q = np.where(last, 0.0, np.log(np.clip(1.0 - p, 1e-12, None)))
    cq = np.cumsum(q)
    S = cq - q - np.repeat(cq[book.starts] - q[book.starts], book.lens())
    return p * np.exp(S)


def path_expected(book: Book, z: np.ndarray) -> np.ndarray:
    P = stop_probs(book, z)
    return np.add.reduceat(P * book.R, book.starts)


def grad_check(seed: int = 0) -> float:
    """Max relative error of `expected_reward`'s analytic gradient vs central differences."""
    rng = np.random.default_rng(seed)
    ls = np.array([5, 1, 7, 3])
    st = np.concatenate([[0], np.cumsum(ls)[:-1]])
    n = ls.sum()
    z = rng.normal(0, 1, n)
    R = rng.normal(0, 0.02, n)
    legal = rng.random(n) > 0.2
    last = np.zeros(n, bool)
    last[st + ls - 1] = True
    w = np.repeat(rng.random(ls.size) + 0.5, ls)
    _, g = expected_reward(z, R, legal, last, st, ls, w)
    wsum = w[st].sum()
    err = 0.0
    for t in range(n):
        e = np.zeros(n)
        e[t] = 1e-6
        jp, _ = expected_reward(z + e, R, legal, last, st, ls, w)
        jm, _ = expected_reward(z - e, R, legal, last, st, ls, w)
        num = (jp - jm) / 2e-6
        err = max(err, abs(num - g[t] / wsum) / max(1e-9, abs(num) + abs(g[t] / wsum)))
    return err


def fit_final(book: Book, R: np.ndarray, w: np.ndarray, spec: dict, seed: int = 0) -> dict:
    """The policy fitted on every non-holdout path, for stage 4 to apply."""
    view = Book(
        book.starts,
        book.stops,
        book.X,
        R,
        book.legal,
        book.minute,
        book.fill,
        book.path,
        book.names,
    )
    pol = Policy(book.X.shape[1], spec["hidden"], seed, b0=-float(np.log(np.median(book.lens()))))
    pol.fit(
        view, w, iters=spec["iters"], lr=spec["lr"], batch=spec["batch"], l2=spec["l2"], seed=seed
    )
    return {
        "spec": spec,
        "features": list(book.names),
        "params": {k: v.tolist() for k, v in pol.p.items()},
    }


def evaluate(
    name: str,
    book: Book,
    R: np.ndarray,
    base: np.ndarray,
    days: np.ndarray,
    months: np.ndarray,
    fold_months: list[str],
    spec: dict,
    weights,
    scr_cols: list[int],
    n_noise: int,
    on_noise=None,
) -> tuple[dict, np.ndarray]:
    """One candidate beside its shuffle-noise best-of-N, all out-of-fold.

    `base` is the baseline's stop step per path (same population, same cost);
    every number is also reported as a paired, day-clustered difference to it.
    """
    oof = np.isin(months, fold_months)
    t0 = time.time()
    ch, ex = run_folds(book, book.X, R, weights, months, fold_months, spec, seed=0)
    diff = R[ch][oof] - R[base][oof]
    real = day_cluster(days[oof], R[ch][oof]) | {
        "diff_vs_base": day_cluster(days[oof], diff),
        "stochastic": day_cluster(days[oof], ex[oof]),
        "stochastic_diff_vs_base": day_cluster(days[oof], ex[oof] - R[base][oof]),
        "mean_minutes_vs_base": float((book.minute[ch] - book.minute[base])[oof].mean()),
        "frac_same_as_base": float((ch == base)[oof].mean()),
        "seconds": round(time.time() - t0),
    }
    log(
        f"{name}: det {real['bp']:+.1f}bp  diff {real['diff_vs_base']['bp']:+.1f} "
        f"+- {real['diff_vs_base']['se_bp']:.1f}  dmin {real['mean_minutes_vs_base']:+.0f}  "
        f"same {real['frac_same_as_base']:.2f} | stoch diff {real['stochastic_diff_vs_base']['bp']:+.1f} "
        f"+- {real['stochastic_diff_vs_base']['se_bp']:.1f}  {real['seconds']}s"
    )
    noise, noise_st = [], []
    for s in range(1, n_noise + 1):
        donor = within_day_donors(days, seed=1000 + s)
        Xn = scramble_X(book, donor, scr_cols)
        chn, exn = run_folds(book, Xn, R, weights, months, fold_months, spec, seed=s)
        del Xn
        noise.append(1e4 * float((R[chn][oof] - R[base][oof]).mean()))
        noise_st.append(1e4 * float((exn[oof] - R[base][oof]).mean()))
        log(
            f"  {name} noise {s}/{n_noise}: det diff {noise[-1]:+.1f}bp  stoch diff {noise_st[-1]:+.1f}bp"
        )
        if on_noise:
            on_noise(noise)
    nz = np.asarray(noise) if noise else np.full(1, np.nan)
    real["noise_diff_bp"] = {
        "n": len(noise),
        "mean": float(np.mean(nz)),
        "p90": float(np.quantile(nz, 0.9)),
        "max": float(np.max(nz)),
        "all": noise,
    }
    real["beats_noise_best_of_n"] = bool(noise) and real["diff_vs_base"]["bp"] > float(np.max(nz))
    ns = np.asarray(noise_st) if noise_st else np.full(1, np.nan)
    real["stochastic_noise_diff_bp"] = {
        "n": len(noise_st),
        "mean": float(np.mean(ns)),
        "p90": float(np.quantile(ns, 0.9)),
        "max": float(np.max(ns)),
        "all": noise_st,
    }
    real["stochastic_beats_noise_best_of_n"] = bool(noise_st) and real["stochastic_diff_vs_base"][
        "bp"
    ] > float(np.max(ns))
    return real, ch
