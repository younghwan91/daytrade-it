"""A genuinely sequential exit policy: hold / add / cut, decided every minute.

Two earlier attempts on this book lost to a fixed grid (`docs/research/
2026-09-22-rl-news-overlap.md` 2.3). Both chose a *parametric family* -- a
stop/target pair, constant or scaled by entry volatility -- and searched its
coefficients. This one searches a lookup table over the state the position is
actually in at each minute, and differs in three ways that were named before
anything ran:

1. **Volume is in the state.** Neither earlier attempt had it. `vsurge` is the
   last three completed bars' mean volume over the session's mean volume per
   bar so far -- the stock against its own baseline, not a raw level -- and
   `vtrend` is that three-bar mean over the three before it. Both are running
   aggregates the live daemon already accumulates for VWAP, so they cost
   nothing at decision time.

2. **The action space is asymmetric by construction.** `cut` is legal ONLY
   below the entry price and `add` ONLY at or above it. That is a structural
   constraint, not something the search is asked to discover: a symmetric
   trailing stop is *unreachable* from this action set, which is the point --
   that family already lost twice.

3. **No worst-day term anywhere in the objective.** The earlier tabular stop
   policy degenerated to cutting 87-92% of positions under a lambda-weighted
   day sum penalised by `min(day_nets)`. `rl_exits.objective` still carries
   exactly that shape (`sum(vals) - 0.5 * max(0, -min(vals))`) while
   `bayes_judge` gate 2 reads `min(net)` over days -- select on X, gate on X.
   Nothing here selects on a worst-day statistic; see `objective`.

Everything the policy reads at minute `m` comes from bars that ENDED BEFORE
`m`, the same `b.ts < minute` filter the live `exit_rule_hit` applies, so the
reduction below is decision-time by construction rather than by inspection.

`gate_exact` is the exactness test: driven with a pure stop-loss the sequential
walk must return the same exit minute as `exit_fast.first_exit`, which
`verify_twin.py` in turn holds against the live `exit_rule_hit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPEN_MINUTE = 9 * 60
FLATTEN_MINUTE = 15 * 60
#: `TradingLimits.entry_end` -- live refuses an add-on at or after 14:30.
ENTRY_END_MINUTE = 14 * 60 + 30
#: `TradingLimits.add_on_price_band`: the add-on limit must sit within 3% of the
#: last completed close, else `_add_on` skips it.
ADD_PRICE_BAND = 0.03
#: `add_on_hit` refuses before 20 completed bars.
ADD_MIN_BARS = 20
#: Per-leg notional. One share is one leg; an add-on is a second leg of the
#: same size, which is what `_add_on` buys (quantity 1, once per ticker).
NOTIONAL = 100_000.0

# ---------------------------------------------------------------- state lattice
#
# Edges are round numbers fixed before the first search and never tuned. The
# `pnl` split at exactly 0 is what makes the action space asymmetric: buckets
# 0-2 are strictly below the entry (cut legal, add illegal) and 3-4 at or above
# it (add legal, cut illegal).

PNL_EDGES = (-0.020, -0.007, 0.0, 0.015)  # -> 5 buckets
SLOPE_EDGES = (-0.003, 0.003)  # -> 3 buckets
VSURGE_EDGES = (0.7, 1.5)  # -> 3 buckets
VTREND_EDGES = (0.8, 1.25)  # -> 3 buckets

N_PNL, N_SLOPE, N_VSURGE, N_VTREND = 5, 3, 3, 3
#: pnl buckets where a cut is legal (strictly below entry).
CUT_BUCKETS = (0, 1, 2)
#: pnl buckets where an add is legal (at or above entry).
ADD_BUCKETS = (3, 4)


@dataclass(frozen=True)
class Book:
    """Every path's decision minutes concatenated, plus the per-path scalars.

    Index `t` runs over (path, decision minute) pairs; `starts[i]` is where path
    `i` begins. `stop[i] = starts[i+1]` so a path's slice is `starts[i]:stop[i]`.
    """

    starts: np.ndarray
    stop: np.ndarray
    fill: np.ndarray  # this minute's open -- what a decision here transacts at
    close: np.ndarray  # last COMPLETED bar's close
    minute: np.ndarray
    pnl_b: np.ndarray
    slope_b: np.ndarray
    vsurge_b: np.ndarray
    vtrend_b: np.ndarray
    add_ok: np.ndarray  # live `add_on_hit` + band + entry_end + >= entry
    valid: np.ndarray  # finite session VWAP, mirroring `exit_fast._fires`
    last_idx: np.ndarray  # per path: index of the flatten minute
    entry: np.ndarray  # per path
    day: np.ndarray  # per path
    ticker: np.ndarray  # per path
    # raw values kept for the exactness gate and for diagnostics
    pnl: np.ndarray
    vsurge: np.ndarray
    elapsed: np.ndarray

    @property
    def n_paths(self) -> int:
        return self.starts.size

    @property
    def n_steps(self) -> int:
        return self.fill.size


def _bucket(x: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Half-open buckets: bucket k is [edges[k-1], edges[k]). `searchsorted` with
    side='right' puts a value exactly ON an edge into the bucket ABOVE it, so
    `pnl == 0` lands in bucket 3 -- at the entry price is NOT a loser, and a cut
    there is illegal."""
    return np.searchsorted(np.asarray(edges), x, side="right").astype(np.int8)


def reduce_row(row: dict) -> dict | None:
    """One recorded entry reduced to its decision minutes.

    Mirrors `exit_fast.reduce_path` for the fields they share -- same decision
    minutes, same fill convention, same VWAP -- and adds the volume and slope
    channels. Divergence there would make the exactness gate meaningless, so
    the shared part is deliberately a transcription rather than a rewrite.
    """
    bars = np.asarray(row["bars"], dtype=float)
    if bars.size == 0:
        return None
    order = np.argsort(bars[:, 0], kind="stable")
    bars = bars[order]
    minutes = bars[:, 0].astype(int)
    opens, highs, lows, closes, vols = bars[:, 1], bars[:, 2], bars[:, 3], bars[:, 4], bars[:, 5]
    del highs, lows
    buy_minute = int(row["buy_minute"])

    cum_vol = np.cumsum(vols)
    cum_pv = np.cumsum(closes * vols)
    n_done = np.arange(1, minutes.size + 1, dtype=float)

    after = minutes[minutes > buy_minute]
    if after.size == 0:
        return None
    at_or_after = after[after >= FLATTEN_MINUTE]
    close_at = int(at_or_after[0]) if at_or_after.size else int(after[-1])
    real_flatten = bool(at_or_after.size)
    sel = (minutes > buy_minute) & (minutes <= close_at)
    if not sel.any():
        return None

    dec_minutes = minutes[sel]
    dec_fill = opens[sel].copy()
    if not real_flatten:
        # The day ran out before 15:00: the leftover is valued at the last
        # close, not sold into an open that does not exist.
        dec_fill[-1] = closes[sel][-1]
    j = np.searchsorted(minutes, dec_minutes, side="left") - 1
    ok = j >= 0
    if not ok.all():
        dec_minutes, dec_fill, j = dec_minutes[ok], dec_fill[ok], j[ok]
        if j.size == 0:
            return None

    vol_j = cum_vol[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vol_j > 0, cum_pv[j] / np.where(vol_j > 0, vol_j, 1.0), np.nan)

    entry = float(row["buy_price"])
    close_j = closes[j]
    pnl = close_j / entry - 1.0

    # --- volume, always against this stock's own running baseline -------------
    # `mean_vol` is the session's mean volume per completed bar. Dividing by it
    # removes both the stock's size and the day's overall activity, so `vsurge`
    # is "is this stock trading harder than it has been today", which is the
    # thing the owner's prior is about. A raw level would be neither comparable
    # across stocks nor stationary within a day.
    mean_vol = cum_vol[j] / n_done[j]
    win3 = _trailing_mean(vols, j, 3)
    prev3 = _trailing_mean(vols, np.maximum(j - 3, 0), 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        vsurge = np.where(mean_vol > 0, win3 / np.where(mean_vol > 0, mean_vol, 1.0), 1.0)
        vtrend = np.where(prev3 > 0, win3 / np.where(prev3 > 0, prev3, 1.0), 1.0)
    vsurge = np.nan_to_num(vsurge, nan=1.0, posinf=1.0)
    # Fewer than 6 completed bars cannot support a 3-vs-3 comparison; neutral.
    vtrend = np.where(j >= 5, np.nan_to_num(vtrend, nan=1.0, posinf=1.0), 1.0)

    j3 = np.maximum(j - 3, 0)
    slope = np.where(j >= 3, close_j / closes[j3] - 1.0, 0.0)

    ma5 = _trailing_mean(closes, j, 5)
    ma20 = _trailing_mean(closes, j, 20)
    # `add_on_hit` verbatim (>= 20 completed bars, close above VWAP, MA5 > MA20)
    # plus the two guards that live applies around it: the 3% band between the
    # add limit and the last close, and `now < entry_end`. `pnl >= 0` is the
    # structural constraint -- never add to a position that is down.
    with np.errstate(divide="ignore", invalid="ignore"):
        band = np.abs(dec_fill / np.where(close_j > 0, close_j, np.nan) - 1.0)
    add_ok = (
        (j + 1 >= ADD_MIN_BARS)
        & (vol_j > 0)
        & (close_j > vwap)
        & (ma5 > ma20)
        & (pnl >= 0.0)
        & (dec_minutes < ENTRY_END_MINUTE)
        & np.isfinite(band)
        & (band <= ADD_PRICE_BAND)
    )

    return {
        "fill": dec_fill,
        "close": close_j,
        "minute": dec_minutes,
        "pnl": pnl,
        "slope": slope,
        "vsurge": vsurge,
        "vtrend": vtrend,
        "add_ok": add_ok,
        "valid": np.isfinite(vwap),
        "entry": entry,
        "day": row["day"],
        "ticker": row["ticker"],
        "buy_minute": buy_minute,
    }


def _trailing_mean(values: np.ndarray, j: np.ndarray, w: int) -> np.ndarray:
    """Mean of `values` over the `w` entries ending at index `j` (inclusive).

    Short windows at the start of the day average what exists rather than
    returning NaN, which matches `add_on_hit`'s `done[-5:]` slicing; the
    `>= 20 bars` guard is what actually keeps a half-formed MA20 out of a
    decision.
    """
    cum = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.maximum(j + 1 - w, 0)
    return (cum[j + 1] - cum[lo]) / (j + 1 - lo)


def build(rows: list[dict]) -> tuple[Book, np.ndarray]:
    """(book, indices of the rows that survived).

    The indices are returned rather than left to be recovered by matching on
    (day, ticker): two articles about the same stock on the same day are two
    separate entries sharing that key, so matching on it silently re-admits a
    dropped row. Every array aligned to the book downstream -- the round-trip
    cost above all -- is built from these indices.
    """
    parts, entries, days, tickers, lens, keep_idx = [], [], [], [], [], []
    for i, row in enumerate(rows):
        red = reduce_row(row)
        if red is None:
            continue
        keep_idx.append(i)
        parts.append(red)
        entries.append(red["entry"])
        days.append(red["day"])
        tickers.append(red["ticker"])
        lens.append(red["fill"].size)

    lens_arr = np.asarray(lens, dtype=np.int64)
    stop = np.cumsum(lens_arr)
    starts = stop - lens_arr
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731

    pnl, slope, vsurge, vtrend = cat("pnl"), cat("slope"), cat("vsurge"), cat("vtrend")
    minute = cat("minute")
    buy = np.repeat(np.asarray([p["buy_minute"] for p in parts], dtype=np.int64), lens_arr)
    book = Book(
        starts=starts,
        stop=stop,
        fill=cat("fill"),
        close=cat("close"),
        minute=minute,
        pnl_b=_bucket(pnl, PNL_EDGES),
        slope_b=_bucket(slope, SLOPE_EDGES),
        vsurge_b=_bucket(vsurge, VSURGE_EDGES),
        vtrend_b=_bucket(vtrend, VTREND_EDGES),
        add_ok=cat("add_ok"),
        valid=cat("valid"),
        last_idx=stop - 1,
        entry=np.asarray(entries, dtype=float),
        day=np.asarray(days),
        ticker=np.asarray(tickers),
        pnl=pnl,
        vsurge=vsurge,
        elapsed=minute - buy,
    )
    return book, np.asarray(keep_idx, dtype=np.int64)


#: Channels the shuffled-feature null may scramble. `pnl` is excluded on
#: purpose: it decides which actions are legal and is mechanically tied to the
#: reward, so scrambling it would test a different (and meaningless) null.
SCRAMBLEABLE = ("slope", "vsurge", "vtrend")


def scramble(book: Book, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Each path's non-price channels replaced by another path's, same fold.

    The null this builds is exactly the claim under test: the price path, the
    legal action set and every reward are untouched, and only the ALIGNMENT
    between a path's volume/slope history and its own prices is destroyed. A
    donor's series is tiled or truncated to the recipient's length, so each
    channel keeps its own marginal distribution and its within-path serial
    correlation -- a null that kept neither would be too easy to beat.
    """
    lens = book.stop - book.starts
    donor = rng.permutation(book.n_paths)
    cols = {"slope": book.slope_b, "vsurge": book.vsurge_b, "vtrend": book.vtrend_b}
    out = {}
    for name, col in cols.items():
        pieces = []
        for i in range(book.n_paths):
            d = donor[i]
            src = col[book.starts[d] : book.stop[d]]
            need = int(lens[i])
            reps = -(-need // src.size)
            pieces.append(np.tile(src, reps)[:need] if reps > 1 else src[:need])
        out[name] = np.concatenate(pieces)
    return out


# ------------------------------------------------------------------ the lattice


@dataclass(frozen=True)
class Lattice:
    """Which channels the policy may read, and how many states that leaves.

    `dims` is a subset of ("pnl", "slope", "vsurge", "vtrend"). `pnl` is always
    present: without it the action space could not be asymmetric. Dropping
    `vsurge`/`vtrend` gives the ABLATION arm -- the same search, same folds,
    same objective, with the volume channels blinded -- which is the comparison
    the volume hypothesis actually turns on.
    """

    name: str
    dims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dims[0] != "pnl":
            raise ValueError("pnl must lead the lattice; the action legality is defined on it")

    @property
    def sizes(self) -> tuple[int, ...]:
        n = {"pnl": N_PNL, "slope": N_SLOPE, "vsurge": N_VSURGE, "vtrend": N_VTREND}
        return tuple(n[d] for d in self.dims)

    @property
    def n_states(self) -> int:
        return int(np.prod(self.sizes))

    def index(self, book: Book, scramble: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Flat state index per step. `scramble` replaces a channel's bucket
        array, which is how the shuffled-feature null is applied without
        touching the price path."""
        cols = {
            "pnl": book.pnl_b,
            "slope": book.slope_b,
            "vsurge": book.vsurge_b,
            "vtrend": book.vtrend_b,
        }
        if scramble:
            cols = {**cols, **scramble}
        idx = np.zeros(book.n_steps, dtype=np.int32)
        for d, size in zip(self.dims, self.sizes, strict=True):
            idx = idx * size + cols[d].astype(np.int32)
        return idx

    def decode(self, state: int) -> dict[str, int]:
        out, rest = {}, state
        for d, size in zip(reversed(self.dims), reversed(self.sizes), strict=True):
            out[d] = rest % size
            rest //= size
        return out

    def pnl_bucket_of(self, state: int) -> int:
        return self.decode(state)["pnl"]


LATTICES = {
    "novol": Lattice("novol", ("pnl", "slope")),
    "vol": Lattice("vol", ("pnl", "slope", "vsurge")),
    "volplus": Lattice("volplus", ("pnl", "slope", "vsurge", "vtrend")),
}


# ------------------------------------------------------------------ simulator


class Sim:
    """Rolls every path forward under a table policy, in a handful of array ops.

    The sequential walk looks like it must be a Python loop over minutes, and it
    is not: `cut` is legal only below the entry and `add` only at or above it,
    so at any one minute at most one of them can fire and neither changes the
    price path. Each is therefore a first-hitting time, which `minimum.reduceat`
    finds for all paths at once. Position size does not enter any state feature,
    so an add cannot change when the cut happens -- that independence is what
    makes the vectorisation exact rather than an approximation.
    """

    BIG = np.iinfo(np.int64).max // 4

    def __init__(self, book: Book, lat: Lattice, cost: np.ndarray, *, scramble=None) -> None:
        self.book = book
        self.lat = lat
        self.cost = cost
        self.state = lat.index(book, scramble)
        pnl_b = book.pnl_b
        self.cut_legal = np.isin(pnl_b, CUT_BUCKETS) & book.valid
        self.add_legal = book.add_ok & np.isin(pnl_b, ADD_BUCKETS) & book.valid
        self.idx = np.arange(book.n_steps, dtype=np.int64)
        self._cut_idx = np.where(self.cut_legal, self.idx, self.BIG)
        self._add_idx = np.where(self.add_legal, self.idx, self.BIG)
        # Which states can ever act at all, so coordinate ascent never spends a
        # sweep on a cell no path visits in a legal position.
        self.cut_states = np.unique(self.state[self.cut_legal])
        self.add_states = np.unique(self.state[self.add_legal])
        self.actionable = np.union1d(self.cut_states, self.add_states)

    def visits(self) -> tuple[np.ndarray, np.ndarray]:
        """(steps, distinct paths) per state, over legal positions only."""
        n = self.lat.n_states
        legal = self.cut_legal | self.add_legal
        steps = np.bincount(self.state[legal], minlength=n)
        path_of = np.repeat(np.arange(self.book.n_paths), self.book.stop - self.book.starts)
        seen = {s: set() for s in range(n)}
        for s, p in zip(self.state[legal], path_of[legal], strict=True):
            seen[s].add(int(p))
        return steps, np.array([len(seen[s]) for s in range(n)])

    def run(self, act: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(net KRW per path, exit step index, add step index or -1)."""
        fires = act[self.state].astype(bool)
        first_cut = np.minimum.reduceat(np.where(fires, self._cut_idx, self.BIG), self.book.starts)
        exit_at = np.minimum(first_cut, self.book.last_idx)
        first_add = np.minimum.reduceat(np.where(fires, self._add_idx, self.BIG), self.book.starts)
        added = first_add < exit_at
        add_at = np.where(added, np.minimum(first_add, self.book.last_idx), -1)

        proceeds = self.book.fill[exit_at]
        leg1 = proceeds / self.book.entry - 1.0 - self.cost
        add_price = self.book.fill[np.maximum(add_at, 0)]
        leg2 = np.where(added, proceeds / add_price - 1.0 - self.cost, 0.0)
        return NOTIONAL * (leg1 + leg2), exit_at, add_at

    def run_fixed_stop(self, barrier_price: np.ndarray) -> np.ndarray:
        """Exit step under a plain stop-loss at `barrier_price` (per step).

        Not part of the policy search -- this is the driver for `gate_exact`,
        which is the whole reason the class exposes it.
        """
        fires = (self.book.close <= barrier_price) & self.book.valid
        first = np.minimum.reduceat(np.where(fires, self.idx, self.BIG), self.book.starts)
        return np.minimum(first, self.book.last_idx)


# ------------------------------------------------------------------ objective


def objective(net: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean realised KRW per trade. The ONLY statistic that selects a policy.

    Deliberately NOT `rl_exits.objective`, which is
    ``sum(day_nets) - 0.5 * max(0, -min(day_nets))``. That shape penalises the
    single worst day, which is (a) an extreme order statistic read off one day
    in a fold and (b) the same quantity `bayes_judge` gate 2 then tests -- so a
    policy chosen under it is chosen partly for clearing its own gate. The
    earlier tabular stop policy degenerated to cutting 87-92% of positions under
    exactly this objective, which is what a broken reward looks like.

    Worst-day and loser-ratio are computed downstream as DIAGNOSTICS and never
    fed back here.

    `mask` selects the training paths. Every fold rolls the WHOLE book and then
    scores the fold, rather than rolling a rebuilt sub-book: a sub-book has to
    re-derive `starts`, the per-path cost and the scrambled channels in step
    with each other, and getting any one of those out of alignment produces a
    plausible-looking number instead of an error.
    """
    return float(net.mean() if mask is None else net[mask].mean())


def day_records(days: np.ndarray, net: np.ndarray) -> list[dict]:
    """`bayes_judge`-shaped day records. `net` is already KRW."""
    out = []
    for day in sorted(set(days.tolist())):
        sel = days == day
        r = net[sel]
        out.append(
            {
                "day": day,
                "net": float(r.sum()),
                "trades": int(sel.sum()),
                "losers": int((r < 0).sum()),
                "worst_trade": float(r.min()) if r.size else 0.0,
                "errors": 0,
                "leftovers": 0,
            }
        )
    return out
