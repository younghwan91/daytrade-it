"""Vectorized twin of `auto_trader.exit_rule_hit`, plus the test that it is a twin.

`exit_rule_hit` recomputes the session VWAP from scratch on every call, so
scoring one configuration over ~4,500 paths costs O(bars^2) per path and a grid
of a few hundred configurations is hours. Every one of its rules, though,
depends only on the *last completed bar* plus running aggregates (cumulative
volume and price x volume, the running peak since entry, the opening-range low).
So each path can be reduced once to a handful of arrays and any configuration
scored with array comparisons.

This is an optimization, not a second implementation of the rules: `verify`
asserts that the twin returns the same exit minute and reason as the live
function on every path for every configuration it is given, and the search
refuses to run if that fails. The live function remains the only thing that
decides anything in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np

from daytrade_it.application.trading.auto_trader import TradingLimits

OPEN_MINUTE = 9 * 60
#: `exit_rule_hit`'s own order; the first rule that fires names the exit.
RULE_ORDER = (
    "stop_loss",
    "stop_vwap",
    "stop_time",
    "stop_orb",
    "take_profit",
    "trailing",
    "vwap_break",
)


@dataclass(frozen=True)
class Reduced:
    """One path reduced to per-decision-minute arrays.

    Index k runs over the minutes at which a sell could actually be placed:
    minutes that have a bar, after the fill minute and at or before the flatten
    minute. Each entry describes the state `exit_rule_hit` would see there.
    """

    minute: np.ndarray  # the decision minute itself
    fill: np.ndarray  # that minute's open -- the modeled sell price
    close: np.ndarray  # last completed bar's close
    vwap: np.ndarray  # session VWAP of the completed bars
    peak: np.ndarray  # running max high from the minute after entry
    has_peak: np.ndarray  # whether any bar after the entry minute is complete
    orb_low: np.ndarray  # min low of the opening-range window (per window width)
    last_minute: np.ndarray  # last completed bar's minute (for the ORB "after" test)
    elapsed: np.ndarray  # decision minute - fill minute, in minutes
    entry: float
    flatten_fill: float  # price if nothing fires: the flatten minute's open
    flatten_minute: int


def reduce_path(
    bars: list,
    buy_minute: int,
    buy_price: Decimal,
    flatten_minute: int,
    orb_widths: tuple[int, ...],
) -> Reduced | None:
    """Reduce one path's bars; `orb_widths` are the opening-range widths to precompute."""
    minutes = np.array([b.ts.hour * 60 + b.ts.minute for b in bars])
    order = np.argsort(minutes, kind="stable")
    minutes = minutes[order]
    closes = np.array([float(bars[i].close) for i in order])
    highs = np.array([float(bars[i].high) for i in order])
    lows = np.array([float(bars[i].low) for i in order])
    opens = np.array([float(bars[i].open) for i in order])
    vols = np.array([float(bars[i].volume) for i in order])

    cum_vol = np.cumsum(vols)
    cum_pv = np.cumsum(closes * vols)
    # Running peak of highs from the minute AFTER entry (exit_rule_hit's trailing rule).
    after_entry = minutes >= buy_minute + 1
    masked_high = np.where(after_entry, highs, -np.inf)
    run_peak = np.maximum.accumulate(masked_high)
    run_has_peak = np.maximum.accumulate(after_entry.astype(int)).astype(bool)

    # Decision minutes: a bar exists, after the fill, up to and including the minute
    # the position is closed. That closing minute is the FIRST print at or after
    # 15:00 -- live retries the exit every minute until it fills, so a ticker with
    # no 15:00 bar leaves on its next print, not at 15:00 exactly. With no print at
    # all at or after 15:00 the leftover is valued at the day's last close.
    after = minutes[minutes > buy_minute]
    if after.size == 0:
        return None
    at_or_after = after[after >= flatten_minute]
    close_at = int(at_or_after[0]) if at_or_after.size else int(after[-1])
    real_flatten = bool(at_or_after.size)
    sel = (minutes > buy_minute) & (minutes <= close_at)
    if not sel.any():
        return None
    dec_minutes = minutes[sel]
    dec_fill = opens[sel].copy()
    if not real_flatten:
        # Settled at the last close, not sold into the open of that minute.
        dec_fill[-1] = closes[sel][-1]
    # j(m) = index of the last bar whose minute is strictly before m.
    j = np.searchsorted(minutes, dec_minutes, side="left") - 1
    ok = j >= 0
    if not ok.all():
        dec_minutes, dec_fill, j = dec_minutes[ok], dec_fill[ok], j[ok]
        if j.size == 0:
            return None
    vol_j = cum_vol[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(vol_j > 0, cum_pv[j] / np.where(vol_j > 0, vol_j, 1.0), np.nan)

    orb = {}
    for w in orb_widths:
        end = OPEN_MINUTE + w
        window = minutes < end
        orb[w] = float(lows[window].min()) if window.any() else np.inf

    flatten_fill = float(dec_fill[-1])
    return Reduced(
        minute=dec_minutes,
        fill=dec_fill,
        close=closes[j],
        vwap=vwap,
        peak=run_peak[j],
        has_peak=run_has_peak[j],
        orb_low=np.array([orb[w] for w in orb_widths]),
        last_minute=minutes[j],
        elapsed=dec_minutes - buy_minute,
        entry=float(buy_price),
        flatten_fill=flatten_fill,
        flatten_minute=close_at,
    )


def _fires(r: Reduced, lim: TradingLimits, orb_widths: tuple[int, ...]) -> dict[str, np.ndarray]:
    e = r.entry
    valid = np.isfinite(r.vwap)
    out: dict[str, np.ndarray] = {}
    z = np.zeros_like(r.close, dtype=bool)

    out["stop_loss"] = (
        (r.close <= e * (1 - float(lim.stop_loss_pct))) & valid
        if lim.stop_loss_pct is not None
        else z
    )
    out["stop_vwap"] = (
        (r.close < e) & (r.close < r.vwap * (1 - float(lim.stop_vwap_gap))) & valid
        if lim.stop_vwap_gap is not None
        else z
    )
    out["stop_time"] = (
        (r.elapsed >= lim.stop_time.total_seconds() / 60)
        & (r.close < e * (1 + float(lim.stop_time_min_gain)))
        & valid
        if lim.stop_time is not None
        else z
    )
    if lim.stop_orb is not None:
        w = int(lim.stop_orb.total_seconds() // 60)
        low = r.orb_low[orb_widths.index(w)]
        out["stop_orb"] = (
            (r.last_minute >= OPEN_MINUTE + w) & (r.close < low) & np.isfinite(low) & valid
        )
    else:
        out["stop_orb"] = z
    out["take_profit"] = (
        (r.close >= e * (1 + float(lim.take_profit_pct))) & valid
        if lim.take_profit_pct is not None
        else z
    )
    out["trailing"] = (
        r.has_peak
        & (r.peak >= e * (1 + float(lim.trail_after_gain)))
        & (r.close <= r.peak * (1 - float(lim.trail_gap)))
        & valid
        if lim.trail_after_gain is not None
        else z
    )
    out["vwap_break"] = (
        (r.close / e - 1 >= float(lim.vwap_exit_min_gain))
        & (r.close / r.vwap - 1 < -float(lim.vwap_exit_gap))
        & valid
        if lim.vwap_exit_gap is not None
        else z
    )
    return out


def first_exit(
    r: Reduced, lim: TradingLimits, orb_widths: tuple[int, ...]
) -> tuple[float, str, int]:
    """(sell price, reason, minute) -- the twin of one `simulate` run."""
    fires = _fires(r, lim, orb_widths)
    any_fire = np.zeros_like(r.close, dtype=bool)
    for name in RULE_ORDER:
        any_fire |= fires[name]
    # The flatten minute always exits, whatever the rules say.
    at_flat = r.minute == r.flatten_minute
    any_fire |= at_flat
    idx = np.argmax(any_fire) if any_fire.any() else None
    if idx is None:
        return r.flatten_fill, "flatten", r.flatten_minute
    if at_flat[idx]:
        # `simulate` flattens at the flatten minute before consulting the rules,
        # so the twin must label it the same way.
        return float(r.fill[idx]), "flatten", int(r.minute[idx])
    reason = next(n for n in RULE_ORDER if fires[n][idx])
    return float(r.fill[idx]), reason, int(r.minute[idx])
