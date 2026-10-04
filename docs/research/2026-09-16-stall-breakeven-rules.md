# The stall and break-even rules on the live pool: both should stay off (2026-09-16)

simnode. Built against `origin/main` `69bfc1a` in a clean worktree (`~/git/daytrade-it-mainrl`),
importing `stall_exit_hit`, `breakeven_stop_hit` and `vwap_exit_hit` from the code live runs and
reproducing `_early_exits`' order exactly: **VWAP → stall → break-even**, then flatten at the
first print at or after 15:00.

**Verdict: on the dashboard pool — the one live trades — the live default with BOTH rules OFF
beats all 27 alternatives, under plain return and under the downside-weighted objective alike.**
Neither rule earns its place on current evidence.

---

## 1. The regression gate runs before the grid

The lab refuses to start unless both rules decline on the case that exposed the original
day-peak bug (a stock running 100 → 110 in the morning, entered at 100 in the afternoon, flat
thereafter). Against `69bfc1a`:

```
regression gate: rules must not read pre-entry bars ...
  breakeven_stop_hit: hit=False  peak_gain=0.001  close=100.0
  stall_exit_hit:     hit=False  gain=0.0
passed
```

`peak_gain` is now 0.001 — the position's own peak since entry — where before the fix it was
0.1000, the day's peak. The fix is verified independently of the unit tests that accompany it.

## 2. Dashboard pool, 154 entries over 7 days

`stall_exit_min_gain = 0` is excluded from the grid: at 0 a flat post-entry window satisfies both
the gain and the stall conditions, degenerating into "sell when it stops moving, profit or not".

| configuration | bp | win rate | worst day | downside-weighted (λ=2) |
|---|---|---|---|---|
| **LIVE DEFAULT (both off)** | **+66** | 0.474 | −19,648 | **+75,280** |
| stall off, breakeven 0.030 | +52 | 0.461 | −19,648 | +54,519 |
| stall 0.020 / 10m, breakeven off | +45 | 0.519 | −17,280 | +51,499 |
| stall 0.020 / 5m, breakeven off | +43 | 0.519 | −16,644 | +49,486 |
| stall off, breakeven 0.020 | +41 | 0.448 | −19,542 | +36,947 |
| stall 0.010 / 5m, breakeven off | +34 | 0.591 | −17,098 | +34,833 |
| stall 0.005 / 10m, breakeven off | +24 | **0.688** | **−15,102** | +21,407 |
| stall off, breakeven 0.010 | +24 | 0.474 | −17,117 | +14,343 |

**The live default is top on both objectives**, and by a wide margin on the downside-weighted one
(+75,280 against +54,519 for the best alternative).

## 3. The stall rule does exactly what it was designed to do, and it costs money

This is the interesting part, and it is not a failure of implementation. Tightening
`stall_exit_min_gain` from off → 0.020 → 0.010 → 0.005 moves every risk metric the right way:

| stall threshold | win rate | worst day | mean return |
|---|---|---|---|
| off | 0.474 | −19,648 | **+66 bp** |
| 0.020 / 10m | 0.519 | −17,280 | +45 bp |
| 0.010 / 5m | 0.591 | −17,098 | +34 bp |
| 0.005 / 10m | **0.688** | **−15,102** | **+24 bp** |

Win rate climbs from 47% to 69% and the worst day improves by 23%, monotonically — the rule
genuinely does "take the profit when the move stalls". And mean return falls by two thirds doing
it, also monotonically.

That is the same trade every risk-cutting rule has made on this pool tonight: trailing,
hold-to-close, pyramiding-off, and now stall. **On a pool with positive drift, cutting early
converts return into win rate.** The user's asymmetric objective favours protection over
expectancy, but not at a 42 bp cost for a 4,546 KRW better worst day — and the λ=2 score says so
directly.

The break-even stop shows no such compensation: it costs return (+66 → +52/+41/+24) while leaving
the win rate flat (0.474 → 0.461/0.448/0.474) and the worst day nearly unchanged. On this sample
it is paying for nothing.

## 4. The caveat that bounds all of this: no folds are possible

The dashboard pool is 7 trading days **inside a single month**, so there is no time-ordered fold
to make. Section 2 is a **full-sample, in-sample** readout, not an out-of-sample result. The fold
table in the script is empty for exactly this reason.

That weakens the numbers but not the conclusion, by the same argument used against the entry
features: **a configuration that cannot beat the default in-sample will not beat it
out-of-sample.** Live-default being top of 28 configurations with no held-out penalty applied is
the strongest form this particular negative can take on 7 days.

The sweep-pool run (95 days, five monthly time-ordered folds) provides the out-of-sample fold
evidence, at the cost of being the wrong pool — and three demonstrations tonight say sweep-pool
exit results do not transfer.

## 5. Recommendation

**Leave `stall_exit_min_gain` and `breakeven_after_gain` at `None`.** Both rules are correctly
implemented and well-motivated, and on the pool live trades neither pays for itself under either
objective.

What would change this: more dashboard days — the constraint is 154 entries in one month, which
cannot even be split into folds. If the pool's profitability weakens (the sweep pool suggests it
might on a wider universe), the trade reverses and the stall rule becomes attractive, because its
risk-side behaviour is real and monotone.

## 6. Artifacts

| what | path |
|---|---|
| lab (imports main's functions, reproduces `_early_exits`) | `scripts/eval/stall_lab.py` |
| regression gate | `scripts/eval/stall_repro.py` |
| dashboard-pool result | `data/eval/rl_minimal/stall_dash.json` |
| dashboard-pool entry paths (154 entries, 7 days) | `data/eval/rl_minimal/paths_dash.jsonl` |

## 7. Concerns

1. **In-sample, single month, 154 entries.** See §4.
2. **Uncapped one-share counterfactual.** These paths lift the concurrent-position and cash caps,
   so the per-trade bp are the transferable quantity and the λ-scores are not live P&L. The capped
   comparison would need the rules ported into a worktree that also has the replay engine —
   neither branch currently has both.
3. **The grid was reduced** from 117 to 28 configurations (`stall_minutes` {5,10},
   `breakeven_buffer` {0.003} only) because simnode was at load 30 with three other agents'
   work. The dropped settings were the extremes; the monotone response in §3 suggests the shape
   is well captured, but `breakeven_buffer` in particular is untested off 0.003.
