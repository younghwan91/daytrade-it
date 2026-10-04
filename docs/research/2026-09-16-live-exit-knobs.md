# The live-expressible exit knobs, tuned on dashboard days (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal`. Real replay engine, 7 dashboard-pool
days, 0 errors. Zero API spend, kr_quant read-only, live repo/daemon/backfill untouched.

**Verdict: `add_on_min_gain = 0.005` is the strongest candidate found in this entire body of
work — and it still fails adoption, on one criterion.** `vwap_exit_gap = 0.01` and
`vwap_exit_min_gain = 0.02` are already at or indistinguishable from their optimum;
`add_on_price_band` is inert.

---

## 1. Why these four knobs, and why a coordinate sweep

Live's news exit path can express exactly six settings: `vwap_exit_gap`, `vwap_exit_min_gain`,
`add_on_min_gain`, `add_on_price_band`, and the two new (currently buggy, see
`2026-09-16-exit-rule-learning.md`) `stall_*` / `breakeven_*` knobs. Everything my earlier grid
searched — `stop_loss_pct`, `take_profit_pct`, `trail_after_gain`, `stop_time`, `stop_orb` — is
**not on main** and cannot be switched on.

The four testable ones are all currently **ON** with values chosen by leave-one-day-out over
6 days — the methodology `learning-curve` established is invalid (LODO folds sharing n−1 of n
training days are one model n times over). So they are exactly the hand-set constants the
standing "no fixed values" rule points at, and nobody had re-examined them.

`vwap_exit_hit` and `add_on_hit` are **byte-identical between this branch and `origin/main`**
(verified by diff, not assumed — that assumption is what cost me the earlier grid). So a result
here transfers to live.

This is a **coordinate sweep, not a grid**: each row moves one knob off its live value with
everything else at default. With 7 days and ~35 trades a full cross would be a
multiple-comparison machine. The question asked is narrower and answerable: *is the hand-set
value near the optimum on this axis, or clearly off it?*

## 2. Results

| knob setting | total | trades | losers | worst day | p(μ>0) | days ≥ 0 |
|---|---|---|---|---|---|---|
| **base (live: gap 0.01, min_gain 0.02, add_on 0.01)** | **+39,915** | 35 | 12 | −2,838 | — | — |
| `vwap_exit_gap = None` | +29,423 | 34 | 11 | −1,601 | 0.121 | 4/7 |
| `vwap_exit_gap = 0.005` | +38,884 | 36 | 13 | −2,838 | 0.358 | 5/7 |
| `vwap_exit_gap = 0.02` | +38,894 | 34 | 11 | −2,838 | 0.283 | 5/7 |
| `vwap_exit_gap = 0.03` | +38,889 | 34 | 11 | −2,838 | 0.280 | 5/7 |
| `vwap_exit_min_gain = 0.01` | +39,746 | 35 | 12 | −2,838 | 0.498 | 6/7 |
| `vwap_exit_min_gain = 0.03` | +38,989 | 34 | 11 | −2,838 | 0.499 | 6/7 |
| `vwap_exit_min_gain = 0.05` | +38,989 | 34 | 11 | −2,838 | 0.499 | 6/7 |
| `add_on_min_gain = None` | +30,027 | 35 | 12 | −1,601 | 0.131 | 4/7 |
| **`add_on_min_gain = 0.005`** | **+42,737** | 35 | 13 | **−2,255** | **0.911** | **7/7** |
| `add_on_min_gain = 0.02` | +37,311 | 35 | 13 | −1,967 | 0.246 | 4/7 |
| `add_on_min_gain = 0.03` | +35,079 | 35 | 14 | −2,767 | 0.195 | 4/7 |
| `add_on_price_band = 0.02` | +39,915 | 35 | 12 | −2,838 | 0.500 | 7/7 |
| `add_on_price_band = 0.05` | +39,915 | 35 | 12 | −2,838 | 0.500 | 7/7 |

## 3. What each knob says

**`vwap_exit_gap`: the live 0.01 is right, and the gap width barely matters.** Every alternative
is worse, but note the shape — 0.005, 0.02 and 0.03 all land within 11 KRW of each other
(−1,031, −1,021, −1,026 against base), while switching the rule **off entirely** costs 10,492.
So it is *having* the VWAP-break exit that matters, not where the threshold sits. That is a
robustness result worth more than a tuned value: the rule is not perched on a hand-picked number.

**`vwap_exit_min_gain`: inert on this sample.** μ ≈ 0 and p ≈ 0.499 for every alternative. Seven
days cannot separate them and there is no reason to touch 0.02.

**`add_on_price_band`: completely inert.** Both alternatives reproduce the base total to the won.
The ±3% band never binds on these days, so this knob does not need tuning — it needs a note that
it is doing nothing.

**`add_on_min_gain`: monotone, and the live value is not at the optimum.**

| None | 0.03 | 0.02 | **0.01 (live)** | **0.005** |
|---|---|---|---|---|
| +30,027 | +35,079 | +37,311 | +39,915 | **+42,737** |

Perfectly monotone across five settings: **the more readily the position pyramids, the better**,
with "no pyramiding at all" the worst. That smooth response is what a real effect looks like, and
it stands in deliberate contrast to the far-target result, where +5%/+8%/+10% zigzagged
(+40,246 / +51,924 / +35,809) on a single day's coincidence.

## 4. `add_on_min_gain = 0.005` against the adoption criteria

| criterion | value | verdict |
|---|---|---|
| 1. P(μ>0) ≥ 0.90 | **0.911** | **pass** |
| 2a. worst day not worse | −2,255 vs −2,838 | **pass** |
| 2b. loser ratio not worse | **0.371 vs 0.343** | **FAIL** |
| 3. days ≥ 0 ≥ ceil(5/7 × 7) = 5 | **7/7** | **pass** |
| 4. forward 5 days | — | not evaluated |

**Three of four, failing only the loser ratio** — and the failure is mechanically honest rather
than incidental: lowering the threshold adds a share to *more* positions, including ones that
subsequently turn down, so a higher share of trades close red even as the total improves. That is
the same shape as the trade-count effects seen throughout: an exit or sizing change alters which
trades exist, not just their outcomes.

Under the user's asymmetric objective this is a genuinely mixed case, and I am not going to
resolve it quietly in either direction. It **improves** the worst day (the loss-tail metric) and
is positive on every single day, while **worsening** the loser ratio. "Capital preservation
outranks expectancy" does not by itself say which of those two risk metrics wins.

**The reason not to act on it tonight is power, not the criterion.** This is 7 days, ~35 trades,
and one of **13 coordinate-sweep comparisons**. A p of 0.911 as the best of 13 is not a p of
0.911; the expected maximum across that many draws is high, and no multiplicity correction was
applied because the sweep was designed as a sensitivity check rather than a selection. The
monotone response across five settings is better evidence than the p-value, and it is still
7 days.

## 5. Recommendation

**Do not change anything for tomorrow**, consistent with every other exit finding tonight.

**`add_on_min_gain = 0.005` is the one candidate worth collecting more dashboard days on** —
replacing `stop_orb = 30`, which I had recommended three times before discovering it is not in
live code. Unlike `stop_orb`, this knob exists in production, its function is byte-identical to
main, its response is monotone across five settings, and it is positive on 7 of 7 days.

What would settle it: the same coordinate sweep on 20+ dashboard days, with the loser-ratio
question decided explicitly rather than inherited from a criterion list.

**Two knobs can be marked settled**: `vwap_exit_gap` at 0.01 (the rule matters, the width does
not) and `add_on_price_band` at 0.03 (inert; it never binds).

## 6. Artifacts

| what | path (simnode `~/git/daytrade-it-rl`) |
|---|---|
| coordinate sweep | `scripts/eval/dash7_exits.py` |
| results | `data/eval/rl_minimal/dash7_live_knobs.json` |

## 7. Concerns

1. **Seven days, 13 comparisons, no multiplicity correction.** The p-values are sensitivity
   readings, not tests. The monotone response is the load-bearing evidence.
2. **Only 2 of the 7 days are true live-archive days** (per prompt-opt's audit of the dashboard
   pool); the rest are reconstructed. The pool is the right one, the sample is thin.
3. **`add_on_min_gain` changes position sizing**, so it interacts with `max_trader_exposure` and
   the deposit-cash check in ways 7 uncontended days may understate. A lower threshold means more
   shares held concurrently; on a busier day that could bind where it did not here.
4. **The two new `stall_*`/`breakeven_*` knobs are excluded** — they measure their peak from
   09:00 rather than from entry, so their thresholds cannot be learned until that is fixed.
