# The third exit result, and three corrections to the second (2026-09-22)

**A genuinely sequential exit policy — hold, add or cut decided every minute from the position's own state, with volume in that state — loses to a fixed stop-loss out of sample on 149 never-fitted days and 29,560 trades. Adding the volume channel makes it worse in 5 of 5 time-ordered folds, and the search earns no more on the real volume channel than on twenty scrambled ones. `bayes_judge` says adopt=false against live, against hold-to-close and against the fixed grid, on a confirm slice frozen before any of this ran.**

This is the third exit negative. It is a companion to `2026-09-22-rl-news-overlap.md` §2.3, and it corrects three things that document and its predecessors left on the record.

1. **Volume was already in an exit state, and it had already lost.** This work was commissioned on the premise that fill strength had never been in an exit state. It had.
2. **"Cut almost everything" is not a symptom of a broken reward.** It was diagnosed as one. Remove the broken reward entirely and the behaviour is unchanged.
3. **The select-on-X / gate-on-X composition is present in the exit lab, and it cost essentially nothing.** Both halves of that sentence matter.

---

## 1. The answer, in one table

Pooled per-trade net, out of sample, at the broker's own round trip. `search-OOS` pools 5 time-ordered monthly folds inside the search window (19,645 trades); `confirm` is the frozen slice, fitted on the whole search window and scored once (30 days, 5,954 trades). **The two columns cover different months and are not comparable to each other** — only down a column. Confirm is the harsher period for everything, live included.

| arm | search-OOS | confirm | `bayes_judge` vs live |
|---|---|---|---|
| **best fixed grid** | **−32.6** | −45.5 | adopt=false, p=0.420 |
| sequential, no volume | −35.2 | −45.6 | adopt=false, p=0.417 |
| sequential, **+ volume** | −36.7 | −44.1 | adopt=false, p=0.392 |
| sequential, + volume + trend | −38.6 | −44.2 | adopt=false, p=0.400 |
| live exit rules | −40.6 | −47.6 | — |
| hold to close | −40.6 | −50.6 | — |
| live pyramiding rule | — | **−81.9** | — |

**Nothing is positive. Nothing is adopted.** Every sequential arm beats live and hold-to-close and loses to a fixed 1% stop on the search window — which is exactly where the two earlier attempts landed.

The ladder down the first column is the whole story:

> **−32.6 → −35.2 → −36.7 → −38.6 as state dimensions are added.** Each extra dimension costs one to two basis points out of sample. That is not a policy learning more; it is a policy overfitting more.

### The confirm slice disagrees, and that is itself the point

On the 30 frozen days the ordering **reverses**: the volume arm (−44.1) edges the no-volume arm (−45.6) and the fixed grid (−45.5). Every one of those gaps is inside noise — `seq_vs_fixed` gives p(mu>0)=0.432 with days≥0 at 14 of 30, and the mean daily delta is −681 KRW against a prior sd of 5,000.

**Do not read the confirm reversal as volume working.** Read it as what it is: on 30 days these arms are indistinguishable, and a 1.5 bp ordering flips on a different draw. The evidence that volume *hurts* is the 5-of-5 fold result and the shuffle premium below, not the confirm slice. The evidence that none of it *helps* is that no arm, on any window, gets anywhere near zero.

---

## 2. Correction 1 — volume had already been tested, and had already lost

`scripts/eval/rl_stop.py` on `feat/rl-minimal` carries

```python
TV_EDGES = (0.7, 1.5)  # minute turnover vs the median of the 20 minutes before it
```

with `use_volume=True` by default, and **both arms were run and saved**: `stop.json` without the volume bin, `stop_vol.json` with it. Pooled over its five monthly time-ordered folds, 3,851 trades, 2026-04..08, at the legacy 23.0 bp cost literal:

| arm | per-trade | `bayes_judge` vs the fixed stop |
|---|---|---|
| best fixed stop (1.0%) | **−18.33** | — |
| learned, no volume | −19.15 | adopt=false, p(mu>0)=**0.0078** |
| learned, **with volume** | −19.96 | adopt=false, p(mu>0)=**0.0039** |

Volume made that policy worse too, and *more significantly* worse. Its docstring reads "minute volume was tested as a proxy and is reported separately", which is how the fact got lost.

So tonight is the **second independent failure of the volume hypothesis**, on a population six times larger, never fitted, with the cost corrected and the objective corrected. That closes the question harder than a first test could have. It should not be recorded as a first test.

Two caveats kept for honesty. The two studies use different volume statistics — `rl_stop` takes one minute's turnover against a trailing 20-bar median, this study takes a three-bar mean against the session's running mean per bar — and different populations. They agree in direction anyway.

---

## 3. Correction 2 — the degeneracy survives the reward fix

The earlier tabular stop policy cut 87–92% of positions, and that was read as "a red flag for a broken reward, not a working policy". The reward *was* broken: `rl_stop.learn` weights every negative reward by λ=2 inside the policy improvement step, and the fold selection uses the same λ-score.

This study removed all of it. `seqexit_core.objective` is

```python
return float(net.mean() if mask is None else net[mask].mean())
```

— mean realised KRW per trade, nothing else. No day aggregation, no λ, no tail term. Worst-day, 5th-percentile day and loser ratio are computed downstream as diagnostics and never fed back.

The learned policy still **cuts 76–87% of positions in every fold**, and the loser ratio still runs to 0.85–0.92 against live's 0.65.

| | prior tabular stop (λ=2, worst-day) | this study (clean mean KRW) |
|---|---|---|
| cut rate | 0.87 – 0.92 | 0.76 – 0.87 |
| loser ratio | 0.908 (live 0.637) | 0.853 – 0.922 (live 0.647) |

**The behaviour survives the fix, so the reward never caused it.** What causes it is arithmetic: this pool drifts down after entry, so cutting is locally correct per trade, and the 35 bp round trip then eats whatever the cut saved. A policy told to maximise mean realised KRW on a pool like this *should* cut nearly everything. The degeneracy is a true statement about the pool, not a bug in the objective.

### The diagnostics look like risk control and are not

On the search window the sequential policy's worst day is −127,377 KRW against live's −395,146, and its 5th-percentile day −106,526 against −233,888. That reads as a large improvement in tail risk.

It is the same trap §2.2 flagged for the entry bandit. Cutting 86% of positions within minutes truncates every tail, good and bad, and the price is paid in the mean. Days with net ≥ 0: **sequential 0 of 99**, fixed 6 of 99, live 22 of 99. The decisive number remains per-trade net.

---

## 4. Correction 3 — select-on-X / gate-on-X: real, and inert

`rl_exits.objective` is literally

```python
return float(sum(vals) - 0.5 * max(0.0, -min(vals)))
```

and `bayes_judge` gate 2 is `worst_day_candidate >= worst_day_baseline`, i.e. `min(net)` over days. The selector and the gate read the same extreme order statistic, and that objective selects for **both** `rl_exits.py` and `rlnews_exit_holdout.py`, for the fixed grid *and* the σ-scaled arm. The same composition was found and fixed earlier tonight in `optimize_policy.py`; this is a second site.

Rather than assert it mattered, `seqexit_compare.py` re-selects the identical folds and grids under plain mean realised KRW:

| arm | published selector | clean mean KRW | difference |
|---|---|---|---|
| best fixed | −35.4 | −35.4 | **+0.0 bp** |
| σ-scaled contextual | −40.5 | −40.7 | **+0.2 bp** |

Configuration stability barely moves either: the fixed grid picks 3 distinct configurations in 7 folds under both selectors; the σ-scaled arm picks 5 under the penalised selector and 4 under the clean one.

**The defect is real and worth fixing for hygiene. It did no damage on this book.** It is not what rejected the σ-scaled exit, and — per §3 — it is not what produced the earlier degeneracy either.

One fact worth keeping from that re-run: **all 7 folds chose a 1.0% stop-loss**, and 4 of 7 chose the identical full configuration `sl=0.01 tp=0.10 tr=0.08/0.03 st=60 orb=30`. The fixed grid is the stable arm here, not the fragile one.

---

## 5. What was built

### Why this is not attempt 1 or attempt 2

| | attempt 1 (fixed grid) | attempt 2 (σ-scaled) | attempt 3 (this) |
|---|---|---|---|
| what is learned | one configuration of 4,320 | two coefficients `a`, `b` | a lookup table over position state |
| decision | thresholds fixed at entry | thresholds fixed at entry | re-decided every minute |
| actions | exit rules | exit rules | **hold / add / cut** |
| volume in state | no | no | **yes** |
| selects on | `sum − ½·max(0, −min)` | same | **mean realised KRW** |

The two earlier attempts searched *parametric families*. This searches a policy: a binary action per discretised state, re-evaluated at every minute the position is open.

### Three properties, fixed in writing before anything ran

**Volume enters against the stock's own baseline.** `vsurge` is the last three completed bars' mean volume over the session's mean volume per bar so far; `vtrend` is that three-bar mean over the three before it. Both are ratios — the stock against itself, never a level — and both are running aggregates the live daemon already accumulates for VWAP, so they cost nothing at decision time. Everything read at minute `m` comes from bars that ended before `m`, the same `b.ts < minute` filter the live `exit_rule_hit` applies.

**The action space is asymmetric by construction.** `cut` is legal *only* strictly below the entry price, `add` *only* at or above it. Structural, not something the search was asked to discover: a symmetric trailing stop is unreachable from this action set, which is the point, because that family had already lost twice. The gate asserts it on every step of every path.

**Nothing selects on a tail statistic** (§4).

### The state lattice

Edges are round numbers fixed before the first search and never tuned.

| channel | edges | buckets |
|---|---|---|
| `pnl` (last close vs entry) | −2.0%, −0.7%, 0, +1.5% | 5 — cut legal in 0–2, add legal in 3–4 |
| `slope` (3-minute return) | −0.3%, +0.3% | 3 |
| `vsurge` | 0.7, 1.5 | 3 |
| `vtrend` | 0.8, 1.25 | 3 |

The split at exactly 0 is what makes the action space asymmetric; a position exactly at the entry price is not a loser and a cut there is illegal.

| arm | state | states | search trades per free parameter |
|---|---|---|---|
| `novol` | pnl × slope | 15 | 1,574 |
| `vol` | pnl × slope × vsurge | 45 | 525 |
| `volplus` | pnl × slope × vsurge × vtrend | 135 | 175 |

Tabular throughout — no function approximation, so the effective degrees of freedom are exactly the state count, and every one of them is reported with its visit count in §7.

### Live parity on `add`

`add` is held to what `AutoTrader._add_on` can actually execute, not to an idealised pyramid: `add_on_hit` verbatim (≥20 completed bars, last close above session VWAP, MA5 > MA20), **one share**, **once per ticker**, before `entry_end` 14:30, and within the 3% `add_on_price_band` between the fill and the last completed close. The policy chooses *among* the minutes live would allow, so every add in the simulation is an order live could have placed.

### Optimiser

Coordinate ascent on the binary action table. Every move is evaluated **exactly**, by re-rolling all 29,560 paths — no fitted value function, no bootstrapping, no discount factor to misspecify. The number being maximised is the number that gets reported. Six restarts; restart 0 from all-hold (hold-to-close, the honest null), the rest random, so a basin only all-hold can reach is not mistaken for the optimum.

The sequential walk vectorises exactly rather than approximately: `cut` is legal only below entry and `add` only at or above it, so at most one can fire at any minute; position size enters no state feature, so an add cannot move the cut. Both are therefore first-hitting times over the whole book.

### Cost

`costs.round_trip` per trade **and per leg** — mean **35.4 bp** on this book (min 32.0, max 53.9). Never the 23.0 bp `replay_baseline` literal, which is what `rl_stop.py`'s numbers in §2 are still charged at.

---

## 6. Gates

### Exactness, run exhaustively

`exit_fast.py` is a verified twin of the live `exit_rule_hit`; `seqexit_core.Sim` is a *third* implementation of the same walk, so it gets the same treatment — and exhaustively, because sampling is precisely how the Decimal-tie defect survived its gate the first time (§5.2 of the prior document: seed 0's 400-path draw contained none of the 6 affected paths).

| stage | scope | result |
|---|---|---|
| usability | 29,800 rows | 29,560 usable, 240 dropped — **agrees with `exit_fast` row for row** |
| reduction identity | all 29,560 paths, elementwise | decision minutes **0**, sell fills **0**, last-completed close **0**, VWAP validity **0** |
| driven identity | all paths × all 4 `STOP_LOSS` levels | **0** mismatching paths |
| unreachable stop | all paths | exit index equals the flatten index exactly |
| asymmetry | every step | no cut above entry, no add below it |

Thresholds go through `exit_fast.barrier` on both sides, so the Decimal tie is *inside* the test rather than underneath it. `verify_twin.py` ties `first_exit` to the live `exit_rule_hit`, so passing these places the third implementation against the live function transitively.

`Sim.visits`, which reports the sample behind each policy cell, was separately checked against a brute-force set count: 0 mismatches.

### Reproduction

Before reading anything new, the published exit-holdout figures were regenerated through this pipeline — full 149 days, 7 monthly folds, 25,599 trades:

| arm | published §2.3 | reproduced |
|---|---|---|
| live exits | −42.2 | **−42.2** |
| hold-to-close | −42.9 | **−42.9** |
| best learned fixed | −35.4 | **−35.4** |
| σ-scaled contextual | −40.5 | **−40.5** |

Four for four. The cost model, the twin and the fold protocol are the published ones.

### The split was read, not chosen

`CONFIRM_WINDOWS.json` froze `track_a_prior` at **2026-09-22T02:52:01** (commit `f9376cf`), before any exit searching was dispatched:

- **search** 2025-08-01..2026-01-27 — 119 days, 23,606 trades
- **confirm** 2026-01-28..2026-03-16 — 30 days, 5,954 trades, **touched once**

`seqexit_search.py` reads that file and refuses to run if the two do not partition the book. Inside the search window: monthly time-ordered folds (fit on months strictly before `m`, score on `m`), plus an inner holdout (fit 2025-08..12, score 2026-01) that the shuffled-feature null is measured against — so establishing the noise floor costs no look at the real confirm slice. **No leave-one-day-out anywhere.**

---

## 7. What the policy actually learned

### Search window, by fold

`seq` is this study; `fixed` is the best grid configuration chosen on the same training months by the same objective.

| fold | n | novol | vol | volplus | fixed | live | hold |
|---|---|---|---|---|---|---|---|
| 2025-09 | 4,367 | −33.9 | −35.7 | −37.8 | −36.4 | −40.1 | −40.0 |
| 2025-10 | 3,570 | −30.7 | −31.3 | −32.4 | −22.7 | −26.6 | −26.7 |
| 2025-11 | 3,962 | −37.8 | −41.5 | −48.4 | −40.2 | −63.9 | −63.7 |
| 2025-12 | 4,164 | −39.8 | −40.3 | −40.4 | −35.2 | −48.8 | −49.5 |
| 2026-01 | 3,582 | −33.2 | −33.7 | −33.1 | −26.4 | −19.8 | −19.5 |
| **pooled** | **19,645** | **−35.2** | **−36.7** | **−38.6** | **−32.6** | **−40.6** | **−40.6** |

**Volume is worse in 5 of 5 folds.** Adding `vtrend` on top is worse again in 4 of 5.

The in-sample/out-of-sample gap names the mechanism directly, and it is monotone in both directions at once. In sample is the objective reached by fitting all 119 search days; out of sample is the fold-pooled figure above.

| arm | states | in sample (119 d) | out of sample | swing vs `novol` |
|---|---|---|---|---|
| novol | 15 | −34.52 | −35.2 | — |
| vol | 45 | −32.83 | −36.7 | **+1.7 in, −1.5 out** |
| volplus | 135 | **−32.09** | **−38.6** | **+2.4 in, −3.4 out** |

**Every state dimension added buys roughly a basis point in sample and costs one and a half out of sample.** The volume arm is better on the data it was fitted to and worse on data it was not, by almost exactly the amount it gained. That is overfitting with the arithmetic shown, and no interpretation of the volume channel is needed to explain it.

### Policy stability

`novol` selects **exactly 4 cut states out of 9 in all five folds** — the same count every time. `vol` selects 11, 10, 7, 9, 11 of 27. Adding the volume dimension destabilises the policy as well as degrading it. Fitted on the full search window the counts are 4 of 9, 6 of 27 and 23 of 81; `volplus` switches on nearly a third of its cells, which is what 175 trades per free parameter buys.

### The learned table (fit on all 119 search days, `novol`)

Cut states in **bold**. "paths" is distinct trades that ever visited the cell — the number that matters, since a position sits in one state for many consecutive minutes.

| pnl | slope | action | paths | steps |
|---|---|---|---|---|
| ≤ −2.0% | down | hold | 5,986 | 126,897 |
| ≤ −2.0% | flat | **CUT** | 5,917 | 529,690 |
| ≤ −2.0% | up | hold | 4,393 | 78,058 |
| −2.0..−0.7% | down | hold | 13,260 | 118,490 |
| −2.0..−0.7% | flat | **CUT** | 14,171 | 1,025,280 |
| −2.0..−0.7% | up | **CUT** | 9,424 | 73,660 |
| −0.7..0% | down | hold | 18,973 | 90,839 |
| −0.7..0% | flat | **CUT** | 24,645 | 1,323,459 |
| −0.7..0% | up | hold | 13,288 | 62,175 |
| all five add-legal cells (pnl ≥ 0) | | hold | 9,238 – 17,863 | — |

Every cell is backed by thousands of distinct trades, so nothing here is a small-sample artifact.

**Read this carefully, because it is not what it looks like.** The policy holds in every `down` cell and cuts in every `flat` cell. That is *not* "cut what is falling". The `flat` bucket is where positions **dwell** — 529k, 1,025k and 1,323k step-visits against 90k–126k for `down` — so "cut when below entry and not moving" is operationally "cut as soon as you are below entry, and keep cutting". The slope dimension is selecting the highest-occupancy bucket, not a direction of travel.

The `vol` policy is scattered by comparison — 6 cut cells with no coherent structure across `vsurge` (it declines to cut in almost every `heavy ≥ 1.5` cell, then cuts in `−0.7..0% / down / heavy`). Visits are in the thousands there too, so this is not noise from thin cells; it is the optimiser fitting training months with a dimension that carries nothing.

### `add` earns nothing

The search switches pyramiding **off entirely** in the final policy of every arm — 0 add states in `novol` and `vol` — and the fold-level add rate never exceeds 0.082. This is not for lack of opportunity: an add is legal at some point on **66.1%** of all paths.

The decomposition explains why, and it is a finding about the **live default**, not about the research arm. On the confirm slice:

| | per entry | per 100k deployed |
|---|---|---|
| hold-to-close, all trades | −50.6 bp | −50.6 bp |
| hold-to-close, on the 4,023 trades where `add_on_hit` fires | **+10.6 bp** | +10.6 bp |
| the same trades, with the add-on leg | **−35.7 bp** | **−17.8 bp** |
| the add-on leg alone (by difference) | | **≈ −46 bp** |

**`add_on_hit` is a good filter and a bad trade.** It picks entries that are 61 bp better than the pool on a hold-to-close basis — the selection is real. But the second share is bought *after* the +1% move and pays its own full 35 bp round trip, so the added leg loses about 46 bp on its own capital and drags a +10.6 bp entry down to −17.8 bp per unit of capital deployed. Across the whole confirm slice the live pyramiding rule reads **−81.9 bp per entry against −50.6 for the same entries held without it.**

Caveats: this is the 2025-08..2026-03 judgment-free population, not the judged live pool; live buys the add at the ask where this models the minute's open; and live's exposure cap can refuse an add this simulation allows. The direction is large enough to survive all three, and **`add_on_min_gain` is ON in the live default** — it is worth re-measuring on the judged days before leaving it there.

---

## 8. The shuffled-feature null

Each repeat reassigns every path's non-price channels (`slope`, `vsurge`, `vtrend`) from another path, tiled or truncated to length, so each channel keeps its own marginal distribution and its within-path serial correlation while losing its alignment to *this* path's prices. `pnl` is never scrambled: it decides action legality and is mechanically tied to the reward. The entire search is then re-run against that scrambled world, 20 times, and fit and test face the *same* scrambled world — scrambling them separately would set the policy a harder task than the real arm faces and understate the floor.

Each run is itself a best-of-a-very-large-N (a hypercube of 2^15 to 2^135 policies), so this is the analogue of the entry work's best-of-200-on-noise.

Measured on the inner holdout, so the real confirm slice stays clean:

| arm | | real | noise mean | noise p90 | noise max | clears? |
|---|---|---|---|---|---|---|
| novol | train | −34.75 | −34.90 | −34.78 | **−34.64** | **no** |
| novol | test | −33.24 | −34.25 | −33.87 | −32.66 | above p90, below max |
| vol | train | −34.50 | −34.53 | −34.18 | **−34.05** | **no** |
| vol | test | **−33.68** | **−33.62** | −32.52 | −30.51 | **no — below the noise MEAN** |

**Neither arm's training objective clears the maximum of its own noise band.** The search earns as much on scrambled features as on real ones.

And the noise band itself moves with the parameter count, exactly as it should:

| arm | free parameters | best-of-search on pure noise (train) |
|---|---|---|
| novol | 15 | −34.64 |
| vol | 45 | **−34.05** |

Tripling the free parameters buys **+0.6 bp of pure search premium** — a policy that fits scrambled volume better than a smaller policy fits scrambled nothing. That is the whole of what the volume channel delivers on the real data too: the `vol` arm's real training objective (−34.50) sits *below* what the same search extracts from noise at the same parameter count (−34.05). The 135-parameter `volplus` scramble set **did not complete**: it was killed by the kernel OOM killer five scrambles into twenty, when this study and another sharing the machine together pushed the user slice past physical memory (`memory.events` oom_kill counter; `memory.peak` 64.4 GB against 61 GB of RAM). An empty `journalctl -k` was first misread as ruling OOM out — `dmesg` is not readable by this user, so the kernel log was simply blind to it. Its fold results and confirm pass had already been written and are reported above; only this one row of the noise table is missing. It was not re-run, because it cannot change the conclusion — the two completed arms already fail their own noise band, and on this trend a 135-parameter search would only raise the floor further.

The last row is the cleanest single statement available on the volume hypothesis:

> **The real volume channel scores worse on a held-out month than the average of twenty scrambled volume channels.**

A result that does not clear its own shuffle premium is not a finding. Neither arm clears it, and the volume arm does not reach the middle of the noise.

---

## 9. `bayes_judge`, on the frozen slice, touched once

30 days, 5,954 trades, 2026-01-28..2026-03-16. Fit on the whole 119-day search window; scored once.

| candidate vs baseline | p(mu>0) | mu (KRW/day) | days ≥ 0 | adopt | binding reasons |
|---|---|---|---|---|---|
| seq(novol) vs live | 0.417 | −1,045 | 10/30 | **false** | p, loser ratio, days |
| seq(novol) vs hold | 0.420 | −1,006 | 9/30 | **false** | p, loser ratio, days |
| seq(novol) vs fixed | 0.420 | −758 | 11/30 | **false** | p, worst day, loser ratio, days |
| seq(vol) vs live | 0.392 | −1,354 | 9/30 | **false** | p, loser ratio, days |
| seq(vol) vs fixed | 0.432 | −681 | 14/30 | **false** | p, worst day, loser ratio, days |
| seq(volplus) vs fixed | 0.474 | −238 | 15/30 | **false** | p, worst day, loser ratio, days |
| **fixed vs live** | 0.420 | −995 | 10/30 | **false** | p, loser ratio, days |

Every p(mu>0) sits below 0.5 — the candidates are, if anything, worse. Note the last row: **on this slice even the fixed grid, the best arm everywhere else, does not beat the live rules.** On 30 days none of these arms is distinguishable from any other.

### What selected, what gated, and whether they are the same

The selection statistic is `net.mean()` — mean realised KRW per trade, on the training months only.

`bayes_judge` gates on four things: p(mu>0) over daily deltas, worst day, loser ratio, and count of days ≥ 0. **Three of the four are untouched by the selector.** The first is not independent of it — selecting the highest-mean policy and then asking whether its mean is positive is the same quantity twice — and that is unavoidable in any adopt decision. It is also why the frozen confirm slice and the shuffle premium exist: they are what make gate 1 mean anything. Nothing here reproduces the `optimize_policy.py` pattern, where the selector carried the worst-day term that gate 2 then re-read.

---

## 10. What this closes, and what it does not

**Closed.** The volume/fill-strength hypothesis for *exits*, asked twice and now tested twice. Two independent studies, different volume statistics, different populations, different objectives, different cost models, one of them with an action space that includes pyramiding: both find volume makes the exit policy worse out of sample, and this one finds it does not clear the noise floor of its own search.

**Closed.** That the earlier degeneracy was a reward artifact. It was not.

**Open, and cheap to settle.** `add_on_min_gain` is ON in the live default and costs ~31 bp per entry on this book while selecting entries that are 61 bp better than the pool. The filter is worth keeping; the second leg is not, on this evidence. Re-measure on the judged days before changing anything.

**Not addressed.** Everything §3 of the prior document lists. In particular, order flow: 1-minute OHLCV bars are not fill strength, and `rl_stop.py` said so explicitly in its own docstring. This study tests the best available *proxy* for the owner's prior and finds it empty; it does not test the prior itself, which needs the collector change (subscribe to the full candidate list from 09:00) and about six months of accrual.

**The arithmetic that has not moved all night.** The best exit anywhere in this study is −32.6 bp per trade. The round trip is 35.4 bp. No exit rule can fix an entry pool whose gross edge is below its own toll; it can only decide how quickly to pay it.

---

## 11. Artifacts

| file | what |
|---|---|
| `scripts/eval/seqexit_core.py` | reduction, state lattice, the vectorised sequential simulator, the objective |
| `scripts/eval/seqexit_gate.py` | the exhaustive exactness gate against `exit_fast` |
| `scripts/eval/seqexit_search.py` | folds, coordinate ascent, shuffled-feature null, the confirm pass |
| `scripts/eval/seqexit_compare.py` | reproduction of the published holdout, and the selection-defect price |
| `data/eval/exitseq/seqexit_gate.json` | gate result |
| `data/eval/exitseq/seqexit_search.json` | folds, pooled, shuffle premium, per arm |
| `data/eval/exitseq/seqexit_confirm.json` | the one-time confirm pass, policy tables, verdicts, daily series |
| `data/eval/exitseq/seqexit_compare.json` | both selectors, both grids, all seven folds |

`data/` is gitignored, as it is for every study here, so the four JSON artifacts live in the worktree `~/git/daytrade-it-exitseq` on simnode rather than in the branch. Only the scripts and this document are committed; the scripts regenerate the JSON from `paths_prior.jsonl` unattended. (The directory was renamed from `data/eval/rlnews/` to `data/eval/exitseq/` mid-study, so this study's files cannot be confused with the neighbouring `rlnews` worktree's.), and `seqexit_search.py --confirm-only` reproduces the confirm pass in minutes from the cached fixed grid.

Run in a worktree on simnode. kr_quant read-only, zero API spend, nothing touched on the trader machine, on `~/git/daytrade-it` there, or on any running daemon or backfill.
