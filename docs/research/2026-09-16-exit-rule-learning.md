# Learning the exit: the biggest lever found so far, and still not adoptable (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal`. Zero API spend, kr_quant read-only,
live repo/daemon/backfill untouched. Monthly time-ordered folds only; LODO banned.

**Verdict: the learned exit roughly halves the per-trade loss — from −52 bp to −29 bp — and
still fails adoption.** `bayes_judge` against the current live exits gives p(μ>0) = 0.871 against
a 0.90 bar, and 50 of 84 days ≥ 0 against a floor of 60. It passes the two loss-prevention
criteria (worst day −214,007 vs −323,420; loser ratio 0.634 vs 0.637).

**On "no fixed constants": context-dependence buys stability, not money.** Making the stop and
target multiples of the entry's own volatility produces *far* more stable parameters across folds
(a = 1.0 in 4 of 5 folds) than the fixed grid (5 distinct configurations in 5 folds) — but it does
not beat the best fixed constant on the adoption criteria, and it is worse on the worst day.

---

## 1. The exactness gate did its job, and found a live-parity bug

The search scores configurations with a vectorized twin of `auto_trader.exit_rule_hit`, because
the production predicate recomputes the session VWAP on every call and a 1,080-configuration grid
over 4,416 paths is otherwise hours. The twin is gated: it must return the same exit minute as
the live function on every sampled path for every sampled configuration, or the search refuses to
run.

**It refused.** The first run reported 5 mismatches out of 13 configurations × 400 paths, all the
same shape: reference minute 919, twin minute 900. After the fix, a wider test passes cleanly:
**25 configurations × 1,500 paths = 37,500 comparisons, 0 mismatches**
(`scripts/eval/verify_more.py`).

The cause was real and neither side had it right. **A stock that prints no 15:00 bar cannot be
sold at 15:00.** Live places the exit at `flatten_at` and retries every minute with a widening
discount until it fills, with continuous trading ending at 15:20 — so the position leaves at the
**first print at or after 15:00**. My reference fell through to the day's last bar; the twin
labelled it 15:00. Both now implement the live semantics, with "no print at all at or after
15:00" (leftover valued at the last close) kept distinct from a real flatten.

This matters more than a labelling nit: it only bites illiquid minutes, which is exactly where a
stop or a trailing exit is most likely to fire and most likely to fill badly. A search run on the
unfixed twin would have mispriced precisely the configurations it existed to evaluate.

## 2. Parity against the replay engine

The offline lab is checked against the engine on the **same policy** (add-ons off, so the
comparison isolates the lab and not the pyramiding setting):

| | |
|---|---|
| entries | 4,416 |
| lab mean | −49 bp |
| engine mean | −62 bp |
| correlation | **0.9996** |
| median abs difference | **12 bp** |
| differing by > 50 bp | **0.2%** |

**The lab is systematically ~13 bp optimistic**, because it models the sell as filling at the
order minute's open while the engine walks the bid with a discount ladder. That bias applies
equally to the candidate and the baseline, so it largely cancels in every comparison below — but
absolute lab figures are optimistic and should not be quoted as P&L. Any configuration that
survives here is re-run through the real replay engine before adoption.

## 3. Global exit search, monthly time-ordered folds

1,080 configurations over `stop_loss_pct`, `take_profit_pct`, `trail_after_gain`/`trail_gap`,
`stop_time`, `stop_orb`. `vwap_exit_gap` is held at its adopted live value. The configuration is
chosen on the months strictly before the evaluation month.

| fold | n | chosen configuration | live | learned |
|---|---|---|---|---|
| 2026-04 | 1,032 | trail 0.01/0.005 | −49 bp | **−40 bp** |
| 2026-05 | 916 | tp 0.01, trail 0.01/0.01, orb 30 | −49 bp | **−31 bp** |
| 2026-06 | 859 | sl 0.02, tp 0.03, trail 0.03/0.01, orb 30 | −36 bp | **−21 bp** |
| 2026-07 | 640 | sl 0.03, tp 0.03, trail 0.03/0.01, time 60, orb 30 | −121 bp | **−39 bp** |
| 2026-08 | 404 | sl 0.015, tp 0.03, trail 0.01/0.005, time 30, orb 30 | −12 bp | **−7 bp** |

**Every fold improves, 5 of 5.** Pooled, the per-trade loss falls from **−52 bp to −29 bp**.

`bayes_judge` against the live exits, 84 out-of-sample days:

```
p(mu>0) = 0.871   mu = +3,545/day   days>=0 = 50/84
worst day  -214,007 vs -323,420  (passes)
losers      0.634 vs 0.637        (passes)
adopt = false
  - p_mu_gt_0=0.8715 < 0.90
  - days_ge_0=50 < ceil(5/7*84)=60
```

**Parameter instability is the weakness.** Five folds chose five different configurations. Only
two things are consistent: **trailing is on in 5 of 5 folds**, and **`stop_orb = 30` in 4 of 5**.
`stop_loss_pct` takes None, 0.02, 0.03, 0.015; `take_profit_pct` takes None, 0.01, 0.03, 0.03,
0.03. This is the same disease A7-prep's stage-2 optimizer had — a family resemblance, not a
policy — and it is why the fold-level improvement does not translate into a day-count majority.

## 4. Context-dependent exit: stop and target as multiples of entry volatility

Per the standing "no fixed constants" rule, the stop and target become `a × σ` and `b × σ`, where
σ is the stock's 1-minute volatility over the ≤30 bars completed **before** the entry minute
(decision-time only; the live daemon already pulls those bars for `ret_5m`). Median σ is 25 bp,
p10 15 bp, p90 60 bp — so a single fixed percentage is a 4× different bet across the pool.

| fold | a (stop) | b (target) | live | best fixed | **context** |
|---|---|---|---|---|---|
| 2026-04 | None | 3.0 | −49 | −40 | **−37** |
| 2026-05 | 1.0 | 1.5 | −49 | −31 | **−23** |
| 2026-06 | 1.0 | 1.5 | −36 | −21 | −23 |
| 2026-07 | 1.0 | 2.0 | −121 | −39 | **−25** |
| 2026-08 | 1.0 | 1.5 | −12 | −7 | −17 |

**The parameters are far more stable than the fixed grid's**: `a = 1.0` in 4 of 5 folds,
`b = 1.5` in 3 of 5. Against five distinct fixed configurations in five folds, that is the
clearest evidence in this work that the volatility-scaled form is the better *parameterisation* —
it is learning one thing rather than five.

But it does not pay for itself:

```
context vs live exits    p(mu>0)=0.868  days>=0=49/84   adopt=false
context vs best fixed    p(mu>0)=0.914  mu=+2,131/day  days>=0=45/84
                         worst day -230,261 vs -214,007  (FAILS)
                         losers     0.578 vs 0.634       (passes)
                         adopt=false
```

Against the best fixed constant it clears p(μ>0) = 0.914 and materially improves the loser ratio
(0.578 vs 0.634), but it loses on the two loss-prevention-and-breadth criteria: a worse worst day
and 45 of 84 positive days. It wins on 3 of 5 folds and loses clearly on 2026-08.

**The honest reading:** the user's instinct that hand-set constants are the wrong shape is
supported on *stability* grounds, and the volatility-scaled form is what I would carry forward.
It is not supported on money, on this sample.

## 4b. Does the corrected bug invalidate the decision that turned trailing off? **No.**

The bug I fixed lived in `scripts/eval/exit_lab.py` and `scripts/eval/exit_fast.py` — both
written today, used only by this offline search, and never by any production decision. The 7-day
test whose result keeps trailing and stop-loss switched off in live ran through the **replay
engine**, which models the exit correctly:

- `auto_trader.manage()` (line 775) calls `flatten()` on **every tick** at or after `flatten_at`,
  and `_exit_or_retry` re-prices an unfilled sell after `exit_retry_after` with a widening,
  capped discount.
- `SimBroker` holds sells as **resting orders**, filling on whichever later bar satisfies
  `b.high >= limit` (line 167).

Neither depends on a bar existing at 15:00. So the engine already did what my lab got wrong.

Rather than rest that on a reading of the code, the candidates were re-run through the real
engine on the 7 dashboard-pool days (`scripts/eval/dash7_exits.py`, 0 errors on all 7 days):

| policy | total | trades | losers | worst day | p(μ>0) | days ≥ 0 | adopt |
|---|---|---|---|---|---|---|---|
| **base (live default)** | **+39,915** | 35 | 12 | −2,838 | — | — | — |
| `trail_2_1` | **−547** | 51 | 21 | −8,560 | **0.016** | 1/7 | false |
| `orb30` | **+42,717** | 39 | 17 | −2,774 | **0.801** | 5/7 | false |
| `consensus` (my folds' 5/5 config) | **−1,973** | 59 | 31 | −4,680 | **0.004** | 0/7 | false |

**The old result reproduces to the won**: base +39,915, `trail_2_1` −547, p = 0.016, μ = −5,043,
1/7 days — identical to the pre-registered table in `2026-09-16-bayes-pipeline-eval.md`. `orb30`
likewise reproduces at p = 0.801, 5/7.

**The evidence that turned trailing off still stands.** It was not an artifact of the bug.

**And my own sweep-pool finding does not transfer.** The configuration my monthly folds selected
5 out of 5 times is the *worst* of the four on the dashboard pool: −1,973, p = 0.004, 0 of 7 days
positive. I am not going to argue for it on the strength of a pool that loses money.

The two pools disagree for a reason that is economic, not methodological. **Trailing caps winners
and cuts losers.** The dashboard pool is profitable (34% losing trades); the sweep pool is not
(64% losing trades, −52 bp/trade). A rule that trades away upside to limit downside helps the
second and hurts the first — which is exactly the asymmetry that makes a trailing stop dangerous
on a pool with positive drift: it exits near the bottom of a retrace on the names that would have
recovered.

One mechanism worth noting: **trailing raises the trade count** (35 → 51 → 59). Exiting early
frees a concurrent-position slot, so more entries get in. Trailing is not only an exit change; it
changes which entries the day takes.

**The one finding with support on both pools is `stop_orb = 30`** — p = 0.801 and a better worst
day on the dashboard pool, chosen in 4 of 5 sweep folds. It is still short of the 0.90 bar, on
both.

### Recommendation for tomorrow

**Do not change the live exit configuration.** The reason trailing is off is real, reproduces
exactly, and my contrary sweep-pool evidence comes from a different pool with opposite drift.
Seven days could not clear the bar anyway, and nothing here asks it to.

If anything is worth queuing for more evidence it is `stop_orb = 30`, the only candidate that
improves the worst day on both pools. That is a "collect more days" recommendation, not a
"change live tomorrow" one.

## 5. Pyramiding, hold-to-close, and the finding that undoes this whole section

### 5a. Two more "cut risk" changes, measured on the sweep pool

**Pyramiding** (`add_on_min_gain = 0.01`), paired engine-against-engine on the same entries so the
add-on effect is not mixed with the lab's fill model:

| | |
|---|---|
| paired entries / days | 4,377 / 95 |
| add-ons **off** | −62 bp |
| add-ons **on** | −93 bp |
| paired difference | **−31 bp (se 1 bp)** |
| entries actually changed | 47.0% |
| days worse with add-ons on | **95 of 95** |

**Hold-to-close.** `vwap_exit_gap` had been pinned at its live value, which kept "switch every
early exit off and simply flatten at 15:00" outside the search space — the same kind of ceiling
as the +5% target cap. Adding `vwap_exit_gap ∈ {0.01, None}`:

| fold | live exits | hold-to-close |
|---|---|---|
| 2026-04 | −49 | −50 |
| 2026-05 | −49 | −48 |
| 2026-06 | −36 | −34 |
| 2026-07 | −121 | −126 |
| 2026-08 | −12 | −9 |

Hold-to-close is **within 1–5 bp of the live exits**, and the learned configuration beats it at
p(μ>0) = 0.835 — essentially the same verdict as against live (0.871). The two baselines are
interchangeable on this pool. (prompt-opt measured hold-to-close 60 bp *better* than live on
overlapping rows; that comparison had pyramiding on for "live" and off for "hold", so roughly
half of it is the add-on effect above and the rest is their label's optimism.)

### 5b. Both reverse on the pool live actually trades

Run through the real replay engine on the 7 dashboard days (`scripts/eval/dash7_exits.py`,
0 errors):

| policy | total | trades | losers | worst day | p(μ>0) | days ≥ 0 |
|---|---|---|---|---|---|---|
| **base (live default)** | **+39,915** | 35 | 12 | −2,838 | — | — |
| `no_add_ons` | **+30,027** | 35 | 12 | −1,601 | 0.131 | 4/7 |
| `hold_to_close` | **+29,423** | 34 | 11 | −1,601 | 0.121 | 4/7 |
| `trail_2_1` | −547 | 51 | 21 | −8,560 | 0.016 | 1/7 |
| `consensus` (my folds' 5/5) | −1,973 | 59 | 31 | −4,680 | 0.004 | 0/7 |

Switching pyramiding off **costs 9,888 KRW** on the live pool; hold-to-close costs 10,492.

### 5c. The pattern, and what it costs this document

**Three independent changes — trailing, hold-to-close, and add-ons-off — all help the sweep pool
and all hurt the dashboard pool.** None is a near miss; all three reverse.

The reading is economic, not methodological. The dashboard pool is profitable (34% losing
trades); the sweep pool is not (64% losers, −52 bp/trade). Every one of these changes trades
upside away for downside protection — which is what a losing pool needs and what a winning pool
is destroyed by. The risk metrics behave exactly as that predicts: on the dashboard pool both
`no_add_ons` and `hold_to_close` **improve** the worst day (−1,601 vs −2,838) and the loser ratio
while giving up a quarter of the profit.

**So §3 and §4 do not license a live change.** The −52 → −29 bp improvement is real for a pool we
do not trade. This is the central caveat of this document, and it was found only because the
trailing re-test (§4b) forced the same check on every subsequent candidate.

## 6. What this does and does not establish

Established:

1. The vectorized twin is exact against the live predicate: **0 mismatches in 37,500
   comparisons**, after it caught a real live-parity bug on its first run.
2. On the **sweep pool**, a learned exit roughly halves the per-trade loss (−52 → −29 bp),
   improving in 5 of 5 folds, and still fails adoption (p = 0.871, 50/84 days).
3. On the **sweep pool**, pyramiding costs 31 bp/trade (se 1) and is worse on 95 of 95 days;
   hold-to-close is indistinguishable from the live exits.
4. **None of that transfers.** All three risk-cutting changes reverse on the dashboard pool,
   which is what live trades. The evidence that turned trailing off in live reproduces to the
   won (§4b) and stands.
5. Volatility-scaled thresholds are a **more stable parameterisation** than fixed percentages
   (a = 1.0 in 4/5 folds vs five distinct fixed configurations in five folds) and do not beat
   them on realized money.

Not established:

- Whether any exit helps the **dashboard** pool. Seven days cannot answer it, and the sweep pool
  cannot stand in for it. This is now the binding question for the whole exit line, and it needs
  dashboard-pool days, not more sweep days.
- The **(a)+(b) combination**. Every figure here holds the entry ranking at the live default.
- Whether a **far target** monetises the travel distance prompt-opt measured (+10% shows 22.2x
  raw / 9.23x σ-normalised lift, against 1.47x / 1.12x at +1%). The grid's +5% ceiling and 3%
  trailing arm made that untestable; extended to +8%/+10% targets and 5%/8% arms, results pending.

**The methodological lesson worth carrying beyond this document:** a sweep-pool optimisation
result must be checked on dashboard days before it is called anything. Three candidates looked
good on the sweep pool and all three would have been harmful in live. That applies to the
optimizer and the label work as much as to exits.

## 7. Artifacts

| what | path (simnode `~/git/daytrade-it-rl`) |
|---|---|
| exit search + context-dependent search | `scripts/eval/rl_exits.py` |
| offline lab calling the live predicate | `scripts/eval/exit_lab.py` |
| vectorized twin + its contract | `scripts/eval/exit_fast.py` |
| wider exactness test, paired add-on cost | `scripts/eval/verify_more.py` |
| results | `data/eval/rl_minimal/exits.json` |
| recorded entry paths (4,416 entries, 97 days) | `data/eval/rl_minimal/paths.jsonl` |

## 8. Concerns

1. **The lab is ~13 bp optimistic** on absolute levels (§2). It cancels in comparisons; it would
   not cancel if anyone quoted a lab figure as expected P&L.
2. **Parameter instability** (§3). Five folds, five configurations. Only "trailing on" and
   "orb 30" survive as consensus, and a policy built from the consensus rather than any one fold
   has not been tested — that is a cheap next step.
3. **MFE/MAE are printed by the script as a diagnostic and are not evidence.** A maximum is
   mechanically ≥ the entry for any stock; the September null showed random tickers reaching
   5.75σ of maximum. No claim in this document rests on them.
4. **84 evaluation days carry fewer than 84 independent observations** — roughly a quarter of
   reward variance is day-level. The `bayes_judge` day series is the right shape, but the failures
   above are, if anything, understated.
5. **Process trap, for whoever comes next:** this repo has a `ci-local` pre-push hook that leaves
   the worktree's HEAD **detached**. Five commits sat unpushed here while `git push` reported
   success — it was pushing a stale branch pointer. `learning-curve` hit the same thing. Treat
   "Everything up-to-date" straight after a commit as an error signal, and check `git status -sb`
   after every push.
