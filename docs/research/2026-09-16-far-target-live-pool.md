# Far targets on the live pool: the best total in this work, and it is one day (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal`. Real replay engine, 7 dashboard-pool
days, 0 errors on every day. Zero API spend, kr_quant read-only, live repo/daemon/backfill
untouched.

**Verdict: no support for a far take-profit on the live pool. Dead rather than "collect more
days" — and the reason is structural, not just small-sample.**

`tp8_orb30` produces **+55,476 KRW against the live default's +39,915**, the largest total of
anything tested in this work. It is not a finding. `bayes_judge` says so (p(μ>0) = 0.550,
4/7 days, loser ratio 0.436 vs 0.343, adopt = false), and the per-day breakdown says why.

---

## 1. The panel

| policy | total | trades | Δ trades | losers | worst day | p(μ>0) | days ≥ 0 | adopt |
|---|---|---|---|---|---|---|---|---|
| **base (live default)** | **+39,915** | 35 | +0 | 12 | −2,838 | — | — | — |
| `tp5` | +40,246 | 36 | +1 | 12 | −2,838 | 0.544 | 5/7 | false |
| `tp8` | **+51,924** | 35 | +0 | 12 | −2,838 | 0.461 | 5/7 | false |
| `tp10` | +35,809 | 35 | +0 | 12 | −2,838 | 0.657 | 6/7 | false |
| `tp5_orb30` | +41,993 | 40 | +5 | 18 | −2,774 | 0.654 | 5/7 | false |
| **`tp8_orb30`** | **+55,476** | 39 | +4 | 17 | −2,774 | **0.550** | 4/7 | false |
| `tp10_orb30` | +38,611 | 39 | +4 | 17 | −2,774 | 0.685 | 5/7 | false |
| `orb30` | +42,717 | 39 | +4 | 17 | −2,774 | 0.801 | 5/7 | false |

Note `tp8` has the second-best total (+12,009 over base) while its posterior mean is
**negative** (μ = −58). A total and a day-level posterior disagreeing that sharply is the
signature the day-count criterion exists to catch.

## 2. It is one day

Per-day change against the live default:

| policy | 09-07 | 09-08 | 09-09 | **09-10** | 09-11 | 09-14 | 09-15 |
|---|---|---|---|---|---|---|---|
| `tp5` | −6,857 | 0 | +798 | **+6,619** | +120 | −349 | 0 |
| `tp8` | −5,987 | 0 | 0 | **+18,011** | 0 | −15 | 0 |
| `tp10` | −4,590 | 0 | 0 | **+200** | 0 | +284 | 0 |
| `tp8_orb30` | **−6,047** | +1,906 | −299 | **+19,771** | +181 | −15 | +64 |

`tp8_orb30`'s entire +15,561 advantage is 2026-09-10's +19,771. **Remove that one day and it is
worse than the live default** (the other six sum to −4,210). And 2026-09-07 goes the other way by
−6,047: the far target sells a position that kept running.

One up a lot, one down a lot, five flat. That is two draws, not an effect.

## 3. The target is also non-monotone, which confirms it

On 2026-09-10 the change is +6,619 at a +5% target, **+18,011 at +8%**, and **+200 at +10%**.

A real policy property would vary smoothly with the threshold. This is a single position that
peaked somewhere between +8% and +10%: an +8% target caught it, a +10% target missed it entirely
and the position went back to exiting as before. The same non-monotonicity shows in the totals
(+40,246 / +51,924 / +35,809 for tp5/tp8/tp10).

## 4. Why "collect more days" is unusually expensive here, not just slow

The trade counts explain the mechanism and set the cost of any future test. **Far targets barely
change the trade count** (+0, +1, +0). I had predicted the opposite — that staying in longer
would occupy concurrent-position slots and *reduce* entries, the mirror of trailing's +16. That
prediction was wrong, and the reason matters: **a far target almost never fires.** On the sweep
pool only 8.5% of entries ever reach +5% unrealized, and the share reaching +8% is lower still.
A rule that binds on a few percent of positions cannot change slot occupancy, and it cannot
change P&L except through those few positions.

So with 35 trades over 7 days, the `tp8` comparison rests on roughly **one position**. Evaluating
a rule that fires this rarely needs far more days than a rule that fires on most trades — the
estimator is dominated by a handful of observations by construction. That is why this is reported
as dead rather than pending: not because 7 days is few, but because this particular family needs
a sample far beyond what more dashboard days will realistically provide.

(Only `orb30` moves the trade count, +4, and it is a stop rather than a target.)

## 5. The σ-scaled far target could not be run, and why

The standing no-fixed-constants rule asks for the target as a multiple of the stock's own
volatility, judged against the best fixed target. **That cannot go through the replay engine.**
`TradingLimits.take_profit_pct` is a single scalar applied to every position; a per-position
target requires `AutoTrader` to accept per-ticker exit configuration, which is precisely the
production change the bandit design deferred ("청산 행동의 밴딧화 ... 다음 단계").

It could be approximated in my offline lab, which does support per-entry thresholds. I did not,
and the reason is §2–§4 rather than effort: any σ-scaled far target inherits the same structure —
it fires on the same handful of large movers, so it would be evaluated on the same one or two
positions, and the offline lab would additionally drop the concurrent-position cap that makes
these dashboard numbers meaningful. A σ-scaled version of a rule that rests on one position is
still a rule that rests on one position.

If per-position exits are ever built, this is the test to run then, on many more days.

## 5b. The sweep pool says the same thing, from the other direction

Extending the sweep-pool grid the same way (targets to +8%/+10%, trailing arms to 5%/8%) doubled
the search space from 2,160 to 4,320 configurations. Out-of-sample it changed nothing, and what
it did change went the wrong way:

| | 2,160 configs | 4,320 configs (far targets added) |
|---|---|---|
| p(μ>0) vs live | 0.871 | **0.867** |
| μ per day | +3,545 | +3,511 |
| days ≥ 0 | 50/84 | 50/84 |

Only one fold (2026-05) selected a far target in training — `take_profit_pct = 0.10` — and its
out-of-sample result got **worse**, −35 bp against −31 bp under the smaller grid. A larger search
space that lowers out-of-sample performance is the textbook overfitting signature, and it is the
independent confirmation that there is nothing at the far targets to find: the search reaches for
them in training and pays for it in the held-out month.

## 6. Where this leaves the exit work

Every exit candidate tested against the live pool, in one place:

| candidate | live-pool total | verdict |
|---|---|---|
| `orb30` | +42,717 | **best supported**: p = 0.801, 5/7, better worst day; also 4/5 sweep folds |
| `tp8_orb30` | +55,476 | one day (§2) |
| `tp5` / `tp8` / `tp10` | +40,246 / +51,924 / +35,809 | one day, non-monotone |
| `no_add_ons` | +30,027 | harmful, −9,888 |
| `hold_to_close` | +29,423 | harmful, −10,492 |
| `trail_2_1` | −547 | harmful, p = 0.016 |
| `consensus` (sweep folds' 5/5) | −1,973 | harmful, p = 0.004, 0/7 days |

**`stop_orb = 30` remains the only candidate positive on both pools**, and it is the only one
still worth days. It is a stop, not a target — it improves the worst day on both pools, and it
is the one change whose mechanism (cut the position that breaks the opening range) does not
depend on rare large movers.

**Recommendation for tomorrow is unchanged: do not change the live exit configuration.**

## 7. Artifacts

| what | path (simnode `~/git/daytrade-it-rl`) |
|---|---|
| dashboard-pool exit panel | `scripts/eval/dash7_exits.py` |
| results | `data/eval/rl_minimal/dash7_far.json` |

## 8. Concerns

1. **Seven days, and only ~35 trades.** Nothing here can clear the adoption bar and nothing here
   tries to. The claim is about which candidates are worth more days, not about adoption.
2. **The pools stay separate.** Every KRW figure in this document is the capped live portfolio on
   dashboard days. The sweep-pool figures elsewhere are the uncapped one-share counterfactual;
   per-trade bp transfer between them, totals do not.
3. **`tp8_orb30`'s +55,476 will look attractive to anyone reading totals.** It is the single most
   misleading number produced in this work and is recorded here so it is not rediscovered as a
   result later.
