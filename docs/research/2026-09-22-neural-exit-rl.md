# A neural-network exit policy, on a fresh holdout: nothing (2026-09-22)

**The first real function-approximation attempt at the exit — REINFORCE-trained networks reading a continuous, per-stock-normalised state with moving averages and volume in it — finds nothing. Every neural arm loses to a 15-state lookup table, and that table is itself identical, to a tenth of a basis point, to the zero-parameter rule "cut at the first minute below entry". No arm clears its shuffled-feature noise; the main network scores 9.5 bp *below* the mean of networks trained on scrambled features. And on the frozen confirm slice every active exit — learned, tabular or fixed — loses to simply holding to the close.**

Companion to `2026-09-22-sequential-exit-rl.md` (the table study). Same simulator, same gates, same selection statistic; a new population and a new frozen holdout.

---

## 1. The answer, in one table

Per-trade net, broker's own round trip (mean 39.0 bp on this book). `search-OOS` pools four time-ordered monthly folds (16,453 trades); `confirm` is the frozen slice, fitted once on the whole search window and scored once (31 days, 6,150 trades). The two columns are different months — compare down a column only.

| arm | params | search-OOS | confirm | shuffle null (inner holdout) |
|---|---|---|---|---|
| hold to close | 0 | −73.8 | **−35.1** | — |
| live exit rules | 0 | −72.1 | −35.4 | — |
| **cut at the first minute below entry** | **0** | **−42.3** | −41.9 | — |
| table, pnl × slope | 15 states | −42.5 | −41.9 | inside noise |
| table, + volume | 45 states | −43.2 | −42.0 | inside noise |
| best fixed grid | 1 of 4,320 | −43.8 | −40.7 | — |
| MLP, MA channels blinded | 193 | −43.6 | −42.1 | inside noise |
| **MLP, full state** | 193 | −47.3 | −42.6 | **9.5 bp below noise mean** |
| MLP, pool-standardised inputs | 193 | −47.5 | −45.3 | not run |
| MLP, volume blinded | 193 | −50.5 | −42.4 | 7.1 bp below noise mean |
| logistic | 14 | −50.0 | −54.9 | above p90, below max |
| live pyramiding rule | 0 | — | **−63.0** | — |

Neural confirm figures are the **median of three seeds**, not the best: choosing the best seed on the confirm slice would be selecting on it.

`bayes_judge` on the confirm slice: **adopt=false for every neural arm against every baseline** — live, hold, the fixed grid and the table. Against the table the networks are significantly *worse*: p(mu>0) = 0.005 (MLP), 0.003 (volume blinded), 0.042 (MA blinded), 0.000 (logistic, pooled).

Three things carry the result.

1. **More capacity does not help.** Zero parameters (−42.3) ≈ 15 states (−42.5) > 45 states (−43.2) > the 193-parameter network (−47.3) — the same shape as the table study's −35.2 → −36.7 → −38.6 on the prior book. The 14-parameter logistic is worst of all (−50.0), and it is the arm that adds most (§6), so this is not a clean capacity curve: what separates the arms is mostly whether they learned to pyramid.
2. **The learned table is a zero-parameter rule.** "Cut at the first minute below entry" scores −42.31 search-OOS and −41.85 on confirm; the learned 15-state table scores −42.5 and −41.9. The search found nothing beyond that corner of the action space, on either window.
3. **On the fresh confirm slice, holding wins.** −35.1 for hold-to-close against −41.9 for the cut-first rule. In the search months the post-entry drift was violently negative (July: hold −99.7) and cutting early saved ~30 bp; in August–September it turned benign and cutting sold dips that recovered. **"Cut losers fast" was a regime, not a rule.**

---

## 2. Normalisation audit — every feature, before any result

Required before the search, and done before it; recorded here first so it can be checked independently of the outcome.

**The rule applied:** every input is either (a) a dimensionless ratio that is cross-stock comparable by construction, or (b) such a ratio divided by *that stock's own* trailing statistic, computed at decision time. No feature is standardised with a pool-wide mean or variance. The network's input scaling is a vector of fixed constants chosen before training (`INPUT_SCALE`), identical for every stock and never estimated from the sample.

| feature | built as | normalisation | verdict |
|---|---|---|---|
| `pnl` | close / entry − 1 | a return; unit-comparable at any price level | correct |
| `pnl_n` | pnl / σ_own | per-stock: own 30-bar pre-entry minute σ, floored 15 bp | correct |
| `dd_n` | (close / peak since entry − 1) / σ_own | per-stock | correct |
| `slope_n` | 3-minute return / σ_own | per-stock | correct |
| `lvsurge` | log(3-bar volume / session mean volume per bar) | ratio vs own day so far | correct* |
| `lvtrend` | log(3-bar volume / previous 3-bar volume) | ratio vs own recent past | correct |
| `ma_frac_i` | share of *available* intraday MAs price is above | fraction in [0, 1] | correct |
| `ma_avail_i` | available intraday MA periods / 7 | fraction | correct |
| `ma_dist_i_n` | mean((close − MA_k) / MA_k) / σ_own | percentage distance, per-stock scaled | correct |
| `ma_frac_d` | share of 7 daily MAs last close is above | fraction | correct |
| `ma_dist_d_n` | mean % distance / own 60-day daily-return sd | per-stock, floored 0.5% | correct |
| `elapsed`, `to_close` | clock / fixed constant | not data-derived | correct |

\* `vsurge`'s denominator averages few bars early in the session; `elapsed` is in the state so the policy can discount that.

**Deliberately excluded:** raw σ as an input. It is literally "which stock is volatile" — the leakage that halved the travel-distance lift — and enters only as a divisor.

**Items checked specifically:**

- **The earlier tabular stop attempt (`rl_stop.py`)** built `pnl = close/entry − 1`, `dd = close/peak − 1`, `slope = close/base − 1`, `tv_rel = turnover / median`. All ratios; none in KRW. No fix needed.
- **The table lattice edges** (−2%, −0.7%, 0.7×, 1.5×) are fixed constants on dimensionless ratios — correct. Scaling the pnl edges by each stock's own σ is not a new idea: it *is* the σ-scaled contextual exit, already tested, and it lost.
- **`application/trading/bandit.py::fit`, lines 277–280:** `means = X.mean(axis=0)`, `stds = X.std(axis=0)` — **confirmed pooled.** That is the entry path, not this exit state.

### The mechanism, measured rather than asserted

Pooled z-scoring of a feature that is *already* a per-stock ratio is one global affine map — the same transform for every stock — so it cannot make one stock's state read differently because of another's. The travel-distance lift halved because its inputs were never divided by the stock's own volatility; pooled standardisation is what hides that defect, because the columns look well-scaled either way. **The defect is the missing per-stock division, not the z-score.**

The `mlp8_pooled` arm tests this directly: `bandit.fit`'s pooled standardisation reproduced on this state, fitted per fold on the training steps only, beside the fixed-constant version.

| | fixed constants (`mlp8`) | pooled z-score (`mlp8_pooled`) |
|---|---|---|
| search-OOS | −47.3 | −47.5 |
| confirm, median seed | −42.6 | −45.3 (seeds −44.4 … −51.6) |

Search-OOS is identical to 0.2 bp, as predicted. The confirm gap (2.7 bp) is inside the pooled arm's own seed spread (7 bp). **On a state already in per-stock ratio form, pooled standardisation neither helps nor measurably hurts — so no correction of the travel-distance kind (3.17× → 1.68×) was available to find here.** A large pooled-vs-per-stock gap on the *entry* side would not contradict this; the entry feature matrix was not in ratio form, and this state was from the start.

---

## 3. Moving averages — which formulation, and what the policy did with them

The owner's instruction: "5, 10, 20, 50, 100, 150, 200 선 기준 지금 위아래에 있냐가 중요해 청산은" — whether price is above or below each of those averages matters for the exit.

**Both readings were built, as separate state variables**, because they are different hypotheses: at one-minute resolution MA200 spans 3.3 hours; on daily bars it spans about ten months.

- **Daily** (`ma_frac_d`, `ma_dist_d_n`) — the classic technical-analysis meaning. From `daily_bars_adjusted` closes **strictly before** the decision day, evaluated at the previous close, so constant within the day: trend *context*. It is kept entirely inside the adjusted series on purpose. Comparing today's raw intraday price against a split/dividend-adjusted average mixes two price scales — the defect that manufactured +8.1 bp of D2 in the hourly transplant and +208 bp/day at the 15:00 seam. Built by `seqexit_ma.py`; 11,316 of 11,421 (day, code) pairs resolved, 105 lacking 201 daily closes. Mean fraction above: 0.556; 14.5% of pairs below all seven, 24.6% above all seven.
- **Intraday** (`ma_frac_i`, `ma_avail_i`, `ma_dist_i_n`) — the same periods over minute closes completed before the decision minute, entirely inside the raw minute series.

**Collapse.** The seven binaries are collapsed to *the fraction above the available averages*, plus the count available — not a 0–7 ordinal. At 09:10 only MA5 and MA10 exist intraday, and an ordinal would make "2 of 2" and "2 of 7" the same state. The continuous companion is the mean percentage distance, per-stock scaled. (Agreed with rl-overlap so the entry and exit studies use the same construction.)

**Live parity.** Daily inputs end at the close of d−1, in `daily_bars_adjusted` before the open. Intraday inputs are completed bars only. Both are computable by `AutoTrader` at decision time.

**What the policy did with them: nothing useful.** Blinding all five MA channels (`mlp8_noma`) *improved* the search-window score (−43.6 vs −47.3) and left confirm unchanged (−42.1 vs −42.6); its shuffle null sits inside noise. The daily and intraday groups were not ablated separately, because the joint ablation shows no contribution for either to lose. **Neither reading carries exit information that a network trained on 18,631 trades can use.**

---

## 4. The holdout, and the gates

**A new population, frozen before any feature on it existed.** `track_a_prior` is spent — its confirm slice was scored by the table study and again by the confirm-only re-run — so it is not read here. `paths_fresh.jsonl` uses the *same* judgment-free construction as `paths_prior` (`news_master_prior` → `prompt_labels` → `build_prior_paths`): 25,000 paths, 125 days, 2026-03-17..2026-09-15, none dropped. The split — search 94 days, confirm 31 days (2026-08-03..09-15) — was committed at `d55dd90` before the moving-average features were built or anything was fitted. The copy is `docs/research/frozen/2026-09-22-fresh-confirm-window.json`.

**Caveat:** these *dates* overlap the 2026-04..08 book that `rl_exits.py` and `rl_stop.py` used. That book was actual live-policy entries, not judgment-free pairs, and no exit policy has ever been fitted on this construction over these days. The population is new; the dates are not.

**Gates passed before training:**

| gate | result |
|---|---|
| exhaustive twin gate on the fresh book, re-run after extending the feature code | 24,781 usable; reduction identity 0 / 0 / 0 / 0; driven identity 0 mismatches at all 4 stop levels; asymmetry holds |
| hand-written backprop vs central finite differences | max relative error 1.2e-9 (logistic), 3.4e-7 (MLP) |

There is no torch on simnode. Rather than add a dependency to a shared environment, the backward pass is written out and gated: a policy-gradient trainer with a sign error still "trains" — it climbs the wrong hill.

**Method.** REINFORCE on the exact simulator (`Sim.run_steps`, the walk the twin gate verified). A Bernoulli action is sampled at every legal step; paths are rolled out by first-hitting time; each path is scored by realised KRW net of the per-leg round trip. The gradient of a Bernoulli log-probability with respect to its logit is `a − p`, weighted by the path's batch-normalised advantage, and applied only on steps the trajectory actually reached. The legality masks live in the simulator, so no weight can cut a winner or add to a loser. Selection statistic: mean realised KRW per trade, nothing else. Hyperparameters (300 iterations, batch 2,000 paths, Adam at 0.01, 8 hidden units, networks initialised near "never act") were fixed before any full run on this book; the only earlier contact was a 5,000-row smoke run inside the search window.

**Degrees of freedom:** logistic 14 parameters (1,331 search trades each); MLP 193 (97 each).

---

## 5. The shuffled-feature null

Each repeat reassigns every path's information channels — everything except `pnl`, `pnl_n` (which decide legality) and the clock — from another path, tiled to length, then retrains from scratch against that scrambled world and scores the same held-out month. Twenty repeats for the logistic and main MLP arms; ten for the two ablations.

| arm | real | noise mean | p90 | max | reading |
|---|---|---|---|---|---|
| table, 15 states | −47.83 | −47.85 | −47.49 | −47.49 | indistinguishable from noise |
| table, + volume | −48.48 | −48.57 | −47.69 | −47.29 | indistinguishable from noise |
| logistic | −55.29 | −59.77 | −58.18 | −54.60 | above p90, below max |
| MLP, full state | **−63.24** | −53.79 | −48.41 | −47.80 | **9.5 bp below the noise mean** |
| MLP, volume blinded | −63.24 | −56.29 | −49.76 | −48.48 | 7.1 bp below the noise mean |
| MLP, MA blinded | −48.91 | −52.86 | −47.94 | −47.58 | between mean and p90 |

**No arm clears the maximum of its own noise.** The main network does substantially *worse* on the held-out month than networks trained on features whose alignment with the prices has been destroyed. The logistic arm's p90 clearance does not survive anywhere else: it is the worst arm search-OOS (−50.0) and on confirm (−54.9).

Two honest notes on reading this table.

- **The inner holdout is the last monthly fold.** Fitting on months before July and scoring July is exactly the fourth fold, so each arm's "real" figure equals its July fold score (e.g. MLP −63.2). The same was true of the table study (its inner −33.24 equals its last fold −33.2). It is the correct noise comparison, but it is one month, not an independent observation on top of the folds.
- **The two −63.24s are the same policy.** The full-state and volume-blinded MLPs share initialisation and seed; in July both learned to add on ~24% of positions and made identical decisions on every path. Blinding volume changed nothing the network was using.

---

## 6. Why the networks lose: they learn to pyramid, and pyramiding loses

Every time a network learns to use the `add` action, it pays for it.

| | add rate | score |
|---|---|---|
| MLP, July fold | 24.3% | −63.2 (other folds −40.8 … −42.4, adds ≤ 2%) |
| logistic, confirm | 21–29% across seeds | −54.9 (the worst confirm arm) |
| volume-blinded MLP, confirm seed 2 | 29.0% | −55.5 (seeds 0 and 1, adds 0.2–0.3%: −42.0, −42.4) |
| live pyramiding rule, confirm | — | **−63.0** vs hold −35.1 |

This replicates the table study's decomposition on a second, independent population. There, on the prior book, the add-on filter selected entries 61 bp better than the pool, but the second share — bought after the +1% move, paying its own full round trip — lost about 46 bp on its own capital, and the live rule read −81.9 against −50.6 held. Here the live rule reads **−63.0 against −35.1**: 28 bp per entry.

**This is the one exit-side result that now holds on two independent books.** `add_on_min_gain` is ON in the live default. Caveats (judgment-free population, add modelled at the minute's open rather than the ask, live's exposure cap) are in the table study §7; none plausibly reverses a gap of this size twice.

---

## 7. Operational note: runs lost, and my part in it

Three batches of runs died mid-flight tonight on simnode — this study's, and the neighbouring `rlnews` worktree's. I first called it "not OOM" (the kernel log was empty), then "confirmed OOM" (the user slice's `memory.events` shows `oom_kill` 1702, `memory.peak` 64.4 GB against 61 GB of RAM). **Both were overreach.** `dmesg` is not readable by this user, so the empty kernel log proved nothing. The third death left `oom_kill` unchanged, so the 1702 may be days old, and `memory.peak` fell to 60.0 GB, meaning something reset it. **The cause is unproven.**

What is proven is my share of the load. Each numpy process ran 17 threads (one BLAS thread per core), so three neural arms took ~14.5 of the 16 cores and the load average reached 40. Since relaunching as systemd user units with `OPENBLAS/OMP/MKL_NUM_THREADS=1`, `CPUQuota=100%` and `Nice=10` — and from a cached reduced book (~1.4 GB per arm, instead of each re-parsing 490 MB of JSON) — nothing has died. Five canaries (bash and python; `nohup`, `setsid` and `systemd-run` launches) logged no signal over 35+ minutes. Single-threaded BLAS also turned out *faster* for these small matrices: an MLP fold went from ~4 minutes to ~2.

---

## 8. What this closes

**Closed: the exit, for this entry pool, at the resolution we have.** Four independent attempts now — a fixed grid, a σ-scaled contextual exit, a sequential lookup table, and REINFORCE-trained networks — on two populations with two frozen holdouts. None clears its noise. The best learned policy on either book is a zero-parameter rule the search could have been handed. Volume has now been tested four times. It made the policy worse in three: the earlier stop policy, the table on the prior book, the table on this book (−43.2 vs −42.5). In the network, blinding it made search-OOS *worse* (−50.5 vs −47.3), so on that one figure volume appears to help by 3.2 bp — but the whole gap is one fold (April: −52.3 blinded vs −41.7), it vanishes on confirm (−42.4 vs −42.6), and on the held-out month the two networks make identical decisions on every path. That is not evidence the network uses volume. The owner's moving-average hypothesis was tested in both of its readings and contributes nothing the network can use.

**Closed: "cut losers fast" as a standing rule.** It saved ~30 bp in the search months and cost ~7 bp on the confirm months. It is a bet on post-entry drift continuing down, and it wins exactly when that happens.

**Open, and cheap: pyramiding.** Now replicated on two books. Worth re-measuring on the judged live days before leaving `add_on_min_gain` on.

**The arithmetic that has not moved.** The best exit anywhere tonight is roughly −35 bp per trade against a round trip of 35–39 bp. An exit decides how quickly to pay the toll on an entry pool whose gross edge sits below it; it cannot make that pool profitable. The lever, if there is one, is upstream.

---

## 9. Artifacts

| file | what |
|---|---|
| `scripts/eval/seqexit_nn.py` | REINFORCE trainer, gradient gate, folds, shuffle null, confirm |
| `scripts/eval/seqexit_ma.py` | daily moving-average position per (day, code) |
| `scripts/eval/seqexit_core.py` | reduction (now with drawdown, σ_own and intraday MA channels), `Sim.run_steps` |
| `scripts/eval/seqexit_nn_report.py` | this document's side-by-side tables and `bayes_judge` verdicts |
| `scripts/eval/seqexit_degenerate.py` | the zero-parameter "cut first minute below entry" reference |
| `docs/research/frozen/2026-09-22-fresh-confirm-window.json` | the frozen split, committed before any fit |
| `data/eval/exitseq/seqexit_nn_<arm>.json`, `seqexit_table_fresh.json`, `seqexit_nn_verdicts.json`, `seqexit_gate_fresh.json` | results (gitignored; on simnode) |

kr_quant read-only, zero API spend, nothing touched on the trader machine or any running daemon.
