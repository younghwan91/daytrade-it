# What is actually testable, and what it says (2026-09-22)

**The usable sample for any LLM-feature policy is 125 days, and that is now a structural ceiling rather than a shortage to wait out. Widening the sample four-fold, using every field we hold, allowing interactions, and adding a latent regime state did not narrow the gap between alpha and the toll.**

This document supersedes one earlier claim (§2.4) and records four defects found along the way (§5).

---

## 1. The usable sample, per track

Counted directly on every axis.

| source | coverage | trading days |
|---|---|---|
| Kiwoom `minute_bars` | 2025-08-01 .. 2026-09-21 | 278 |
| Yahoo hourly, clean subset | 2023-09-19 .. 2026-09-21 | 722 |
| `daily_bars_adjusted` | 2016-09-09 .. 2026-09-21 | 2,459 |
| Toss news archive | 2023-03-16 .. 2026-09-17 07:49 | — |
| **LLM judgments** | **2026-03-17 .. 2026-09-15** | **125** |

**Track A — minute, intraday.** 278 bar-days, of which 125 judged. Every judged day has bars, so for the current entry policy — which ranks on eight LLM-derived features — **the 125 judged days are the entire sample, and the 14-month bar window adds zero.** Kiwoom's 13-month depth is server-side retention, confirmed by probing `ka10080` at 30- and 60-minute scope and finding both exhaust at the same 2025-09-01 floor. **That makes the limit structural, not a shortage to wait out.**

**Track B — hourly.** 722 clean days (`identity_verified & ~irregular_session`, 9,313,689 rows, 2,050 codes). 124 judged; **598 with news and prices but no judgment.**

**Track B — daily.** 2,459 days, 854 of them with news.

**Permanently unusable at intraday resolution: 285,826 articles** (2023-03-16..2023-09-19), before Yahoo's rolling 730-trading-day window, which recedes one day per day.

### The judgment-free holdout

The 125 judged days are the entire sample only for a policy that *needs* judgments. The matched-null construction does not. That yields two never-fitted holdouts, at zero API cost:

- **149 days, 2025-08-01..2026-03-16, minute resolution** — 122,877 pairs priced, 4.19% dropped, every drop accounted for by bar coverage.
- **598 days, 2023-09-19..2026-03-26, hourly resolution** — 285,440 pairs on 596 days with pairs.

Every published result in this project was fit on 2026-03..09. Nothing had ever looked at these.

**Confirmation windows were frozen before any searching**, committed at `f9376cf` (02:52) with the first search dispatched after: Track A judged 2026-08-19..2026-09-15 (20 d), Track A prior 2026-01-28..2026-03-16 (30 d), Track B hourly 2026-03-27..2026-09-21 (120 d), daily 2025-03-26..2026-09-21 (365 d).

---

## 2. What was found — with each number's method named

### 2.1 Selection alpha, matched null

Null matched on day x acting bar x trailing-volatility quartile x prior-day traded-value quartile, quartiles formed inside each day from the **previous** session only. `D2 = pool - universe`; cost cancels in the difference and is neither added nor subtracted. Day-clustered bootstrap.

| sample | resolution | pairs | days | D2 / pair | se | days > 0 |
|---|---|---|---|---|---|---|
| judged, fitted | minute | 131,712 | 125 | **+1.7 bp** | 3.4 | 69/125 |
| prior holdout | minute | 108,603 | 129 | **-0.3 bp** | 2.0 | 67/129 |
| shared days | hourly | 124,446 | 124 | +2.7 bp | 3.65 | 73/124 |
| **never-fitted** | **hourly** | **285,440** | **596** | **-2.0 bp** | **1.17** | **288/596** |

Reproduction gate passed before anything else: the published +1.6 bp / se 3.4 / 69-of-125 reproduces at +1.7 / 3.4 / 69. The hourly-vs-minute comparison on the shared 124 days (+2.7 vs +1.8) sits inside one standard error, so **the resolutions agree** and the hourly result is not a data pathology.

Against a **36.7 bp** round-trip toll (`scripts/eval/costs.py`), all four are zero.

**By year, hourly, never-fitted:** 2023 **-4.2**, 2024 **-5.1**, 2025 **-0.9**, 2026 **+4.2**. Monotone in time; negative in every year before the judgment window opens. **The fitted +1.7 bp sits at the top of a four-year drift, not in the middle of a stable one.** Two artifacts run the same direction and cannot be separated from it: the archive doubles in depth from 2025-10, and the survivorship hole shrinks 8.0% -> 1.2%. Do not read the trend as regime alpha.

Robustness: splitting the minute holdout by bar-coverage quality gives -3.2 +/- 1.9 (88-94% coverage) and +3.5 +/- 4.3 (97%+). Not a pricing artifact.

### 2.2 Entry

**Bandit, 10 judgment-free decision-time features, monthly time-ordered folds, nested hyperparameter choice, no LODO.** On the 149-day holdout, **no fold produces a positive per-trade net.** Five of six folds select nothing; the one that trades returns **-46.95 bp per trade, 8.8 bp worse than the pairs it rejected**. `bayes_judge` adopt=false (p=0.577, days>=0 66/107, loser ratio 0.719 > 0.627). Lambda pins to the grid maximum in 6 of 6 folds; 4 of 11 coefficients sign-stable.

`worst_day_candidate` reads 22x better than baseline. **That is abstention, not risk control: 93 of 107 candidate days have zero trades.** The decisive number is per-trade net.

**Wide search: 6 feature families x 10 configs x 2 targets x 2 periods = N = 200**, plus 8 confirm evaluations from a shortlist fixed before the window was read. Families span all ten v2 judgment fields, volume relative forms, free deterministic attributes, and gradient-boosted trees so interactions are reachable.

**Selection premium, labels shuffled within each day, 20 repeats per search (~16,000 fits):**

| cell | best real | shuffle best-of-N mean / p90 / max | verdict |
|---|---|---|---|
| judged, alpha | +15.99 | **+9.56 / +18.08 / +23.43** | below p90 — dead |
| prior, alpha | +18.09 | +2.95 / +7.15 / +13.54 | above max, barely |
| judged, reward | -50.66 | -111.23 / -104.44 / -96.65 | beats max, still -51 |
| prior, reward | -5.69 | -33.14 / -26.12 / -19.47 | beats max, still negative |

A search of this shape earns **+9.6 bp on pure noise on average and +18 bp one time in ten.** Random top-5%-per-day selection earns +6.78 bp; the perfect-foresight oracle earns +842. The best result anywhere recovers **1.9% of the oracle gap**. The claimed "GBM 9% vs linear 3%" did not reproduce at any point.

**The GBM interaction gain does not reproduce on the never-fitted holdout** — the GBM-vs-ridge ordering flips between periods and between targets.

### 2.3 Exits

All figures at the broker's own round trip (mean 38.3 bp), not the 23.0 bp literal the exit lab had been charging.

**Holdout, 7 folds, 25,599 trades:** live **-42.2**, hold-to-close **-42.9**, best learned fixed **-35.4**, sigma-scaled **-40.5** bp per trade. **No arm is positive.** Sigma-scaled vs best fixed `p(mu>0)=0.002` — significantly worse out of sample. Best fixed vs live `p=0.546`. All adopt=false.

**The worst-day gate diagnosis.** `bayes_judge` gate 2 is `min(net)` over days — on 84 days, one day decides. The sigma-scaled exit cleared p(mu>0) and improved the loser ratio and died only here. Three penalties, same gate direction, after reproducing the recorded verdict exactly on 84 days:

| penalty | candidate | baseline | margin | passes | bootstrap |
|---|---|---|---|---|---|
| `raw` min(net) | -239,836 | -223,547 | -16,289 | no | 0.364 |
| `weighted` min(net/trades) | -3,997 | -3,726 | -272 | no | 0.364 |
| `robust` q05(net) | -35,401 | -59,229 | **+23,828** | **yes** | 0.997 |

**`weighted` agrees with `raw`; `robust` disagrees with both.** The thin-day size confound was **not** the operating defect — normalising by trade count neither rescues the candidate nor moves the bootstrap. **The defect was single-day extremeness.** And the gate as written passes in only **36% of day-resamples**: a different draw of the same 84 days reverses it one time in three.

Two conclusions that must not be merged: **the worst-day gate has a real defect worth fixing**, and **the sigma-scaled exit is not the candidate to fix it for** — on the 598-day holdout all three penalties reject it (`robust` -11,444, bootstrap 0.129), and the stability argument inverts (the fixed grid chose 3 configurations in 7 folds, against 5 in 5 in sample).

### 2.4 Horizons — supersedes the earlier +40d claim

Stratum-matched null rebuilt at every horizon; 854 days; day-level se inflated by sqrt(h) for window overlap.

| h | 2023 | 2024 | 2025 | 2026 | pooled | se_ov | net bp/day |
|---|---|---|---|---|---|---|---|
| 1 | -3.8 | +1.7 | +2.8 | -1.4 | +0.1 | 1.1 | -39.17 |
| 5 | -2.0 | +4.1 | +1.0 | -3.1 | +0.4 | 5.9 | -7.78 |
| 10 | +5.5 | +7.1 | +3.1 | -5.2 | +3.2 | 11.9 | -3.61 |
| 20 | -7.0 | -0.6 | +12.3 | +11.7 | +3.9 | 22.7 | -1.77 |
| 40 | +41.8 | -0.5 | +24.0 | +33.2 | +22.5 | 43.4 | -0.42 |

Gross alpha never reaches its own overlap-adjusted standard error at any horizon, and the sign is not stable by year. Cost amortisation works exactly as advertised (39 -> 0.4 bp/day) and buys nothing.

> **Supersedes:** the earlier "+96.6 bp at +40d, alpha growing with horizon". That came from **86 common days in a single hostile year** and predates the sqrt(h) overlap inflation, without which the standard error reads ~sqrt(40) too small. On the multi-year sample it is **+22.5 bp against se 43.4 — half its own error.** The two figures should not both circulate.

**Dividend handling. Do not quote an h=40 number without naming the method.**

| method | h=20 | h=40 | h=40 windows dropped |
|---|---|---|---|
| A, pool everything | +3.9 | +22.5 | 0% |
| B, drop flagged `adjustment_step` windows | — | +24.0 | 1.48% |
| C, drop windows touching 20 Dec - 10 Jan | +13.5 | +39.1 | ~20%, clustered Nov-Jan |

**C moves the point estimate by +17 bp — about 40% of its value and more than the year-to-year variation. B barely moves it.** B keys off the first plateau in the Yahoo/ours ratio, which for an annual dividend lands in early January, several sessions after the true ex-date, and knows only 315 codes. C is code-blind and full-history but trades a dividend confound for a seasonal one. Neither is a clean correction; neither is significant.

### 2.5 `persistence` — ordered in search, sign-reversed in confirm

The only LLM field that has ever ordered anything. Per-day paired differences; split frozen at `f9376cf`.

**Search, 105 judged days:** ordered at 5 of 6 horizons, spread widening monotonically with horizon, sign fraction climbing 0.53 -> 0.72. Every paired difference inside one standard error.

**Confirm, 20 judged days, touched once:**

| h | structural | multi_quarter | one_off | s-o | se |
|---|---|---|---|---|---|
| 1 | **-13.5** | -2.4 | -10.6 | -2.9 | 11.6 |
| 2 | **-14.0** | -1.2 | -12.9 | -1.1 | 21.1 |
| 5 | **-45.3** | -26.8 | -21.6 | -23.8 | 38.7 |
| 10 | **-72.3** | -21.8 | -30.3 | -42.0 | 87.5 |

**0 of 4 ordered. `structural - one_off` flips sign at 4 of 4, and `structural` goes from highest group to lowest at every horizon; at h=5 the full three-way ordering inverts exactly.** All three groups are negative in confirm — a different regime from search. h=20/40 have fewer than 5 usable confirm days.

**Not demonstrated. And 20 days cannot distinguish "hypothesis fails" from "one hostile regime" — which is what §4 is for.**

### 2.6 What is robust, and none of it is about the news

Daily, top-300, 2016-09..2026-09: **overnight +25.7 bp/day, intraday -28.6, sign stable 11 of 11 years on both legs.** Hourly over 713 days agrees: **+30.6 / -22.5**, and +27.2 on the adjustment-clean subset. **Cross-resolution agreement on an 11-year fact is the strongest evidence in this document.**

Movers vs calm, split on decision-time-clean prior-day absolute move: mover overnight **+33.9** vs calm +17.2; mover intraday **-35.6** vs calm -21.2. Signs stable 11/11 years, and the ordering reproduces at hourly resolution. **These are properties of the top-300 universe, not of the news.** Pool alpha by mover bucket is noise.

### 2.7 Regime, inferred causally

Hand-written EM plus forward filter. States K in {2,3,4} x 5 observation sets, all elements functions of bars at or before the close of d-1. BIC and out-of-sample predictive likelihood are monotone in K in every set, which is a mixture artifact contradicted by visit counts (one K=4 state has 7 visits), so all three were carried and the shuffle adjudicated.

**Filtered states only.** Parameters refit on an expanding window every 63 days, so the model at time t saw only data before t. **Label agreement filtered vs smoothed is 0.721** — 28% of days change label once look-ahead is removed. Smoothed (+21.68) and full-fit-filtered (+22.82) beat the admissible expanding-filtered (+18.23) and are reported only as the non-admissible upper bound.

**Overnight state ordering holds 6 of 7 years, including both held-out years**; s0 is the top overnight state 7/7. Intraday ordering 3/7 — no stable structure. **State visits behind the conditional net: 41 visits / 83 days in search, 34 visits / 98 days in confirm.** That is the binding sample, not the 2,388 days the parameters saw.

**Held out, scored once:** conditioned +6.86 vs unconditioned +4.24 bp/day. `bayes_judge` **adopt=false, p(mu>0)=0.009**; median delta exactly 0.00; days>=0 205/365 against 261 required; the top 10 days carry +7.42 of the +2.62 mean.

**Selection premium, N=15:** under a **21-day circular block shuffle** — the correct null, since it preserves the serial structure an HMM exists to exploit — the observed +18.23 sits at the **58th percentile of noise, p=0.424**. The iid permutation p=0.004 destroys autocorrelation and must not be quoted. Also, **14 of 15 configurations degenerate to "trade nothing"**, which scores +17.71 purely because the unconditioned strategy loses that much; the winner beats not trading by **+0.52 bp/day**.

**A causal regime state is estimable and its overnight ordering survives out of sample. Conditioning on it clears nothing.**

---

## 3. What was NOT tested

1. **The v2 judgment fields on unseen data.** Both holdouts carry no judgments, so they establish that **price and timing features** carry no alpha in an unseen window. They say nothing about the LLM fields, which remain tested only on the 125 judged days. This asymmetry is what §4 would fix.
2. **Sub-hour behaviour before 2025-08.** Fill timing, queue position and the exit families at minute granularity are confined to the 13-month Kiwoom window — one regime, permanently.
3. **Order flow.** 21 collected days, but **acting-minute coverage is 4.7%** (~31 pairs/day): the collector subscribes reactively, after the decision minute has passed. One fold's worth is ~2.6 years at that rate. More decisively, **every collected day falls inside or after the frozen confirm window**, so it cannot be searched on without burning it. The fix is a collector change — subscribe to the full candidate list from 09:00 — after which ~6 months gives a searchable structure in a future frozen split. Accrual alone fixes neither problem.
4. **Delisted names at hourly resolution.** Absent from the Yahoo pull and unrecoverable. Measured at daily resolution, where the table retains them: removing delisted names moves h=40 by ~1 bp. The hourly numbers carry the unquantified version; its hole is 8.0% (2023) falling to 1.2% (2026).
5. **Anything at all before 2023-03-16**, where the news archive begins, and before 2023-09-19 at intraday resolution.

---

## 4. The purchase decision

The binding limit on Track B is no longer prices — it is that judgments exist for 125 days. `persistence` is a **daily-resolution** test and `daily_bars_adjusted` reaches 2016, so the news archive (2023-03-16) is the only constraint.

At the measured **$0.000734/pair** (1,127 in / 68 out tokens) and the adopted prefilter's 84.3% keep rate:

| year | days | pairs/day | kept pairs | cost |
|---|---|---|---|---|
| 2023 (from 03-16) | 195 | 436 | 71,631 | **$53** |
| 2024 | 244 | 469 | 96,544 | **$71** |
| 2025 | 242 | 686 | 140,033 | **$103** |
| 2026 (remaining) | 48 | 992 | 40,134 | **$29** |
| **all** | **729** | — | **413,127** | **$256** |

**$256 takes `persistence` from one search window and one 20-day confirm to four independent years.**

### This is a discriminating test, not a search for a positive

`persistence` reversed sign at 4 of 4 confirm horizons on 20 days with all three groups negative. That is either the hypothesis failing or one hostile regime, and 20 days cannot tell them apart. The decision rule, registered **before** any spend:

- **Kills the hypothesis:** `sign(structural - one_off)` holds in **2 or fewer of 4 years** at h=5 and h=10, each year scored on its own frozen confirm sub-window.
- **Keeps it alive:** sign consistent in **3 or more of 4 years** at both h=5 and h=10, **and** the effect exceeds the shuffled-label selection premium computed as in §2.2.
- Ordering counts in search windows count toward neither. Only frozen confirm sub-windows do.

### Dividend method, declared before buying

Primary is **method B** — drop windows spanning a flagged `adjustment_step` break — because it keys to actual recorded breaks. **C is a robustness check only.** The primary claim is restricted to **h <= 10**, where the two are indistinguishable (B touches 0.06% of windows at h=1 and 0.21% at h=5). At h >= 20 the dividend choice dominates the estimate, and no `persistence` claim should rest there.

### What it does not buy

Not a change in the arithmetic. Selection alpha is a factor of twenty below the toll, and $256 does not move that. It buys a verdict on the one field that has ever ordered anything, in place of a coin flip.

---

## 5. Defects found

1. **The literal hourly transplant of SimBroker's quote rule fabricates +8.1 bp of D2 at 5.7 se.** "This bar's open, else the last completed close" prices an intraday article at a bar opening up to 59 minutes **before** the news. At minute resolution the same mechanism is bounded by ~30 seconds and is worth ~0.1 bp; at hourly it manufactures a highly significant edge from nothing. **Any coarse-resolution study here must use a strictly executable entry.** Entry is now the open of the first bar at or after `decide_at`.

2. **Float-vs-Decimal in the exit twin.** The live predicate holds prices and thresholds as `Decimal`, so 345,000 x 1.10 is exactly 379,500 and a bar closing there is a hit; float64 gives 379,500.00000000006, misses the tie, and exits a minute late. KRW prices are integers on a tick grid, so the tie is common. Affects 1.08 and 1.10 only: **28 of 29,800 holdout paths (0.09%), 6 of 4,416 in sample (0.14%)**, all on the far-target arms. **The published run passed its verification gate only by sampling luck** — seed 0's 400-path draw contained none of the 6. Fixed in both float implementations (`exit_fast._fires` and `Stack.rewards`) plus `peak x (1 - trail_gap)`; both books now verify at 0 mismatches. **Re-running 2026-04..08 reproduces the published verdict digit for digit**, so the defect was real and immaterial to the conclusion.

3. **The reconstructed 15:00 hourly bar is on a different price scale** from the Yahoo bars around it — its close comes from the non-dividend-adjusted daily table. Mean `close(15)/close(14) - 1` is **+968 bp**; 104 codes are wrong every day. A first pass that used it reported the top-300 earning +208 bp/day, entirely artifact. The exit is now the 14:00 bar's close, one price scale end to end. **This is a defect in the new hourly dataset and belongs back with whoever built it.**

4. **`replay_baseline`'s 23.0 bp cost literal** — sell tax plus one side of commission, spread and slippage zero — underprices every exit-lab result by ~15 bp against the measured 38.3.

Two smaller ones: **`MIN_EDGES = (0.005, 0.002, 0.0)`** are thresholds on expected net return sitting ~130 bp above a pool mean of -38 bp, which manufactures abstention; and **a sample conditioned on its own outcome does not produce a fake mean here, it produces fake precision** — the calm bucket's standard error collapsed from 45 bp to 14 bp at h=40 while the means stayed put. Check any future result whose se looks too tight for its n.

---

## 6. Does anything clear the toll?

No, and the gap did not narrow when the never-fitted sample grew four-fold, when every field was used, when interactions were allowed, or when a latent regime state was added. The one candidate that beat its shuffle premium did not reproduce out of sample; the one that looked strongest in sample was **below the p90 of its own noise band**.

What would change it is not more effort on this pool:

- **$256 of judging** makes 2023-2025 testable and gives `persistence` a verdict (§4).
- **A collector change** — subscribe to the full candidate list at 09:00 rather than reactively — makes order flow searchable in about six months.
- **Accruing forward is the only way to extend minute resolution at all**, since Kiwoom's 13 months is a hard floor and Yahoo's 730-day window recedes daily. That is an argument for continuing to collect regardless of whether anything is trading.
