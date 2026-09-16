# Is there an edge, and where does it come from? (2026-09-16)

Everything ran on simnode in `~/git/daytrade-it-edge` (branch `feat/edge-hunt`), `data` symlinked to `~/git/daytrade-it/data`.
- The DB was read-only throughout (`scripts/eval/edge_db.py` opens `set_session(readonly=True)`).
- No Anthropic calls, no Toss calls, no Kiwoom calls. Nothing was written outside `data/eval/edge/`.
- Nothing in `~/git/daytrade-it`, the live daemon or the running backfill was touched.

**Answer, up front.** There is no demonstrated edge in the selection. Against a null matched on the day, the acting minute, the volatility bucket and the liquidity bucket, the whole judged pool is worth **+1.6 bp per pair (se 3.4) over 125 days** — zero. The bar it has to clear is **about 36.7 bp**, not the 21.5 bp every table in this repo has been netting out (§2.5): the research constant omits spread and slippage entirely and counts commission once instead of twice. The dashboard pool's celebrated +18 bp is two-thirds calendar and one-third reconstructed data; on the only two days that are genuine live archive it is −33.4 bp, and **−48.6 bp** once costed properly. One conditioning rule survives the search, it is an exclusion rather than a pick, and it then fails `bayes_judge` at p(μ>0)=0.8635.

The single most useful number in this document is not about selection at all. Buying any top-300 name at the 09:00 open and holding it to the close lost **56.8 bp** per name per day, averaged over 125 days. 42% of our pairs act at that minute. That headwind, not our stock picking, is what the system has been fighting.

---

## 1. The instrument, and why it should be believed

Every previous comparison in this repo has compared our pool to *something else measured differently*. This one does not.

`scripts/eval/edge_null.py` takes every acting minute that appears in a pool and prices **the entire top-300 cross-section** at that minute with `prompt_labels.py`'s own rules — `select_universe` on the previous day's `daily_bars_adjusted.trade_value` for the universe, SimBroker's quote rule for the entry (that minute's bar open, else the last completed regular close, with the pre-09:00 floor), and the day's last regular bar for the exit. Not a sample: the full cross-section, which is cheap because `ret_close` is just `day's final close / entry(minute) - 1` and only the denominator moves with the minute.

Two calibrations before any result was read:

| check | result |
|---|---|
| `edge_verify.py`: re-price the pool's own pairs through the null's machinery and compare to `labels.jsonl` (3,054 pairs, 2026-03-17 / 05-20 / 07-15 / 09-10) | entry mismatches **0**, `ret_close` mismatches **0**, worst absolute error **0.00e+00** |
| `edge_dash.py`: re-label the dashboard pool from `data/eval/replay_events/` with the same rules | **594 / 594** labelled, and the ladder reproduces the published −52 bp / +18 bp / +39 bp exactly |

(The first pass of `edge_null.py` had a real bug — the backward scan for the exit bar kept the earliest close instead of the latest — and it was `edge_verify.py` that caught it, at 2,912 mismatches out of 3,054. The verification is load-bearing, not ceremonial.)

Inference is clustered on the **day**, because the day is the unit that repeats. CIs are a bootstrap over days. Where a table shows 2 or 7 days, the interval is decoration and the point estimate is the content; that is called out each time it happens.

---

## 2. The ladder

Four rungs, each a mean over pairs, gross, hold to close. `net` subtracts the 21.5 bp round trip. Every delta is formed **pair by pair against that pair's own minute**, so the day and the minute are differenced out exactly and costs cancel.

```
R0 index       069500 held from the decision minute to the close
R1 universe    the equal-weight top-300 cross-section at the same minute
R2 pool        the news-tagged pairs we actually judged
R3 buy         the subset with score > 0.15

D1 = R1 - R0   universe vs index       cross-sectional drift
D2 = R2 - R1   news tag vs no news     is the NEWS the edge
D3 = R3 - R2   our score within the tag is our CHOOSING the edge
```

All figures in bp, per pair.

| | sweep 125d | sweep, the 7 dash days | dashboard 7d | dashboard, **archive 2d** | dashboard, reconstructed 5d |
|---|---:|---:|---:|---:|---:|
| n pairs | 131,820 | 4,547 | 594 | 271 | 323 |
| R0 index | −18.9 | +13.1 | −3.3 | **−72.9** | +55.1 |
| R1 universe | −38.5 | +18.7 | +11.8 | −22.2 | +40.3 |
| R2 pool gross | −30.4 | +15.1 | +39.3 | −11.9 | +82.1 |
| **R2 pool net** | **−51.9** | **−6.4** | **+17.8** | **−33.4** | **+60.6** |
| R3 buy net | — | — | +39.4 | **+1.7** | +75.1 |
| D1 universe − index | −19.5 | +5.6 | +15.1 | +50.7 | −14.8 |
| **D2 pool − universe** | **+8.1** | −3.7 | +27.5 | **+10.4** | +41.9 |
| D3 buy − pool | — | — | +24.5 | +34.9 | +14.7 |

The sweep pool's headline decomposes exactly:

```
-18.9  index intraday drift, decision minute to close
-19.5  equal-weight universe underperforms the index
 +8.1  the news tag, over the unmatched universe null
-21.5  cost
------
-51.8  = the published -52 bp
```

### 2.1 Most of the dashboard pool's advantage is the calendar

Run the same ladder on the **sweep** pool restricted to the seven dashboard days. The universe rung there is **+18.7 bp**, against −38.5 over the full 125 days: those seven days were a 57 bp favourable window for everything in the universe. Sweep net over the same window is **−6.4 bp**, not −51.9.

So of the ~70 bp gap between the two pools, **~45 bp is calendar** — it has nothing to do with the dashboard, the news, or our score.

### 2.2 The remaining ~24 bp lives entirely in reconstructed data

`data/eval/replay_events/meta.json` says 09-14 and 09-15 are genuine live archive and 09-07..09-11 are reconstructed from kr_quant articles placed on the daemon's poll grid. Split on that line and the pool splits with it:

| day | source | pairs | net bp | universe null | index | D2 | buys | buy net |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-09-07 | reconstructed | 80 | +74.9 | +56.1 | +101.4 | +40.3 | 35 | **+238.0** |
| 2026-09-08 | reconstructed | 54 | −45.0 | −129.8 | −150.0 | +106.2 | 28 | −29.8 |
| 2026-09-09 | reconstructed | 64 | +99.1 | +68.6 | +62.4 | +52.0 | 33 | +67.7 |
| 2026-09-10 | reconstructed | 39 | +95.0 | +118.2 | +83.8 | −1.6 | 25 | +104.2 |
| 2026-09-11 | reconstructed | 86 | +69.5 | +75.9 | +122.3 | +15.2 | 33 | −23.4 |
| 2026-09-14 | **archive** | 145 | **−30.6** | −19.6 | −52.2 | +10.5 | 82 | +0.5 |
| 2026-09-15 | **archive** | 126 | **−36.5** | −25.3 | −96.8 | +10.3 | 64 | +3.4 |

Reconstructed: +60.6 bp net, D2 +41.9. Archive: **−33.4 bp net, D2 +10.4**. The archive D2 sits right on the +8.1 bp the sweep pool shows over 125 days, which is what you would expect if the archive days are honest measurements of the same thing and the reconstructed ones are not.

Two circumstantial notes on why the reconstructed days may be contaminated, neither of them proven here:
- The reconstructed days carry **39–86 pairs each** against 126–145 on the archive days. Per-publication-day dashboard collection counts were 1, 0, 2, 29, 29, 45 and only then 129–188 — early collection was partial. A partial capture of a *dashboard* plausibly retains the items that stayed on it, and items stay on a dashboard when the stock is moving.
- 37% of reconstructed pairs act at the 09:00 open against 14% of archive pairs, and the reconstructed publish→decide distribution has a fat tail (p90 916 minutes vs 47.6). The null differencing removes the minute, so this does not explain D2, but it shows the two groups are not the same object.

**Conclusion for question 1.** The dashboard pool's positive drift is (a) about 45 bp of favourable calendar in a seven-day window, (b) about 24 bp that exists only in the five reconstructed days, and (c) on the two real days, nothing — the pool loses 33 bp per pair net and the BUY subset earns +1.7 bp, which is zero. "Three independent systems agree in sign" should be restated as "three independent systems read the same five reconstructed days."

### 2.3 And the news tag itself is worth nothing once the names are matched

The +8.1 bp in D2 is measured against the *whole* top-300. News-tagged names are the big, volatile, heavily traded ones, and on an up day a high-beta name beats the cross-section for reasons that are not news.

`scripts/eval/edge_strata.py` buckets each day's universe 4×4 on two values known before the open — the population sd of the trailing 20 close-to-close returns ending at the **previous** session, and the previous day's traded value — and computes the cross-section inside each cell separately. Comparing each pair only to its own cell:

**Pool alpha over 125 days and 132,077 pairs: +1.6 bp, se 3.4, positive on 69 of 125 days.**

The +8.1 bp was composition. Properly matched, the news tag buys nothing.

(On the dashboard's seven days, stratum matching moves D2 only from +27.5 to +23.8 — the dashboard's residual is not a volatility-bucket artifact. It is a reconstructed-data artifact, per §2.2.)

### 2.4 The founding premise does not replicate

`docs/research/2026-09-13-news-prompt-eval.md` opens with "기준선이 0이 아니다": the average excess of every Toss-tagged article was **+1.27%**, i.e. news-tagged stocks are strong that day, and the prompt has to beat that baseline. That number is the reason this system trades news at all.

It was measured on **209 labelled pairs over 10 days**, with `excess` defined as the return minus that day's median open-to-close across all codes — structurally the same quantity as `D2` here. Its own caveats are the relevant ones: 137 of the 353 intraday pairs were dropped for lack of tick data, and the doc says the surviving intraday sample is skewed toward scalp-it's subscribed names, which are surging stocks.

The same quantity on 132,185 pairs over 125 days:

| | 09-13 eval | this measurement |
|---|---:|---:|
| pairs / days | 209 / 10 | 132,185 / 125 |
| tagged excess over a cross-sectional centre | **+127 bp** | **−11.5 bp** (vs 069500) |
| tagged excess over a day × minute × volatility × liquidity matched null | — | **+1.6 bp** (se 3.4) |

At 632× the sample the baseline is not +1.27%, and it is not positive. The most likely explanation is the one the original doc already names — the intraday half that survived the tick-data filter was a sample of stocks that were already moving — but whatever the cause, the premise that news-tagged names carry a large same-day drift does not hold on the full pool.

---

## 2.5 The cost floor, which is 15 bp higher than every table in this repo assumes

If the whole question is a few bp of signal against a round trip, the round trip has to be right. `COST = 0.00215` — used by `prompt_ceiling.py`, `prompt_oracle_mine.py`, `prompt_dashboard_fanout.py` and every table above — is not the cost this system pays. It is the 0.20% sell tax plus **one side** of commission, with **no spread and no slippage at all**.

The broker's own model (`krx_quant_core.costs.model`, which `daytrade_it` configures and `PaperBroker` charges) says:

| | tax (sell only) | commission (both sides) | slippage (both sides) | **round trip** |
|---|---:|---:|---:|---:|
| 거래소 (81.7% of entries) | 20.0 | 3.0 | 10.0 | **33.0** |
| 코스닥 (18.3%) | 20.0 | 3.0 | 30.0 | **53.0** |
| **pool-weighted** | | | | **36.7** |
| research constant in use | 20.0 | 1.5 | 0.0 | 21.5 |
| **understated by** | | | | **15.2** |

The slippage line is a configuration choice, so it is worth a second, independent route that no configuration can argue away. A KRX quote moves on a fixed price grid, and crossing a one-tick spread costs exactly one tick per round trip — buy the ask, sell the bid. Measured on the 132,185 actual entry prices:

| market | n | median | mean | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| 거래소 | 107,967 | 11.5 | 12.4 | 16.4 | 20.1 |
| 코스닥 | 24,218 | 12.2 | 12.8 | 16.8 | 19.6 |
| **pooled** | **132,185** | **11.7** | **12.5** | 16.5 | 20.0 |

tax 20.0 + commission 3.0 + **one tick of spread 12.5 = 35.5 bp**, against the model's 36.7. Two independent routes, 1.2 bp apart. And a one-tick spread is the *floor*: it assumes we never cross a wider book and never move the price, on 100,000 KRW orders placed at the open when the book is thinnest.

**The bar is therefore about 36 bp of selection alpha. The measured selection alpha on this pool is +1.6 bp (se 3.4).** The required signal is roughly twenty times what exists.

Every net figure in this document and in the prompt-v3 and dashboard work is **15.2 bp optimistic**. Restated at the true cost:

| | at COST = 21.5 | **at the real ~36.7** |
|---|---:|---:|
| sweep pool, 125 days | −51.9 | **−67.1** |
| dashboard pool, 7 days | +17.8 | **+2.6** |
| dashboard BUY, 7 days | +39.4 | **+24.2** |
| dashboard, archive 2 days | −33.4 | **−48.6** |
| dashboard BUY, archive 2 days | +1.7 | **−13.5** |
| pre-move gate, kept half | −43.8 | **−59.0** |

The dashboard pool's headline "+18 bp" is **+2.6 bp** once the trade is costed the way the broker costs it, and the BUY subset on the two genuine archive days goes from +1.7 to **−13.5**. Nothing in this investigation survives the correction, but several things that looked marginal become clearly negative, and the pre-move gate's bayes_judge failure gets worse rather than better.

I have not re-run the tables at 36.7 bp, because the deltas are unaffected — cost cancels in every `alpha` and every D-row, which is exactly why the decomposition was built on differences. Only the `net` columns move, and they move by a constant.

---

## 3. The structural headwind

`alpha` is flat, so everything the pool earns or loses is the null. The null, by acting minute, over 125 days:

| acting minute | pool pairs | universe | index | universe − index |
|---|---:|---:|---:|---:|
| **09:00 open** | **55,350** | **−56.8** | −24.7 | −32.1 |
| 09:01–09:29 | 16,580 | −30.4 | −21.6 | −8.8 |
| 09:30–09:59 | 14,105 | −34.3 | −17.3 | −17.0 |
| 10:00–10:59 | 19,449 | −27.0 | −18.6 | −8.4 |
| 11:00–12:59 | 12,287 | −6.3 | +0.3 | −6.6 |
| 13:00+ | 14,414 | **+5.6** | +4.5 | +1.1 |

Monotone, and large. A long-only intraday position opened at the Korean open and held to the close carried a 56.8 bp headwind over this sample, before costs and before any stock is chosen; by the afternoon the headwind is gone. 42% of our pairs act at that worst minute — which is not the same as the 61% whose *article* is pre-market, since a judge backlog can push a pre-market article's decision well into the session.

This is also the mechanical explanation for the `after_10am` result that was correctly retracted as position shrinkage: later entries are not better selections, they are entries into a smaller headwind.

### 3.1 The 09:00 batch effect is calendar, not selection

`rl-builder` found the 09:00–09:01 batch worth +166 bp on the dashboard pool and everything after 10:00 negative. The dashboard pool has 78 pairs at that minute; the sweep pool has 55,308 over 125 days. Ran once, here, as agreed.

| acting minute | n | alpha | net | days ≥ 0 | months |
|---|---:|---:|---:|---:|---|
| 10:00–10:59 | 19,438 | +7.9 | −35.5 | 55/125 | `+-++-++` |
| 09:30–09:59 | 14,091 | +5.2 | −42.1 | 48/125 | `++++--+` |
| 11:00–12:59 | 12,279 | +2.4 | −24.1 | 51/125 | `+--++-+` |
| 13:00+ | 14,408 | −0.5 | −16.4 | 42/125 | `+-+-+-+` |
| 09:01–09:29 | 16,553 | −3.0 | −43.1 | 54/125 | `+-+---+` |
| **09:00 open** | **55,308** | **−4.8** | **−82.6** | 48/125 | `+++---+` |

On 125 days the open batch is the **worst** bucket on net and the worst on alpha. It is not +166 bp; the sign is reversed.

It is also **not** the reconstructed-day artifact I expected. Splitting the dashboard's own 09:00 batch by provenance gives archive +134.8 bp net (n=13) and reconstructed +128.3 (n=65) — the effect is present in both, so the archive/reconstructed split does not explain it away. What explains it is the window. In the sweep pool, the 09:00 batch on those same five reconstructed days ran +45.2, −215.6, +67.3, +35.7, +114.3 bp per day; September as a whole is −22.8 bp net at the open, and the monthly series is −7.2, −19.7, −108.1, −148.6, −154.0, +7.0, −22.8. The 09-07..09-11 open was an unusually good five-day window in **both** pools.

So the finding is real for those days and does not generalise: 78 pairs on 7 days against 55,308 on 125, pointing opposite ways. `minutes_since_open` having sign stability in the bandit work is consistent with §3 — it is tracking the headwind curve, not a selection signal.

I am flagging this to `rl-builder` rather than acting on it, since buy timing is that lane. The point here is a decomposition, not a proposed feature: **the −52 bp is not bad stock picking. It is a falling universe, a falling time of day, and a 21.5 bp toll.**

---

## 4. Does any conditioning produce an edge?

Twenty-two axes over 132,077 pairs and 125 days, dependent variable `alpha` (day, minute, volatility and liquidity already differenced out), with monthly sign strings printed alongside every level — `scripts/eval/edge_cond.py`, full output reproducible with no arguments.

### 4.1 The honest out-of-sample test

`scripts/eval/edge_folds.py`. For each month from the second, rank every (axis, level) with ≥500 pairs and ≥20 days on the months **strictly before** it, take the best, apply it to that month alone. No fold shares a day with its own evaluation and no fold sees the future; this is not LODO.

| | n | alpha | net | days ≥ 0 | worst day | losers |
|---|---:|---:|---:|---:|---:|---:|
| BASE, all pairs | 132,077 | +1.6 (se 3.4) | −51.4 | 46/125 | −580.4 | 0.595 |
| OOS best-1 gate | 5,092 | **−16.7** (se 14.2) | **−69.1** | 35/91 | −802.8 | 0.640 |
| OOS best-3 union | 13,871 | +6.2 (se 10.3) | **−69.6** | 33/92 | −781.2 | 0.613 |

Fold by fold: 건설 (train +54.0) → test −23.6; 전기/전자 (+32.1) → −17.7; 보험 (+39.1) → +26.3; 의료/정밀기기 (+46.8) → −72.6; 보험 (+29.0) → +6.8. Two of five.

Both searched rules are **worse than doing nothing** on net, on worst day and on loser ratio. A search over a few hundred levels on a pool with no edge returns exactly this.

### 4.2 Market regime does not change selection, and no regime is worth trading

`scripts/eval/edge_regime.py`, six pre-open day features (previous session's universe return, 5- and 20-session returns, position against the 20-session mean, trailing 20-session volatility, previous day's breadth), each split into terciles, all built only from sessions strictly before the day being scored. 069500 is not in `daily_bars_adjusted`, so the market here is the universe's own chained equal-weight top-300 index.

**Every one of the 18 buckets loses money.** Best is `idx_5d` high at −27.6 bp and `idx_vol20` high at −27.4; worst is `idx_vol20` mid at −87.3. And in all 18, `net − null` sits between −19 and −26 bp, i.e. exactly the cost drag: regime moves `net` and `null` together and leaves `alpha` untouched. Whatever regime predicts, it is market timing with no news content, and it does not reach a level where a day is worth trading.

### 4.3 The LLM judgment fields, against a matched null

For completeness, since this is the third independent route to the same place. Alpha in bp, with monthly signs:

| field | best level | worst level | spread |
|---|---|---|---|
| `score` | 0–0.15 +7.5 | ≥0.5 **−2.2** (`++-----`) | non-monotone again |
| `subject` | main +2.4 | partial +1.0 | 1.4 |
| `material_type` | 실적 +15.5 | 루머 −29.1 (n=375) | — |
| `persistence` | multi_quarter +2.1 | one_off +0.9 | 1.2 |
| `freshness` | follow_up +4.4 | repeat +0.3 | 4.1 |
| `scale_vs_size` | small +3.0 | large −7.9 (`+-----+`) | 10.9 |
| `reports_price_move` | False +2.4 | True −8.6 | 11.0 |
| `sentiment_direction` | 0 +2.3 | −1 −0.9 | 3.2 |
| `surprise` | 0 +2.7 | 2 −15.7 (n=654) | — |
| `specificity` | 1 +5.9 | 3 −2.9 | 8.8 |

Standard errors on these are 3–8 bp. The score is non-monotone against alpha exactly as it was against raw return, with the ≥0.5 bucket the worst and negative in 5 of 7 months. Nothing here is separable from zero and nothing is remotely near 21.5 bp.

### 4.4 The one survivor

The pre-entry move — `entry / previous close - 1`, a pure decision-time input — is the only axis in the scan that is **monotone across all its buckets**, and it is the exclusion side that is stable.

| pre-entry move | n | alpha | net | months |
|---|---:|---:|---:|---|
| −1 .. −0.3% | 9,587 | **+16.0** | −43.2 | `+++++-` |
| < −3% | 16,277 | +12.2 | **−21.1** | `+++++-` |
| −3 .. −1% | 16,965 | +8.1 | −45.0 | `++--+-` |
| +0.3 .. +1% | 11,294 | +0.4 | −66.3 | `-+-++-` |
| flat (±0.3%) | 11,440 | −9.1 | −86.4 | `----+-` |
| **+1 .. +3%** | 21,990 | **−16.8** | −94.9 | **`------`** |
| **> +3%** | 23,482 | **−21.5** | −104.4 | **`------`** |

A monotone response over an ordered variable is a far stronger prior than "this level scored highest", so `scripts/eval/edge_gate.py` evaluates cut points stated in advance rather than searched for. Of the 132,077 pairs, 111,035 have a known previous close (`data/eval/prev_close/*.json` stops at 2026-08-14, so 24 days have none), covering 2026-03..08:

| gate | n kept | alpha | net | days ≥ 0 | worst day | losers | months |
|---|---:|---:|---:|---:|---:|---:|---|
| ALL | 111,035 | +3.0 (se 3.8) | −61.3 | 33/101 | −593.2 | 0.604 | `+++++-` |
| keep ≤ **+3%** (live's current cap) | 87,553 | +3.2 | −53.9 | 29/101 | −563.9 | 0.606 | `+++++-` |
| ↳ dropped | 23,482 | −21.5 | −104.4 | 27/101 | −849.0 | 0.596 | `------` |
| keep ≤ **+1%** | 65,563 | **+7.5** (se 4.2) | **−43.8** | 36/101 | −652.1 | 0.600 | `+++++-` |
| ↳ dropped | 45,472 | **−18.4** | −101.6 | 26/101 | −729.3 | 0.611 | **`------`** |
| keep ≤ +0.3% | 54,269 | +6.4 | −42.9 | 34/101 | −693.5 | 0.596 | `+-+++-` |
| keep ≤ 0% | 49,450 | +7.1 | −39.9 | 37/101 | −712.1 | 0.592 | `++-++-` |

Frozen transfer to the dashboard pool, which none of this was fitted on (`--rows rows_dash.jsonl`):

| | n | alpha | net | losers |
|---|---:|---:|---:|---:|
| dashboard 7d, all | 579 | +32.3 (se 18.7) | +34.9 | 0.603 |
| dashboard 7d, keep ≤ +1% | 316 | +46.8 | +56.6 | 0.557 |
| dashboard **archive 2d**, all | 271 | +19.0 | −33.6 | 0.716 |
| dashboard **archive 2d**, keep ≤ +1% | 148 | +19.2 | −20.9 | 0.662 |
| dashboard **archive 2d**, keep ≤ +3% | 178 | +30.8 | −8.7 | 0.680 |

Sign is consistent everywhere and the loser ratio improves everywhere. The seven-day standard errors are 17–27 bp, so this transfer is corroboration, not evidence.

**What this is and is not.** It is a real, sign-stable, free exclusion: the half of the pool that has already run up more than 1% is worth −18.4 bp of alpha and is negative in **6 of 6 months**. Tightening the live `MAX_PRE_ENTRY_MOVE` from 3% to 1% improves net by 17.5 bp per pair and the loser ratio slightly, and it replaces a value that was chosen by leave-one-day-out on six days — the methodology this session has banned — with one chosen on 101 days under monthly time-ordered sign stability.

It is **not** an edge. The kept half still loses 43.8 bp per pair. Alpha at +7.5 ± 4.2 bp is a third of the round trip. This is harm reduction, and it should be adopted as harm reduction or not at all.

#### The adoption verdict: no

Pair-level bp is not what the adoption rule asks about, so `scripts/eval/edge_budget.py` spends both gates through the live capital budget — 100,000 KRW flat notional on at most 5 BUY signals (`score > 0.15`) per day, selected in the live order (earliest acting minute, higher score breaking ties, which is the ordering already known to be 35 bp/trade worse than random and is used here only because candidate and baseline must differ *solely* in the gate). It is a budget model, not the replay engine: no fill model, no VWAP exit, no add-on, no contention, and it should not be read against replay KRW figures.

101 common days. The 5/day cap binds, so both gates take exactly **505 trades** — the gate changes *which* five, not how many.

| | net KRW | trades | losers |
|---|---:|---:|---:|
| candidate, cap +1% | **−592,680** | 505 | 322 |
| baseline, cap +3% (live) | −741,227 | 505 | 330 |

`scripts/eval/bayes_judge.py`:

```
p_mu_gt_0             0.8635      <- the only failing criterion (needs >= 0.90)
mu_mean               +623 KRW/day
days_ge_0             73 / 101    (floor 73)              pass
worst_day_candidate   -60,955  =  worst_day_baseline      pass
loser_ratio           0.6376   <=  0.6535                 pass
adopt                 false
reasons               ["p_mu_gt_0=0.8635 < 0.90"]
```

**Three of four criteria pass and the fourth misses at 0.8635.** So the verdict on the only survivor of the whole search is: **not adopted**, narrowly, on 101 days. It improves the worst day not at all and the loser ratio by 1.6 points. If it is taken up anyway it should be as a config change justified by the 6/6-month sign stability of the excluded half, with the failed adoption test stated alongside — not as an edge.

There is a second, smaller directional finding inside the same table that the current one-sided cap ignores: the **down** buckets are the good ones. `< −3%` has the best net of any bucket (−21.1) and positive alpha in 5 of 6 months. The live filter only refuses upward extension; a version that also *prefers* weakness is untested and is the natural next hypothesis, stated here so that it is pre-registered rather than mined later.

### 4.5 Ranked list of everything tested

| rank | conditioning | verdict |
|---|---|---|
| 1 | **pre-entry move ≤ +1%** | survives the scan — monotone, exclusion side negative 6/6 months, transfers in sign to the dashboard — and then **fails `bayes_judge` at p(μ>0)=0.8635**. Harm reduction of 17.5 bp per pair, not an edge, not adopted. |
| 2 | publisher / wire (`특징` −31.1 alpha 0/7 months, `윤서` −48.8 1/7) | exclusion-only, n = 1,290–1,529, no positive counterpart. Worth a prefilter line, not a strategy. |
| 3 | title tag (`오늘의 IR` −73.6, `오늘의 주요공시` −36.2, `표` +22.3) | the big positive (`표`) is 3/7 months and is the same anti-correlated tail the oracle mining already warned about. Exclusions only. |
| — | sector | **fails** — best training sectors transfer 2/5 folds; `건설` +30.2 and `보험` +27.1 have se 19 and 16. |
| — | market regime (6 features × terciles) | **fails** — all 18 buckets lose, and none moves alpha. |
| — | volatility quartile, liquidity quartile | **fails** — spread ±14 bp, no monotonicity, s3 is `+++++--`, s2 is `+--+--+`. |
| — | fan-out | **fails** as a positive; `41+` at −31.4 (n=3,196) is a weak exclusion already covered by the live prefilter. |
| — | session, weekday, market (거래소/코스닥) | **fails** — all within ±10 bp, no month stability. |
| — | every v2 LLM field | **fails** — all within ±8 bp against the matched null; `score` non-monotone. |
| — | greedy best gate / best-3 union from history | **fails badly** — worse than the base pool on net, worst day and loser ratio. |

---

## 5. What is not covered here

- **Volume, turnover and buy timing** are `rl-builder`'s lane and were deliberately left alone. `minute_bars.trade_value` still has never been used at decision time; §3 says why that lane is the interesting one.
- **Exits** are `rl-builder`'s; everything above is hold-to-close. `ret_1520` is in the data and `edge_null.py` emits the matching null (`null_mean_1520`, `mkt_ret_1520`), but no exit is decomposed here.
- **The learning curve** is `learning-curve`'s.
- `pre_move` on September rests on the previous session's *adjusted* close rather than the frozen vintage, because `data/eval/prev_close/*.json` ends 2026-08-14. Only the adjustment factor is a later vintage; over a seven-day window it moves nothing unless a name splits inside it. The 125-day tables use the frozen files and therefore cover 101 days.
- September labels still sit on an incomplete minute-bar backfill (13.1% of September pairs had no bars when `labels.jsonl` was built). Every September figure here should be re-read after the backfill lands.

---

## 6. If nothing has an edge, what would give one

Ordered by how much the answer would move per unit of effort.

**1. Real archive days. This is the binding constraint and it is nearly free.**
Every claim that the live pool differs from the sweep pool rests on **two days**. They accrue at one per trading day and cost nothing but leaving the collector running. Twenty real archive days would settle §2 in a month, and until then the "live pool makes money" premise — which is the premise under which live trading was paused rather than stopped — is unsupported. Nothing else in this list is worth doing before this.

**2. Do not buy more Haiku labelling to answer *this* question.**
The directive says to buy labels rather than accept a thin sample, and that is right in general, but it does not apply to the entry-selection question. The 132,185 already-judged pairs put the whole LLM field block at ±8 bp against a matched null with `score` non-monotone, and the character-n-gram bound on the raw text caps any possible prompt at 1.9% of the oracle gap. Buying labels buys more of a quantity that has now been measured at zero three independent ways, and a fourth measurement against a properly matched null. Labels **are** the binding constraint for the learning curve and for a bandit with enough days; they are not the binding constraint for "does the news tell us which stock to buy".

**3. Attack the ~36.7 bp, because the signal is not going to grow to meet it.**
Per §2.5 the bar is about 36.7 bp, not 21.5, and the honest central estimate of our selection alpha is +1.6 ± 3.4 — a factor of roughly twenty. The entire cross-sectional spread of conditional means found anywhere in §4 is about ±30 bp, so even a perfect, costless selector over the free attribute set would not clear the floor. No feature search can close a gap this size; only the cost term can move.

Where it can move, in order of size:
- **Slippage, 10–30 bp.** The largest single controllable line, and the one the research constant zeroed. We enter at the 09:00 open with market orders — the thinnest book of the day — and 18.3% of entries are KOSDAQ at triple the KOSPI assumption. Limit orders, or simply not entering in the opening minute, act directly on this. This is also where §3 and this section point at the same change for different reasons.
- **The 0.20% sell tax, 20 bp.** Irreducible within a same-day round trip and paid in full on every entry. It amortises only over a longer holding period, which is a different strategy, but it is the one place the arithmetic genuinely changes.
- **KOSDAQ exposure.** 53.0 bp round trip against 33.0 for 거래소. A market filter is worth 20 bp on 18.3% of entries — about 3.7 bp pool-wide — which is small, but it is larger than any selection feature found here and it costs nothing to implement.

The uncomfortable implication is that this is not a modelling problem. At a 36.7 bp floor, same-day long-only news trading on KRX top-300 needs an entry signal stronger than anything in the literature on daily-resolution news features, and the measurement says we have none at all.

**4. Tick and order-book coverage on the names we actually trade.**
21.4% of our entries have tick data. That is a collection-scope problem, not a modelling one. §3 says the money is in the minute, not the article, and the minute is where we have the least data. Extending tick collection to the traded names is the only proposal here that could plausibly produce 25 bp, and it should be scoped before any further feature search on daily-resolution attributes.

**5. A different universe, or a different side.**
The 56.8 bp open-to-close headwind is a property of long-only top-300 Korean equity at that minute. It is not noise and it is not going away. Any strategy that opens long positions at the open is starting 56.8 bp behind; that fact deserves to be confronted directly rather than optimised around.

---

## 7. Reproduction

From `~/git/daytrade-it-edge` on simnode, `data` symlinked to `~/git/daytrade-it/data`. Credentials come from `~/git/quant-airflow/.env` via `edge_db.py`; nothing is written into the repo.

```
uv run python scripts/eval/edge_verify.py                       # calibrate against labels.jsonl
uv run python scripts/eval/edge_null.py                         # -> null_sweep.jsonl
uv run python scripts/eval/edge_strata.py                       # -> null_sweep_strat.jsonl, strata_sweep.jsonl
uv run python scripts/eval/edge_dash.py                         # -> dash_labels.jsonl
uv run python scripts/eval/edge_null.py   --labels data/eval/edge/dash_labels.jsonl --out data/eval/edge/null_dash.jsonl
uv run python scripts/eval/edge_strata.py --labels data/eval/edge/dash_labels.jsonl \
    --out data/eval/edge/null_dash_strat.jsonl --strata-out data/eval/edge/strata_dash.jsonl

uv run python scripts/eval/edge_ladder.py --pool data/eval/prompt_v3/labels.jsonl \
    --null data/eval/edge/null_sweep.jsonl --label "SWEEP 125d"
uv run python scripts/eval/edge_ladder.py --pool data/eval/edge/dash_labels.jsonl \
    --null data/eval/edge/null_dash.jsonl --days 2026-09-14,2026-09-15 --label "DASHBOARD archive"

uv run python scripts/eval/edge_cond.py   --dump data/eval/edge/rows_sweep.jsonl
uv run python scripts/eval/edge_folds.py  --rows data/eval/edge/rows_sweep.jsonl
uv run python scripts/eval/edge_regime.py
uv run python scripts/eval/edge_gate.py   --rows data/eval/edge/rows_sweep.jsonl --label "SWEEP 125d"
uv run python scripts/eval/edge_budget.py --rows data/eval/edge/rows_sweep.jsonl
uv run python scripts/eval/bayes_judge.py --candidate data/eval/edge/budget_candidate.json \
                                          --baseline  data/eval/edge/budget_baseline.json
uv run python scripts/eval/edge_cost.py                         # the cost floor (no DB)
```

## 8. Coordination

- `rl-builder` confirmed the lane split and supplied two corrections used above: `minute_bars.trade_value` is exactly `close × volume` (a derived column, not exchange-reported 거래대금), and the `ci-local` pre-push hook leaves a worktree HEAD detached — `git rev-parse HEAD origin/<branch> | uniq -c` must print `2` on one line after every push. Their `dash_forward.py` independently found random tickers beating our own pairs (r_close +21 bp vs +11) on the September window, which is the same conclusion as §2.3 from a different direction.
- The 09:00 batch check in §3.1 was the one the team lead asked to be run once; it was run here and not duplicated.
- Exit knobs `stop_loss_pct`, `take_profit_pct`, `trail_after_gain`, `stop_orb`, `exit_rule_hit` are **not on main** — any result about them describes a research branch's configuration surface, not live.
- 2026-09-16 was deliberately **not** added as an archive day: the live scoring path failed 86 times between 09:07 and 14:58 (Anthropic credit exhaustion), leaving 20 judgments against 147 and 144 on the two real days. It is a degraded day and would contaminate exactly the comparison §2.2 rests on.
