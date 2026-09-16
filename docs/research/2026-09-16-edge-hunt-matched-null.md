# Is there an edge, and where does it come from? (2026-09-16)

Everything ran on simnode in `~/git/daytrade-it-edge` (branch `feat/edge-hunt`), `data` symlinked to `~/git/daytrade-it/data`.
- The DB was read-only throughout (`scripts/eval/edge_db.py` opens `set_session(readonly=True)`).
- No Anthropic calls, no Toss calls, no Kiwoom calls. Nothing was written outside `data/eval/edge/`.
- Nothing in `~/git/daytrade-it`, the live daemon or the running backfill was touched.

**Answer, up front.** There is no demonstrated edge in the selection. Against a null matched on the day, the acting minute, the volatility bucket and the liquidity bucket, the whole judged pool is worth **+1.6 bp per pair (se 3.4) over 125 days** — zero. The bar it has to clear is **about 36.7 bp**, not the 21.5 bp every table in this repo has been netting out (§2.5): the research constant omits spread and slippage entirely and counts commission once instead of twice. The dashboard pool's celebrated +18 bp is two-thirds calendar and one-third reconstructed data; on the only two days that are genuine live archive it is −33.4 bp, and **−48.6 bp** once costed properly. One conditioning rule survives the search, it is an exclusion rather than a pick, and it then fails `bayes_judge` at p(μ>0)=0.8635.

The single most useful number in this document is not about selection at all. Buying any top-300 name at the 09:00 open and holding it to the close lost **56.8 bp** per name per day, averaged over 125 days. 42% of our pairs act at that minute. That headwind, not our stock picking, is what the system has been fighting.

---

> ## Two rules this investigation earned, before any result
>
> **1. No sweep-pool result transfers to live.** Six times in one session a candidate passed on the 132,185-pair sweep pool and reversed on the dashboard pool — trailing, hold-to-close, add-ons-off, two stop families, and the movers filter of §4.5. At six this stops being a caveat and becomes a rule: **nothing is a candidate until it has been frozen onto dashboard days**, and every sweep-pool number in this document is labelled as such.
>
> **2. But the dashboard pool cannot currently validate anything either.** Seven days, two of them genuine archive, 35 trades in a capped book, measured against a baseline that is itself the September 09:00-batch artifact. Candidates pass or fail there by noise against a bent yardstick. **Fixing the yardstick outranks generating candidates**, and it is free: real archive days accrue at one per trading day for as long as the daemon runs. Twenty of them is four weeks and no money.
>
> Rule 2 is why rule 1 currently disqualifies everything without confirming anything. That is the state of the project, not a defect in a particular result.

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

### 2.5 The cost floor, which is 15 bp higher than every table in this repo assumes

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

### 2.6 Does saving spread cost signal?

Tick width is a function of price, price correlates with company size, and small caps react harder to news — so a universe chosen to save spread might quietly be selling the signal. Our measured selection alpha is +1.6 ± 3.4 bp, so there is probably nothing to sell, but that is an assumption and it is one line of analysis to test.

Quintiles cut on our own tick/price distribution (edges 7.1 / 10.3 / 13.4 / 17.8 bp; mean 12.5), `scripts/eval/edge_tick.py`:

| tick bucket | n | mean tick | alpha | se | months |
|---|---:|---:|---:|---:|---|
| <7 bp (highest-priced) | 26,388 | 5.9 | **−4.2** | 6.6 | `--++--+` |
| 7–10 | 26,405 | 8.7 | **+10.8** | 4.5 | `+++++++` |
| 10–13 | 26,393 | 11.7 | +5.6 | 5.2 | `+++++--` |
| 13–18 | 26,429 | 15.4 | +1.6 | 6.0 | `++--+--` |
| >18 (lowest-priced) | 26,462 | 20.7 | −1.8 | 6.3 | `+-+---+` |

**No.** The narrowest-tick bucket is the worst, not the best, so moving toward high-priced names to save spread does not look like it discards signal. The spread saving is realisable and the alpha differences here sit on the same ±3–6 bp noise floor as everything else in this pool.

The 7–10 bp bucket at +10.8 bp with **7 of 7 months positive** is the only cell worth a second look, and it is one of five buckets chosen after the fact, so it is recorded as an observation rather than a candidate.

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

### 4.4 The survivor of the scan: the pre-entry move

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

**Read §4.7 before applying this.** The result below is *same-day*, and the sign of `pre_move` inverts across the middle of its range at multi-day horizons. Anyone who takes "tighter no-chase is better" from here into a multi-day design will have it backwards.

**What this is and is not.** It is a real, sign-stable, free exclusion: the half of the pool that has already run up more than 1% is worth −18.4 bp of alpha and is negative in **6 of 6 months**. Tightening the live `MAX_PRE_ENTRY_MOVE` from 3% to 1% improves net by 17.5 bp per pair and the loser ratio slightly, and it replaces a value that was chosen by leave-one-day-out on six days — the methodology this session has banned — with one chosen on 101 days under monthly time-ordered sign stability.

It is **not** an edge. The kept half still loses 43.8 bp per pair. Alpha at +7.5 ± 4.2 bp is a third of the round trip. This is harm reduction, and it should be adopted as harm reduction or not at all.

#### It does not validate on the pool live trades

Everything above is the **sweep pool**. Five exit families were retracted tonight for passing there and reversing on the dashboard pool, so the candidate was frozen and run on the dashboard days before being proposed. Per pair, all judged dashboard pairs (more power than the 35-trade book), provenance split:

| | n | alpha | se | net |
|---|---:|---:|---:|---:|
| all 7 days — all | 579 | +32.3 | 18.7 | +34.9 |
| calm two-thirds | 385 | **+15.1** | 16.2 | +16.7 |
| movers third | 194 | **+65.8** | 52.0 | +70.5 |
| archive 2 days — all | 271 | +19.0 | 17.6 | −33.6 |
| calm two-thirds | 181 | **+13.2** | 9.5 | −41.0 |
| movers third | 90 | **+30.3** | 72.1 | −18.9 |

**The sign reverses.** On the sweep pool the movers third is worse; on the dashboard pool it is better, in both the full and archive-only cuts. The book agrees: drop-top-33% takes all-7-days net from +189.5 to +172.0 and archive-only from +347.8 to +152.2.

With n = 194 and 90 and se 52 and 72 bp, this does not *refute* the filter — the difference is inside noise in both directions. But the burden was on the candidate and it did not clear it, and an underpowered test pointing the wrong way is not neutral evidence. Adopting on the sweep-pool result alone would repeat exactly what the five retracted exit families did.

**And the dashboard baselines are themselves unusable.** A 7-day book baseline of +189.5 bp net, and +347.8 bp on **ten** archive-day trades, are not plausible figures; they are the 09:00 batch on a favourable window, which §3.1 showed was +134.8 bp on those days and is a September artifact. The book fills earliest-minute-first, so it loads precisely the trades that artifact inflates.

**And the reversal is measured against the same bent yardstick.** The filter failing to validate is the right conclusion — the burden was on the candidate — but the *magnitude* of that failure is not interpretable either. Nobody should read "+65.8 bp for the movers third on live" as evidence **for** buying yesterday's movers; it is a number from a pool whose baseline is an artifact, and it says only that the filter did not clear its burden.

**The larger point, which outranks the filter.** The dashboard pool cannot currently validate anything: seven days, two real, 35 trades in a capped book, measured against a baseline that is itself an artifact. That is the binding constraint on every candidate this project produces — ahead of labelling, compute or modelling. Twenty real archive days would make it an instrument instead of a rumour, and they accrue at one per trading day for free.

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

### 4.5 The one live-expressible candidate: drop yesterday's biggest movers

Three independent routes arrived at the same line — the ten-year universe table (§7.1, movers third positive in 6 of 121 months), the news-pool horizon split (§8.9, where the multi-day alpha lives entirely in the calm two-thirds), and the dashboard's own profile (§7.1.1). Unlike everything else in this document it has a decade behind it, it needs no new data, and the previous day's bar is available at decision time.

But a filter cannot be judged on the names it removes. The live book is capped at five positions a day, so removing candidates does not shrink the book — it changes **which** names fill it, and this project has twice found a filter whose replacements were worse than what it excluded. `scripts/eval/edge_filter.py` fills the book the live way (earliest acting minute, higher score breaking ties, capped at five) and measures it before and after, restricted to the live BUY gate:

| filter | candidates | book | alpha | net | worst day | losers |
|---|---:|---:|---:|---:|---:|---:|
| none (baseline) | 22,690 | 625 | −16.2 | −112.2 | −1,219 | 0.614 |
| drop top 1% movers | 22,399 | 625 | −8.5 | −103.6 | −1,219 | 0.605 |
| drop top 5% | 21,497 | 625 | −4.5 | −97.2 | −1,219 | 0.600 |
| drop top 10% | 20,368 | 625 | −6.3 | −98.0 | −1,219 | 0.592 |
| drop top 20% | 18,104 | 625 | +3.9 | −87.1 | −1,219 | 0.589 |
| **drop top 33%** | 15,142 | 625 | **+9.4** | **−71.4** | **−848** | **0.582** |
| drop top 50% | 11,316 | 625 | −2.9 | −80.9 | −715 | 0.584 |

**The cut is the top third**, removing 33% of candidates and 0% of the book. Net improves 40.8 bp, the worst day improves by 371 bp, and the loser ratio falls 3.2 points.

**The refill is ordinary, and that is the honest mechanism.** 295 slots refill with alpha −5.8 and net −77.6 — the replacements are unremarkable. The entire gain is that what left was much worse than what arrived, not that the filter finds better names. It should be described that way.

#### The adoption verdict

```
p_mu_gt_0             0.9576     >= 0.90                       PASS
mu_mean              +1,705 KRW/day
worst_day_candidate  -42,410  vs baseline -60,955              PASS
loser_ratio           0.5824  vs baseline  0.6144              PASS
days_ge_0             77 / 125,  floor ceil(5/7 * 125) = 90    FAIL
adopt                 false
```

**Three of four, and the two that pass are the ones the objective weights above expectancy.** This is materially stronger than §4.4's `pre_entry_move ≤ +1%`, which failed p(μ>0) at 0.8635 *and* left the worst day untouched. Here p(μ>0) is 0.958 and the worst day improves by 18,500 KRW.

The failure is consistency: the filter beats the baseline on 62% of days against a required 71%. Worth noting about the rule itself — `days_ge_0` is a consistency criterion, so a change whose gain is concentrated in avoiding bad days will tend to fail it even when it is right on both expectancy and tail. That is a reasonable thing to demand and it is not a complaint, but it means `adopt=false` here does not mean the same thing as `adopt=false` on a candidate that also failed on expectancy.

#### On the post-hoc charge

Six cut points were scanned and the best one won. That is a search and it should be said plainly. The defence is that the top-third cut was **specified in advance** by the ten-year universe analysis — the movers third — before this book simulation existed, and independently reached by the horizon split. Three routes, one cut, named before the test that selected it. Both facts belong on the record and neither cancels the other.

### 4.6 Ranked list of everything tested

| rank | conditioning | verdict |
|---|---|---|
| 1 | **drop yesterday's top-third movers** (§4.6) | the strongest strongest sweep-pool candidate here (§4.5) — book net +40.8 bp, worst day +371 bp, `bayes_judge` 3 of 4 — and then **the sign reverses on the dashboard pool**. Not adopted, and not proposed. |
| 2 | **pre-entry move ≤ +1%** | survives the scan — monotone, exclusion side negative 6/6 months, transfers in sign to the dashboard — and then **fails `bayes_judge` at p(μ>0)=0.8635**. Harm reduction of 17.5 bp per pair, not an edge, not adopted. Note it is intraday-only: the separation is gone by +40d (§8.5), so it does not compose with the multi-day result. |
| 3 | publisher / wire (`특징` −31.1 alpha 0/7 months, `윤서` −48.8 1/7) | exclusion-only, n = 1,290–1,529, no positive counterpart. Worth a prefilter line, not a strategy. |
| 4 | title tag (`오늘의 IR` −73.6, `오늘의 주요공시` −36.2, `표` +22.3) | the big positive (`표`) is 3/7 months and is the same anti-correlated tail the oracle mining already warned about. Exclusions only. |
| — | sector | **fails** — best training sectors transfer 2/5 folds; `건설` +30.2 and `보험` +27.1 have se 19 and 16. |
| — | market regime (6 features × terciles) | **fails** — all 18 buckets lose, and none moves alpha. |
| — | volatility quartile, liquidity quartile | **fails** — spread ±14 bp, no monotonicity, s3 is `+++++--`, s2 is `+--+--+`. |
| — | fan-out | **fails** as a positive; `41+` at −31.4 (n=3,196) is a weak exclusion already covered by the live prefilter. |
| — | session, weekday, market (거래소/코스닥) | **fails** — all within ±10 bp, no month stability. |
| — | every v2 LLM field | **fails** — all within ±8 bp against the matched null; `score` non-monotone. |
| — | greedy best gate / best-3 union from history | **fails badly** — worse than the base pool on net, worst day and loser ratio. |

---

### 4.7 "No-chase" is two rules with opposite signs

This has its own section because reading §4.4 alone and then designing a multi-day strategy gets the sign backwards, and that is a foreseeable and expensive misreading.

Alpha in bp by pre-article move, across the full range including the region live refuses, 91,914 pairs with a known prior close over 84 common days (`scripts/eval/edge_nochase.py`):

| pre_move bucket | n | **same day** | +2d | +5d | **+10d** | +40d | months (d0) |
|---|---:|---:|---:|---:|---:|---:|---|
| −inf .. −3.0% | 13,414 | **+21** | −35 | −74 | **−128** | −304 | `+++++` |
| −3.0 .. −1.0% | 14,156 | +9 | +8 | +5 | +10 | −49 | `++--+` |
| −1.0 .. +0.3% | 17,388 | +4 | −1 | +1 | +11 | +5 | `+--++` |
| +0.3 .. +1.0% | 9,375 | +5 | +11 | +41 | **+46** | +117 | `-+-++` |
| +1.0 .. +2.0% | 10,823 | −3 | +16 | +39 | **+45** | +23 | `--+-+` |
| +2.0 .. +3.0% | 7,432 | −28 | +6 | +6 | +7 | +56 | `-----` |
| +3.0 .. +5.0% | 8,211 | −9 | −29 | +29 | +39 | +99 | `---+-` |
| +5.0% .. | 11,115 | **−37** | −50 | −72 | **−110** | −67 | `--+--` |

**Same day the response is monotone decreasing**: the further a name has already run, the worse it does. Tighter no-chase is better, which is §4.4's finding.

**Multi-day it inverts across the middle of the range.** Names that fell hardest are the *worst* at +10d (−128 bp), and mild-to-moderate up-moves are the *best* (+45 to +46). The same variable that says "avoid what has risen" at a session horizon says "prefer what has risen modestly" at ten days.

Only the extreme holds its sign: above +5% is negative at every horizon.

So `pre_move` is not one rule. It is a same-day mean-reversion signal and a multi-day momentum signal wearing the same name, and which one applies depends entirely on how long the position is meant to live. Any future use of it has to state the horizon it was fitted at.

#### The filter itself is safe, and safe for an unusual reason

| cut | kept % | kept alpha d0 | removed d0 | kept +10d | removed +10d |
|---|---:|---:|---:|---:|---:|
| ≤ 0% | 44.7% | +9.6 | −9.8 | −6.1 | −0.4 |
| ≤ 1% | 59.1% | +11.0 | −17.4 | +8.9 | −3.1 |
| ≤ 2% (as fitted) | 70.9% | +9.4 | −24.2 | **+28.4** | −26.7 |
| ≤ 3% (**live now**) | 79.0% | +6.8 | −23.5 | **+30.2** | −36.7 |
| ≤ 5% | 87.9% | +8.5 | −37.4 | **+30.5** | −109.6 |

Every cut removes a negative-alpha population at both horizons, so there is no line at which the filter discards something that works. And between 2% and 5% the kept alpha is **flat** (+28.4 / +30.2 / +30.5 at +10d).

That flatness is worth stating positively rather than as an absence of harm. `MAX_PRE_ENTRY_MOVE` was set to 2% on 209 pairs over 10 days, then raised to 3% on six days of leave-one-day-out — a fitting procedure this investigation has otherwise spent the night discrediting. It survives not because the fit was good but because **it landed in a plateau**, so the fragility of the procedure did not propagate into the result. It is rare to be able to report a hand-set constant as safe *because* the surface is flat, and it is the honest reason to leave it alone rather than re-fit it, which would be optimising noise.

#### And the hypothesis this was built to test is refuted

The proposal was that the pool's alpha lives in names that gap and run, so the no-chase filter removes the only part that works. It conflates two variables:

- `pre_move` — entry versus the *prior* close, what the filter sees
- **never dipped** — never traded a tick below entry *after* the decision minute, the +246 bp population from §8.11's fill split

If they were the same names, "never dipped" would concentrate at high `pre_move`. It does not: **2.1% to 4.1% in every bucket, with no trend** — 3.6% in the +5% bucket against 3.8% in the −3% bucket. A name that gapped up is no likelier to be one a resting bid misses than a name that gapped down. The inference never had a bridge.

Buyability was measured anyway, since it was the stated crux: the rejected region fills at **96–98%** with a resting bid. Those names are perfectly purchasable. They simply do not pay — which closes the escape hatch where someone later argues the filter is protecting us from something unbuyable.

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

**2. Do not buy more Haiku labelling to answer *this* question — but see §8.7, which finds the one question where it would help.**
The directive says to buy labels rather than accept a thin sample, and that is right in general, but it does not apply to the entry-selection question. The 132,185 already-judged pairs put the whole LLM field block at ±8 bp against a matched null with `score` non-monotone, and the character-n-gram bound on the raw text caps any possible prompt at 1.9% of the oracle gap. Buying labels buys more of a quantity that has now been measured at zero three independent ways, and a fourth measurement against a properly matched null. Labels **are** the binding constraint for the learning curve and for a bandit with enough days; they are not the binding constraint for "does the news tell us which stock to buy".

That holds for the *entry-selection* question and for the intraday horizon, which is what this section is about. §8 opened a question it does not cover: `persistence` is the one judgment field that earns anything, it only earns it on a multi-day horizon, and the labels that carry it exist for 2026-03..09 only. Extending them is the single labelling purchase in this project with a live question attached to it.

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

## 7. The arithmetic

Everything above reduces to one sum. A trade earns the drift of its bucket, plus whatever our selection adds, minus the toll:

```
net  =  alpha  +  null  -  cost
```

This is an identity, not an estimate: `alpha` is *defined* as `ret - null`, so the three terms cannot double-count. What can go wrong is mixing components from different populations, and the numbers below are matched — each row's alpha and null come from the same acting-minute bucket.

Per pair, in bp, over 125 days and 132,077 pairs, at the real round trip of 36.7 bp (§2.5). Every quantity is pair-weighted and day-clustered, and both the alpha and the drift come from the same stratum-matched null, so each row is an exact identity rather than three numbers from three places:

| acting minute | n | share | alpha | se | null (drift) | se | t | **net** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **09:00 open** | 55,308 | 41.9% | −4.8 | 7.0 | **−56.3** | 18.8 | **−3.00** | **−97.8** |
| 09:01–09:29 | 16,553 | 12.5% | −3.0 | 4.4 | −18.6 | 16.5 | −1.13 | −58.3 |
| 09:30–09:59 | 14,091 | 10.7% | +5.2 | 5.2 | −25.8 | 15.1 | −1.71 | −57.3 |
| 10:00–10:59 | 19,438 | 14.7% | +7.9 | 3.1 | −21.8 | 13.3 | −1.64 | −50.7 |
| 11:00–12:59 | 12,279 | 9.3% | +2.4 | 2.9 | −5.0 | 10.2 | −0.49 | −39.3 |
| **13:00+ (best case)** | 14,408 | 10.9% | −0.5 | 1.6 | **+5.6** | 5.1 | 1.10 | **−31.6** |

**No row is positive, and the best row is not close.** The best case available — perfect timing, entering only after 13:00, and granting ourselves the drift at its point estimate — is **−31.6 bp per trade**. The worst case, where 42% of our pairs actually act, is **−97.8 bp**.

(The drift figures here differ from §3's curve because the weighting differs and both are correct for their own question. §3 weights each acting minute once per day, which is what "what does the headwind look like through the session" asks. This table weights by pairs, which is what "what did we earn" asks. They agree at the open, where there is only one minute, and diverge intraday because our pairs cluster on particular minutes.)

Three corrections to the version of this sum that was circulating:

1. **The cost term is 36.7 bp, not 21.5.** Per §2.5 the research constant omits spread and slippage and counts commission once. This alone moves every row by −15.2 bp.
2. **The afternoon drift does not survive its own standard error.** +5.6 bp has se 5.1, t = 1.10, and is positive on 73 of 125 days. It should be read as zero, which makes the best case **≈ −37 bp** rather than −31.6. The only rung in the table that is statistically solid is the one that hurts: the open at t = −3.00.
3. **Alpha must come from the same bucket as the drift.** Pairing the pool-wide +1.6 bp with the 13:00+ drift overstates the best case; the 13:00+ bucket's own alpha is −0.5 bp.

On order size: 36.7 bp is if anything optimistic for us. The tick-grid floor is size-independent — a 100,000 KRW order crosses the same one-tick spread as a large one — and a position that small cannot be worked across the book to do better.

**So the conclusion is not "we have not found the edge yet."** It is:

> Long-only intraday trading on this universe cannot clear its costs even with perfect entry timing, because the selection alpha is zero at every hour and the day's drift never exceeds the toll.

Each of those qualifiers — *this universe*, *this horizon*, *long-only*, *a 21.8-minute lag*, *a 36.7 bp toll* — is a thing that could be changed rather than a law of nature, and §7.1 through §8 change them one at a time.

### 7.1 Is the headwind ours or the market's? It is ours.

Our universe is the top 300 by the *previous* day's traded value — a momentum screen wearing a liquidity costume. `scripts/eval/edge_universe.py`, equal-weight open-to-close, **2,454 days over 121 months, 2016-09 .. 2026-09**, with the round trip priced per universe from its own tick grid:

| universe | drift | se | spread | cost | **net** | months > 0 |
|---|---:|---:|---:|---:|---:|---:|
| all traded codes | **−6.4** | 2.3 | 22.7 | 45.7 | −52.1 | 44/121 |
| **top 300 by prior-day value (ours)** | **−28.6** | 2.6 | 13.4 | 36.4 | **−65.0** | **11/121** |
| rank 301–600 | −16.4 | 2.8 | 13.6 | 36.6 | −52.9 | 30/121 |
| rank 601–900 | −8.1 | 2.8 | 13.7 | 36.7 | **−44.8** | 45/121 |
| our 300, **movers** third (top 1/3 by yesterday's |move|) | **−42.0** | 3.3 | 13.8 | 36.8 | −78.8 | **6/121** |
| our 300, **calm** two-thirds | −21.8 | 2.5 | 13.2 | 36.2 | −58.0 | 20/121 |

**This table corrects an earlier version of this document.** Measured over the six months of the news sample alone, the market-wide drift looked like −29.8 bp and the conclusion drawn was "about 30 bp is the market and is irreducible for any long-only intraday strategy". That was regime. Over ten years the market's intraday drift is **−6.4 bp** — small, if statistically negative. The earlier claim is withdrawn.

What the ten-year view shows instead is worse for us and better for the project:

- **The headwind is almost entirely our own screen.** Ours is −28.6 bp against the market's −6.4 — **22.2 bp of self-harm**, negative in **110 of 121 months**. That is structural, not regime, and it is a choice we made.
- **The ranking decay is real** — monotone −28.6 → −16.4 → −8.1, with 11/121 → 30/121 → 45/121 positive months. (A six-month version of this table showed the decay reversing in three of seven months. That too was noise.)
- **The movers third is the single most robust result in this investigation**: −42.0 bp, positive in **6 of 121 months**, across 2,454 days.
- **The cheapest universe is not the best one.** "All traded codes" has the smallest drift but a 22.7 bp spread, because it includes low-priced names on a coarse relative tick grid; that eats the advantage and leaves it at −52.1 net. The spread penalty is real and it is why `net` is the column that matters.

**By net the best available universe is rank 601–900 at −44.8 against our −65.0 — +20.2 bp, free.** Dropping the movers third from our own universe is worth +20.2 bp on the drift and keeps the universe's shape.

It does not flip the sign: −44.8 is a smaller loss, not a profit, and nothing here is a path back to trading. But the framing changes. It is not that the market is against us. It is that **the universe was chosen for sweep lap time rather than for returns, and has been costing about 22 bp a trade in every backtest in this document.** That is a fixable mistake rather than a law of nature — but see §7.1.1 before applying it to live, which does not use this screen and carries a smaller version of the same drag for a different reason.

### 7.1.1 Does LIVE trade this universe? No — and it still inherits most of the drag

§7.1 is a statement about the **sweep pool**, and it was nearly mis-applied. `select_universe`'s top-300-by-prior-day-turnover exists to bound the 1-req/s company-news sweep's lap time. **It has never gated a live entry.** Live trades whatever tickers Toss tags on a dashboard article, filtered only by the stock master. So "we chose a bad universe and have paid 22 bp a trade for the life of the project" is true of every sweep-pool backtest number in this document — which is most of them — and says nothing by itself about live.

The dashboard-tagged ticker set exists for seven days of 2026-09, two of them genuine archive, so it cannot be measured over a decade directly. It can be *characterised* by properties observable before the open, and that characterisation can be measured over a decade.

**Profile** (percentile within that day's traded names; 1.0 = most traded / biggest mover):

| pool | prior-day turnover pct (p25 / med / p75) | prior-day \|move\| pct |
|---|---|---|
| dashboard, all 7 days | 0.72 / **0.90** / 0.97 | 0.32 / 0.59 / 0.82 |
| dashboard, archive days only | 0.70 / 0.89 / 0.96 | 0.33 / 0.56 / 0.81 |
| dashboard, reconstructed days | 0.73 / 0.90 / 0.98 | 0.29 / 0.60 / 0.82 |
| sweep pool, same 7 days | 0.87 / 0.94 / 0.98 | 0.34 / 0.60 / 0.81 |
| sweep pool, all 125 days | 0.87 / 0.93 / 0.97 | 0.34 / 0.62 / 0.85 |

Archive and reconstructed days give the same profile, so this is not a provenance artifact.

**Implied drift**: weighting the ten-year drift of each (turnover percentile × prior-move percentile) cell by the dashboard's own weight in it gives **−20.9 bp**, against market −6.8, our top-300 −28.6, movers third −42.0, calm two-thirds −21.8.

**So the Toss dashboard is a momentum screen by editorial accident.** Its median tagged name sits at the 90th percentile of prior-day turnover. Live inherits roughly **14 bp** of drag against the market that nobody chose — less than the sweep pool's 22 bp, and nowhere near neutral. The correction to §7.1 is therefore a narrowing, not a withdrawal: the magnitude belongs to research, the direction belongs to both.

#### The direct measurement is uninterpretable, and it is the founding trap again

On those seven days the dashboard-tagged set drifted **+233.5 bp**, against +23.1 for the top-300 and +3.4 for all traded codes. That number must not be used, and it is worth keeping precisely because of *why*.

Dashboard membership is determined **intraday**. A stock jumps at 10:00, news follows at 10:05, the article tags it, and its open-to-close return is large and positive by construction. Measuring "the drift of names the dashboard tagged today" therefore conditions on the day's outcome.

This is the same mechanism that produced the +1.27% founding baseline retracted in §2.4, arriving in a new place four sections later. It is the most persistent error in this project and it has now been caught three times: once in the founding eval, once in the dashboard pool's forward maximum, and once here.

#### What is actionable, and it is live-expressible

The ten-year cells are monotone in **prior-day |move| percentile**, and the top 1% is severe — **−93.9, −114.0 and −147.3 bp** depending on the turnover bucket, against roughly zero for the bottom half of movers. Prior-day bars are available at decision time, so excluding yesterday's biggest movers is a filter live can apply today without new data, new labelling, or any model.

That is the movers/calm finding arriving for the third time by an independent route — first in the ten-year universe table (§7.1, 6 of 121 months positive), then inside our own judged pool across horizons (§8.9), and now on the pool live actually trades.

### 7.2 Trading less, and whether more data can rescue this

**Fewer, larger trades.** Cost is charged per trade and alpha is zero, so trading less is strictly better and the limit of the argument is trading nothing. That is not a rhetorical flourish, it is the boundary condition: with `alpha ≈ 0`, expected P&L is `n × (null − cost)`, and since `null < cost` in every hour of the day (§7), the optimum over `n` is `n = 0`. Any proposal that increases trade count — pyramiding, add-ons, re-entries — moves the wrong way twice, because each additional round trip pays the full 36.7 bp rather than the 21.5 it was evaluated against.

**Can more data rescue this?** No, and the power arithmetic is worth stating because it answers the question directly rather than by assertion. The day-level standard deviation of the pool's alpha is 38 bp (se 3.4 over 125 days). To detect an effect at t = 2:

| effect to detect | trading days needed |
|---:|---:|
| **36.7 bp (the cost floor — the smallest tradeable effect)** | **4** |
| 20 bp | 14 |
| 12 bp | 40 |
| 8 bp | 90 |
| 5 bp | 231 |
| **1.6 bp (what we actually measure)** | **2,256 (≈ 9 years)** |

**An effect large enough to trade would have been visible in four days. We have 125.** The sample is not the constraint and never was; the effect is roughly twenty times too small. This is the quantitative form of the §6 argument against buying more labelling for the entry question, and it applies equally to buying more days of anything at daily or minute resolution.

The corollary is the one useful thing in the table: because the bar is high, ruling things out is cheap. Forty days of any new data source is enough to detect a 12 bp effect and to reject a 36.7 bp one decisively.

**Order flow, sized rather than argued.** `ticks` and `quotes` currently hold **18 trading days**, 2026-08-24 to 2026-09-16, at roughly 120–250 codes per day. A month of further collection takes that to about 38 days — which, per the table above, is enough to detect a 12 bp effect and more than enough to rule out a tradeable one. So the honest sizing of this lever is:

- it is the only untested hypothesis class, since every result in this document lives at minute or daily resolution;
- a month of collection makes it a **well-powered test**, not a well-powered strategy;
- it costs nothing but leaving the collector running, and it runs in parallel with everything else;
- and it should be scoped as "rule it in or out" rather than "build on it". If microstructure carried a 36.7 bp same-day edge on KRX top-300, it would be the best-documented anomaly in the market.

The day-level sd of a tick-feature-selected subset will be larger than the pool's, since fewer pairs per day means a noisier daily mean, so 38 days is the optimistic end of that estimate.

---

### 7.3 Which conclusions in this project rest on six months

Three published claims were retracted in one night, and all three failed the same way: a pooled average over 2026-03..09 that did not survive a decade. That is a pattern, not three accidents, and it deserves an inventory rather than a resolution to be careful.

`daily_bars_adjusted` reaches back to **2016-09-09**, so anything that depends only on prices can be re-checked for the cost of a few minutes of compute. Anything that depends on the LLM judgments cannot, because those exist only for 2026-03..09.

**Checkable on price history alone — and what happened when checked:**

| claim | six-month version | ten-year version | status |
|---|---|---|---|
| market-wide intraday drift | −29.8 bp | **−6.4 bp** | **retracted** (§7.1) |
| our universe's intraday drift | −61.1 bp | −28.6 bp, 110/121 months negative | confirmed, magnitude halved |
| turnover-ranking decay | reversed in 3 of 7 months | monotone, 11/121 → 30/121 → 45/121 | **restored** — the six-month reversal was noise |
| movers vs calm third | 7 of 7 months | 6 of 121 months positive | **confirmed and strengthened** |
| multi-day drift by horizon (the `null` term in §8) | −697 bp at +40d | §8 addendum | **in progress — this one carries §8's arithmetic** |
| tick-grid spread floor by universe | 12.7–13.9 bp | price levels are available for the whole window | checkable, not yet run |

**Not checkable without labels**, and therefore still resting on 2026-03..09:

- everything involving `alpha`, since it needs the judged pairs: the +1.6 bp selection result (§2.3), the 22-axis conditioning scan (§4), the `pre_entry_move` survivor (§4.4), and the entire `persistence` × horizon result (§8.3).
- the dashboard pool comparison (§2), which rests on **two** genuine archive days.
- the founding +1.27% baseline replication (§2.4).
- the headwind-by-acting-minute curve (§3), which needs minute bars — those begin far later than the daily series, so it cannot be extended even though it is price-only.

**The rule I would draw from this.** The retracted claims were all *magnitudes* of a pooled average. The claims that survived the decade were *orderings* — which universe is worse than which, whether movers are worse than calm. Orderings held; levels did not. That is worth carrying into how the remaining label-bound results are read: §8.3's contribution is that `persistence` **orders** the horizon response at all seven horizons, and I have deliberately declined to claim its magnitudes for exactly this reason.

---

## 8. Persistent news over multi-day horizons

> **Movement pays for the toll somewhere between +5 days and +10 days. Intraday it never does.**

That is the deliverable of this investigation. Everything else in this document is either support for that sentence or a door it closed.

> This is a different strategy from day trading. It should be chosen knowingly, not drifted into. Everything below is a multi-day hold, which the live system cannot currently execute or manage, and which consumes capital across days rather than releasing it at the close.

The crossover, for our judged names minus yesterday's biggest movers (`scripts/eval/edge_curve.py`, `data/eval/edge/holding_curve.csv`). `minutes` is exposure, not calendar — a same-session hold is entered mid-session, so it is only the remaining minutes, and each extra day adds a whole session:

| hold | minutes | alpha | se | cost | **alpha − cost** |
|---|---:|---:|---:|---:|---:|
| rest of day | 322 | +4.6 | 4.3 | 36.7 | **−32.1** |
| +1d | 712 | +14.8 | 11.3 | 36.7 | −21.9 |
| +2d | 1,102 | +18.5 | 16.4 | 36.7 | −18.2 |
| +5d | 2,272 | +28.3 | 29.1 | 36.7 | **−8.4** |
| **+10d** | 4,222 | +46.9 | 46.6 | 36.7 | **+10.2** |
| +20d | 8,122 | +60.3 | 89.2 | 36.7 | +23.6 |
| +40d | 15,922 | +134.3 | 76.7 | 36.7 | +97.6 |

The toll is charged once per round trip and does not amortise until the horizon lengthens; the signal grows with horizon. Those two lines cross between 2,272 and 4,222 minutes of exposure. **No point on this curve is statistically significant** — the errors are moving-block bootstrapped at the holding length and the best t is 1.38 — so the crossing is a shape, not a level.

Two of this project's binding constraints weaken only on that side of the crossover. We are ~21.8 minutes late to every article and most of that lag is upstream of us, so an impulse is unreachable — but if the news has *lasting* impact, arriving late costs little. And the sell tax is charged per disposal, so it amortises. Neither weakens intraday, which is why §7's arithmetic was closed.

This is also the first time `persistence` has been scored on a horizon where it could possibly express itself. Every evaluation in this project's history has been same-day — a "structural" judgment has no way to show up by 15:00 on the day of the article — and the prompt that extracts it was validated on 209 pairs over 10 days.

### 8.0 What the net number depends on, in the table rather than in a caveat

`net = alpha + drift − cost`. The alpha term is five months of 2026. The drift term has ten years behind it and **swings enormously by year**, and 2026 is among the worst in the decade. Any net figure taken from this section carries a year range with it, so the range is printed beside it:

Calm two-thirds, alpha from our 86 common days, drift from `edge_years.py` 2016–2026, cost 36.7 bp:

| hold | alpha | drift, decade pooled | drift, worst year | drift, best year | net at pooled | net range across years |
|---|---:|---:|---:|---:|---:|---|
| rest of day | +4.6 | −21.8 | −32 (2016) | −13 (2025) | **−53.9** | −64 .. −45 |
| +2d | +18.5 | −26.0 | −70 (2022) | +16 (2025) | −44.2 | −88 .. −2 |
| +5d | +28.3 | −26.2 | −118 (2022) | +61 (2025) | −34.6 | −126 .. +53 |
| +10d | +46.9 | −24.3 | −188 (2022) | +145 (2025) | −14.1 | −178 .. +155 |
| +20d | +60.3 | −17.5 | −282 (2022) | +354 (2025) | +6.1 | −259 .. +378 |
| +40d | +134.3 | **+4.3** | **−401 (2022)** | **+827 (2025)** | **+101.9** | **−303 .. +925** |

Read the last row honestly: the same alpha produces −303 bp in a 2022-like year and +925 bp in a 2025-like one. **The multi-day result is a bet on the drift regime at least as much as on the news.** Intraday there is no such ambiguity — the drift is between −13 and −32 bp in all eleven years, which is why §7 closes and this does not.

### 8.1 Two corrections without which this section would have been wrong

**Common days.** A +40d exit needs 40 trading days of future. The sample ends 2026-09-15 and today is 2026-09-17, so only the early days resolve at long horizons. Left alone, "alpha grows with horizon" and "the long horizons are March and April" would have been the same statement. Everything below is restricted to the **86 days where every horizon resolves** (2026-03-17 .. 2026-07, 97,271 pairs), 39 days dropped.

**Overlapping windows.** A +40d hold started today and one started tomorrow share 39 of their 40 days. Treating those as independent understates the standard error by roughly √40. All errors below come from a **moving-block bootstrap** with the block set to the holding length, so the overlap is carried into the resamples. This is the single most important number in the section and it is what stops the result being oversold: on the raw day-clustered error the +40d figure looked like t = 2.4; corrected, it does not.

### 8.2 Does alpha grow with horizon? Yes — and so does the error, faster

All pairs, 86 common days, `alpha` = return minus its own volatility × liquidity cell at the same horizon:

| hold | n | alpha | se | t | z (σ) | net | net/day | worst day | losers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same day | 97,271 | +5.9 | 3.8 | **1.55** | +0.015 | −74.9 | −74.9 | −595.6 | 0.639 |
| +1d | 97,271 | +13.4 | 9.8 | 1.37 | +0.024 | −89.0 | −89.0 | −756.6 | 0.606 |
| +2d | 97,271 | +16.2 | 14.7 | 1.10 | +0.023 | −101.8 | −50.9 | −1039.3 | 0.603 |
| +5d | 97,271 | +19.0 | 28.5 | 0.67 | +0.025 | −159.7 | −31.9 | −1445.8 | 0.601 |
| +10d | 97,271 | +27.4 | 49.7 | 0.55 | +0.028 | −250.7 | −25.1 | −1914.3 | 0.625 |
| +20d | 97,271 | +23.6 | 85.4 | 0.28 | +0.031 | −290.1 | −14.5 | −2084.3 | 0.622 |
| +40d | 97,271 | +96.6 | 69.9 | 1.38 | +0.071 | −687.3 | −17.2 | −3401.0 | 0.685 |

Alpha rises from +5.9 to +96.6 bp, and in σ units from +0.015 to +0.071 — so it is not merely the extra variance a longer hold brings. **But no horizon reaches t = 1.6.** The point estimates grow and the errors grow faster, which is what an underpowered measurement of a real effect and a measurement of nothing both look like.

**Cost amortisation is real and does exactly what was predicted.** Net per capital-day improves more than fourfold, −74.9 → −17.2 bp/day. That is the only mechanism found tonight that moves the bar rather than trying to clear it, and it works. It is simply not enough on its own, because the drift term grows with horizon too.

### 8.3 `persistence` orders the horizon response, at every horizon

This is the first v2 field in this project to behave the way the prompt intends.

`alpha` in bp, by `persistence` × horizon:

| hold | one_off (n=43,405) | multi_quarter (n=35,662) | structural (n=18,204) | structural − one_off |
|---|---:|---:|---:|---:|
| same day | +3.9 | +7.3 | +8.4 | +4.5 |
| +1d | +9.1 | +16.0 | +21.7 | +12.6 |
| +2d | +11.0 | +17.9 | +26.8 | +15.8 |
| +5d | +8.7 | +26.0 | +26.2 | +17.5 |
| +10d | +5.6 | +45.1 | +42.1 | +36.5 |
| +20d | −35.2 | +63.9 | +82.0 | +117.2 |
| +40d | −9.7 | +151.6 | +247.0 | **+256.7** |

The ordering `structural ≥ multi_quarter > one_off` holds at **all seven horizons**, the spread widens monotonically, and `structural` rises at every step while `one_off` flattens and turns negative. In σ units `structural` goes +0.020 → +0.133 and `one_off` +0.009 → +0.027.

Month by month, the contrast (multi_quarter + structural, minus one_off), on the five months that resolve at every horizon:

| hold | pooled | 03 | 04 | 05 | 06 | 07 | + months |
|---|---:|---:|---:|---:|---:|---:|---:|
| same day | +8.1 | +6 | +24 | +4 | +6 | −3 | 4/5 |
| +2d | +20.1 | +29 | +68 | +17 | −9 | +19 | 4/5 |
| +10d | +48.7 | +243 | +62 | −87 | +39 | +113 | 4/5 |
| +20d | +106.0 | +433 | +62 | −132 | +161 | +231 | 4/5 |
| +40d | +187.2 | +473 | +87 | +19 | +307 | +260 | **5/5** |

**What I am willing to claim.** The *ordinal* fact — that `persistence` ranks the horizon response in the direction the prompt intends, at every horizon, widening monotonically, 4/5 to 5/5 months — is the strongest signal this investigation has produced. It is seven consistent checks of one ordering rather than one lucky bucket.

**What I am not willing to claim.** The magnitudes. At +20d and +40d there are roughly **two independent windows** in 86 days, and no standard error computed from two observations is trustworthy — including the flattering ones the bootstrap produces for `structural`. March carries a large share of the pooled contrast at every long horizon (+243, +433, +473). Treat the long-horizon numbers as ordinal evidence, not as effect sizes.

### 8.4 The arithmetic, and the one honest route to a positive number

Net is **negative at every horizon and in every bucket**. The best is `structural` at +20d, −10.7 bp per capital-day against −74.9 for the same-day pool. Better, never positive.

The reason is not the cost any more — it is the drift. By construction `net = alpha + null − cost`, and over this sample the universe fell steadily, so the `null` term swamps everything at long horizons. For `structural` at +40d: alpha **+247.0**, null **≈ −697**, cost −36.7, net **−486.9**.

Which points at the one thing that would make this a strategy rather than a curiosity. `alpha` is market-neutral by construction. **If the market exposure were hedged, the return would be `alpha − cost`** — for `structural` at +40d, +247.0 − 36.7 = **+210 bp per trade over 40 days, about +5.3 bp per capital-day.** KRX retail cannot short individual names, but index futures and inverse ETFs are available, so a hedged version is executable in principle.

I am flagging that rather than recommending it. It is untested, it rests on the same two independent windows, the hedge carries its own round trip and tracking error, and a market-neutral multi-day book is a different business from news day trading in every operational respect.

### 8.5 The intraday survivor does not survive

`pre_entry_move ≤ +1%`, the survivor of §4's scan, is an intraday effect only:

| hold | keep ≤ +1% | drop > +1% | spread |
|---|---:|---:|---:|
| same day | +11.0 | −17.4 | **+28.4** |
| +2d | +12.1 | −13.3 | +25.4 |
| +10d | +8.9 | −3.1 | +12.0 |
| +40d | +26.8 | +28.9 | **−2.1** |

The separation decays monotonically and is gone by +40d. That is consistent with it being what it looked like — a short-horizon mean-reversion effect on names that have already run — and it means the two findings do not compose: the gate belongs to the intraday strategy that §7 closed, not to the multi-day one.

### 8.6 What a longer hold brings with it

Being adversarial about the costs that are not in the mean:

- **The tail gets much worse, and this is a trade the user has to be shown explicitly.** Worst day **−595.6 → −3,401.0 bp**; loser ratio **0.639 → 0.685**. The stated objective is asymmetric — 잃는 게 안 버느니만 못하다, losing is worse than not earning — so the dimension that matters most moves the wrong way as the horizon extends. Whatever the significance question resolves to, **a longer horizon buys expectancy by selling tail protection.** The mean improves per capital-day (−74.9 → −17.2) and the worst day gets 5.7× deeper at the same time. Those two facts belong in the same sentence whenever this strategy is described, because the improving mean is the part that is easy to quote and the deepening tail is the part the objective actually weights.
- **Capital.** A multi-day position occupies a slot for the whole holding period. The live 5-slot budget would support 5 positions per 40 days rather than 5 per day — a ~40× reduction in trade count. `net/day` is the right comparison and it is the one reported, but the operational consequence is that the strategy is almost entirely idle.
- **Live parity: the current system cannot do this.** The news exit path flattens by the close and `SESSION_CLOSE`/signal expiry are same-day; a multi-day hold would need new position management, overnight risk handling, and corporate-action handling that does not exist. This is a build, not a config change.
- **Market beta.** The null does more work at longer horizons, so any mismatch in it matters more. This is why §8.2 reports σ-normalised alpha alongside raw — the growth survives normalisation, which is the check that matters.
- **Sample.** Five months, ~two independent 40-day windows, in a period when the market fell. Everything here would need re-measuring on a longer history before it could carry weight.

### 8.7 Verdict

The strategy class **does not close**, which is a different answer from every other avenue tonight, and I want to be precise about why rather than let it read as encouragement.

- Alpha grows with horizon, in bp and in σ. Not significant at any horizon once window overlap is honestly priced.
- `persistence` orders that growth correctly at all seven horizons, 4/5 to 5/5 months. This is the first time an LLM judgment field has earned anything in this project.
- Long-only it lost at every horizon **in this sample**, because the 2026 drift grew faster than the alpha — but §8.8 shows that drift is regime, not structure, and the decade-typical +40d drift for the calm two-thirds is +4.3 bp rather than −697.
- Hedged, the arithmetic would be positive on these point estimates — and those point estimates rest on two independent windows.

So: not demonstrated, and not excluded. That is genuinely different from the intraday case, where the confidence interval excludes anything tradeable. The next step is not to build it; it is to measure it on a longer history, where the 40-day window has enough independent observations to carry a standard error. `daily_bars_adjusted` reaches back to 2016-09-09, so the drift and the horizon arithmetic can be extended immediately — but the `persistence` labels only exist for 2026-03..09, and extending those is the one place in this project where buying more Haiku labelling would answer the question that is actually open.

---

### 8.8 Addendum: ten years of the drift term, and what it does to §8.4

§8.4 concluded that a multi-day hold loses because the drift term swamps the alpha — `structural` at +40d had alpha +247.0, null ≈ −697, and net −486.9. That null was measured on 2026-03..07. Per §7.3's own rule, it needed checking on a decade before it could carry a conclusion, and it does not survive.

`scripts/eval/edge_years.py`, equal-weight, open of day *d* to close of day *d+h*, **2,454 days by year**. (069500 is absent from `daily_bars_adjusted` and `sector_index` begins 2024-01-26, so "all traded codes" stands in for the market and is labelled as the proxy it is.)

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | pooled | se |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all, same day** | −9 | −8 | −5 | −5 | −0 | −5 | −8 | −3 | −9 | −7 | −24 | **−6.8** | 2.4 |
| **top300, same day** | −34 | −25 | −24 | −31 | −28 | −24 | −33 | −27 | −32 | −20 | −44 | **−28.6** | 2.6 |
| **movers, same day** | −39 | −41 | −28 | −39 | −46 | −34 | −47 | −45 | −43 | −36 | −71 | **−42.0** | 3.3 |
| **calm, same day** | −32 | −17 | −22 | −26 | −20 | −20 | −27 | −17 | −26 | −13 | −31 | **−21.8** | 2.5 |
| all, +40d | −107 | +216 | −39 | +18 | +776 | +170 | −161 | +108 | −148 | +436 | −366 | **+116.5** | 17.9 |
| top300, +40d | −468 | +101 | −360 | −351 | +534 | −264 | −479 | −32 | −238 | +689 | −313 | −73.2 | 19.7 |
| movers, +40d | −663 | −58 | −406 | −417 | +477 | −309 | −637 | −202 | −478 | +413 | −780 | **−228.1** | 22.3 |
| **calm, +40d** | −369 | +181 | −337 | −318 | +562 | −242 | −401 | +53 | −118 | +827 | −79 | **+4.3** | 19.3 |

Two facts, pointing opposite ways:

**1. The intraday drift is structural.** Every universe is negative in **11 of 11 years**, with our top-300 between −20 and −44 bp every single year. §7's conclusion rests on the one thing in this investigation that genuinely does not move. That is now settled.

**2. The multi-day drift is not structural — it is regime, and 2026 is among the worst years in the decade.** At +40d the market proxy swings from −366 (2026) to +776 (2020), pooling to **+116.5**. Our own top-300 pools to −73.2 with a −479..+689 range. So §8.4's "the drift swamps the alpha" was a statement about the sample, not about multi-day holding.

**3. And the screen's penalty compounds with horizon.** The gap between our top-300 and the market proxy is −21.8 bp same-day and **−189.7 bp at +40d**. Choosing a momentum screen costs little intraday and a great deal over weeks. The calm two-thirds at +40d drifts **+4.3 bp** over the decade — essentially zero — against −228.1 for the movers third, a 232 bp separation.

### 8.9 The same split, measured directly inside our own pool

§8.8's split is a property of the universe. Whether it survives *our selection* is a separate question, and it can be answered without any transfer assumption or new labelling, because both groups are names we already judge. `--movers` on `edge_persist.py`, same 86 common days, same matched null, same block bootstrap:

| hold | **calm two-thirds** (n=69,041) | | | | **movers third** (n=18,270) | | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | alpha | net/day | worst day | losers | alpha | net/day | worst day | losers |
| same day | +4.6 | −72.1 | −569 | 0.641 | +7.5 | −94.9 | −766 | 0.637 |
| +2d | +18.5 | −43.3 | −1,003 | 0.597 | +19.3 | −68.7 | −1,253 | 0.624 |
| +5d | +28.3 | −24.8 | −1,372 | 0.592 | +6.1 | −48.6 | −1,938 | 0.625 |
| +10d | +46.9 | −18.4 | −1,794 | 0.608 | −7.0 | −39.9 | −2,503 | 0.676 |
| +20d | +60.3 | **−8.7** | −2,018 | 0.602 | −84.2 | −28.9 | −3,638 | 0.695 |
| +40d | +134.3 | −12.9 | −3,036 | 0.660 | **−156.0** | −32.9 | −4,737 | **0.767** |

**The multi-day alpha lives entirely in the calm two-thirds.** Its alpha grows monotonically at every horizon (+4.6 → +134.3) while the movers third turns negative from +10d onward and reaches −156.0. The movers third is worse on every dimension at every horizon — alpha, net per capital-day, worst day, and loser ratio (0.767 vs 0.660 at +40d).

This is the same finding as §7.1's screen result, arriving independently through the news pool, and it is the one recommendation in this document that requires no new data, no new labelling, and no transfer assumption: **yesterday's movers should not be bought, and the reason to exclude them gets stronger the longer you intend to hold.**

### 8.10 The counterfactual, stated as a counterfactual

Putting §8.8 and §8.9 together gives the first positive number this investigation has produced, and it needs its caveats attached to it rather than following it.

For the calm two-thirds at +40d, using the decade-typical drift instead of 2026's:

```
alpha   +134.3   measured on our pool, 86 common days, se 76.7  (t = 1.75)
null      +4.3   the calm two-thirds' +40d drift, 2016-2026, se 19.3
cost     -36.7   one round trip
        --------
net     +101.9   per trade over 40 trading days  =  +2.5 bp per capital-day
```

**Why this is not a result.** It composes an alpha measured over five months of 2026 with a drift measured over a decade — legitimate as arithmetic, but the two terms come from different populations and the alpha term has roughly **two independent 40-day windows** behind it. The drift term ranges from −369 to +827 by year, so "decade-typical" describes no actual year. And `structural` at +40d would give a larger number still (+247.0 alpha), on even less data.

**What it does establish.** The sign of the project's arithmetic is no longer determined by structure. Intraday it is: the drift is negative in 11 of 11 years and no conditioning reaches the toll, so §7 stands. Multi-day it is determined by the sample, and on a decade of drift the terms are close enough together that the answer depends on an alpha we have measured for five months.

That is what makes the labelling question in §8.7 worth funding rather than a hunch: it is the only term in this sum we do not have a decade of.

---

### 8.11 Limit entry, and why the retry that was supposed to fix it breaks it

We send marketable orders on news, so we pay the spread on every entry — 14 to 31 bp by measurement (§2.5), the largest controllable line in the toll. Joining the bid instead of lifting the ask saves it, and at a multi-day horizon the saving stays in the position. scalp-it proposed that the usual objection — you miss the winners — is weak at ten days, because we can simply retry the next day.

**The retry is the part that does not work, and it is the part that was supposed to make it work.**

Alpha in bp, 97,271 pairs over 86 common days, matched null, moving-block errors:

| | n | same day | +5d | +10d | +20d | +40d |
|---|---:|---:|---:|---:|---:|---:|
| **market** (today's policy) | 97,271 | +5.9 | +19.0 | +27.4 | +23.6 | +96.6 |
| **limit, any retry** | 97,271 | **+6.2** | +18.1 | **+26.6** | **+23.2** | **+96.4** |
| filled day 0 | 94,105 | +10.8 | +23.3 | +32.0 | +28.2 | +103.5 |
| **filled on a retry** | 3,166 | **−133.8** | −151.4 | −167.7 | −183.0 | −158.1 |
| through-fill | 93,608 | −0.5 | +11.9 | +19.4 | +13.3 | +84.2 |
| rested-fill | 3,663 | +203.0 | +198.2 | +186.9 | +313.4 | +449.4 |

A retry acquires almost everything — 96.7% on day 0, 99.9% by day 1, 100% by day 3 — and acquires it at **−134 to −183 bp**. The reason is mechanical: a name that did not dip to our price on day 0 and does dip on day 1 is a name that has turned. We do not get the asset we wanted late; we get a different and worse asset. **Limit-with-retry is indistinguishable from market at every horizon** (+6.2 vs +5.9; +26.6 vs +27.4; +96.4 vs +96.6).

So §8.11's predecessor measurement was right for the wrong reason. Its +4.5 bp came from *declining to buy* the names a bid missed, not from buying them cheaper. Skipping is worth ~+4.5 bp before the queue correction and ~+2.8 after; retrying is worth nothing.

**Through-fills are not a special subset — they are the normal case.** 96.2% of fills happen because price traded *through* our limit rather than resting on it, and they run 5 to 10 bp below market at every horizon. scalp-it's short-horizon finding that dips revert (+14 bp against random at five minutes) does not survive to ours.

**Which answers the question this experiment existed to ask.** Does the spread/adverse-selection cancellation decay with horizon? **No.** It holds from the same session out to forty days. The adverse selection in a limit fill is not a short-horizon artifact that a patient holder can wait out.

The one striking cell is the 3.8% of fills that rested — price touched our limit exactly and never went below — at +203 to +449 bp. That is essentially the same population as §8.9's "never dipped" names, and it is not knowable in advance: "the order that filled without the price going through it" is a description of what happened, not a signal.

---

## 9. Reproduction

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
uv run python scripts/eval/edge_spread.py                       # measured touch, by market and hour
uv run python scripts/eval/edge_universe.py --start 2016-09-12  # ten years, by month
uv run python scripts/eval/edge_years.py                        # ten years, by year and horizon
uv run python scripts/eval/edge_horizon.py                      # -> horizon_pairs.jsonl
uv run python scripts/eval/edge_movers.py                       # -> movers.jsonl
uv run python scripts/eval/edge_persist.py --cell --common-days \
    --movers data/eval/edge/movers.jsonl
uv run --with pytest --with pytest-cov pytest tests/unit/test_eval_costs.py --no-cov
```

## 10. Coordination

- `rl-builder` confirmed the lane split and supplied two corrections used above: `minute_bars.trade_value` is exactly `close × volume` (a derived column, not exchange-reported 거래대금), and the `ci-local` pre-push hook leaves a worktree HEAD detached — `git rev-parse HEAD origin/<branch> | uniq -c` must print `2` on one line after every push. Their `dash_forward.py` independently found random tickers beating our own pairs (r_close +21 bp vs +11) on the September window, which is the same conclusion as §2.3 from a different direction.
- The 09:00 batch check in §3.1 was the one the team lead asked to be run once; it was run here and not duplicated.
- Exit knobs `stop_loss_pct`, `take_profit_pct`, `trail_after_gain`, `stop_orb`, `exit_rule_hit` are **not on main** — any result about them describes a research branch's configuration surface, not live.
- 2026-09-16 was deliberately **not** added as an archive day: the live scoring path failed 86 times between 09:07 and 14:58 (Anthropic credit exhaustion), leaving 20 judgments against 147 and 144 on the two real days. It is a degraded day and would contaminate exactly the comparison §2.2 rests on.
