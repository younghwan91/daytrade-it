# Decontaminating the entry label: the features can rank news, but it is not money yet (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal`. Zero API spend, kr_quant read-only,
live repo/daemon/backfill untouched. Monthly time-ordered folds only; LODO banned.

**Verdict: the relabelling hypothesis is half right, and the half that is right does not pay.**

On a σ-normalised post-news-maximum label the 13 decision-time features **can** rank pairs
out-of-sample — the top-ranked decile beats its month's mean by **+1.61σ (se 0.61)**, where the
earlier realized-reward label produced a negative in-sample ceiling. So the label was part of
the problem. But when that same ranking is spent through the live capital budget and scored on
**realized KRW**, it improves on the live ordering by +1,073 KRW/day and **still fails
adoption**: p(μ>0) = 0.837 < 0.90 and 49/84 days ≥ 0 against a floor of 60. Pooled it is
**−185,150 KRW against the live ordering's −295,643** — less bad, not good.

**And the signal is not in the news.** Split by feature family, the LLM judgment features
contribute essentially nothing (−0.06σ on the label, −289,373 KRW realized, barely distinguishable
from the baseline). Everything that ranks is in the price and timing features.

---

## 1. What changed and why

Previous work labelled "was this news good?" with the realized per-pair reward, which bakes in
the exit rule: a good article whose gain was given back is labelled bad. The new label is the
best move available after the decision minute — no exit involved.

Three guards, because the raw maximum is a trap (established on the September window, where
random tickers reached 5.75σ of maximum — see `2026-09-16-dashboard-edge-test.md` §5):

1. **σ-normalised.** The primary label is `mfe / σ`, σ being the stock's 1-minute volatility over
   the ≤30 completed bars **before** the entry minute. The normalisation works: raw
   `corr(σ, mfe_close) = +0.346` falls to `corr(σ, z_close) = −0.093`.
2. **Four horizons** (+30m, +60m, +120m, to 15:00), so the label is not one arbitrary window.
3. **Time-ordered monthly folds.** λ chosen on an inner time-ordered split of the training
   months, never on the evaluation month.

Sample: **4,298 of 4,416 recorded entries** over 95 days, 2026-03..2026-08. The 118 dropped had
no pre-entry bars to measure σ from — these are 09:00 entries, i.e. the premarket batch, which
is where the label is hardest to decontaminate and which is therefore under-represented below.

## 2. The label is not as independent of the exit as hoped

| | |
|---|---|
| `corr(realized, mfe_close)` | **+0.662** |
| `corr(realized, z_close)` | **+0.657** |

This matters and cuts against the strong form of the hypothesis. If the decontaminated label
correlated weakly with realized reward, "the exit destroyed the label" would be a complete
explanation. At 0.66 it is a partial one: a ranking that sorts the maximum well should already
sort the realized reward moderately well. §5 is where that expectation is tested, and it is why
the realized result is not a surprise.

Label scale:

| horizon | mean mfe | mean z | sd z |
|---|---|---|---|
| +30m | +98 bp | 3.15 | 3.39 |
| +60m | +127 bp | 4.13 | 4.89 |
| +120m | +160 bp | 5.23 | 6.04 |
| to close | +202 bp | 6.65 | 7.97 |

## 3. Minutes to the peak — "빨리 먹고 나와야지", with a caveat

| p25 | median | p75 | share peaking within 5 min |
|---|---|---|---|
| 9 min | 47 min | 157 min | **20.8%** |

A fifth of entries make their high within five minutes of the fill, and half within 47 minutes.
That supports a short holding horizon.

**But it is not evidence about our news.** The September null group — random tickers, same days,
same minutes — had a median peak at 27 minutes, i.e. *earlier* than our 47. Fast peaks are a
property of intraday price paths, not of the articles we select. The design implication (do not
hold for the whole session) stands; the "our news spikes early" reading does not.

## 4. Can the features rank the decontaminated label? Yes.

Top-ranked 10% of the evaluation month, minus that month's mean, in σ:

| family | 04 | 05 | 06 | 07 | 08 | **pooled** | se |
|---|---|---|---|---|---|---|---|
| **all13** | +2.92 | +1.54 | +1.44 | −0.19 | +1.57 | **+1.61** | 0.61 |
| price4 | +2.71 | +1.89 | +0.26 | −2.09 | +0.44 | +0.95 | 0.60 |
| **judgment7** | +0.25 | −1.33 | +0.41 | +0.10 | +0.84 | **−0.06** | 0.39 |

`price4` = `pre_entry_move, ret_5m, mkt_move, minutes_since_open`.
`judgment7` = `score, surprise, specificity, sentiment_strength, persistence_one_off,
persistence_multi_quarter, subject_main`.

Same on the raw (un-normalised) label, which flatters everything because it lets the model
select volatility: all13 +113 bp, price4 +124 bp, judgment7 +61 bp. The gap between the two
tables is the volatility channel, and it is why σ-normalisation is not optional.

**Caveat on the se.** It treats pairs as independent. About 24% of reward variance is
day-level, so the effective sample is smaller than 372 and +1.61σ / 0.61 = 2.6 se is optimistic.
Treat §4 as suggestive. §5 uses day-level `bayes_judge`, which does handle the clustering, and
it is the binding result.

## 5. Realized KRW, out-of-sample — the number that decides

Rank by the model **trained on `z_close`**, spend the live budget (top 5/day), score with the
**realized** reward at a flat 100,000 KRW notional. Baseline is the live ordering (chronological
first-5, since live cash runs out long before the candidate list).

| family | 04 | 05 | 06 | 07 | 08 | **pooled** |
|---|---|---|---|---|---|---|
| all13 | −5,463 | −19,904 | −16,809 | −117,308 | −25,665 | **−185,150** |
| price4 | −20,851 | −13,398 | −55,600 | −103,404 | −9,970 | −203,223 |
| judgment7 | −54,770 | −106,236 | +928 | −94,458 | −34,836 | −289,373 |
| **BASE (live order)** | | | | | | **−295,643** |

`bayes_judge` against the live ordering, 84 out-of-sample days:

| family | p(μ>0) | μ/day | days ≥ 0 | worst day | loser ratio | adopt |
|---|---|---|---|---|---|---|
| **all13** | **0.837** | +1,073 | 49/84 | −26,795 vs −49,700 ✓ | 0.605 vs 0.621 ✓ | **false** |
| price4 | 0.759 | +761 | 45/84 | −28,358 ✓ | 0.600 ✓ | false |
| judgment7 | 0.665 | +435 | 44/84 | −34,519 ✓ | 0.633 ✗ | false |

`all13` passes two of four criteria — it beats the baseline's worst day (−26,795 vs −49,700) and
its loser ratio (0.605 vs 0.621). It fails the two that matter most: **p(μ>0) = 0.837 < 0.90**,
and **49/84 days ≥ 0 against a floor of ceil(5/7 × 84) = 60**. Every month is negative; 2026-07
alone is −117,308.

**Not adopted.** The improvement is real (+110,493 KRW over 84 days against the live ordering)
and it is in the right direction, but it is an improvement from losing a lot to losing less.

## 6. The LLM judgment features are not carrying this

Worth stating on its own, because it bears on what the v2 prompt is buying:

- On the decontaminated label, `judgment7` ranks at **−0.06σ** — indistinguishable from random
  ordering.
- On realized KRW it returns **−289,373** against the baseline's −295,643, i.e. a 6,270 KRW
  improvement over 84 days, p(μ>0) = 0.665.
- `price4` alone reaches +0.95σ and −203,223 KRW.
- `all13` beats both, so the judgment features are not worthless in combination — but alone they
  rank nothing.

This does not say the v2 prompt fails to separate tradable from untradable articles; the pool is
already filtered to score > 0.15, so the question here is only whether the judgment features
rank *within* that admitted set. Within it, they do not.

## 7. What this changes and what is next

The relabelling was worth doing: it moved the entry problem from "cannot rank even in-sample"
to "ranks out-of-sample at +1.61σ, worth +1,073 KRW/day, still not adoptable". The remaining
gap is not obviously a ranking problem any more.

The next test is the combination the brief specified: **rank by this model, then apply a learned
exit**, and judge realized KRW. The exit is still the live default in every number above, so
2026-07's −117,308 is being taken with the current stop-less, take-profit-less configuration.
That is the one untested lever left before this line is called.

Order from here: (b) exit search on the 97-day paths with the vectorized-twin exactness gate,
then the (a)+(b) combination, then `add_on_min_gain` across the full day set.

## 8. Artifacts

| what | path (simnode `~/git/daytrade-it-rl`) |
|---|---|
| relabelling + folds + realized verification | `scripts/eval/rl_relabel.py` |
| results | `data/eval/rl_minimal/relabel.json` |
| recorded entry paths (4,416 entries, 97 days) | `data/eval/rl_minimal/paths.jsonl` |

## 9. Concerns

1. **λ in §5 is fixed at 10.0**, not chosen by the inner time-ordered split used in §4. It is
   still never chosen on evaluation data, so the result is out-of-sample, but the two sections
   do not run an identical procedure. Re-running §5 with the inner selection is cheap and should
   be done before any adoption decision.
2. **The 118 dropped entries are 09:00 fills.** The premarket batch is a large share of live
   entries and is systematically under-represented here, because σ needs pre-entry bars. A
   premarket-specific σ (previous session's close-to-close, say) would restore them.
3. **§4's se ignores day clustering** and overstates significance; §5 does not, and §5 is the
   binding number.
4. **The label is never achievable.** Nobody sells at the maximum. It appears here only to rank
   news, and no figure in §4 should be read as P&L.
