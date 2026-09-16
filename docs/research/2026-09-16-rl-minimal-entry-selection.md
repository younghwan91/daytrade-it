# Minimal-data RL for entry selection — a clean negative (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal` off `feat/a-plan` (`6b8f0f1`).
**Zero Anthropic calls, zero Toss, zero Kiwoom. `kr_quant` not touched** (the analysis reads
one existing JSONL file). `~/git/daytrade-it` and the live daemon untouched; nothing committed
in `~/git/daytrade-it-aplan`, whose `data/` was read only.

**Verdict: no selection rule beats the current live policy out-of-sample, and the evidence
says none built on these features ever will.** The ceiling is below zero: a model *fitted on
the very month it is scored on* still loses money on four of six months. This is not a
regularization problem, a feature-count problem, or a sample-size problem that more judged
days would fix. The 13 decision-time features carry almost no information about the per-pair
reward.

---

## 1. Why this could be measured now

Day-level *portfolio* backtests need fully-judged days, and we have zero (Addendum H: median
coverage 29.4%, no day above 34%). But a **per-pair** reward — what one ticker did after one
article — does not depend on which other pairs that day were judged. So a ~30% stratified
sample of each day's pairs is still a usable training set for a *selection* rule: the
conditional mean reward given decision-time features is unbiased under that sampling even
though the day's P&L is not.

That is the whole opening, and it is a real one. It is also the limit of what follows: nothing
here measures capital contention or the concurrent-position cap across a whole day's candidates.
Those still need complete days.

## 2. Reward definition

Reused as-is from `scripts/eval/bandit_dataset.py` — the counterfactual dataset the A7-prep run
already built (`data/eval/bandit_a7prep/dataset.jsonl`), because it is already the live-parity
definition and rebuilding it would have cost hours of replay for an identical answer.

Each day is replayed once with an **uncapped** copy of the live-default policy (position, entry
and exposure caps lifted, SimBroker cash 10¹², bandit off) so that every candidate reaching
`trader.enter` trades one share-sized entry independently of the others. For each such
candidate:

> **r = net / buy_value** of that ticker's trade — entry at the decision minute's executable
> price (ask-side limit, 5-minute TTL, fill when a minute's low ≤ the limit), exit under the
> current live exit rules (VWAP break, +2% target, 15:00 liquidation), net of the modeled costs
> (0.20% sell tax + 0.015% commission per side). 0.0 if entered or deferred but never filled.

Features come from `news_entry.bandit_features` — the live verdict's own function — evaluated at
the pair's `decide_at` on a fresh broker over that day's data: only completed bars strictly
before the decision minute, and the 069500 quote at that instant. Live parity holds because the
live path and the replay call the same function.

**What r does not capture** — stated plainly because it bounds every number below:

- **Capital contention.** One share each, no competition for cash. Live has ~5 concurrent slots
  and a 1,000,000 KRW exposure cap; in the A7-prep baseline 6,531 of 10,585 skips were
  `concurrent position cap reached` or `insufficient deposit cash`. A rule that looks good per
  pair may be unreachable once the day's cash is gone.
- **The 30% sample.** Per-pair conditional means survive it; per-day *rankings* do not. A
  "top-5 of the day" computed on ~46 judged pairs is not the top-5 of the ~150 the live daemon
  would have seen.
- **Universe vintage.** The dataset was built on the pre-N=300 (top-600) universe events
  (`replay_events_universe`, 97 days, 2026-03-17..2026-08-14). N=300 was adopted afterwards
  (Addendum D). Rows are conditioned on passing the live selection filters either way, so the
  conclusion transfers, but the exact bp figures are for the older pool.

Dataset: **4,496 trainable pairs over 97 days**, 2026-03-17..2026-08-14, ~46/day.

## 3. The base rate

| | |
|---|---|
| mean reward | **−94 bp** per pair |
| win rate | **27.4%** (1,230 / 4,496) |
| average win / average loss | +179 bp / −198 bp |
| positive days | **20 / 97** |
| day share of reward variance | 0.24 |

Monthly: Mar −60, Apr −86, May −90, Jun −88, Jul −170, Aug −56 bp. Negative every month.

A selection rule has to close a 94 bp gap before it makes its first won.

## 4. Do any features move with the reward?

`scripts/eval/rl_diag.py`, month by month, on disjoint months. Spearman ρ(feature, reward):

| feature | 03 | 04 | 05 | 06 | 07 | 08 | pooled | sign stability |
|---|---|---|---|---|---|---|---|---|
| `score` | +0.005 | −0.012 | −0.004 | −0.061 | −0.093 | +0.035 | −0.025 | 0.67 |
| `surprise` | −0.039 | +0.158 | +0.004 | −0.159 | −0.067 | −0.062 | −0.059 | 0.67 |
| `specificity` | +0.007 | +0.010 | −0.035 | −0.098 | −0.014 | +0.026 | −0.012 | 0.50 |
| `sentiment_strength` | +0.059 | +0.133 | +0.006 | −0.162 | −0.041 | +0.040 | −0.032 | 0.67 |
| `persistence_one_off` | −0.048 | +0.123 | −0.043 | −0.092 | +0.061 | +0.025 | +0.052 | 0.50 |
| `persistence_multi_quarter` | −0.061 | +0.003 | −0.023 | +0.020 | −0.049 | −0.085 | −0.025 | 0.67 |
| `persistence_structural` | −0.012 | +0.093 | −0.029 | +0.027 | +0.021 | +0.005 | −0.030 | 0.67 |
| `subject_main` | +0.112 | −0.039 | −0.071 | +0.008 | −0.029 | −0.101 | −0.042 | 0.67 |
| `pre_entry_move` | +0.050 | −0.054 | −0.087 | +0.011 | +0.089 | −0.031 | +0.004 | 0.50 |
| `premarket` | −0.101 | −0.153 | −0.121 | +0.058 | −0.136 | −0.032 | −0.062 | **0.83** |
| `minutes_since_open` | −0.068 | +0.084 | +0.167 | +0.044 | +0.114 | −0.022 | +0.074 | 0.67 |
| `ret_5m` | −0.066 | +0.047 | +0.036 | +0.031 | −0.002 | −0.020 | +0.018 | 0.50 |
| `mkt_move` | +0.126 | −0.044 | −0.054 | −0.031 | +0.096 | −0.086 | +0.038 | 0.67 |

No |ρ| reaches 0.17 in any single month; the largest pooled |ρ| is 0.074. Nine of thirteen
features are at 0.50–0.67 sign stability, which for six months is a coin flip. The two that
look stable — `premarket` negative (5/6) and `minutes_since_open` positive (4/6) — are the same
signal wearing two hats: premarket articles are decided at 09:01, so `minutes_since_open` ≈ 0.

Tercile spread (high tercile mean − low tercile mean) for `minutes_since_open`, the best
feature in the table: −39, +35, +83, +15, +88, −7 bp — averaging ~+29 bp against the 94 bp that
has to be closed, and flipping sign in two of six months. That is the strongest feature
available, it covers at most a third of the gap in its good months, and section 7 shows even
that third is dispersion shrinking rather than better picks.

## 5. Time-ordered folds

`scripts/eval/rl_folds.py`. Fold *m* trains on every row strictly before month *m* and
evaluates on month *m*. No fold shares a training row with its evaluation set — unlike LODO,
where 40 folds share 39/40 of their training days and are one model forty times over (Addendum
H1 established that; this run does not repeat the mistake).

Two readouts:

- **Gate** — mean reward of the pairs the rule enters, against the month's pool mean. Robust to
  the 30% sampling.
- **Top-5/day** — the fixed-capital comparison, at 100,000 KRW flat notional per position (the
  A7-prep baseline's −465,246 KRW over 456 trades at −0.94% implies a ~108,500 KRW average buy
  value, so the scale is calibrated). Two no-skill baselines at the **same trade count**:
  `first5` (chronological — closest to live, whose cash runs out long before its candidate list)
  and `rand5` (5 × the day's mean, the expected value of a random pick). Comparing a rule
  against the whole pool would let it "win" by trading nine times less; that comparison is not
  reported.

Rule families, each trained per fold: the existing 13-feature bandit; reduced 5/3/2-feature
versions; a monotone score (each feature standardized and signed by its train-set correlation,
no fitted magnitudes); and two hand-fixed time-of-day thresholds. λ is chosen inside each fold
on a time-ordered inner split (the last quarter of the training days), never on the evaluation
month.

### Per-fold gate delta (bp, selected mean − pool mean; `sel` = pairs entered)

| rule | 04 sel/Δ | 05 sel/Δ | 06 sel/Δ | 07 sel/Δ | 08 sel/Δ |
|---|---|---|---|---|---|
| `bandit13` | 134 / +1 | 2 / +63 | 0 / — | 0 / — | 0 / — |
| `bandit5` | 23 / +18 | 7 / +125 | 9 / −118 | 0 / — | 0 / — |
| `bandit3` | 13 / −73 | 3 / +217 | 8 / −92 | 0 / — | 0 / — |
| `bandit2` | 8 / −33 | 0 / — | 1 / −443 | 0 / — | 0 / — |
| `mono3` | 646 / −15 | 541 / +7 | 481 / −3 | 350 / +18 | 216 / +3 |
| `mono2` | 548 / −14 | 470 / +23 | 451 / +10 | 337 / +25 | 202 / +5 |
| `intraday_only` | 384 / +25 | 306 / +31 | 359 / −10 | 302 / +67 | 152 / −12 |
| `after_10am` | 403 / +12 | 397 / +35 | 356 / −2 | 291 / +46 | 139 / −10 |
| `after_11am` | 172 / +48 | 176 / +55 | 157 / −31 | 127 / +124 | 67 / −11 |

The fitted bandit **abstains**: 0 rows entered in three of five folds, and the deltas that look
large (`bandit3` +217 bp) rest on three rows. This is the same degenerate abstention A7-prep
found under LODO, reproduced under an honest split — λ is driven to the top of the grid, every
posterior mean lands near zero, and `decide` never fires. The rules that do trade in volume
(`mono*`, the time thresholds) move the mean by −15 to +67 bp against a pool that is −56 to
−170 bp. None crosses zero.

### Pooled out-of-sample, 86 evaluation days

| rule | pairs entered | gate mean | top-5 net (KRW) | bp/trade |
|---|---|---|---|---|
| `bandit13` | 136 | −83 | −282,140 | −66 |
| `bandit5` | 39 | −81 | −323,760 | −75 |
| `bandit3` | 24 | −130 | −337,237 | −78 |
| `bandit2` | 9 | −165 | −266,379 | −62 |
| `mono3` | 2,234 | −98 | −324,515 | −75 |
| `mono2` | 2,008 | −90 | −277,561 | −65 |
| `intraday_only` | 1,503 | −79 | −455,318 | −106 |
| `after_10am` | 1,586 | −81 | −216,996 | **−50** |
| `after_11am` | 699 | −60 | −216,996 | **−50** |
| **BASE `first5`** (≈ live) | — | — | −547,125 | **−127** |
| **BASE `rand5`** (no skill) | — | — | −393,735 | −92 |

Every rule loses money on every fold. The best, −50 bp/trade, still burns half a percent per
entry.

One thing in this table is worth keeping even though no rule passes: **`first5` (−127 bp) is
worse than `rand5` (−92 bp)**. Live's "take whatever arrives first" ordering is 35 bp/trade
worse than picking at random within the day. That is a property of the current policy, not a
new rule, and it is the only unambiguous finding here.

## 6. The ceiling: even in-sample fitting loses

The decisive test. Fit the 13-feature bandit on a month and score its own top-5/day on that
same month — an in-sample number, reported only as an upper bound:

| month | pool | in-sample best-5/day | oracle top-5 (perfect hindsight) |
|---|---|---|---|
| 2026-03 | −60 | **+42** | +293 |
| 2026-04 | −86 | **−6** | +245 |
| 2026-05 | −90 | **−13** | +324 |
| 2026-06 | −88 | **−28** | +353 |
| 2026-07 | −170 | **−93** | +286 |
| 2026-08 | −56 | **+54** | +357 |

The oracle column proves the spread is there: a perfect selector would earn +245 to +357 bp per
trade. The in-sample column proves the features cannot find it — with hindsight, with no
regularization penalty to pay, fitting and scoring on the same data, the model is negative on
four months of six.

**That closes the line of work the brief asked for.** Out-of-sample performance cannot exceed
the in-sample ceiling, so none of the minimal fixes on the list — fewer features, stronger
shrinkage, coarser buckets, a monotone 2–3 feature rule — can lift a family whose in-sample
optimum is already below zero. Every one of those is a *restriction* of the linear family
tested here, and restricting a family lowers its ceiling. More judged days would sharpen the
estimate of a negative number; they would not make it positive.

The strict claim is about the linear/monotone family over `bandit.FEATURES`. It does not
formally exclude a nonlinear model finding an interaction a linear fit misses. But with every
single-feature |ρ| under 0.17 in every month and six of thirteen coefficient signs at coin-flip
stability, a richer model on 4,496 rows would be fitting noise with more parameters — which is
the failure mode the spec's "머신러닝은 표본 대비 위험하다" rule exists to prevent.

## 7. Where `after_10am`'s apparent edge comes from

`after_10am` reached p(μ>0) = 0.988 against `first5`, which would look like a candidate if the
mechanism were not checked. It is not selection skill. Reward by time of day:

| bucket | n | mean | sd | mean\|r\| | win rate | p5 | p95 |
|---|---|---|---|---|---|---|---|
| 09:00–10:00 | 2,643 | −105 | 295 | 220 | 0.29 | −570 | +349 |
| 10:00–11:00 | 1,033 | −92 | 236 | 175 | 0.25 | −465 | +213 |
| 11:00–13:00 | 433 | −69 | 205 | 141 | 0.25 | −336 | +196 |
| 13:00–14:30 | 367 | −57 | 141 | 99 | 0.22 | −232 | +162 |

The mean improves from −105 to −57 bp, but the standard deviation falls from 295 to 141 and
mean |r| from 220 to 99 — the entire move is the distribution shrinking toward zero as the
15:00 liquidation approaches and positions have less time to travel. The **win rate falls**
(0.29 → 0.22), and mean/sd barely moves (−0.36 → −0.40), i.e. risk-adjusted it is marginally
*worse*. "Trade later" buys a smaller loss by taking a smaller position in time, which is the
same thing the A7-prep stage-2 optimizer did ("the optimizer's whole gain is trading less").
Shrinking exposure is available directly, without a model, and should not be sold as a learned
policy.

## 8. Readiness criteria, applied honestly

Spec `2026-09-16-bandit-entry-design.md`, criteria as restated in Addendum H3.

| criterion | verdict | evidence |
|---|---|---|
| **1. λ stable across time-ordered folds (≤ one grid step)** | **FAIL** | `bandit13` λ = 30, 1000, 1000, 1000, 100 — both ends of the grid. `bandit5`/`bandit3`/`bandit2`: 100, 1, 1, 1000, 1. Worse than the LODO figure, as expected once folds stop sharing training data. |
| **2. coefficient signs stable across time-ordered folds** | **FAIL** | Only `pre_entry_move` (−) and `premarket` (−) hold 5/5. `score`, `specificity`, `minutes_since_open`, `ret_5m`, `mkt_move`, `persistence_multi_quarter` at 4/5. **Six of thirteen at 3/5 — a coin flip**: `surprise`, `sentiment_strength`, `persistence_one_off`, `persistence_structural`, `subject_main`. |
| **3. `bayes_judge` adopt** | **FAIL** | `adopt=false` for all 9 rules against both baselines. |
| **4. forward 5 trading days agree** | not reached | no candidate to carry forward. |

### `bayes_judge`, top-5/day, 86 out-of-sample days

Against `first5` (the live-like baseline):

| rule | P(μ>0) | μ (KRW/day) | days ≥ 0 | worst day | losers | adopt |
|---|---|---|---|---|---|---|
| `after_10am` | 0.988 | +2,739 | 53/86 | −12,401 | 0.756 | **false** |
| `mono2` | 0.982 | +2,376 | 53/86 | −32,859 | 0.719 | **false** |
| `bandit13` | 0.982 | +2,017 | 52/86 | −35,256 | 0.686 | **false** |
| `bandit5` | 0.976 | +2,018 | 55/86 | −37,924 | 0.702 | **false** |
| `bandit2` | 0.973 | +2,224 | 51/86 | −32,859 | 0.726 | **false** |
| `mono3` | 0.957 | +1,887 | 50/86 | −36,494 | 0.712 | **false** |
| `bandit3` | 0.924 | +1,508 | 50/86 | −37,842 | 0.721 | **false** |
| `intraday_only` | 0.674 | +434 | 45/86 | −37,488 | 0.744 | **false** |
| baseline | — | — | — | −61,888 | 0.695 | — |

(`after_11am` is omitted: it picks the same top-5 as `after_10am` and its row is identical.)

Eight of the nine rules clear criterion 1 (P(μ>0) ≥ 0.90) against `first5` — everything but
`intraday_only`. **Every one fails the day-count floor** (ceil(5/7 × 86) = 62; the best is
`bandit5` at 55), and on the loser-ratio gate only `bandit13` (0.686) beats the baseline's
0.695 — the other eight are worse. Against the no-skill `rand5` baseline the probabilities collapse —
`bandit3` 0.756, `mono3` 0.793, `bandit5` 0.877, `intraday_only` 0.017 — confirming that most
of the apparent gain against `first5` is just "do not systematically take the worst-ordered
candidates", not selection.

`after_10am` and `bandit13` do beat the baseline's worst day (−12,401 and −35,256 vs −61,888)
and `bandit13` slightly beats its loser ratio. Neither earns anything; both simply lose less.

## 9. Candidate policy to hand forward

**None.** There is nothing to hand to the complete-day verification when the batch lands.

Recommending `after_10am` because it clears P(μ>0) would be the exact failure the brief warned
about: it fails two of four criteria, its mechanism is position shrinkage rather than
selection, and it is risk-adjusted flat. Handing it over would spend the complete-day budget
confirming a −50 bp/trade policy.

## 10. What this does and does not settle

Settled, on 97 days and 4,496 pairs:

1. The live candidate pool loses **−94 bp per pair**, 27% win rate, negative every month.
2. No decision-time feature has a month-stable relationship with that reward; the strongest
   moves the mean ~29 bp and flips sign twice in six months.
3. **The in-sample ceiling is negative.** These features cannot be assembled into a profitable
   entry rule by any method, which makes further model search on this feature set pointless.
4. The live chronological ordering is 35 bp/trade worse than random within the day.

Not settled:

- Whether the *news edge itself* exists on a fully-judged universe day. Every number here is
  conditioned on the current pool with ~30% of pairs judged; the A7-prep reading (a) — "the
  dashboard pool was a selected pool and the edge does not survive widening" — is still the
  open question, and per-pair rewards cannot close it.
- Whether a **different feature set** would work. Everything tested here lives inside
  `bandit.FEATURES`. The exit rule, the entry limit placement, and the stop are all untested
  levers, and Addendum H's point stands: the per-pair reward is dominated by the exit, and a
  27% win rate against a −198 bp average loss says more about where the stop sits than about
  which article was picked.

The honest next lever is **not** a better selector over these 13 features. It is either (a)
complete days, to find out whether the pool has an edge at all, or (b) the exit and stop rules,
where a 27%/−198 bp asymmetry suggests the damage is being done after entry, not at it.

## 11. Artifacts

| what | path (simnode `~/git/daytrade-it-rl`) |
|---|---|
| univariate diagnostics | `scripts/eval/rl_diag.py` |
| time-ordered fold evaluation | `scripts/eval/rl_folds.py` |
| fold output (console) | `data/eval/rl_minimal/folds.txt` |
| fold output (JSON, per-rule per-fold + verdicts) | `data/eval/rl_minimal/folds.json` |
| input dataset (read-only, built by A7-prep) | `~/git/daytrade-it-aplan/data/eval/bandit_a7prep/dataset.jsonl` |

Reproduce:

```
cd ~/git/daytrade-it-rl
PYTHONPATH=scripts/eval uv run python scripts/eval/rl_folds.py \
    --dataset ~/git/daytrade-it-aplan/data/eval/bandit_a7prep/dataset.jsonl \
    --out data/eval/rl_minimal/folds.json
uv run python scripts/eval/rl_diag.py \
    --dataset ~/git/daytrade-it-aplan/data/eval/bandit_a7prep/dataset.jsonl \
    --out data/eval/rl_minimal/diag.json
```

## 12. Concerns

1. **The dataset is the older top-600 universe vintage.** N=300 was adopted after it was built.
   Rebuilding costs hours of replay and the section-6 ceiling argument does not depend on the
   pool's width, so it was not rebuilt. If the N=300 pool is later shown to have a materially
   different reward distribution, section 3's base rate would need re-measuring — section 6's
   conclusion would not.
2. **Three days were excluded by `bandit_dataset`'s all-or-nothing error guard** (06-19, 07-21,
   07-27 — limit-lock `NO BID` exits in the uncapped replay). Inherited from A7-prep; noted
   there as worth classifying separately rather than dropping the day.
3. **86 evaluation days is the fold count, not 86 independent observations.** Day-level rewards
   carry 24% of their variance at the day level, so the effective sample for the `bayes_judge`
   day series is smaller than 86. This makes the failures *more* secure, not less — the
   P(μ>0) values above are, if anything, overstated.
4. **`after_10am` will keep reappearing.** Any search that optimizes a mean without checking
   dispersion will rediscover it, and it will keep clearing P(μ>0). The diagnostic in section 7
   should be run on any future candidate before it is called a selection rule.
