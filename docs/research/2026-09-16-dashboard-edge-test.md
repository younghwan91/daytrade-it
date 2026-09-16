# Is Toss's dashboard selection itself the edge? (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal`. **Zero Anthropic spend**, zero Toss,
zero Kiwoom. kr_quant read with `default_transaction_read_only = on` throughout — query only,
never written. `~/git/daytrade-it`, the live daemon and the minute-bar backfill untouched;
nothing committed in `~/git/daytrade-it-aplan`.

**Verdict: no detectable edge from dashboard membership, and the test the brief specified
cannot be run at all on current data.** The specified live-parity comparison yields **n = 0**.
A judgment-free proxy that sidesteps both blockers does run, on 327 vs 14,666 pairs, and finds
**+13 bp ± 23 bp** — indistinguishable from zero, against the ~64 bp this sample could have
detected. What dashboard membership does change is **dispersion**: sd 415 bp vs 218 bp at a
*lower* win rate. That is a more-volatile-names signature, not a better-news signature.

---

## 1. The overlap window

`news_articles` holds dashboard-sourced articles (`source = 'toss'`) for **2026-09-01..09-16,
1,346 rows** in that range. The sweep pool (`universe_pairs_n300.jsonl`) covers trading days
through 09-15.

Trading days where both pools are present: **10** — 09-01, 09-03, 09-04, 09-07, 09-08, 09-09,
09-10, 09-11, 09-14, 09-15. Excluded: weekends (09-05, 06, 12, 13), 09-16 (the sweep has not
run), and **09-02, where kr_quant holds no dashboard article at all** — a one-day collection
gap worth someone checking separately.

## 2. Containment: 17% headline, 75% on the assumption that matters

Matched on the live article key. `news_articles.id` is already stored in `toss:<md5(url)[:12]>`
form, so the key is `uuid5(TOSS_NEWS_NAMESPACE, id)`; re-deriving the digest from the stored
url reproduced the id on **1,346 of 1,346** rows, so nothing below is a key-derivation artifact.

Containment was measured against the **full sweep pool before judging**, not against the judged
events — matching on judged events would conflate "the sweep never saw this article" with "the
sweep saw it but it was not in the ~30% sampled for judging".

| bucket | n | share |
|---|---|---|
| no ticker mapping in kr_quant | 620 | 0.461 |
| tickers all outside the N=300 universe | 254 | 0.189 |
| **in the sweep pool** | **232** | **0.172** |
| day not swept (weekend / 09-16) | 163 | 0.121 |
| has a universe ticker but the sweep never saw it | 77 | 0.057 |

The 17.2% headline is misleading and should not be quoted on its own. **46% of dashboard
articles are not poolable in principle** — they carry no ticker at all (the samples are general
market copy: "코스피, '네 마녀의 날' 경계감에…", "로빈후드 CEO…"), and another 19% are about
tickers outside the top-300 universe.

The number that actually tests the brief's assumption is containment *conditional on being
poolable*: of the 309 dashboard articles on swept days that have a universe ticker, the sweep
pool carries **232 — 75.1%**. So the containment assumption is broadly sound; the sweep feed's
genuine miss rate is 5.7% of all dashboard articles, 25% of the poolable ones.

**What this already says about A-plan:** the sweep universe and the dashboard are largely
*different objects*. Two thirds of what the dashboard surfaces could never enter the sweep pool
as constructed, mostly because it is not about one top-300 company. Any expectation that
widening the sweep reproduces the dashboard's behaviour has to contend with that first.

## 3. The specified test yields n = 0

Step 3 asked for per-pair rewards under the live-parity counterfactual, split by membership.
Two independent blockers stop it:

**Blocker A — the minute-bar backfill.** Of the 10 overlap days, 7 fail `prepare_day`'s
prev-close coverage gate (>5% of event tickers have no prior-session bar): 09-01 (9/135),
09-08 (26/143), 09-09 (34/145), 09-10 (32/139), 09-11 (30/153), 09-14 (29/150), 09-15 (20/149).
Of the remaining three, 09-03 (28/147) and 09-07 (24/160) then fail the missing-bars gate inside
the replay. Only **09-04** replays cleanly — and it produces **0 entries**, because no candidate
that day cleared the live gates and filled.

```
2026-09-03: SKIPPED -- 28/147 event tickers have no minute bars (> 5%)
2026-09-04: 0 entries
2026-09-07: SKIPPED -- 24/160 event tickers have no minute bars (> 5%)
0 entries -> data/eval/rl_minimal/paths_sept.jsonl
```

**Blocker B — the ~30% judging sample, which is the binding one.** Even with every day
replayable, only **31 dashboard pairs are judged** in the sweep events on those days, against
**62,839 judged sweep pairs**. At the historical rate at which judged pairs become filled
entries (4,496 / 78,859 ≈ 5.7%), 31 judged pairs is an expected **1 to 2 entries**. Fixing the
backfill alone would not fix this; the stratified 30% sample was drawn without regard to
dashboard membership, so it thins the treatment group exactly as hard as the control.

**`news_judgments` does not rescue it.** kr_quant does hold judgments covering the window
(~50–90 news rows/day), but they are `prompt_version = v1` with a different schema
(`event_type`, `price_impact_likely`, `confidence`) — not the v2 `ArticleFeatures` the live
`score_article` produces. Using them would break live parity at the first step.

So the brief's step 3/4 comparison is reported as **not runnable**, not as a null result.

## 4. A judgment-free test that does run

Both blockers are about *judgments and replay*, not about prices. The economic core of the
question — are dashboard-selected articles better news? — can be asked with prices alone: for
every (article, ticker) pair on those days, the stock's forward return from the first completed
minute at or after publication (pre-open publications enter at 09:00, as the live premarket
batch does). Same days, same construction for both groups.

**This is a proxy and the gaps are large:** no entry filter (the live policy trades a pair only
if the v2 score clears 0.15 and the stock has not already run), no exit rule, no fill model, no
costs (~23 bp of tax and commission alone), no concurrency or caps. It can settle the direction
and rough size of a membership effect. It cannot price one.

| group | n | r+30m | r+60m | r+120m | r→15:00 | win@close | sd(r→close) |
|---|---|---|---|---|---|---|---|
| dashboard (all, incl. non-universe) | 433 | +23 | +36 | +49 | **+55** | 0.416 | 521 |
| **dashboard ∩ sweep** | **327** | −5 | +8 | +15 | **+23** | 0.401 | 415 |
| **sweep-only** | **14,666** | +4 | +2 | +8 | **+11** | 0.441 | 218 |

All figures in bp. The apples-to-apples rows are the middle two — same days, same universe,
same construction. The top row is higher but not comparable: it includes the non-universe
tickers, which are smaller and more volatile.

**Observed gap: +13 bp. Standard error of that gap: 23 bp.** 80% power at 5% would need a true
gap of about **64 bp**. The comparison cannot distinguish the groups, and +13 bp should be read
as "no measurable difference", not as a small positive.

`bayes_judge` at the day level (mean forward return per day, 9 days with both groups present,
`prior_sd` 50 bp to match the bp scale):

```
p(mu>0) = 0.389   mu = -9.0 bp   days>=0 = 4/9   adopt = false
  - p_mu_gt_0=0.3889 < 0.90
  - worst_day_candidate=-190.8 < worst_day_baseline=-106.1
  - loser_ratio_candidate=0.4037 > loser_ratio_baseline=0.3914
  - days_ge_0=4 < ceil(5/7*9)=7
```

Note the day-level mean is *negative* (−9.0 bp) where the pair-level mean is positive (+13 bp):
weighting days equally rather than pairs equally flips the sign. An effect that changes sign
under a reweighting that mild is not an effect.

## 5. What did differ: dispersion, not mean

The one clearly non-null difference is second-moment. Dashboard-present pairs have **sd 415 bp
against sweep-only's 218 bp** — close to double — while their **win rate is lower** (0.401 vs
0.441). Dashboard selection is picking names that move more in both directions, slightly more
often down.

For a strategy with a stop and a 15:00 flatten that is not automatically good news, and it is a
plausible mechanical explanation for why the 7-day dashboard backtest looked strong: higher
dispersion with the same slight positive drift produces a wide spread of daily outcomes, and 7
days is few enough to land on a good draw. It does not establish that; it offers a reading that
+39,915 KRW over 7 days cannot rule out.

## 6. Answering the brief

| step | result |
|---|---|
| 1. overlap window | 10 trading days, 09-01..09-15; 09-02 has no dashboard rows (collection gap) |
| 2. containment | **17.2% overall; 75.1% conditional on being poolable.** Key derivation verified on 1,346/1,346 |
| 3. two-group per-pair reward | **not runnable — n = 0** (7 days fail the bar backfill, 2 more fail in-replay, the 1 clean day has no entries; and only 31 dashboard pairs are judged at all) |
| 4. `bayes_judge` | on the price proxy: `adopt = false`, p(μ>0) = 0.389. Explicitly underpowered: se 23 bp, detectable gap ~64 bp, observed +13 bp |
| 5. is membership predictable? | **not attempted** — the brief gated this on the edge being real, and it is not detectable |

**Recommendation on A-plan spend:** this is a clean negative on the specific hypothesis, and it
argues against buying more sweep-pool judging days *for the purpose of chasing the dashboard's
7-day result*. It does not close the question, because the proxy has no entry filter and so
cannot rule out an interaction between membership and the v2 score.

The cheap way to close it properly, when spend is available again: **judge the ~232
dashboard ∩ pool articles' pairs directly** (roughly 378 pairs on these 10 days) instead of
waiting for a stratified sample to cover them. That is targeted, small, and it is the only thing
that turns step 3 from n = 0 into a real comparison. It also needs the minute-bar backfill to
reach 09-01..09-15 first — currently ~1,050 codes/day there against ~1,290 in late August.

## 7. Artifacts

| what | path (simnode `~/git/daytrade-it-rl`) |
|---|---|
| containment | `scripts/eval/dash_overlap.py` → `data/eval/rl_minimal/dash_overlap.json` |
| gap decomposition | `scripts/eval/dash_decomp.py` → `data/eval/rl_minimal/dash_decomp.json` |
| forward-return test | `scripts/eval/dash_forward.py` → `data/eval/rl_minimal/dash_forward.json` |
| live-parity attempt (empty, deliberately kept) | `data/eval/rl_minimal/paths_sept.jsonl` |
| usable/attempted day lists | `data/eval/rl_minimal/days_sept.txt`, `days_sept_ok.txt` |

## 8. Concerns

1. **The proxy has no entry filter.** The live edge, if any, would come from membership *and* a
   high v2 score together. This design cannot see that interaction, and says so rather than
   claiming to have ruled it out.
2. **10 days, 9 usable for the day-level judge.** Thin. The pair-level n (327 vs 14,666) is what
   carries the power statement; the day-level `bayes_judge` is reported for consistency with the
   adoption rules, not because 9 days settles anything.
3. **`news_article_tickers` is kr_quant's mapping, not the live daemon's.** For the
   dashboard-only rows I had no alternative. The dashboard ∩ sweep group uses the *sweep's* own
   ticker mapping and publication time, so the two compared groups are constructed identically;
   only the "dashboard (all)" row mixes mappings, which is one more reason not to read it.
4. **09-02's missing dashboard rows** were treated as a collection gap and the day dropped. If
   that gap is systematic rather than a one-off, the window is 9 days, not 10.
