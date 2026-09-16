# Is Toss's dashboard selection itself the edge? (2026-09-16)

simnode, `~/git/daytrade-it-rl`, branch `feat/rl-minimal`. **Zero Anthropic spend**, zero Toss,
zero Kiwoom. kr_quant read with `default_transaction_read_only = on` throughout — query only,
never written. `~/git/daytrade-it`, the live daemon and the minute-bar backfill untouched;
nothing committed in `~/git/daytrade-it-aplan`.

**Verdict: no edge from dashboard membership.** The live-parity comparison the brief specified
cannot be run at all — it yields **n = 0**. A judgment-free test does run, on 327 vs 14,666
pairs, and on the *decontaminated* news-quality label (post-article maximum) dashboard
membership first looks like a decisive win: **+102 bp, `bayes_judge` p(μ>0) = 0.963, 9/9 days,
adopt = true**.

That win is an artifact. Dashboard-selected names carry **twice the pre-entry minute volatility**
(41 bp vs 21 bp), and a maximum scales with volatility mechanically. Measured per unit of
volatility, dashboard is **−0.15σ against sweep-only and −0.46σ against random tickers**, and
`bayes_judge` flips to **adopt = false, p(μ>0) = 0.432**. Had dashboard articles merely been
*sweep-only quality* at their own volatility, their maximum would have been **+294 bp**; the
observed figure is **+252 bp**, i.e. below what their volatility alone implies.

This is the brief's **third outcome**: no edge. The dashboard pool's +39,915 KRW over 7 days is
most economically explained as a volatility draw, and the live policy's evidence base is weaker
than we have been treating it.

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

## 4. Retraction: the MFE framing in my earlier note was wrong

I previously offered "mean MFE +154 bp against −51 bp realized" as motivation for exit work.
**That was not evidence and I withdraw it.** A maximum over a path is at or above the entry for
any stock and any entry, so the gap is partly just what maxima do. Every MFE figure below is
therefore paired with a **null group**: entries on the same days, at the same minutes, on random
tickers from that day's tradable universe (3 draws per real pair, 37,100 draws).

The null settles it immediately. Random entries reach **mfe_close +204 bp** — and our sweep-pool
pairs reach only **+150 bp**, i.e. **53 bp below random** (se 2 bp). Our entries do not have
unusual unrealized upside; they have *less* than random draws from the same universe at the same
minutes. The "we are giving back large gains" story is retired.

## 5. A judgment-free test that does run

Both blockers in §3 are about judgments and replay, not prices. So: for every (article, ticker)
pair on those days, the stock's forward path from the first completed minute at or after
publication (pre-open publications enter at 09:00, as the live premarket batch does). Same days,
same construction, three groups.

**This is a proxy.** No entry filter, no exit rule, no fill model, no costs (~23 bp of tax and
commission alone), no concurrency or caps. It settles direction and size, not price.

### Endpoint return (bp)

| group | n | r+30m | r+60m | r+120m | r→15:00 | win |
|---|---|---|---|---|---|---|
| dashboard ∩ sweep | 327 | −5 | +8 | +15 | +23 | 0.401 |
| sweep-only | 14,702 | +3 | +2 | +8 | +11 | 0.441 |
| **NULL (random)** | 37,100 | +10 | +10 | +16 | **+21** | 0.426 |

Note the null already beats the sweep pool on the endpoint too (+21 vs +11).

### Post-entry maximum — a RANKING label, never P&L (bp)

| group | mfe+30m | mfe+60m | mfe+120m | mfe→close | **excess over null** |
|---|---|---|---|---|---|
| dashboard ∩ sweep | +135 | +167 | +205 | +252 | **+48** (se 23) |
| sweep-only | +83 | +101 | +120 | +150 | **−53** (se 2) |
| NULL (random) | +114 | +141 | +167 | +204 | — |

### The volatility control, which decides it

| group | n | pre-entry σ | sd(r→close) | **mfe/σ** | r/σ |
|---|---|---|---|---|---|
| dashboard ∩ sweep | 204 | 41 bp | 415 bp | **5.29** | 1.01 |
| sweep-only | 10,537 | 21 bp | 218 bp | **5.44** | 0.36 |
| NULL (random) | 26,590 | 25 bp | 312 bp | **5.75** | 0.45 |

dashboard − sweep-only: **−0.15σ** (se 0.68). dashboard − null: **−0.46σ** (se 0.68).

Two things follow, and the second is the more important one.

1. **Dashboard's raw MFE advantage is entirely a volatility effect.** Its names have ~2× the
   minute volatility. Scaling sweep-only's +150 bp by that ratio predicts +294 bp; the observed
   +252 bp is *below* it. Per unit of risk, dashboard articles are not better news.
2. **All three groups sit at 5.3–5.8 σ of maximum**, random tickers included. That narrow band
   *is* "what a maximum does". Once volatility is divided out, the post-article maximum carries
   almost no cross-sectional information — which is a direct warning about relabelling entry
   selection with MFE: a selector trained on raw MFE will mostly learn *which stock is volatile*,
   not *which news is good*. Any MFE-labelled selector must be σ-normalised from the start.

### Minutes from entry to the peak

| group | p10 | p25 | median | p75 | p90 | mean |
|---|---|---|---|---|---|---|
| dashboard ∩ sweep | 0 | 0 | 14 | 177 | 319 | 94 |
| sweep-only | 0 | 0 | 34 | 162 | 275 | 89 |
| NULL (random) | 0 | 0 | 27 | 148 | 262 | 83 |

The distribution is strongly bimodal: a quarter of entries peak in the *first minute*, and the
upper quartile not until 2.5–5 hours later. "빨리 먹고 나와야지" is supported for the early mode —
but the same shape appears in the random group (median 27 min), so it is a property of intraday
price paths, not of our news.

### `bayes_judge`, dashboard ∩ sweep vs sweep-only (9 days with both present)

| label | p(μ>0) | μ | days ≥ 0 | pair-level gap ± se | adopt |
|---|---|---|---|---|---|
| realized endpoint | 0.389 | −9.0 bp | 4/9 | +13 ± 23 bp | **false** |
| post-entry maximum | **0.963** | +56.6 bp | 9/9 | +102 ± 23 bp | **true** |
| **maximum, in σ** | 0.432 | −0.1 | 4/9 | **−0.15 ± 0.68 σ** | **false** |

The middle row is the one that would have been reported as a discovery without the volatility
control. The bottom row is the same data with the mechanical channel removed.

## 6. Answering the brief

| step | result |
|---|---|
| 1. containment | **17.2% overall; 75.1% conditional on being poolable.** Key derivation verified 1,346/1,346 |
| 2. does membership carry the edge? | **No.** Realized: +13 ± 23 bp (nothing). Raw maximum: +102 bp, adopt=true — but −0.15σ once volatility is divided out, adopt=false |
| 3. is membership predictable? | **not attempted** — gated on the edge being real |

**On the subset argument.** "Dashboard ⊂ sweep, so sweep can never be worse — just discard the
rest" is sound *where containment holds*, but it holds for a smaller slice than it sounds: two
thirds of dashboard articles are not poolable at all (46% carry no ticker, 19% are outside
N=300), and of the poolable remainder the sweep carries 75%. So the argument applies to roughly
a fifth of the dashboard. And within the days we can measure, there is no quality difference to
go and find — which makes "we are failing to identify the subset" the less likely reading.

**Which of the three outcomes:** the third. No edge. Reported plainly, as asked.

**What would still change this.** The proxy has no entry filter, so a membership × v2-score
interaction is not excluded — dashboard articles *that also score highly* were never tested
separately. Closing that needs the ~232 dashboard ∩ pool articles' pairs (≈378 on these days)
judged directly rather than waiting for a stratified sample, plus the minute-bar backfill
reaching 09-01..09-15.

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
