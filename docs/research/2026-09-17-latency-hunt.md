# Can we learn about the news faster, and would it pay? (2026-09-17)

Everything ran on simnode in `~/git/daytrade-it-latency` (branch `feat/latency-hunt`), `data`
symlinked to `~/git/daytrade-it/data`.
- kr_quant was read-only throughout (`scripts/eval/edge_db.py` opens `set_session(readonly=True)`).
- No Anthropic calls, no Toss calls, no DART calls, no Kiwoom calls. Nothing was written outside
  `data/eval/latency/`. Nothing in `~/git/daytrade-it`, the live daemon or the running backfill was
  touched; the live archive was copied to `~/latency_archive/` and read from there.

**Answer, up front.** We *are* very late — the median article reaches our judge **21.8 minutes**
after it was published, and essentially all of that is upstream of us; our own pipeline spends
**under one minute** end to end. But being early would not have helped. Repricing all 131,741
sweep pairs from the **article's own publication minute** against the same matched null
`edge-hunter` built, the selection alpha is **−0.2 bp at publication + 1 minute (se 0.8)** and
stays inside noise at every horizon out to the close. The latency ladder — enter at pub+0, +1,
+2, +3, +5, +10, +15, +22, +30, +45, +60, +90 and hold to the close — is **flat**: +2.3 bp at
L=0 against +3.7 bp at L=22. The best cell in the whole study, the intraday-published BUY subset
entered at the publication minute itself (which no live system can do), is **+7.3 bp (se 3.4)**
against a **36.7 bp** round trip.

Latency is not the missing variable. **Even at zero latency this does not pay**, and by a factor
of five in the most favourable corner and a factor of sixteen on the pool.

One thing this does *not* close, stated plainly in §6.

---

## 1. The instrument, and why it should be believed

`scripts/eval/lat_path.py` reprices every pair from `published_at` instead of from `decide_at`,
using `prompt_labels.py`'s rules copied verbatim out of `edge_null.py`'s `Day` class — a minute's
quote is that minute's bar open, else the last completed regular bar's close — and prices the
whole top-300 cross-section at each anchor minute, inside each day × volatility-quartile ×
liquidity-quartile cell, exactly as `edge_strata.py` does.

Before any result was read, the two paths were joined pair by pair:

| check | result |
|---|---|
| my `ret_close` at our own acting minute vs `edge-hunt`'s `rows_sweep.jsonl` | 131,634 pairs joined, **max absolute difference 0.00e+00** |
| my matched-null `alpha` at that minute vs `edge-hunt`'s | **median difference exactly 0.0 bp**, mean +0.28 bp |

Two independently written pricing paths, same prices, same alpha. The headline it reproduces is
`edge-hunt`'s: alpha at our acting minute **+2.1 bp (se 3.4)** against their +1.6 bp (se 3.4).

Inference is clustered on the **day**, bootstrap over 125 days, same as `edge-hunt`.

---

## 2. Where the 21.8 minutes go

Live archive only — `data/news_judgments/2026-09-1{4,5,6}.jsonl`, the detail archive under
`data/toss_detail/`, and the daemon log's own `Toss news poll complete` events. 258 distinct
articles, 311 pairs, all three genuine live days. Nothing reconstructed.

`scripts/eval/lat_decompose.py` splits the chain at the four points the archive can actually see:

```
A  published_at -> the poll that first emitted it     upstream: Toss + our poll grid
B  that poll    -> our loop reaching that article     our in-poll queue
C  reached      -> judged (detail fetch + Claude)     our scoring
D  judged       -> signal persisted                   our signal path
E  signal       -> order submitted                    our execution path
```

Medians, in minutes:

| segment | all 258 articles | intraday only (201) |
|---|---:|---:|
| **A published → we saw it** | **21.87** | **18.32** |
| B our in-poll queue | 0.15 | — |
| C detail fetch + Claude v2 + v4 shadow | 0.04 | — |
| D judged → signal persisted | 0.0017 (0.10 s) | — |
| E signal → order submitted | 0.62 (37.4 s) | — |
| **total published → judged** | **21.78** | **18.72** |

**Everything we own is worth about 50 seconds**, and 37 of those are segment E — `enter_batch`
deliberately waiting until the whole poll is judged so it can order by `rank_batch`. The poll
interval was 190 s median (183 s p50 on 09-14, 192 s on 09-15, 190 s on 09-16), so the poll grid
can account for **at most 3.2 minutes** of segment A.

### 2.1 It is Toss, and the floor is 8.7 minutes

If an article was already on the dashboard at the *previous* poll we would have taken it then.
So for every intraday article, (previous poll − published_at) is a **lower bound** on how long
Toss took to show it:

| intraday articles, n=201 | median | p25 | p75 |
|---|---:|---:|---:|
| lower bound on Toss dashboard lag | **15.28** | 11.60 | 22.90 |
| upper bound (= segment A) | 18.32 | 14.72 | 26.11 |

So of the 18.3 minutes, **about 15 are Toss deciding to show the article** and about 3 are our
poll grid. And this is not a tail effect — `scripts/eval/lat_by_kind.py`:

**Not one of the 201 intraday articles reached us in under 8.7 minutes.** The floor is the same
for `breaking` (min 8.7) and `stock` (min 8.8), the same for every title tag, and the same across
all thirteen wires — medians run 14.5 min (한국경제) to 27.0 min (조선비즈), with no fast subset
anywhere. There is no low-latency corner of the Toss dashboard to poll harder.

### 2.2 And `published_at` is itself a few minutes late

Most Korean wire ids embed the filing minute (`asiae_2026091608340116027` → 08:34:01). On the 94
archive articles whose id parses, `published_at` runs a **median 4.05 minutes** after that minute
(p10 0.10, p90 72.0). So the real chain, from the reporter pressing send, is roughly:

```
  +0.0  wire files
  +4.1  publisher's own site (this is what Toss reports as published_at)
 +19.4  Toss dashboard shows it            <- 15.3 min, and nothing we do touches it
 +22.4  our poll takes it                  <- 3.2 min, ours
 +22.6  judged                             <- 0.2 min, ours
 +23.2  order submitted                    <- 0.6 min, ours
```

---

## 3. The decisive measurement: there is no impulse to be early for

`scripts/eval/lat_agg.py`, whole sweep pool, 131,643 pairs over 125 days. Alpha is the pair's own
return from the publication minute minus the **same day, same minute, same volatility quartile,
same liquidity quartile** cross-section's return over the identical window. Day-clustered,
bootstrap CI.

| window from publication | alpha (bp) | se | 95% CI | days ≥ 0 |
|---|---:|---:|---|---:|
| pub → pub+1 min | **−0.2** | 0.8 | [−1.9, +1.4] | 72/125 |
| pub → pub+2 | −0.0 | 1.1 | [−2.3, +2.0] | 68/125 |
| pub → pub+3 | +0.4 | 1.3 | [−2.2, +2.9] | 77/125 |
| pub → pub+5 | −0.2 | 1.5 | [−3.3, +2.5] | 75/125 |
| pub → pub+10 | −1.9 | 1.8 | [−5.5, +1.6] | 65/125 |
| pub → pub+15 | −0.8 | 2.0 | [−4.9, +3.1] | 59/125 |
| pub → pub+22 | −1.2 | 2.2 | [−5.7, +3.0] | 59/125 |
| pub → pub+30 | −1.4 | 2.2 | [−5.8, +3.0] | 58/125 |
| pub → pub+60 | −3.1 | 2.7 | [−8.6, +2.1] | 62/125 |
| pub → pub+120 | −1.4 | 3.2 | [−7.9, +4.9] | 65/125 |
| pub → close | +2.3 | 3.8 | [−5.3, +9.8] | 73/125 |

**Zero at every horizon, starting one minute after the article exists.** The window we have never
once traded — the first minute — is the flattest number in the table, and it is measured on 131,643
events over 125 days with a standard error of 0.8 bp. This is not "we arrive after the impulse";
there is no directional impulse at any point.

Raw (no null) the same rows run **−6.7 bp at +1 min** and **−34.8 bp to the close**: the pool falls
from publication onward, and the fall is the market and the cross-section, which is why it
disappears under matching.

### 3.1 There *is* an impulse — in size, not in direction

Conditioning on the event having moved at all, and asking what share of the pub→pub+120min excess
move had already happened by pub+k. For a driftless walk conditioned on its endpoint that share is
exactly k/120, so the benchmark column is what "no news effect" looks like:

| by pub+k | share, moves ≥1% (n=72,206) | ≥2% (n=40,594) | ≥5% (n=10,066) | driftless benchmark |
|---|---:|---:|---:|---:|
| +1 min | 11.1% | 11.7% | 11.0% | 0.8% |
| +2 | 17.3% | 19.0% | 18.3% | 1.7% |
| +5 | 27.4% | 30.4% | 30.1% | 4.2% |
| +10 | 35.7% | 39.3% | 40.5% | 8.3% |
| +15 | 44.3% | 47.9% | 48.3% | 12.5% |
| **+22** (our latency) | **52.4%** | **55.8%** | **56.9%** | 18.3% |
| +30 | 59.6% | 62.6% | 63.1% | 25.0% |
| +60 | 78.5% | 80.0% | 80.7% | 50.0% |

Strongly front-loaded — 11% of a two-hour move in the first minute against a 0.8% benchmark — and
the shape is identical whether the move is 1% or 5%. So news-tagged names really do move on their
news, and at our 22-minute latency we arrive with **about half of the move already gone**.

That is the team-lead hypothesis, and it is half right. The move is front-loaded and we miss half
of it. What the hypothesis predicts next — that capturing the early half would produce alpha — is
what §4 tests, and it is false. The early half is as directionally unpredictable as the late half:
we cannot tell in advance which way the 11% in minute one is going to go, which is exactly what
§3's −0.2 bp says.

---

## 4. The latency ladder: what a faster path is worth

`scripts/eval/lat_ladder.py`. Enter at pub+L, hold to the close, each L against the matched null
**at that very minute and cell**, so the ladder is not confounded by the intraday drift that
`edge-hunt` §3 measured. Alpha in bp, day-clustered over 125 days.

| entry lag L | whole pool (n=131,645) | intraday-published (50,594) | BUY score>0.15 (25,854) | intraday BUY (11,244) |
|---|---:|---:|---:|---:|
| **+0 min** (publication minute; unreachable) | +2.3 (3.8) | +4.1 (2.2) | +5.5 (4.7) | **+7.3 (3.4)** |
| +1 | +2.5 (3.6) | +3.5 (2.2) | +4.4 (4.5) | +5.1 (3.3) |
| +2 | +3.1 (3.4) | +3.4 (2.2) | +4.8 (4.2) | +4.3 (3.3) |
| +3 | +2.3 (3.3) | +3.0 (2.2) | +4.2 (4.2) | +3.7 (3.2) |
| +5 | +3.1 (3.2) | +2.7 (2.2) | +4.5 (4.0) | +3.1 (3.2) |
| +10 | +4.4 (3.0) | +2.7 (2.1) | +6.3 (3.8) | +3.0 (3.1) |
| +15 | +3.3 (2.9) | +3.1 (2.0) | +6.3 (3.7) | +3.3 (3.1) |
| **+22** (today) | +3.7 (2.7) | +2.8 (2.0) | +5.4 (3.5) | +2.1 (3.0) |
| +30 | +3.7 (2.6) | +2.8 (2.0) | +5.5 (3.2) | +1.6 (3.0) |
| +45 | +4.7 (2.4) | +3.1 (1.9) | +5.7 (3.0) | +2.1 (2.9) |
| +60 | +5.4 (2.3) | +2.6 (1.8) | +7.3 (2.9) | +2.7 (2.8) |
| +90 | +4.5 (2.1) | +3.2 (1.7) | +6.3 (2.5) | +3.0 (2.5) |

Flat. On the whole pool and on the BUY subset the ladder is if anything *rising* with lag, which is
the intraday-drift effect `edge-hunt` §3 already identified leaking through the hold-to-close
horizon, not a latency effect. On the only cut where latency should matter most — articles actually
published during the session, that our score wanted to buy — going from 22 minutes late to the
publication minute itself buys **+7.3 − 2.1 = 5.2 bp**, with a standard error of 3.4 on each leg.

**The bar is 36.7 bp.** Every cell in that table, at every latency including the physically
impossible one, is between one fifth and one sixteenth of it. Net of the true round trip, the best
row in the whole study is **−37.7 bp**.

Restated as the question that was asked: *would instant news change the economics?* No. It is worth
somewhere between 0 and 5 bp, against a 36.7 bp cost floor and a selection alpha that is zero to
begin with.

---

## 5. Is there a faster source? Enumerated and measured

| source | earlier than Toss by | live-consumable? | covers our pool? | verdict |
|---|---|---|---|---|
| **publishers' own feeds** (RSS / 네이버) | **~15 min** (§2.1) | yes; the 13 wires with ≥4 intraday articles cover 197/201 | yes, by construction | the only real option, and §4 prices it at ≤5 bp |
| shorter Toss poll interval | ≤3.2 min, ~1.6 min on average | yes, today | yes | worth <1 bp by §4 |
| our own pipeline (mostly `enter_batch`'s 37 s) | 0.6 min | yes, today | yes | worth <1 bp |
| **DART** | **not measurable, and almost certainly not applicable** | see below | **2.4% of the pool** | dead end |
| `vi_events` | — | — | — | **table is empty: 0 rows** |
| `quote_events` | — | — | 164 codes, one day (09-16) | not a news source; it is price |
| `ticks` | — | — | 846 codes, 2026-08-24..09-16 | not a news source; it is price |
| `supply_demand`, `short_selling`, `consensus`, `earnings`, `theme_members`, `credit_balance` | — | — | — | all **daily** granularity; cannot carry an intraday timestamp |

### 5.1 DART, specifically

The brief assumed we hold a DART key, that the daemon polls disclosures every 300 s, and that a
DART filing timestamp would front-run the Toss article. All three are wrong:

- **The live daemon does not poll DART at all.** `scripts/news_signal_poller.py` constructs
  `TossNewsClient` directly, never `NewsPoller` (which is the class that owns the DART half). There
  is no `GPTQ_KRX_NEWS_DART_API_KEY` line in `daytrade-it/.env`; `news_poller.py`'s own docstring
  calls this "an environment with no DART key configured (this one)". The 300 s interval is a
  default on a code path that never runs.
- **DART's list API carries no time.** `krx_news_client/scrapers/dart.py:375` —
  「rcept_dt 는 접수'일'(KST)이라 시각이 없다 — 00:00 KST 로 둔다」. `rcept_dt` is `YYYYMMDD`.
  A disclosure's minute is simply not in the feed.
- **The stored table confirms it.** All 4,141 rows of kr_quant `disclosures` have
  `published_at` = 00:00 KST exactly; `collected_at` clusters at 07/16/23 KST (a daily batch), and
  the table is eight trading days deep (2026-09-06..09-15). It cannot support a latency claim.
- **And it would not cover us.** Only **2.4%** of the 135,946 sweep pairs, and **2.9%** of the BUY
  subset, are disclosure-shaped by title (공시/유상증자/무상증자/자기주식/단일판매·공급계약/
  전환사채/감자결정/주요사항보고…). What we actually trade is 52% 테마, 23% 기타, 11% 정책.
  A disclosure feed is a different strategy, not a faster version of this one.

Polling DART's list endpoint *would* still timestamp filings by our own poll time, the way Toss
does. That is a real option for a disclosure strategy. It is not an option for making *this* pool
faster, and §4 says the pool has nothing to be faster for.

### 5.2 Ranked by (time saved) × feasibility

1. **Publishers' own feeds** — 15.3 min saved, high feasibility (13 wires, RSS, no key, no rate
   limit like Toss's). Measured value: **≤5.2 bp**, se 3.4. Not worth building.
2. **Poll interval 190 s → seconds** — 1.6 min average, trivially feasible, already ours. Measured
   value: **<1 bp**. Not worth the Toss rate-limit risk.
3. **`enter_batch` ordering delay** — 37 s, trivially feasible. Measured value: **<1 bp**. Leave it;
   `rank_batch` ordering is worth more than 37 seconds of latency that is worth nothing.
4. **DART** — unmeasurable lead, 2.4% coverage, not currently wired. Dead end for this pool.
5. **VI / exchange feeds** — no data exists.

---

## 6. What this does not close

This measures the pool **Toss carried**. If there is an event class that moves prices hard and that
Toss's dashboard never shows — or shows so late it falls outside the trading day — then nothing
here sees it, because every pair in the 125-day sweep came through Toss in the first place. A
publisher-feed or DART-feed strategy could in principle surface *different events*, not merely the
same events sooner.

What §3 and §4 do close is the version of the question that was asked: **for the events we
already act on, earliness is worth between 0 and 5 bp against a 36.7 bp floor.** Any future
latency work has to justify itself by the events it would *add*, and would have to clear 36.7 bp on
those, with the pool's own history — zero alpha at every horizon from the publication minute — as
its prior.

Two carried caveats from `edge-hunt`, both unchanged here because this reuses its inputs:
- September labels sit on an incomplete minute-bar backfill (13.1% of September pairs had no bars
  when `labels.jsonl` was built).
- The sweep's `decide_at` replays a *sweep* crawler's judge queue, not the live dashboard poller,
  so its acting minute is not the live 21.8-minute latency. The ladder in §4 is immune to this: L
  is fixed relative to `published_at` and never touches `decide_at`.

---

## 7. Reproduce

```
# publication-anchored repricing of all 131,741 pairs (~40 min, kr_quant read-only)
python scripts/eval/lat_path.py

# section 3 and the calibration against edge-hunt
python scripts/eval/lat_agg.py

# section 4's ladder and section 3.1's move-shape table
python scripts/eval/lat_ladder.py

# sections 2 and 2.1, on the live archive copied to ~/latency_archive/
python scripts/eval/lat_decompose.py
python scripts/eval/lat_by_kind.py
```
