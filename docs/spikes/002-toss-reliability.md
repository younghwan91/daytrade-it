# Spike 002: Toss ingestion reliability

**Status:** DONE (single-session evidence only — see Limitations)
**Date:** 2026-09-07
**Goal:** confirm `krx-news-client`'s `TossScraper` is stable enough to build on
before committing further engineering time, given it wraps an unofficial/undocumented
Toss API with no SLA.

## TL;DR

- **23 live `TossScraper().scrape_news()` calls in this session, 0 exceptions, 0
  rate-limiting/blocking observed.** Every call returned exactly 76 articles across
  the three feeds (`ALL_HIGHLIGHT`/market, `HOT`/breaking, `SOARING_STOCK`/stock),
  with a consistent shape every time.
- **No per-feed errors either** — `TossScraper.scrape_news()` catches exceptions
  per feed internally and logs+continues rather than raising (see
  `krx_news_client/scrapers/toss.py`); this spike hooks the library's logger to
  catch those too, and it never fired.
- **Article turnover is slow relative to a multi-minute polling window**: across 8
  polls at a 30s interval (~4 min total) the article set never changed (duplicate
  rate 100% after the first poll). Across 15 polls at a 2s interval (~1 min total,
  deliberately hammering the endpoint) one poll picked up 7 new articles as the
  feed rotated, otherwise 100% duplicate. This is a *feed content* behavior, not a
  reliability signal — the dashboard endpoint returns "current highlights," not a
  paginated/incremental feed.
- **Latency**: 1.4s–2.7s per call (mean 2.1s) across all 23 calls, including the
  library's built-in throttle (`min_delay`/`max_delay` = 0.5s/1.5s jitter between
  requests) — no elevated latency or timeouts under the 2s-interval burst.
- **Recommendation:** stable enough to build Phase 1 ingestion on, but do not treat
  this session's result as proof of production-grade reliability — see Limitations
  and the known-risk findings below before committing further engineering time.

## Method

`scripts/spikes/toss_reliability_spike.py` polls `TossScraper().scrape_news()` on
an interval and logs, per poll: timestamp, article count (total + per category),
any exception raised out of `scrape_news()`, any per-feed error the library
swallowed internally (captured via a logging handler attached to
`krx_news_client.scrapers.toss`'s logger), and duplicate-id rate against the
running set of article IDs seen so far in the session (`NewsArticle.id` is
`f"{source}:{md5(url)[:12]}"`, stable across polls for the same article since the
URL embeds the Toss `newsId`).

Two runs, back-to-back in this session:

1. **Steady-interval run** — 8 polls, 30s apart (~4 min wall time):
   `docs/spikes/002-poll-log.jsonl`.
2. **Burst run** — 15 polls, 2s apart (~1 min wall time, deliberately faster than
   any plausible production polling cadence, to probe for rate-limiting):
   `docs/spikes/002-poll-log-burst.jsonl`.

Combined: 23 live calls, run via `/tmp/spike-venv` (a venv with `krx-news-client`
0.2.0 installed from the local sibling repo — `gpt-quant-v2-krx`'s own `uv`
environment does not have `krx-news-client` as a declared dependency yet, which is
a Phase 1 setup item, not a spike concern).

## Results

| Run | Polls | Interval | Exceptions | Swallowed feed errors | Article count (min/max/mean) | Duplicate rate (min/max/mean) |
|---|---|---|---|---|---|---|
| Steady | 8 | 30s | 0 | 0 | 76 / 76 / 76.0 | 0.0 / 1.0 / 0.875 |
| Burst | 15 | 2s | 0 | 0 | 76 / 76 / 76.0 | 0.0 / 1.0 / 0.927 |
| **Combined** | **23** | — | **0** | **0** | **76 / 76 / 76.0** | — |

- Exception rate: **0/23 = 0%**.
- Latency across all 23 calls: 1392ms–2724ms, mean 2150ms.
- No HTTP 429s observed, no empty/short-shaped responses, no schema surprises
  (every `NewsArticle` had a non-empty `id`/`title`/`url` and a valid `category`).

Raw per-poll records: `docs/spikes/002-poll-log.jsonl`,
`docs/spikes/002-poll-log-burst.jsonl`.

## Known-blocking-risk search (sibling repos)

Searched `~/Documents/git/*/docs/` for prior documentation of Toss/KRX scraper
blocking risk, per the plan's hint that this may already exist:

- **`krx-fundamentals-client/docs/krx-blocked.md`** (also present in
  `krx-fundamentals-client-refactor`) documents a **real, currently-observed
  blocking issue** — but for a *different* data source and *different* failure
  mode than this spike covers. It's about `KrxScraper` (KRX's own MDCSTAT
  investment-ratio endpoint, not Toss), which now requires a login; the OTP
  endpoint silently returns the string `"LOGOUT"` instead of a real token, passes
  the library's length check, and the scraper ends up returning an **empty list
  with no exception** — a silent-failure mode, not a rate limit. Worth noting as a
  *pattern* to watch for (unofficial endpoints degrading to "looks successful,
  returns nothing" rather than an honest error), but it does not describe Toss.
- **`quant-airflow/docs/superpowers/specs/2026-09-06-news-llm-judgments-design.md`**
  documents two things directly relevant to Toss specifically, from the team
  already running `krx-news-client`'s `TossScraper` in a production Airflow DAG
  (`collectors/news_toss.py`, ingesting into `news_articles`):
  - **Toss ToS compliance for automated access is explicitly unverified**: "이건
    HTML을 긁는 게 아니라 JSON API라 robots.txt는 안 걸리지만, 토스 ToS상 자동화
    접근이 허용되는지는 검증되지 않았다" (it's a JSON API, not HTML scraping, so
    robots.txt doesn't apply — but whether Toss's ToS actually permits automated
    access hasn't been checked).
  - **No backfill capability**: the dashboard endpoint only returns current
    highlights, with no parameter to page into history — consistent with what
    this spike observed (a mostly-static set across a multi-minute window).
  - Their error-handling design treats "API-level failure (rate limit, auth
    error, network)" as a distinct, real category worth a dedicated failure
    path (task fails, Airflow retries) — implying they anticipate this as a
    live risk for production polling cadence, even though this spike didn't
    trigger it in 23 calls.
- No other sibling repo's `docs/` mentions Toss- or krx-news-client-specific
  blocking/rate-limiting.

## Limitations

- **This is single-session evidence only.** 23 calls over roughly 5 minutes of
  wall time cannot rule out rate-limiting, IP-based blocking, or API changes that
  only manifest over hours/days of sustained polling, or under a production
  polling cadence sustained across market hours. **A longer multi-day run is
  recommended before Phase 1 truly commits to this as the sole ingestion path.**
- Ran from a single IP/session with the library's default throttle
  (0.5s–1.5s jitter) always active, even in the "burst" run — this spike did not
  test disabling the throttle or hitting the endpoint from multiple concurrent
  clients, which is a more realistic way to trigger rate-limiting than a single
  well-behaved client.
- Toss ToS for automated access is unverified (see above) — this is a policy/legal
  risk, not something a reliability spike can resolve; flagged for whoever owns
  that decision before Phase 1 ships.
- No backfill: any production design built on this scraper must treat it as a
  live/incremental feed only, not a historical data source.

## Files

- `scripts/spikes/toss_reliability_spike.py` — the polling/logging script
- `docs/spikes/002-poll-log.jsonl` — 8-poll steady run (30s interval)
- `docs/spikes/002-poll-log-burst.jsonl` — 15-poll burst run (2s interval)
- `docs/spikes/002-toss-reliability.md` — this file
