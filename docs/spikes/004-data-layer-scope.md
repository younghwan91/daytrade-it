# Spike 004: quant-airflow / kr-quant overlap — data-layer scope for Phase 1

**Status:** DONE
**Date:** 2026-09-07
**Goal:** answer the plan's open scope question — should the redeveloped
gpt-quant-v2 ingest Toss/DART/market data itself, or read from the user's existing
`quant-airflow` → TimescaleDB pipeline? — with an actual comparison of what
`quant-airflow` and `kr-quant` do today, so Phase 1's data-layer design isn't
redone later.

This is a read-only research spike. `quant-airflow` and `kr-quant` were only read,
never modified (`git status --short` clean in both after this spike, verified
below).

## TL;DR

- **quant-airflow already ingests everything gpt-quant-v2 would otherwise have to
  build itself for batch/EOD purposes**: KOSPI/KOSDAQ daily bars + supply/demand +
  sector index (via `kiwoom-client`), DART earnings/disclosures, Toss news (via
  `krx-news-client`), consensus, short interest/credit balance, delisted-stock
  backfill — all landing in a LAN-reachable TimescaleDB the user already reads
  read-only from `kr-quant`.
- **The plan's working hypothesis is CONFIRMED, and confirmed by quant-airflow's
  own design docs, not just inferred from its schedule.** News-to-signal latency
  is explicitly out of scope for quant-airflow: its own spec document says polling
  is "fundamentally unable to keep up" (`근본적으로 못 따라간다`) with
  second-level reaction and defers that to "a separate project... a different kind
  of system that doesn't fit this repo's batch-DAG architecture." quant-airflow's
  news pipeline runs 3x/weekday (08:45, 10:05, 16:05) — batch cadence, by design,
  not an oversight.
- **Recommendation: hybrid.** Consume quant-airflow's TimescaleDB read-only for
  everything it already provides (EOD bars, fundamentals, historical news/judgments,
  supply-demand). Build gpt-quant-v2's own lightweight, higher-frequency Toss/DART
  poller only for the specific latency-sensitive path (news → signal → trade
  decision) that quant-airflow's own docs say it deliberately does not cover.
  Details and a concrete Phase 1 shape are in [Recommendation](#recommendation).

## What quant-airflow provides today

Evidence is quoted/cited directly from the repo; nothing here is inferred from the
repo's name or general purpose.

### Sources and schedule (`quant-airflow/README.md`, `docs/operations.md`, `dags/*.py`)

quant-airflow runs 16 Airflow DAGs, all "매일 증분 + 주간 깊이 재수집" (daily
incremental + weekly deep re-collection), all idempotent `(code, date)` upserts
(`README.md:139`). Relevant schedule table (`README.md:141-157`, KST):

| DAG | Schedule | Collects |
|---|---|---|
| `daily_collection` | weekdays 16:00 | daily bars + supply/demand (kiwoom) + sector index |
| `daily_collection_catchup` | weekdays 10:05 | prior-day failures only |
| `daily_short_credit` | Tue-Sat 10:00 | short selling + credit balance (kiwoom) |
| `daily_earnings` | weekdays 16:00 | DART earnings, incremental |
| `daily_price_adjust` | weekdays 16:55 | rebuild `daily_bars_adjusted` |
| `daily_consensus` | weekdays 17:00 | Naver analyst consensus |
| `daily_news` | **weekdays 10:05 · 16:05** | Toss + DART news/disclosures + LLM judgment (`news_judgments`) |
| `premarket_news_judgment` | **weekdays 08:45** | same news collection, pre-open judgment |
| `earnings_backfill` | Sun 10:00 | full DART earnings history backfill |
| `weekly_history_backfill` | Sun 11:00 | sector index/short/credit history depth |
| `weekly_delisted_stocks` | Sat 10:05 | delisted-stock master + historical bars/shares backfill |
| `monthly_listed_shares_backfill` | monthly | DART listed-shares historical backfill |

`daily_collection.py` (`dags/daily_collection.py:1-7`): "storage.py가 Postgres
DSN을 받으면 TimescaleDB에 직접 upsert" — direct-to-TimescaleDB writes, no
separate sync step, using `kiwoom-client` under the hood (`collect_both()` calls
`collectors.combined --market all --prod`).

`daily_news.py` (`dags/daily_news.py:1-14`) confirms it uses **krx-news-client**
directly (the same library this repo's own spikes 001/002 exercise): "krx-news-client
(pip, 서버 아님)가 호출 시점에 소스를 직접 불러 정규화된 결과를 돌려준다" — and
states its own limitation in the module docstring:

> "토스 뉴스: 최신 스냅샷만 주고 페이지네이션이 없어, 창 사이 공백이 길수록 그
> 사이 쌓인 기사가 밀려 유실될 수 있다 — 오전 10:05/저녁 16:05 두 창 모두에
> 걸어 공백을 줄인다... **트레이딩 판단에 분 단위 최신성이 필요해지면 이 배치
> 방식 자체를 재검토해야 한다.**"

That last sentence is quant-airflow's own author naming exactly the gap this
spike was asked to check.

### Schema (`docs/schema.md`)

TimescaleDB hypertables (PK `(code, date)`, 1-year chunks, `compress_segmentby='code'`
after 30 days): `daily_bars`, `daily_bars_adjusted`, `supply_demand`,
`short_selling`, `credit_balance`, `sector_index`, `shares_outstanding_history`,
`consensus`, `news_articles` (Toss, PK `(id, published_at)`). Plain tables:
`news_article_tickers`, `stocks`, `earnings` (point-in-time `knowledge_date`),
`delisted_stocks`, `disclosures` (DART), `news_judgments` — LLM sentiment/event
judgments over `news_articles`/`disclosures`, upsert key
`(source_type, source_id, ticker, prompt_version)`, with `judged_at` (TIMESTAMPTZ,
schema note: stored in **UTC**, a real gotcha for any consumer joining KST-naive
columns — `docs/schema.md:36`).

Access is read-only over LAN: `docs/operations.md` / `README.md:36-46` — "TimescaleDB
는 `<spare-pc-ip>:5432` 로 LAN 에 열려 있고 메인 PC 가 여기에 질의한다," with a
`psycopg2.connect(...)` example reading `daily_bars_adjusted` directly.

### Design intent — news_judgments is explicitly scoped to exclude real-time

`quant-airflow/docs/superpowers/specs/2026-09-06-news-llm-judgments-design.md`
(quant-airflow's own accepted design spec for the `news_judgments` table/DAG task,
written by the same author, 2026-09-06 — this is not third-party speculation)
states the scope boundary directly, under "실행 흐름" (execution flow):

> "(범위 밖) 장중 초단위 반응 — scalp-it 핵심 진입은 '대장 상한가 3호가 3~5초'
> 틱 단위인데, 수집기가 폴링 기반(news_toss/news_dart 둘 다 웹훅 아님)이라
> **근본적으로 못 따라간다** — scalp-it도 이미 '기계화 불가'로 분류해 사람
> 판단 영역으로 남겨둔 지점과 일치. **별도 프로젝트로 미룬다**(상시 구동
> 프로세스 + 실시간 이벤트 소스 확보부터 필요, 이 레포의 배치 DAG 아키텍처와
> 안 맞는 다른 종류의 시스템)."

Translation: quant-airflow's news pipeline is polling-based by construction and
its own author has already decided second-to-minute reaction latency needs "a
different kind of system" — an always-on process with a real-time event source —
that does not fit quant-airflow's batch-DAG architecture. The same doc also notes
the same news-collection tasks are reused verbatim by both `daily_news` (swing
signal, 10:05/16:05) and `premarket_news_judgment` (pre-open signal, 08:45) — three
fixed points a day, nothing continuous or event-driven in between.

Also relevant: quant-airflow's own price data (`daily_bars`, `supply_demand`,
`sector_index`) is **end-of-day only** — `daily_collection` runs once at 16:00 after
market close (15:30) plus a 30-minute settle wait (`dags/daily_collection.py:21`).
There is no intraday tick/quote path in quant-airflow at all, for any asset class —
this is not unique to news.

## What kr-quant depends on (`~/Documents/git/kr-quant/README.md`)

kr-quant is explicitly downstream of quant-airflow and does no ingestion of its
own:

> "**데이터 수집은 이 저장소에 없다.** 수집 로직(DART·키움·KRX·네이버 커넥터,
> TimescaleDB 적재)은 [quant-airflow]로 분리되어 있다. 본 저장소는 DB를 읽기
> 전용으로 사용한다." (`README.md:15-17`)

Its architecture diagram (`README.md:184-208`) shows `quant-airflow → TimescaleDB
→ storage.py (읽기 전용) → daily_report.sh (cron) → sector_flow.py/... →
reports/<date>/{payload.json, numbers.html, viewer.html} → kq-flow / kq-ledger
(TUI, no DB access)`. Only `kq-pead` (the one alpha strategy that passed kr-quant's
gate) reads TimescaleDB directly and read-only; the two interactive TUIs
(`kq-flow`, `kq-ledger`) read pre-baked daily report JSON/HTML, not the DB, and
are explicitly a snapshot ("받은 폴더는 그날의 사진이다", `README.md:175`) —
kr-quant has no live/intraday consumption path either.

kr-quant confirms it depends on quant-airflow's *existing* output shape (same
tables cited above: `supply_demand`, `daily_bars`) but is not itself a source of
evidence about news latency — it doesn't touch `news_articles`/`news_judgments` at
all in the README (`grep` for `news` in kr-quant's README returns nothing beyond
the collector-boundary sentence above). This spike found no kr-quant dependency
that changes the recommendation below.

## Confirming/refuting the plan's working hypothesis

**Hypothesis (from the task brief):** nightly/batch data can come from
quant-airflow, but time-sensitive news-to-signal latency needs its own lighter
poller.

**Confirmed**, on two independent legs of evidence:

1. **Schedule fact**: quant-airflow's news pipeline runs at three fixed points a
   day (08:45, 10:05, 16:05 KST) via `daily_news`/`premarket_news_judgment` — not
   continuous, not event-driven, not sub-minute. Everything else (bars, earnings,
   consensus, short interest) is daily-or-slower batch too.
2. **Design-intent fact** (stronger than schedule alone): quant-airflow's own
   accepted design spec explicitly rules out sub-minute reaction as a goal for this
   repo and defers it to "a different kind of system" that would need to be a
   separate always-on process with a real-time event source. This isn't a gap the
   plan is guessing at — it's a boundary quant-airflow's own author already drew.

So the hypothesis doesn't just hold, it's already been independently reached and
documented by quant-airflow's design process. gpt-quant-v2 doesn't need to
convince anyone that a lighter/faster poller is needed alongside quant-airflow —
that's the explicit gap quant-airflow's own spec names and defers.

One nuance the hypothesis didn't originally name: quant-airflow also has **no
intraday price/quote path** for any asset — `daily_bars` is EOD-only, from a
16:00 batch job. If gpt-quant-v2's signal generation ever needs live price context
alongside fast news (e.g., current price vs. last EOD close, or intraday volume),
that's a second gap of the same shape (quant-airflow doesn't have it, by design),
not just a news-specific one.

## Recommendation for Phase 1

**Hybrid, cleanly split by data class — do not rebuild what quant-airflow already
owns, and do not force-fit the low-latency path into quant-airflow's DAG model.**

1. **Consume quant-airflow's TimescaleDB read-only** for everything it already
   provides and keeps current: EOD daily bars (raw + adjusted), supply/demand,
   sector index, earnings/consensus/short-credit, delisted-stock/survivorship data,
   and — importantly — historical `news_articles`/`disclosures`/`news_judgments`
   for backtesting and as a slower-but-free second opinion signal. This is a
   `psycopg2`/read-only connection exactly like kr-quant's `storage.py` pattern
   (`README.md:40-46`); no new ingestion code, no duplicated DART/kiwoom API load,
   and it inherits quant-airflow's survivorship-bias handling and point-in-time
   discipline (`earnings.knowledge_date`, `news_judgments.judged_at`) for free.
   Gotcha to carry over: `news_judgments.judged_at` is UTC-stored — convert to KST
   before joining against anything KST-naive (quant-airflow's own schema doc flags
   this as a real bug they hit, `docs/schema.md:36`).
2. **Build gpt-quant-v2's own lightweight Toss/DART poller** for the one thing
   quant-airflow deliberately does not do: news-to-signal at sub-daily-batch
   latency. This should be scoped as an always-on (or frequently-scheduled, e.g.
   every 1-5 minutes) process using `krx-news-client` directly — the same library
   quant-airflow uses, so no new scraping logic needs inventing, just a different
   run cadence and consumer. It should **not** try to write into quant-airflow's
   `news_articles`/`news_judgments` tables on quant-airflow's upsert keys (that's
   quant-airflow's schema to own per its own `CLAUDE.md` — see
   `2026-09-06-news-llm-judgments-design.md`'s "소유권" section, which explicitly
   reasons DAG/schema ownership stays with quant-airflow even when other repos are
   the consumers). Keep gpt-quant-v2's fast-path news store separate (its own table
   or even in-memory/ephemeral if signals are short-lived), and treat it as a
   parallel, faster, lossier feed alongside — not a replacement for —
   quant-airflow's authoritative slower one.
3. **Do not build a new kiwoom/DART/KRX EOD collector.** There is no gap there to
   fill; quant-airflow's daily batch already covers it and kr-quant already proves
   the read-only-consumer pattern works in production.
4. **If gpt-quant-v2 ever needs live/intraday price context** (not just fast news),
   that is a second, separate poller of the same shape as #2 — kiwoom's real-time
   WebSocket path (mentioned in kr-quant's "관련 프로젝트" table as available in
   `kiwoom-client`) rather than quant-airflow's batch collectors. Out of this
   spike's immediate scope (the brief asked about news latency specifically) but
   worth flagging now so Phase 1 doesn't rediscover it mid-implementation.
5. **Practical dependency**: TimescaleDB is reachable only on LAN
   (`<spare-pc-ip>:5432`, `README.md:36`). Phase 1's design needs to account for
   gpt-quant-v2 running somewhere with LAN access to that host, or needs a
   documented tunnel/VPN plan — this spike did not verify gpt-quant-v2-krx's
   deployment target has that reachability; flagging as a Phase 1 prerequisite to
   check, not something this read-only spike could test.

## Self-review

- Every schedule/schema claim above cites a specific file and line range in
  quant-airflow or kr-quant (README, `docs/schema.md`, `docs/operations.md`,
  `dags/*.py`, the design spec doc) — none are inferred from repo names or
  descriptions.
- The hypothesis confirmation is backed by quant-airflow's *own* design doc
  language ("근본적으로 못 따라간다... 별도 프로젝트로 미룬다"), which is
  stronger evidence than inferring intent from the DAG schedule alone.
- The recommendation names concrete mechanics for Phase 1 (read-only `psycopg2`
  connection reusing kr-quant's pattern; separate fast-path table/store; don't
  write into quant-airflow's schema; LAN-reachability prerequisite) rather than
  restating "hybrid" abstractly.
- Confirmed via `git status --short` in both `~/Documents/git/quant-airflow` and
  `~/Documents/git/kr-quant` that neither repo was modified by this spike (both
  clean).

## Files

- `docs/spikes/004-data-layer-scope.md` — this file (only file this spike adds)
