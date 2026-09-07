# Spike 001: Korean news sentiment model validation

**Status:** DONE (with a gap — see Limitations)
**Date:** 2026-09-07
**Goal:** produce a real accuracy number for Korean-language financial-news sentiment
classification, so Phase 4's risk-gate design has evidence instead of an assumption.

## TL;DR

- **LLM approach (OpenAI, Korean prompt): not measured.** No `OPENAI_API_KEY` was
  available in this environment (see Limitations). The script (`sentiment_spike.py`)
  implements this path and will run it the moment a key is exported — this remains
  an open action item, not a completed measurement.
- **Local model approach (`snunlp/KR-FinBert-SC`, Korean FinBERT): 60.5% accuracy**
  (46/76) against hand labels. Recall on BEARISH — the class a risk-gate cares about
  most — is only **18.2%** (2/11). This does not clear a usable bar for a live
  risk-gate on its own.
- **Recommendation:** do not wire FinBERT (or any single approach) into the Phase 4
  risk-gate yet. Get an OpenAI API key and re-run this spike's LLM path before
  deciding between "LLM-only," "FinBERT as a second signal," or "ensemble" — the
  data needed to make that call doesn't exist yet.

## Method

1. Pulled 76 recent Toss news articles via `krx-news-client`'s `TossScraper`
   (`scripts/spikes/fetch_sample.py` → `docs/spikes/001-sample.json`), covering the
   three categories the live feed actually exposes: `market` (50), `breaking` (20),
   `stock` (6). See **Deviation from brief** below re: the missing `economy` category.
2. Hand-labeled all 76 articles BULLISH/BEARISH/NEUTRAL myself, reading each
   article's text and judging likely stock-market sentiment impact
   (`docs/spikes/001-hand-labels.json`). Label distribution: 33 BULLISH, 11 BEARISH,
   32 NEUTRAL.
3. Scored the same 76 articles with `scripts/spikes/sentiment_spike.py`:
   - **OpenAI approach**: Korean system prompt (translated/adapted from
     `openai_client.py`'s `SENTIMENT_SYSTEM_PROMPT`, written standalone in the spike
     script — `prompts.py` was not touched), same tool-calling/structured-output
     pattern as the production client, reimplemented standalone (no import of
     production code) so the spike has zero coupling to app config/DI.
   - **FinBERT approach**: `snunlp/KR-FinBert-SC`, a BERT-base checkpoint fine-tuned
     for 3-way Korean financial sentiment (`negative`/`neutral`/`positive`, mapped
     1:1 to BEARISH/NEUTRAL/BULLISH), loaded via `transformers`, run fully on CPU.
4. Computed accuracy, confusion matrix, precision/recall against hand labels for
   each approach that actually ran (`docs/spikes/001-results.json`).

Text scored was the Toss `summary` field, not `content` — every one of the 76
articles in this pull returned an empty `content` field from the Toss dashboard
feed; `summary` is the substantive body text and is what a real integration would
have to use too, at least for this feed type.

## Results

### FinBERT (`snunlp/KR-FinBert-SC`)

- **Accuracy: 60.5% (46/76)**, ~30s CPU inference for all 76 articles (batch size 8,
  after a ~4-minute one-time model download).
- Baseline for reference: random guessing among 3 classes ≈ 33%; always predicting
  the majority class (BULLISH, 33/76) ≈ 43%. FinBERT clears both, but not by a wide
  margin, and the class-level breakdown is the real problem:

| Gold \ Predicted | BULLISH | BEARISH | NEUTRAL | Recall |
|---|---|---|---|---|
| BULLISH (33) | 18 | 0 | 15 | 54.5% |
| BEARISH (11) | 0 | 2 | 9 | **18.2%** |
| NEUTRAL (32) | 4 | 2 | 26 | 81.2% |
| Precision | 81.8% | 50.0% | 52.0% | |

Macro F1 ≈ 0.52 (BULLISH 0.65, BEARISH 0.27, NEUTRAL 0.63).

**The model almost never predicts BEARISH** (4 predictions total across 76
articles) and instead defaults heavily to NEUTRAL for genuinely negative articles
(9 of 11 hand-labeled BEARISH articles were scored NEUTRAL). For a risk-gate whose
entire purpose is to catch and act on negative signal, an 18% recall on the bearish
class is disqualifying on its own, independent of the 60.5% headline number.

### OpenAI (Korean prompt) — not run

No `OPENAI_API_KEY` (or `GPTQ_OPENAI_API_KEY`) was present in this environment.
Checked: `echo $OPENAI_API_KEY` (unset), `~/Documents/git/gpt-quant-v2/.env` (does
not exist), `~/Documents/git/gpt-quant-v2-krx/.env` (does not exist — only
`.env.example` is checked in). I did find an `OPENAI_API_KEY` in
`~/Documents/git/gpt_quant/.env` — a different, unrelated sibling repo (legacy
`gpt_quant`, not `gpt-quant-v2`) — but deliberately did not use it: it's outside
the locations this task authorized me to check, and spending another project's
API credit without asking first isn't a call I should make unilaterally. An
`ANTHROPIC_API_KEY` was also present in the environment, but the brief specifically
calls for reusing the OpenAI structured-output pattern from `openai_client.py`
(the actual production client this feeds into), so I did not substitute a
different vendor — that would answer a different question than the one this spike
was scoped to answer.

**This means the spike's most important comparison — LLM vs. local model — is
incomplete.** `scripts/spikes/sentiment_spike.py --sample docs/spikes/001-sample.json
--labels docs/spikes/001-hand-labels.json` will run both approaches as soon as
`OPENAI_API_KEY` is exported (or pass `--skip-finbert` to only run the LLM leg).

## Limitations

- **Single-rater labels.** All 76 hand labels were produced by me (the implementing
  agent) alone, reading each article once. No second rater, no adjudication, no
  inter-rater agreement statistic. Some of my own labels were judgment calls on
  ambiguous articles (e.g. macro/geopolitical stories with no obvious single
  Korean-listed beneficiary) — see `docs/spikes/001-hand-labels.json` for the full
  label set if a second pass is wanted.
- **Small, single-day, single-language-family sample.** 76 articles pulled on one
  day (2026-09-07), all from Toss's live feed at that moment — heavily weighted
  toward that day's dominant story (AI/semiconductor rally). Not a stratified or
  time-diverse sample.
- **No `economy` category in the live sample.** `TossScraper.scrape_news()` only
  aggregates three dashboard feed types today — `ALL_HIGHLIGHT`→`market`,
  `HOT`→`breaking`, `SOARING_STOCK`→`stock` (see
  `krx_news_client.scrapers.toss.FEED_TYPES`). There's no standalone `economy` feed
  to pull from, so this sample doesn't include that category despite the task brief
  naming it — not a bug in this spike, just what the current library exposes.
- **LLM leg unmeasured** (see above) — the FinBERT number alone cannot answer
  "which approach should Phase 4 use," only "is FinBERT alone good enough" (answer:
  no).

## Recommendation

1. **Do not wire FinBERT, or any sentiment classifier, into the Phase 4 risk-gate
   yet.** 60.5% accuracy with 18% recall on BEARISH is not a usable bar for a gate
   whose job is specifically to catch bearish signal before it reaches live trading.
2. **Before Phase 4 starts:** get an `OPENAI_API_KEY` into the environment and
   re-run `scripts/spikes/sentiment_spike.py` against the same
   `docs/spikes/001-sample.json` / `001-hand-labels.json` to get the missing LLM
   number. That comparison — not the FinBERT number alone — is what should decide
   LLM-only vs. FinBERT-as-second-signal vs. ensemble.
3. If the LLM number also comes back weak, get a second rater to check a subset of
   my hand labels before concluding the *models* are the problem rather than my
   single-rater label quality.

## Files

- `scripts/spikes/fetch_sample.py` — pulls the Toss sample (data collection only,
  not one of the two approaches under test)
- `scripts/spikes/sentiment_spike.py` — the two scoring approaches + accuracy
  computation
- `docs/spikes/001-sample.json` — 76 raw articles
- `docs/spikes/001-hand-labels.json` — hand labels (single rater)
- `docs/spikes/001-results.json` — raw scoring output (FinBERT run; OpenAI marked
  skipped)
- `docs/spikes/001-sentiment-validation.md` — this file
