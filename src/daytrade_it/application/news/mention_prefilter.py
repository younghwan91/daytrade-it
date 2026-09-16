"""Deterministic pre-LLM filter: which (article, ticker) pairs are worth judging.

Toss tags several companies per article, and ~74% of judged pairs score 0 -- most
of them passing mentions inside syndicated market roundups and price tables.
Judging those costs real money, so this module decides, without any model call
and without any I/O, whether a pair can ever become a BUY.

The rule is derived from 97,721 judged pairs, not guessed: every bracketed title
prefix with n >= 30 was measured against the 17.30% base BUY rate, and the
roundup/table columns below all sit at ~0%. See
`.superpowers/sdd/2026-09-16-bayes-stages/prefilter-report.md` for the full
table, the rejected candidates (company-name matching loses 25-44% of BUYs), and
the dropped-BUY inspection.

A second, independent rule uses article **fan-out**: how many companies Toss
tags on one article. It is the strongest deterministic signal in the data --
one-code articles are 41.3% BUY, >20-code articles 0.07% -- and it is known
*before* judging in both worlds: live from the detail response's
`companyCodes`, historically from the C1 backfill's related-code grouping.
Measured over the same 97,721 pairs, combined with the roundup tags:

    N     skip     BUY loss
    5    38.71%     3.211%
    8    30.32%     0.757%
    10   26.74%     0.331%
    11   25.78%     0.254%   <- adopted
    12   24.65%     0.219%
    15   22.10%     0.130%
    20   15.99%     0.065%

A third rule, **index commentary**, catches what fan-out alone misses: an
index name (`코스피`/`코스닥`) in the title of an article tagged to
`INDEX_FANOUT_MIN` or more companies is market-wide commentary, not company
news. Alone, marginally over roundup + fan-out(15): +5.55% of pairs skipped for
+0.219% BUY loss. Its threshold is 5 rather than 3 or 4 because the cheaper
thresholds cost more BUYs than they buy skips.

Measured with all three rules on (fan-out 11, index 5): **29.00% of pairs
skipped, 0.373% BUY (score > 0.15) recall loss**, 0.371% weighted by |score|.
Split in half by publication time the loss is 0.259% (older half) and 0.460%
(newer half), so the rule set stays inside the 0.5% bar out of sample.
Roundup alone is 13.34% / 0.024%; roundup + fan-out(15) is 22.10% / 0.130%.

**Rejected candidates**, all measured on the same 97,721 pairs: publisher
(`source`) -- the worst publisher still converts at 16.0% against a 17.3% base,
so there is nothing to cut; summary length -- every Toss summary is truncated
to 101-120 characters, so the feature has no variance; publication hour --
the overnight block (21:00-06:00) holds 8.5% of pairs but 6.8% of BUYs;
near-duplicate titles -- same code + normalised title within 24h is only 0.05%
of pairs, and the loose variant (same title, any code) loses 18.6% of BUYs;
per-code trailing zero-BUY history -- a 60-day window skips 0.82% of pairs for
0.822% of BUYs, worse than 1:1. A greedy search over 54 title n-grams was
rejected outright: fitted on the older half it looked like 0.490% loss and cost
1.046% on the newer half, i.e. the token picks do not generalise.

`should_judge` reads the **title and the related-code count only** -- never the
body. Live judging and backtests do not pass the same body text (see the
parity section of the report), so any body-dependent rule would skip different
pairs in the two worlds. `name`, `code` and `body` stay in the signature so
both callers share one call shape.

`related_count=None` means "the caller cannot count related codes", and the
fan-out rule then never fires -- a caller that does not know stays exactly as
permissive as before.

Default is OFF. Live and backtest must read the same switch -- use
`prefilter_enabled()`; the threshold likewise comes from `fanout_max()`.
"""

from __future__ import annotations

import os
import re

# Syndicated market-roundup / price-table columns. Each measured at <= 0.22% BUY
# rate against a 17.30% base, over 97,721 judged pairs.
ROUNDUP_TAGS: frozenset[str] = frozenset(
    {
        "표",  # n=7395, 0.00% BUY -- price/flow tables
        "개장시황",  # n=646,  0.00%
        "마감시황",  # n=631,  0.16%
        "오늘의 IR",  # n=434,  0.00%
        "오늘증시",  # n=253,  0.00%
        "티타임",  # n=246,  0.00%
        "食전食후",  # n=183,  0.00%
        "오늘의 증시일정",  # n=126,  0.00%
        "연합뉴스 이 시각 헤드라인",  # n=124,  0.00%
        "AI MY 증시전망",  # n=108,  0.00%
        "장중시황",  # n=54,   0.00%
        "ETF 시황",  # n=45,   0.00%
        "주간수급리포트",  # n=38,   0.00%
        "주간증시전망",  # n=32,   0.00%
    }
)

# Dated variants of the same columns: "[0519마감체크]", "[0810개장체크]", ...
DATED_ROUNDUP_RE = re.compile(r"^\d{4}(?:마감체크|개장체크|주요일정|장마감|마감|개장)$")

# A leading bracketed tag, in the half-width, full-width and 【】 forms Korean
# wires use interchangeably.
BRACKET_RE = re.compile(r"^\s*[\[\(（【]([^\]\)）】]{1,24})[\]\)）】]")

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Module-level default. Flip this (or set DAYTRADE_NEWS_PREFILTER) to turn the
# filter on for live judging, the sweep shadow and batch judging at once.
PREFILTER_ENABLED = False

_ENV_VAR = "DAYTRADE_NEWS_PREFILTER"

# Articles tagged with more than this many companies are dropped. 11 leaves
# enough of the 0.5% BUY-loss budget for the index rule below; N=10 alone costs
# 0.331% and would push the three-rule total over the bar on the newer half of
# the sample. 0 disables the fan-out rule.
FANOUT_MAX = 11

_FANOUT_ENV_VAR = "DAYTRADE_NEWS_PREFILTER_MAX_CODES"

# Index names. A title naming one of these on an article tagged to
# INDEX_FANOUT_MIN+ companies is index commentary ("코스피, 7999 찍고 7643
# 마감…"), which never moves one ticker on its own. Measured at 5 rather than 3
# or 4: at 3 the extra skip costs more BUYs per point than fan-out does.
INDEX_TOKENS: frozenset[str] = frozenset({"코스피", "코스닥"})

#: Fan-out floor for the index rule. 0 disables it.
INDEX_FANOUT_MIN = 5

_INDEX_ENV_VAR = "DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES"


def index_fanout_min(override: int | None = None) -> int:
    """The fan-out floor of the index-commentary rule.

    Same contract as :func:`fanout_max`: `override`, then
    `DAYTRADE_NEWS_PREFILTER_INDEX_MIN_CODES`, then :data:`INDEX_FANOUT_MIN`.
    A non-integer env value is ignored rather than raising.
    """
    if override is not None:
        return int(override)
    raw = os.environ.get(_INDEX_ENV_VAR)
    if raw is not None:
        try:
            return int(raw.strip())
        except ValueError:
            return INDEX_FANOUT_MIN
    return INDEX_FANOUT_MIN


def index_token(title: str | None, related_count: int | None, floor: int) -> str | None:
    """The index name that makes this a market-commentary mention, or None."""
    if floor <= 0 or related_count is None or related_count < floor:
        return None
    text = title or ""
    for token in sorted(INDEX_TOKENS):
        if token in text:
            return token
    return None


def fanout_max(override: int | None = None) -> int:
    """The related-code ceiling live judging and backtests both read.

    `override` wins; otherwise `DAYTRADE_NEWS_PREFILTER_MAX_CODES`; otherwise
    `FANOUT_MAX`. A non-integer env value is ignored rather than raising, so a
    typo cannot take the live daemon down.
    """
    if override is not None:
        return int(override)
    raw = os.environ.get(_FANOUT_ENV_VAR)
    if raw is not None:
        try:
            return int(raw.strip())
        except ValueError:
            return FANOUT_MAX
    return FANOUT_MAX


def prefilter_enabled(override: bool | None = None) -> bool:
    """The one switch live judging and backtests both read.

    `override` (a CLI flag or a policy field) wins; otherwise the
    `DAYTRADE_NEWS_PREFILTER` env var; otherwise `PREFILTER_ENABLED` (False).
    """
    if override is not None:
        return bool(override)
    raw = os.environ.get(_ENV_VAR)
    if raw is not None:
        return raw.strip().lower() in _TRUTHY
    return PREFILTER_ENABLED


def roundup_tag(title: str | None) -> str | None:
    """The roundup column tag this title belongs to, or None.

    Returned rather than a bare bool so callers can log *which* column was
    skipped and notice when a publisher renames one.
    """
    match = BRACKET_RE.match(title or "")
    if match is None:
        return None
    tag = match.group(1).strip()
    if tag in ROUNDUP_TAGS or DATED_ROUNDUP_RE.match(tag):
        return tag
    return None


def skip_reason(
    title: str | None,
    related_count: int | None,
    *,
    enabled: bool | None = None,
    max_codes: int | None = None,
    index_min_codes: int | None = None,
) -> str | None:
    """Why this pair is skipped, or None to judge it.

    `"roundup:<tag>"`, `"fanout:<n>"` or `"index:<token>:<n>"`, so every skip
    can be logged with the rule and value that caused it and a drifting
    publisher tag or an unexpected fan-out distribution shows up in the
    counters. The rules are checked most-specific first (roundup, then fan-out,
    then index), so a pair that trips several reports the narrowest one.
    """
    if not prefilter_enabled(enabled):
        return None
    tag = roundup_tag(title)
    if tag is not None:
        return f"roundup:{tag}"
    ceiling = fanout_max(max_codes)
    if ceiling > 0 and related_count is not None and related_count > ceiling:
        return f"fanout:{related_count}"
    token = index_token(title, related_count, index_fanout_min(index_min_codes))
    if token is not None:
        return f"index:{token}:{related_count}"
    return None


def should_judge(
    name: str | None,
    code: str | None,
    title: str | None,
    body: str | None,
    related_count: int | None = None,
    *,
    enabled: bool | None = None,
    max_codes: int | None = None,
    index_min_codes: int | None = None,
) -> bool:
    """True if this (article, ticker) pair is worth spending a judgment on.

    `related_count` is how many companies the article is tagged with -- live,
    `len(detail["companyCodes"])`; historically, the length of the C1
    related-code tuple. `None` means unknown and never drops the pair.

    Pure: no I/O, no clock, no model call. When the filter is off (the default)
    this is unconditionally True, so behaviour is byte-identical to not having
    the filter at all.
    """
    del name, code, body  # title + fan-out only by design; see the module docstring
    return (
        skip_reason(
            title,
            related_count,
            enabled=enabled,
            max_codes=max_codes,
            index_min_codes=index_min_codes,
        )
        is None
    )
