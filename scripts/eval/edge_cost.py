"""What does a round trip actually cost, and what must a signal clear?

Every bp figure in this investigation nets out `COST = 0.00215`, and the whole
question has become "is +1.6 bp of selection alpha worth anything against that
round trip". So the round trip is now load-bearing and it is worth checking
that 21.5 bp is the right number. It is not.

Four components, from the broker's own model
(`krx_quant_core.costs.model`, which `daytrade_it` configures and
`PaperBroker` charges):

  tax         0.20% on the SELL only -- KOSPI 0.05% 증권거래세 + 0.15% 농특세,
              KOSDAQ 0.20% 증권거래세. Same total either way.
  commission  0.015% per side, so 0.03% per round trip.
  slippage    0.05% one way on KOSPI, 0.15% on KOSDAQ -- a modelling choice,
              checked below against the exchange's own tick grid.
  spread      not a separate line in the model; it is what `slippage` stands
              in for. The floor is mechanical: crossing a one-tick spread
              costs one tick per round trip (buy the ask, sell the bid), and
              the tick is set by KRX's price bands.

The research constant omits slippage entirely and counts commission once
instead of twice. This prints the gap, per market and weighted by the pool's
own mix, and then the tick-grid floor measured on the actual entry prices --
which is the part no configuration can argue away.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BP = 10_000

RESEARCH_COST = 0.00215
TAX = 0.0020  # sell only, both markets
COMMISSION_PER_SIDE = 0.00015
SLIPPAGE_ONE_WAY = {"거래소": 0.0005, "코스닥": 0.0015}

#: KRX price bands (2023 revision), won. A quote moves in these steps, so the
#: narrowest possible spread is one of them.
TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
TOP_TICK = 1_000


def tick_for(price: float) -> int:
    for bound, tick in TICKS:
        if price < bound:
            return tick
    return TOP_TICK


def modelled(market: str) -> dict[str, float]:
    slip = SLIPPAGE_ONE_WAY.get(market, SLIPPAGE_ONE_WAY["거래소"])
    return {
        "tax": TAX,
        "commission": 2 * COMMISSION_PER_SIDE,
        "slippage": 2 * slip,
        "total": TAX + 2 * COMMISSION_PER_SIDE + 2 * slip,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/eval/prompt_v3/labels.jsonl")
    ap.add_argument("--master", default="data/eval/prompt_v3/master.jsonl")
    args = ap.parse_args()

    market_of: dict[str, str] = {}
    with Path(args.master).open() as f:
        for line in f:
            m = json.loads(line)
            market_of[m["key"]] = m.get("market") or "거래소"

    entries: dict[str, list[float]] = defaultdict(list)
    ticks: dict[str, list[float]] = defaultdict(list)
    mix: Counter[str] = Counter()
    with Path(args.labels).open() as f:
        for line in f:
            r = json.loads(line)
            e = r.get("entry")
            if not e or e <= 0:
                continue
            mkt = market_of.get(r["key"], "거래소")
            mix[mkt] += 1
            entries[mkt].append(e)
            ticks[mkt].append(tick_for(e) / e)

    total = sum(mix.values())
    print(
        f"\n{total:,} entries   " + "  ".join(f"{k} {v / total:.1%}" for k, v in mix.most_common())
    )

    print(f"\n--- modelled round trip, bp        {'tax':>8}{'comm':>8}{'slip':>8}{'TOTAL':>9}")
    weighted = 0.0
    for mkt, n in mix.most_common():
        c = modelled(mkt)
        weighted += c["total"] * n / total
        print(
            f"{mkt:<34}{c['tax'] * BP:>8.1f}{c['commission'] * BP:>8.1f}"
            f"{c['slippage'] * BP:>8.1f}{c['total'] * BP:>9.1f}"
        )
    print(f"{'pool-weighted':<34}{'':>8}{'':>8}{'':>8}{weighted * BP:>9.1f}")
    print(
        f"{'research constant in use':<34}{TAX * BP:>8.1f}{COMMISSION_PER_SIDE * BP:>8.1f}"
        f"{0.0:>8.1f}{RESEARCH_COST * BP:>9.1f}"
    )
    print(f"{'understated by':<34}{'':>8}{'':>8}{'':>8}{(weighted - RESEARCH_COST) * BP:>9.1f}")

    print("\n--- tick-grid floor: one tick / entry price, bp (the narrowest possible spread)")
    print(f"{'market':<12}{'n':>9}{'median':>9}{'mean':>9}{'p75':>9}{'p90':>9}")
    all_t: list[float] = []
    for mkt, n in mix.most_common():
        t = sorted(ticks[mkt])
        all_t += t
        print(
            f"{mkt:<12}{n:>9,}{statistics.median(t) * BP:>9.1f}{statistics.fmean(t) * BP:>9.1f}"
            f"{t[3 * len(t) // 4] * BP:>9.1f}{t[9 * len(t) // 10] * BP:>9.1f}"
        )
    all_t.sort()
    print(
        f"{'pooled':<12}{len(all_t):>9,}{statistics.median(all_t) * BP:>9.1f}"
        f"{statistics.fmean(all_t) * BP:>9.1f}{all_t[3 * len(all_t) // 4] * BP:>9.1f}"
        f"{all_t[9 * len(all_t) // 10] * BP:>9.1f}"
    )

    floor = TAX + 2 * COMMISSION_PER_SIDE + statistics.fmean(all_t)
    print(
        f"\ntax + commission + ONE tick of spread = {floor * BP:.1f} bp"
        f"   (the model's slippage assumption gives {weighted * BP:.1f} bp)"
    )
    print(f"\nthe bar: a strategy must earn more than {weighted * BP:.0f} bp of selection alpha")
    print("measured selection alpha on this pool: +1.6 bp (se 3.4)\n")


if __name__ == "__main__":
    main()
