"""The long end of the holding-time curve, in a form scalp-it can concatenate.

scalp-it has measured the short end (seconds to minutes) and we have the long
end (a session to forty days). They are two halves of one question: **how much
price movement is available per unit of holding time, against a tax-and-
commission toll that is charged per round trip and does not amortise until the
horizon lengthens.**

Emitted per horizon, so the two series can be laid end to end:

  minutes        holding time, so a 10-second point and a 40-day point sit on
                 one axis
  alpha_bp       our selection's excess over a matched null at that horizon --
                 the same day, the same acting minute, the same volatility x
                 liquidity cell. NOT an oracle: the exit is fixed in advance at
                 the close of day d+h, never chosen with hindsight.
  se_bp          moving-block bootstrap at the holding length, because
                 consecutive windows of a multi-day hold overlap
  cost_bp        one round trip, from the broker's own model
  net_bp         alpha - cost
  per_day_bp     net / holding days -- the comparison that matters when
                 capital is occupied rather than released at the close

Two universes are emitted: our whole judged pool, and the same pool minus
yesterday's biggest movers, which is where the multi-day alpha turns out to
live.

The drift term is deliberately NOT in `net`. It is regime, not structure --
at +40d the market swings from -366 bp (2026) to +776 (2020) by year -- so
folding it in would make the curve a statement about 2026. `edge_years.py`
carries it separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_persist import COST, HORIZONS, summarise  # noqa: E402

BP = 10_000
#: KRX continuous session, 09:00-15:30
MINUTES_PER_DAY = 390


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/eval/edge/horizon_pairs.jsonl")
    ap.add_argument("--movers", default="data/eval/edge/movers.jsonl")
    ap.add_argument("--sigma", default="data/eval/prompt_v3/sigma.json")
    ap.add_argument("--out", default="data/eval/edge/holding_curve.csv")
    args = ap.parse_args()

    sigma = json.loads(Path(args.sigma).read_text()) if Path(args.sigma).exists() else None
    movers: dict[tuple[str, str], bool] = {}
    with Path(args.movers).open() as f:
        for line in f:
            m = json.loads(line)
            movers[(m["day"], m["code"])] = m["movers"]

    rows = []
    with Path(args.pairs).open() as f:
        for line in f:
            r = json.loads(line)
            r["movers"] = movers.get((r["day"], r["code"]))
            rows.append(r)
    # only days where every horizon resolves, so a longer horizon is not also
    # an earlier sample
    ok = {r["day"] for r in rows if all(h in r["r"] for h in HORIZONS)}
    rows = [r for r in rows if r["day"] in ok]

    # Exposure, not calendar. A same-session hold is entered mid-session, so it
    # is only the REMAINING minutes; each extra day adds a whole session. Using
    # calendar days would put "same session" and "+1d" at the same point on a
    # shared time axis, which is exactly the axis this file exists to build.
    mean_act = sum(r["act_min"] for r in rows) / len(rows)
    same_session = MINUTES_PER_DAY - mean_act

    pools = {
        "all judged": rows,
        "ex-movers (calm two-thirds)": [r for r in rows if r.get("movers") is False],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["pool", "days", "minutes", "n", "alpha_bp", "se_bp", "cost_bp", "net_bp", "per_day_bp"]
        )
        for label, pool in pools.items():
            print(f"\n=== {label}   ({len(ok)} common days)")
            print(
                f"{'hold':<12}{'minutes':>9}{'n':>9}{'alpha':>9}{'se':>7}"
                f"{'cost':>7}{'net':>9}{'per day':>9}"
            )
            for h in HORIZONS:
                s = summarise(pool, h, True, sigma)
                if s is None:
                    continue
                minutes = same_session + MINUTES_PER_DAY * int(h)
                days = minutes / MINUTES_PER_DAY
                net = s["alpha"] - COST
                w.writerow(
                    [
                        label,
                        round(days, 3),
                        round(minutes),
                        s["n"],
                        round(s["alpha"] * BP, 2),
                        round(s["alpha_se"] * BP, 2),
                        round(COST * BP, 2),
                        round(net * BP, 2),
                        round(net / days * BP, 3),
                    ]
                )
                print(
                    f"{('rest of day' if h == '0' else f'+{h}d'):<12}{minutes:>9,.0f}{s['n']:>9,}"
                    f"{s['alpha'] * BP:>+9.1f}{s['alpha_se'] * BP:>7.1f}{COST * BP:>7.1f}"
                    f"{net * BP:>+9.1f}{net / days * BP:>+9.2f}"
                )
    print(f"\n-> {out}")
    print(
        "\nalpha is excess over a matched null with a pre-committed exit, not an "
        "oracle ceiling;\nthe drift term is excluded on purpose (regime, not "
        "structure -- see edge_years.py).\n"
    )


if __name__ == "__main__":
    main()
