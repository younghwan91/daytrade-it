"""Is the travel-distance lift real, or only "this stock is volatile"?

rl-builder's challenge: on a September window every group including random
tickers sat at 5.3-5.8 sigma of post-entry maximum, so an unnormalised travel
measure can look predictable while only tracking which stock moves a lot.

This re-runs the MFE fold against a sigma-normalised target. Sigma is the
code's trailing 20-session close-to-close volatility **as of the previous
trading day** -- decision-time available, no look-ahead. Step 1 adds it to the
label file; step 2 refits.

If the lift survives normalisation, the model is finding which article's stock
travels further than its own baseline. If it collapses, it is only a
volatility detector and the doc's section 8 has to say so.

READ-ONLY on kr_quant.
"""

# Analysis script: the target/summary lambdas share one signature on purpose.
# ruff: noqa: ARG005

from __future__ import annotations

import datetime as dt
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

import psycopg2

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import COST, D, featurise, ridge, title_tag  # noqa: E402

LOOKBACK = 20
SIGMA_OUT = D / "sigma.json"


def build_sigma() -> dict[str, float]:
    """(code|day) -> trailing 20-session sigma, using closes strictly before `day`."""
    if SIGMA_OUT.exists():
        return json.loads(SIGMA_OUT.read_text())
    days_needed: set[str] = set()
    codes: set[str] = set()
    with (D / "labels.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            days_needed.add(r["day"])
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            codes.add(r["code"])
    lo = dt.date.fromisoformat(min(days_needed)) - dt.timedelta(days=60)
    hi = dt.date.fromisoformat(max(days_needed))
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT code, date, close FROM daily_bars_adjusted "
            "WHERE code = ANY(%s) AND date >= %s AND date <= %s ORDER BY code, date",
            (sorted(codes), lo, hi),
        )
        rows = cur.fetchall()
    conn.close()
    series: dict[str, list[tuple[dt.date, float]]] = defaultdict(list)
    for code, d, c in rows:
        if c:
            series[code].append((d, float(c)))
    out: dict[str, float] = {}
    for code, ser in series.items():
        rets: list[float] = []
        for i in range(1, len(ser)):
            prev, cur_c = ser[i - 1][1], ser[i][1]
            rets.append(cur_c / prev - 1 if prev else 0.0)
            # sigma available ON day ser[i][0] uses returns up to ser[i-1]
            window = rets[:-1][-LOOKBACK:]
            if len(window) >= 10:
                out[f"{code}|{ser[i][0]}"] = st.pstdev(window)
    SIGMA_OUT.write_text(json.dumps(out))
    print(f"sigma keys: {len(out)}")
    return out


def main() -> None:
    sigma = build_sigma()
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    rows = []
    n_nosig = 0
    with (D / "labels.jsonl").open() as f:
        for line in f:
            lb = json.loads(line)
            m = master[lb["key"]]
            s = sigma.get(f"{m['code']}|{m['day']}")
            if not s or s <= 0:
                n_nosig += 1
                continue
            rows.append((m, lb, s))
    print(f"{len(rows)} pairs with a trailing sigma ({n_nosig} without)")
    sig = [s for *_, s in rows]
    print(
        f"sigma: median {st.median(sig):.4f}, p10 {sorted(sig)[len(sig) // 10]:.4f}, "
        f"p90 {sorted(sig)[9 * len(sig) // 10]:.4f}"
    )
    print(
        f"pool mfe {st.mean([lb['mfe'] for _, lb, _ in rows]) * 1e4:+.0f} bp   "
        f"pool mfe/sigma {st.mean([lb['mfe'] / s for _, lb, s in rows]):+.3f}"
    )

    tags = [t for t, n in Counter(title_tag(m["title"]) for m, _, _ in rows).items() if n >= 200]
    sectors = [s for s, n in Counter(str(m.get("sector")) for m, _, _ in rows).items() if n >= 500]
    months = sorted({m["month"] for m, _, _ in rows})
    bym: dict[str, list] = defaultdict(list)
    for r in rows:
        bym[r[0]["month"]].append(r)

    def run(target, label: str, show) -> None:
        print(f"\n=== target: {label} ===")
        print(f"{'month':9s} {'top5':>9s} {'pool':>9s} {'lift':>7s}")
        allsel: list = []
        allpool: list = []
        for i, mo in enumerate(months):
            if i == 0:
                continue
            train = [r for mm in months[:i] for r in bym[mm]]
            test = bym[mo]
            if len(train) < 3000:
                continue
            w = ridge(
                [featurise(m, lb, tags, sectors) for m, lb, _ in train],
                [target(lb, s) for _, lb, s in train],
                lam=len(train) * 1e-3,
            )
            days: dict[str, list] = defaultdict(list)
            for m, lb, s in test:
                x = featurise(m, lb, tags, sectors)
                p = sum(a * b for a, b in zip(x, w[:-1], strict=True)) + w[-1]
                days[m["day"]].append((p, lb, s))
            sel = [
                (lb, s) for ds in days.values() for _, lb, s in sorted(ds, key=lambda z: -z[0])[:5]
            ]
            pool = [(lb, s) for _, lb, s in test]
            allsel += sel
            allpool += pool
            a, b = show(sel), show(pool)
            print(f"{mo:9s} {a:9.3f} {b:9.3f} {a / b if b else 0:6.2f}x")
        a, b = show(allsel), show(allpool)
        print("-" * 38)
        print(f"{'POOLED':9s} {a:9.3f} {b:9.3f} {a / b if b else 0:6.2f}x")

    run(
        lambda lb, s: lb["mfe"],
        "raw MFE (what the doc reported)",
        lambda xs: st.mean([lb["mfe"] for lb, _ in xs]) * 1e4,
    )
    run(
        lambda lb, s: lb["mfe"] / s,
        "MFE / trailing sigma",
        lambda xs: st.mean([lb["mfe"] / s for lb, s in xs]),
    )
    print("\n(the raw row is in bp; the normalised row is in sigmas)")

    # where does the raw lift concentrate -- which target levels?
    print("\n=== hit rate at each target, model top-5 vs pool (last fold trained OOS) ===")
    i = len(months) - 1
    train = [r for mm in months[:i] for r in bym[mm]]
    test = bym[months[i]]
    for tname, tgt in (
        ("raw MFE", lambda lb, s: lb["mfe"]),
        ("MFE/sigma", lambda lb, s: lb["mfe"] / s),
    ):
        w = ridge(
            [featurise(m, lb, tags, sectors) for m, lb, _ in train],
            [tgt(lb, s) for _, lb, s in train],
            lam=len(train) * 1e-3,
        )
        days: dict[str, list] = defaultdict(list)
        for m, lb, _s in test:
            x = featurise(m, lb, tags, sectors)
            days[m["day"]].append((sum(a * b for a, b in zip(x, w[:-1], strict=True)) + w[-1], lb))
        sel = [lb for ds in days.values() for _, lb in sorted(ds, key=lambda z: -z[0])[:5]]
        pool = [lb for _, lb, _ in test]
        print(f"  --- ranked by {tname} ---")
        print(f"  {'target':>8s} {'top5 hit':>9s} {'pool hit':>9s} {'lift':>6s}")
        for t in (0.01, 0.02, 0.03, 0.05, 0.10):
            a = sum(1 for lb in sel if lb["mfe"] >= t) / len(sel)
            b = sum(1 for lb in pool if lb["mfe"] >= t) / len(pool)
            print(f"  {t:8.0%} {a:9.1%} {b:9.1%} {a / b if b else 0:5.2f}x")
        print(
            f"  top5 close return {st.mean([lb['ret_close'] - COST for lb in sel]) * 1e4:+.0f} bp"
            f"   pool {st.mean([lb['ret_close'] - COST for lb in pool]) * 1e4:+.0f} bp"
        )


if __name__ == "__main__":
    main()
