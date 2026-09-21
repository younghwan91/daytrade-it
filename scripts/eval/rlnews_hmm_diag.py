"""Step 3: diagnostics on the one config the search picked (legs, K=3).

Answers three questions the headline numbers leave open:
  - is the confirm-span state-0 result a centre-of-mass effect or a few days?
  - does the per-state ordering of the next-day legs hold in the confirm years
    too, and in the early years the expanding-window burn-in cost us?
  - why does the judge say p(mu>0)=0.009 on a +2.62 bp/day arithmetic delta?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/young/git/daytrade-it-rlnews")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from hmm_causal import (  # noqa: E402
    OBS_SETS,
    WIN,
    expanding_filter,
    hard,
    load,
    nets_bp,
)
from hmm_core import visit_counts  # noqa: E402

BP = 10_000.0


def main():
    rows = load()
    days = [r["day"] for r in rows]
    yr = np.array([int(d[:4]) for d in days])
    lo_s, hi_s = WIN["track_c_daily"]["search"]
    lo_c, hi_c = WIN["track_c_daily"]["confirm"]
    in_s = np.array([lo_s <= d <= hi_s for d in days])
    in_c = np.array([lo_c <= d <= hi_c for d in days])

    net = nets_bp(rows)
    night = np.array([r["t_night"] * BP for r in rows])
    intra = np.array([r["t_intra"] * BP for r in rows])
    nm = (
        np.array([r["t_night_mover"] if r["t_night_mover"] is not None else np.nan for r in rows])
        * BP
    )
    ncm = (
        np.array([r["t_night_calm"] if r["t_night_calm"] is not None else np.nan for r in rows])
        * BP
    )

    alpha, fits = expanding_filter(rows, OBS_SETS["legs"], 3)
    lab = hard(alpha)
    ok = lab >= 0
    print(
        f"expanding filter: first classified day {days[int(np.argmax(ok))]}, "
        f"{len(fits)} refits, fit sizes {fits[0]['n_fit']}..{fits[-1]['n_fit']} days"
    )

    print("\n--- fitted transition matrix at the FIRST and LAST refit (rows sum to 1)")
    for tag, f in (("first " + fits[0]["day"], fits[0]), ("last " + fits[-1]["day"], fits[-1])):
        A = np.array(f["A"])
        print(f"  {tag}: " + " | ".join("[" + " ".join(f"{x:.3f}" for x in r) + "]" for r in A))

    print(
        "\n--- state-visit counts in the data the PARAMETERS were fit on "
        "(last refit, in-sample labels)"
    )
    d_, v_ = visit_counts(lab[ok & (np.arange(len(rows)) < fits[-1]["n_fit"])])
    print(f"  days {d_}  distinct visits {v_}")

    for tag, mask in (("SEARCH", in_s & ok), ("CONFIRM", in_c & ok)):
        print(f"\n=== {tag} span, per filtered state")
        for k in range(3):
            m = mask & (lab == k)
            if m.sum() == 0:
                continue
            s = net[m]
            srt = np.sort(s)
            top5 = srt[-5:].sum() / len(s)
            print(
                f"  s{k}: n={m.sum():4d}  night {night[m].mean():+6.1f}  "
                f"intra {intra[m].mean():+6.1f}  net mean {s.mean():+6.1f}  "
                f"median {np.median(s):+6.1f}  10%-trim {s[(s >= np.percentile(s, 10)) & (s <= np.percentile(s, 90))].mean():+6.1f}  "
                f"frac>0 {np.mean(s > 0):.2f}  top-5-days contribute {top5:+6.1f} bp/day of the mean"
            )
            b = np.array([np.mean(np.random.default_rng(i).choice(s, len(s))) for i in range(2000)])
            print(
                f"       bootstrap mean 5-95%: [{np.percentile(b, 5):+6.1f}, {np.percentile(b, 95):+6.1f}]"
            )

    print("\n--- ordering of next-day legs by state, EVERY year the filter covers")
    rank = [0, 1, 2]
    okn = oki = ny = 0
    for y in sorted(set(yr[ok])):
        m = ok & (yr == y)
        if m.sum() < 60:
            print(f"  {y}  n={int(m.sum()):3d}  (skipped, <60 classified days)")
            continue
        ny += 1
        gn = {
            k: (night[m & (lab == k)].mean() if (m & (lab == k)).sum() >= 5 else None) for k in rank
        }
        gi = {
            k: (intra[m & (lab == k)].mean() if (m & (lab == k)).sum() >= 5 else None) for k in rank
        }
        cn = [gn[k] for k in rank if gn[k] is not None]
        ci = [gi[k] for k in rank if gi[k] is not None]
        a = all(cn[i] >= cn[i + 1] for i in range(len(cn) - 1))
        b = all(ci[i] >= ci[i + 1] for i in range(len(ci) - 1))
        okn += a
        oki += b
        tail = " CONFIRM" if y >= 2025 else ""
        print(
            f"  {y}  n={int(m.sum()):3d}  night "
            + " ".join(f"s{k}:{'   .  ' if gn[k] is None else f'{gn[k]:+6.1f}'}" for k in rank)
            + f"  mono={a}   intra "
            + " ".join(f"s{k}:{'   .  ' if gi[k] is None else f'{gi[k]:+6.1f}'}" for k in rank)
            + f"  mono={b}{tail}"
        )
    print(f"  overnight monotone in state rank [0,1,2]: {okn}/{ny} years; intraday {oki}/{ny}")

    print("\n--- mover vs calm overnight by state (does the state add to the known split?)")
    for tag, mask in (("SEARCH", in_s & ok), ("CONFIRM", in_c & ok)):
        for k in range(3):
            m = mask & (lab == k) & np.isfinite(nm) & np.isfinite(ncm)
            if m.sum() < 20:
                continue
            print(
                f"  {tag} s{k}: n={m.sum():4d} mover-night {nm[m].mean():+6.1f} "
                f"calm-night {ncm[m].mean():+6.1f}  (mover>calm = {nm[m].mean() > ncm[m].mean()})"
            )

    print("\n--- why the judge disagrees with the arithmetic mean on CONFIRM")
    mc = in_c & ok
    pick = np.isin(lab[mc], [0])
    d = net[mc] * pick - net[mc]  # candidate - baseline, per day
    print(
        f"  n={len(d)}  mean {d.mean():+6.2f}  median {np.median(d):+6.2f}  "
        f"days>=0 {int((d >= 0).sum())}/{len(d)}  (judge needs ceil(5/7*n)="
        f"{int(np.ceil(5 / 7 * len(d)))})"
    )
    print(
        f"  the mean is carried by the tail: top-10 |d| days sum "
        f"{np.sort(d)[-10:].sum() / len(d):+6.2f} bp/day of it"
    )
    print("  Student-t(4) likelihood centres on the bulk, not the tail -> p(mu>0) small")

    # 125 judged news days: which regime were they?
    jd = sorted(
        {
            json.loads(line)["day"]
            for line in (REPO / "data/eval/rlnews/feats_judged.jsonl").read_text().splitlines()
            if line.strip()
        }
    )
    jset = set(jd)
    mj = np.array([d in jset for d in days]) & ok
    dj, vj = visit_counts(lab[mj])
    print(f"\n--- the 125 judged news days ({jd[0]}..{jd[-1]}) sit inside the CONFIRM span")
    print(f"    filtered-state days {dj} visits {vj}  (n classified = {int(mj.sum())})")


if __name__ == "__main__":
    main()
