"""Row-for-row isolation check: feats3_prior.jsonl vs feats5_prior.jsonl.

Every field except the MA ordinal (ma_count_* -> ma_frac_*) must be identical,
compared both as parsed values (NaN == NaN) and as serialized JSON text.
Also checks ma_frac == bucket(ma_count / ma_n_avail) on every row.

    python data/eval/rlnews/feats5_isolation_check.py
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

OUT = Path("/home/young/git/daytrade-it-rlnews/data/eval/rlnews")
sys.path.insert(0, str(OUT))
from feats5_build import frac_bucket  # noqa: E402

OLD_ONLY = {"ma_count_daily", "ma_count_intraday"}
NEW_ONLY = {"ma_frac_daily", "ma_frac_intraday"}


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b and type(a) is type(b)


def main() -> None:
    f3 = (OUT / "feats3_prior.jsonl").open(encoding="utf-8")
    f5 = (OUT / "feats5_prior.jsonl").open(encoding="utf-8")
    n = 0
    diffs: collections.Counter = collections.Counter()
    text_diffs: collections.Counter = collections.Counter()
    examples: list = []
    key_order_ok = True
    frac_mismatch = 0
    old_zero_new_nan = collections.Counter()
    denoms_by_count = {
        "daily": collections.defaultdict(set),
        "intraday": collections.defaultdict(set),
    }
    navail_d = collections.Counter()
    for l3, l5 in zip(f3, f5, strict=True):
        a, b = json.loads(l3), json.loads(l5)
        n += 1
        top = (set(a) | set(b)) - {"features"}
        for k in top:
            if not same(a.get(k), b.get(k)):
                diffs[k] += 1
                if len(examples) < 5:
                    examples.append((n, k, a.get(k), b.get(k)))
            if json.dumps(a.get(k)) != json.dumps(b.get(k)):
                text_diffs[k] += 1
        fa, fb = a["features"], b["features"]
        if set(fa) - OLD_ONLY != set(fb) - NEW_ONLY:
            diffs["__feature_keyset__"] += 1
        la = [k if k not in OLD_ONLY else k.replace("count", "frac") for k in fa]
        if la != list(fb):
            key_order_ok = False
        for k in set(fa) - OLD_ONLY:
            if not same(fa[k], fb.get(k)):
                diffs["f:" + k] += 1
                if len(examples) < 5:
                    examples.append((n, k, fa[k], fb.get(k)))
            if json.dumps(fa[k]) != json.dumps(fb.get(k)):
                text_diffs["f:" + k] += 1
        for res in ("daily", "intraday"):
            cnt, nav = fa[f"ma_count_{res}"], fa[f"ma_n_avail_{res}"]
            new = fb[f"ma_frac_{res}"]
            denoms_by_count[res][cnt].add(nav)
            exp = frac_bucket(cnt / nav) if nav > 0 else float("nan")
            if not same(float(exp), float(new)):
                frac_mismatch += 1
            if nav == 0:
                old_zero_new_nan[(res, cnt, "nan" if new != new else new)] += 1
        navail_d[fa["ma_n_avail_daily"]] += 1

    conflate = {}
    for res, d in denoms_by_count.items():
        conflate[res] = {
            "values_with_multiple_denominators": sum(1 for v in d.values() if len(v) > 1),
            "n_values": len(d),
            "denominators_by_count": {str(k): sorted(v) for k, v in sorted(d.items())},
        }
    report = {
        "n_rows": n,
        "row_count_equal_and_zip_strict": True,
        "non_ma_value_diffs": dict(diffs),
        "non_ma_text_diffs": dict(text_diffs),
        "examples": examples,
        "feature_key_order_preserved": key_order_ok,
        "frac_bucket_consistency_mismatches": frac_mismatch,
        "n_avail_zero_rows": {
            f"{k[0]}|old_count={k[1]}|new={k[2]}": v for k, v in old_zero_new_nan.items()
        },
        "old_encoding_conflation": conflate,
        "ma_n_avail_daily_dist": {str(k): v for k, v in sorted(navail_d.items())},
        "ma_n_avail_daily_eq7_share": navail_d[7.0] / n,
        "verdict": "ISOLATED" if not diffs and not text_diffs and frac_mismatch == 0 else "MOVED",
    }
    (OUT / "feats5_prior_isolation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
