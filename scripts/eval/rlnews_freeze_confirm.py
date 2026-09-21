"""Freeze the final-confirmation windows BEFORE any search touches the data.

Written and committed first, on purpose. A holdout chosen after seeing results
is not a holdout, and "we set one aside" is only checkable if the day list is
in version control with a timestamp that precedes the search.

One window per track, each the most recent days of that track's usable range,
so the confirmation is always forward in time relative to everything fitted.
Feature selection, family selection and hyperparameters all happen on the
earlier days only; these are touched once, at the end.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path.home() / "git" / "daytrade-it-rlnews"
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import edge_db  # noqa: E402

#: Track A judged days are scarce (125), so 20 is about the most that can be
#: held back while leaving the search something to work with.
A_JUDGED_CONFIRM = 20
#: Track A's judgment-free prior holdout is larger; 30 days is ~20% of it.
A_PRIOR_CONFIRM = 30
#: Track B has 722 hourly days, so 120 (~17%) costs the search little.
B_HOURLY_CONFIRM = 120

cur = edge_db.connect().cursor()
out: dict = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds")}

judged = sorted(
    {json.loads(line)["day"] for line in (ROOT / "data/eval/prompt_v3/labels.jsonl").open()}
)
out["track_a_judged"] = {
    "all_days": len(judged),
    "search": [judged[0], judged[-A_JUDGED_CONFIRM - 1]],
    "confirm": judged[-A_JUDGED_CONFIRM:],
}

prior = sorted(
    {json.loads(line)["day"] for line in (ROOT / "data/eval/rlnews/labels_prior.jsonl").open()}
)
out["track_a_prior"] = {
    "all_days": len(prior),
    "search": [prior[0], prior[-A_PRIOR_CONFIRM - 1]],
    "confirm": prior[-A_PRIOR_CONFIRM:],
}

d = ds.dataset(
    Path.home() / "git/daytrade-it/data/yahoo_hourly/bars", format="parquet", partitioning="hive"
)
t = d.to_table(columns=["ts", "identity_verified", "irregular_session"])
clean = t.filter(
    pc.and_(pc.equal(t["identity_verified"], True), pc.equal(t["irregular_session"], False))
)
hdays = sorted(set(pc.strftime(clean["ts"], format="%Y-%m-%d").to_pylist()))
out["track_b_hourly"] = {
    "all_days": len(hdays),
    "search": [hdays[0], hdays[-B_HOURLY_CONFIRM - 1]],
    "confirm": hdays[-B_HOURLY_CONFIRM:],
}

p = ROOT / "data/eval/rlnews/CONFIRM_WINDOWS.json"
p.write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if isinstance(v, dict):
        print(
            f"{k}: {v['all_days']} days, search {v['search'][0]}..{v['search'][1]}, "
            f"confirm {v['confirm'][0]}..{v['confirm'][-1]} ({len(v['confirm'])} days)"
        )
print(f"\n-> {p}")
