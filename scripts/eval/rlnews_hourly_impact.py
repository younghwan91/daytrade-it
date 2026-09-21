"""Which reported hourly numbers moved when the 15:00 bar was rebuilt?

The fix changed the synthetic 15:00 bar only. Anything that never read hour 15
must be bit-identical; anything that did must be re-run. This proves which is
which rather than assuming it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

YAHOO = Path("/home/young/git/daytrade-it/data/yahoo_hourly")
OUT = Path.home() / "git/daytrade-it-rlnews/data/eval/rlnews"

d = ds.dataset(YAHOO / "bars", format="parquet", partitioning="hive")

# -- 1. the frozen confirm window is a DAY LIST; did any day vanish? --------
t = d.to_table(columns=["ts", "identity_verified", "irregular_session", "recovery_ok"])
old = t.filter(
    pc.and_(pc.equal(t["identity_verified"], True), pc.equal(t["irregular_session"], False))
)
new = t.filter(
    pc.and_(
        pc.and_(pc.equal(t["identity_verified"], True), pc.equal(t["irregular_session"], False)),
        pc.equal(t["recovery_ok"], True),
    )
)
dold = set(pc.strftime(old["ts"], format="%Y-%m-%d").to_pylist())
dnew = set(pc.strftime(new["ts"], format="%Y-%m-%d").to_pylist())
win = json.loads((OUT / "CONFIRM_WINDOWS.json").read_text())["track_b_hourly"]
print(f"trading days  old filter {len(dold)}  new filter {len(dnew)}  lost {sorted(dold - dnew)}")
print(
    f"frozen confirm window still valid: "
    f"{set(win['confirm']).issubset(dnew)}  ({len(win['confirm'])} days)"
)

# -- 2. exactness gate: recompute entry/ret_close for the matched-null pool --
#    on the REBUILT bars, slots 0..5 only, and compare to what was reported.
lab = [json.loads(line) for line in (OUT / "labels_hourly.jsonl").open()]
print(f"\nlabels_hourly rows: {len(lab):,}")
days = sorted({r["day"] for r in lab})
sample_days = days[::37]  # deterministic spread over the span
sub = [r for r in lab if r["day"] in set(sample_days)]
print(f"exactness sample: {len(sub):,} pairs over {len(sample_days)} days")

df = d.to_table(
    columns=["code", "ts", "open", "close", "identity_verified", "irregular_session"],
    filter=(ds.field("identity_verified") == True)  # noqa: E712
    & (ds.field("irregular_session") == False),  # noqa: E712
).to_pandas()
df["day"] = df.ts.dt.strftime("%Y-%m-%d")
df["slot"] = df.ts.dt.hour - 9
df = df[(df.slot >= 0) & (df.slot < 6)]  # 09:00..14:00 -- never hour 15
df = df[df.day.isin(sample_days)]
op = {(r.day, r.code, r.slot): r.open for r in df.itertuples()}
# exit = close of the LAST available slot 0..5
last = df.sort_values("slot").groupby(["day", "code"]).close.last().to_dict()

bad_e = bad_r = miss = 0
for r in sub:
    code = r["key"].split("|")[1]
    e = op.get((r["day"], code, r["act_slot"]))
    x = last.get((r["day"], code))
    if e is None or x is None:
        miss += 1
        continue
    if abs(e - r["entry"]) > 1e-9:
        bad_e += 1
    if abs((x / e - 1) - r["ret_close"]) > 1e-12:
        bad_r += 1
print(f"entry mismatches {bad_e}   ret_close mismatches {bad_r}   unmatched {miss}")

# -- 3. did the manifest's adjustment_step set change? ----------------------
man = json.loads((YAHOO / "_MANIFEST.json").read_text())
flags = man.get("adjustment_step") or man.get("defects", {}).get("adjustment_step")
n = len(flags) if hasattr(flags, "__len__") else "n/a"
print(f"\nmanifest adjustment_step entries: {n} (dividend method B keys off these)")
