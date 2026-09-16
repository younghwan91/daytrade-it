"""Master table of every judged (article, code) pair: metadata + v2 judgment + decide_at.

Read-only over existing files. No DB, no API.
Out: data/eval/prompt_v3/master.jsonl
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

D = Path("data/eval")
OUT = Path("data/eval/prompt_v3/master.jsonl")

META_FIELDS = (
    "code",
    "name",
    "market",
    "sector",
    "title",
    "summary",
    "category",
    "day",
    "month",
    "session",
    "published_at",
    "related_n",
)


def main() -> None:
    # 1. judgments
    judged: dict[str, dict] = {}
    with (D / "cache/universe_v2.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            judged[row["key"]] = row["v2"]
    print(f"judged: {len(judged)}")

    # 2. metadata (n300 first: it carries related_n; a6 fills the rest)
    meta: dict[str, dict] = {}
    for name in ("universe_pairs_n300.jsonl", "universe_pairs_a6.jsonl"):
        n_new = 0
        with (D / name).open() as f:
            for line in f:
                row = json.loads(line)
                k = row["key"]
                if k not in judged:
                    continue
                if k in meta:
                    if meta[k].get("related_n") is None and row.get("related_n") is not None:
                        meta[k]["related_n"] = row["related_n"]
                    continue
                meta[k] = {fld: row.get(fld) for fld in META_FIELDS}
                n_new += 1
        print(f"{name}: +{n_new} (total {len(meta)})")

    # 3. decide_at from the two event dirs
    decide: dict[str, tuple[str, str]] = {}
    for d in ("replay_events_universe_all", "n300_t30/events_all"):
        n_new = 0
        for path in sorted(glob.glob(str(D / d / "*.jsonl"))):
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    k = row["key"]
                    if k in decide:
                        continue
                    decide[k] = (row["decide_at"], row["batch_at"])
                    n_new += 1
        print(f"{d}: +{n_new} (total {len(decide)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_out = n_nometa = n_nodecide = 0
    with OUT.open("w") as out:
        for k, v2 in judged.items():
            m = meta.get(k)
            if m is None:
                n_nometa += 1
                continue
            dec = decide.get(k)
            if dec is None:
                n_nodecide += 1
                continue
            row = {"key": k, **m, "decide_at": dec[0], "batch_at": dec[1], "v2": v2}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"written: {n_out}  (no meta {n_nometa}, no decide_at {n_nodecide}) -> {OUT}")


if __name__ == "__main__":
    main()
