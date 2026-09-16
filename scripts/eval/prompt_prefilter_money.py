"""Retune the prefilter against money instead of BUY recall.

`mention_prefilter` is tuned to keep BUY *recall*: every dropped BUY counts as a
cost, and FANOUT_MAX=11 was chosen as the smallest N inside a 0.5% recall bar.
With realised returns the assumption is testable, and it is wrong -- the BUYs
the filter drops lose more than the ones it keeps. This sweeps the threshold
against the kept BUYs' mean return instead, with a time split so the pick is not
fitted on the months it is judged on.
"""

from __future__ import annotations

import json
import statistics as st
import sys

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import COST, D  # noqa: E402

from daytrade_it.application.news.mention_prefilter import should_judge  # noqa: E402

SPLIT = "2026-07"  # months before this are the older half


def main() -> None:
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    buys = []
    with (D / "labels.jsonl").open() as f:
        for line in f:
            lb = json.loads(line)
            m = master[lb["key"]]
            if (m["v2"].get("score") or 0) > 0.15:
                buys.append((m, lb["ret_close"] - COST))
    print(f"BUY pairs with an outcome: {len(buys)}")

    def report(name: str, n_max: int | None) -> None:
        rows = []
        for m, rt in buys:
            ok = should_judge(
                m.get("name") or "",
                m["code"],
                m["title"],
                m.get("summary") or "",
                related_count=m.get("related_n"),
                enabled=True,
                max_codes=n_max,
            )
            rows.append((m["month"], ok, rt))
        kept = [rt for _, ok, rt in rows if ok]
        old = [rt for mo, ok, rt in rows if ok and mo < SPLIT]
        new = [rt for mo, ok, rt in rows if ok and mo >= SPLIT]
        drop = 1 - len(kept) / len(rows)
        print(
            f"  {name:16s} kept {len(kept):6d} ({1 - drop:5.1%})  "
            f"mean {st.mean(kept) * 1e4:+7.0f} bp   "
            f"older {st.mean(old) * 1e4:+7.0f}   newer {st.mean(new) * 1e4:+7.0f}"
        )

    base = [rt for _, rt in buys]
    print(
        f"\n  {'no prefilter':16s} kept {len(base):6d} (100.0%)  "
        f"mean {st.mean(base) * 1e4:+7.0f} bp   "
        f"older {st.mean([rt for m, rt in buys if m['month'] < SPLIT]) * 1e4:+7.0f}   "
        f"newer {st.mean([rt for m, rt in buys if m['month'] >= SPLIT]) * 1e4:+7.0f}"
    )
    for n in (30, 20, 15, 11, 8, 6, 5, 4, 3, 2, 1):
        report(f"fan-out <= {n}", n)


if __name__ == "__main__":
    main()
