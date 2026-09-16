"""Is the prefilter's 'BUY recall loss' actually a loss of money?

The prefilter is tuned to keep BUY *recall*, treating every dropped BUY as a
cost. With realised returns on 132,185 pairs that assumption is testable: a
dropped BUY only costs something if the BUYs it drops made money.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, "scripts/eval")
from prompt_ceiling import COST, D, title_tag  # noqa: E402

from daytrade_it.application.news.mention_prefilter import should_judge  # noqa: E402


def main() -> None:
    master = {}
    with (D / "master.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            master[r["key"]] = r
    rows = []
    with (D / "labels.jsonl").open() as f:
        for line in f:
            lb = json.loads(line)
            rows.append((master[lb["key"]], lb, lb["ret_close"] - COST))

    buys = [(m, lb, rt) for m, lb, rt in rows if (m["v2"].get("score") or 0) > 0.15]
    print(
        f"BUY pairs with an outcome: {len(buys)}  mean {st.mean([r for *_, r in buys]) * 1e4:+.0f} bp"
    )
    print(f"whole pool: {len(rows)}  mean {st.mean([r for *_, r in rows]) * 1e4:+.0f} bp")

    kept, dropped = [], []
    for m, lb, rt in buys:
        ok = should_judge(
            m.get("name") or "",
            m["code"],
            m["title"],
            m.get("summary") or "",
            related_count=m.get("related_n"),
            enabled=True,
        )
        (kept if ok else dropped).append((m, lb, rt))
    print(
        f"\nprefilter on the BUYs: kept {len(kept)}, dropped {len(dropped)} "
        f"({len(dropped) / len(buys):.2%})"
    )
    if kept:
        print(
            f"  kept BUYs    mean {st.mean([r for *_, r in kept]) * 1e4:+7.0f} bp  "
            f"win {sum(1 for *_, r in kept if r > 0) / len(kept):.1%}"
        )
    if dropped:
        print(
            f"  dropped BUYs mean {st.mean([r for *_, r in dropped]) * 1e4:+7.0f} bp  "
            f"win {sum(1 for *_, r in dropped if r > 0) / len(dropped):.1%}"
        )
        tags: dict[str, list[float]] = defaultdict(list)
        for m, _, rt in dropped:
            tags[title_tag(m["title"])].append(rt)
        print("  dropped BUYs by title tag:")
        for t, v in sorted(tags.items(), key=lambda kv: -len(kv[1]))[:10]:
            print(f"    {t:26s} n={len(v):4d}  mean {st.mean(v) * 1e4:+7.0f} bp")

    print("\n=== 오늘의 주요공시, specifically ===")
    dis = [(m, lb, rt) for m, lb, rt in rows if "오늘의 주요공시" in m["title"]]
    dbuy = [x for x in dis if (x[0]["v2"].get("score") or 0) > 0.15]
    print(
        f"  all pairs n={len(dis)}  mean {st.mean([r for *_, r in dis]) * 1e4:+.0f} bp"
        if dis
        else "  none"
    )
    if dbuy:
        print(
            f"  of which BUY n={len(dbuy)}  mean {st.mean([r for *_, r in dbuy]) * 1e4:+.0f} bp"
            f"  win {sum(1 for *_, r in dbuy if r > 0) / len(dbuy):.1%}"
        )


if __name__ == "__main__":
    main()
