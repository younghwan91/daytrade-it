"""Why 58k judged pairs got no label. Counts drop reasons per day. READ-ONLY."""

from __future__ import annotations

import datetime as dt
import os
import sys
from collections import Counter, defaultdict

import psycopg2

sys.path.insert(0, "scripts/eval")
from prompt_labels import MARKET, entry_price, fetch_bars, load_master, window


def main() -> None:
    by_day = load_master()
    conn = psycopg2.connect(os.environ["KR_QUANT_DB"], connect_timeout=5)
    conn.set_session(readonly=True, autocommit=True)
    reasons: Counter[str] = Counter()
    by_month: dict[str, Counter[str]] = defaultdict(Counter)
    score_drop: Counter[str] = Counter()
    with conn.cursor() as cur:
        for day in sorted(by_day):
            rows = by_day[day]
            codes = sorted({r["code"] for r in rows} | {MARKET})
            bars = fetch_bars(cur, day, codes)
            mon = str(day)[:7]
            for r in rows:
                buy = (r["v2"].get("score") or 0) > 0.15
                bs = bars.get(r["code"])
                if not bs:
                    why = "no_bars_for_code"
                else:
                    decide = dt.datetime.fromisoformat(r["decide_at"]).replace(tzinfo=None)
                    minute = decide.replace(second=0, microsecond=0)
                    if minute.date() != day:
                        why = "decide_not_on_day"
                    elif not entry_price(bs, minute):
                        why = "no_entry_price"
                    elif not window(bs, minute):
                        why = "no_forward_bars"
                    else:
                        why = "ok"
                reasons[why] += 1
                by_month[mon][why] += 1
                if buy:
                    score_drop[why] += 1
            print(f"{day} done", flush=True)
    conn.close()
    print("\n== overall ==")
    tot = sum(reasons.values())
    for k, v in reasons.most_common():
        print(f"{k:20s} {v:7d}  {v / tot:6.2%}")
    print("\n== BUY pairs (score>0.15) ==")
    t2 = sum(score_drop.values())
    for k, v in score_drop.most_common():
        print(f"{k:20s} {v:7d}  {v / t2:6.2%}")
    print("\n== by month (ok share) ==")
    for m in sorted(by_month):
        c = by_month[m]
        t = sum(c.values())
        print(
            f"{m}  n={t:6d}  ok={c['ok'] / t:6.2%}  no_bars={c['no_bars_for_code'] / t:6.2%} "
            f"no_entry={c['no_entry_price'] / t:6.2%}  no_fwd={c['no_forward_bars'] / t:6.2%} "
            f"off_day={c['decide_not_on_day'] / t:6.2%}"
        )


if __name__ == "__main__":
    main()
