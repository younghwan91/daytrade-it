"""Build decision-time-clean features for the rlnews bandit study.

Usage: build_feats.py <period>   where period in {prior, judged}
Writes data/eval/rlnews/feats_<period>.jsonl
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "scripts/eval")
import edge_db  # noqa: E402
from costs import round_trip  # noqa: E402

REPO = Path("/home/young/git/daytrade-it-rlnews")
OUT = REPO / "data/eval/rlnews"
SENTINEL_MIN = 10080.0  # 7 days, for "no previous article for this code"

PATHS = {
    "prior": {
        "labels": OUT / "labels_prior.jsonl",
        "master": OUT / "master_prior.jsonl",
        "strata": OUT / "strata_prior.jsonl",
    },
    "judged": {
        "labels": REPO / "data/eval/prompt_v3/labels.jsonl",
        "master": REPO / "data/eval/prompt_v3/master.jsonl",
        "strata": REPO / "data/eval/edge/strata_sweep.jsonl",
    },
}


def jl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    period = sys.argv[1]
    p = PATHS[period]
    labels = list(jl(p["labels"]))
    master = {r["key"]: r for r in jl(p["master"])}
    strata = {(r["day"], r["code"]): r for r in jl(p["strata"])}

    # --- news-burst features from master (publish times) ---
    by_code: dict[str, list[tuple[dt.datetime, str]]] = collections.defaultdict(list)
    for k, r in master.items():
        pub = dt.datetime.fromisoformat(r["published_at"]).replace(tzinfo=None)
        by_code[r["code"]].append((pub, k))
    burst: dict[str, tuple[float, float]] = {}
    for items in by_code.values():
        items.sort()
        prev_pub: dt.datetime | None = None
        same_day: list[dt.datetime] = []
        for pub, k in items:
            if prev_pub is None:
                gap = SENTINEL_MIN
            else:
                gap = min(SENTINEL_MIN, (pub - prev_pub).total_seconds() / 60.0)
            n_today = sum(1 for q in same_day if q.date() == pub.date() and q < pub)
            burst[k] = (float(n_today), float(gap))
            prev_pub = pub
            same_day.append(pub)

    days = sorted({r["day"] for r in labels})
    codes_by_day: dict[str, set[str]] = collections.defaultdict(set)
    for r in labels:
        codes_by_day[r["day"]].add(r["key"].split("|", 1)[1])

    conn = edge_db.connect()
    cur = conn.cursor()
    cur.execute("select code, market from stocks")
    market = dict(cur.fetchall())

    # --- previous-day traded value, one query for the whole span ---
    cur.execute(
        "select code, date, trade_value from daily_bars_adjusted "
        "where date between %s::date - 12 and %s::date",
        (days[0], days[-1]),
    )
    dv: dict[str, list[tuple[dt.date, float]]] = collections.defaultdict(list)
    for code, d, tv in cur.fetchall():
        dv[code].append((d, float(tv or 0.0)))
    for v in dv.values():
        v.sort()

    def prev_value(code: str, day: str) -> float | None:
        d = dt.date.fromisoformat(day)
        seq = dv.get(code)
        if not seq:
            return None
        prev = [tv for dd, tv in seq if dd < d]
        return prev[-1] if prev else None

    out = (OUT / f"feats_{period}.jsonl").open("w", encoding="utf-8")  # noqa: SIM115
    n_out = 0
    missing = collections.Counter()
    for day in days:
        codes = sorted(codes_by_day[day] | {"069500"})
        cur.execute(
            "select code, ts, open, close from minute_bars "
            "where ts::date = %s::date and code = any(%s) "
            "and ts::time >= '09:00' and ts::time <= '15:30' order by code, ts",
            (day, codes),
        )
        bars: dict[str, list[tuple[int, float, float]]] = collections.defaultdict(list)
        for code, ts, o, c in cur.fetchall():
            m = ts.hour * 60 + ts.minute - 540
            bars[code].append((m, float(o), float(c)))

        mkt = bars.get("069500", [])
        mkt_open = mkt[0][1] if mkt else None

        for r in labels:
            if r["day"] != day:
                continue
            key = r["key"]
            code = key.split("|", 1)[1]
            act = int(r["act_min"])
            st = strata.get((day, code))
            if st is None or st.get("sigma") is None or st.get("sig_q") is None:
                missing["strata"] += 1
                continue
            pv = prev_value(code, day)
            if pv is None or pv <= 0:
                missing["prev_value"] += 1
                continue
            b = bars.get(code, [])
            done = [x for x in b if x[0] < act]
            win = done[-5:]
            if win:
                base = win[0][1] if len(win) == 1 else win[0][2]
                ret_5m = (win[-1][2] / base - 1.0) if base > 0 else 0.0
            else:
                ret_5m = 0.0
            if mkt_open and mkt_open > 0:
                mdone = [x for x in mkt if x[0] < act]
                mkt_move = (mdone[-1][2] / mkt_open - 1.0) if mdone else 0.0
                has_mkt = True
            else:
                mkt_move = 0.0
                has_mkt = False
            n_today, gap = burst.get(key, (0.0, SENTINEL_MIN))
            mk = market.get(code, "거래소")
            cost = round_trip(day, mk, at_open=act < 5)
            out.write(
                json.dumps(
                    {
                        "key": key,
                        "day": day,
                        "code": code,
                        "month": day[:7],
                        "market": mk,
                        "ret_close": r["ret_close"],
                        "cost": cost,
                        "reward": r["ret_close"] - cost,
                        "has_mkt": has_mkt,
                        "features": {
                            "act_min": float(act),
                            "premarket": float(bool(r["premarket"])),
                            "sigma": float(st["sigma"]),
                            "sig_q": float(st["sig_q"]),
                            "val_q": float(st["val_q"]),
                            "log_prev_value": math.log(pv),
                            "ret_5m": ret_5m,
                            "mkt_move": mkt_move,
                            "n_articles_today_before": n_today,
                            "minutes_since_prev_article": gap,
                        },
                    }
                )
                + "\n"
            )
            n_out += 1
        print(f"{day} done n={n_out}", file=sys.stderr)
    out.close()
    print(json.dumps({"period": period, "n": n_out, "missing": dict(missing)}))


if __name__ == "__main__":
    main()
