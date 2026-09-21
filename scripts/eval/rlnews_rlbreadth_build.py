"""Breadth features for the rlnews generalisation study.

Adds, on top of the 10 existing decision-time-clean features:
  (a) all TEN v2 judgment fields   (judged period only -- prior has no judgments)
  (b) volume / turnover forms from minute_bars, relative to the code's own baseline
  (c) free deterministic attributes

Writes data/eval/rlnews/feats2_<period>.jsonl
Usage: build2.py <prior|judged>
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts/eval")
import edge_db  # noqa: E402
from costs import round_trip  # noqa: E402

REPO = Path("/home/young/git/daytrade-it-rlnews")
OUT = REPO / "data/eval/rlnews"
SENTINEL_MIN = 10080.0

PATHS = {
    "prior": {
        "labels": OUT / "labels_prior.jsonl",
        "master": OUT / "master_prior.jsonl",
        "strata": OUT / "strata_prior.jsonl",
        "null": OUT / "null_prior_strat.jsonl",
    },
    "judged": {
        "labels": REPO / "data/eval/prompt_v3/labels.jsonl",
        "master": REPO / "data/eval/prompt_v3/master.jsonl",
        "strata": REPO / "data/eval/edge/strata_sweep.jsonl",
        "null": REPO / "data/eval/edge/null_sweep_strat.jsonl",
    },
}

TAG_RE = re.compile(r"^\s*\[([^\]]{1,20})\]")


def tick_size(px: float) -> float:
    if px < 2000:
        return 1.0
    if px < 5000:
        return 5.0
    if px < 20000:
        return 10.0
    if px < 50000:
        return 50.0
    if px < 200000:
        return 100.0
    if px < 500000:
        return 500.0
    return 1000.0


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

    # stratum-matched null: (day, act_min) -> {"sig|val": mean}
    nullcell: dict[tuple[str, int], dict[str, float]] = {}
    for r in jl(p["null"]):
        nullcell[(r["day"], int(r["act_min"]))] = {
            k: v["mean"] for k, v in (r.get("cells") or {}).items() if v.get("mean") is not None
        }
    sorted({a for _, a in nullcell})

    # ---- news-burst + fan-out from master ----
    by_code: dict[str, list[tuple[dt.datetime, str]]] = collections.defaultdict(list)
    for k, r in master.items():
        pub = dt.datetime.fromisoformat(r["published_at"]).replace(tzinfo=None)
        by_code[r["code"]].append((pub, k))
    burst: dict[str, tuple[float, float]] = {}
    for items in by_code.values():
        items.sort()
        prev_pub = None
        same_day: list[dt.datetime] = []
        for pub, k in items:
            gap = (
                SENTINEL_MIN
                if prev_pub is None
                else min(SENTINEL_MIN, (pub - prev_pub).total_seconds() / 60.0)
            )
            n_today = sum(1 for q in same_day if q.date() == pub.date() and q < pub)
            burst[k] = (float(n_today), float(gap))
            prev_pub = pub
            same_day.append(pub)

    # fan-out: how many codes this article names
    art_n: collections.Counter = collections.Counter()
    for k, r in master.items():
        aid = r.get("news_id") or k.split("|", 1)[0]
        art_n[aid] += 1

    days = sorted({r["day"] for r in labels})
    codes_by_day: dict[str, set[str]] = collections.defaultdict(set)
    for r in labels:
        codes_by_day[r["day"]].add(r["key"].split("|", 1)[1])
    labels_by_day: dict[str, list] = collections.defaultdict(list)
    for r in labels:
        labels_by_day[r["day"]].append(r)

    conn = edge_db.connect()
    cur = conn.cursor()
    cur.execute("select code, market, sector from stocks")
    market, sector = {}, {}
    for c, m, s in cur.fetchall():
        market[c] = m
        sector[c] = s or "?"

    cur.execute(
        "select distinct ts::date from minute_bars where ts::date between %s::date - 20 and %s::date order by 1",
        (days[0], days[-1]),
    )
    tdays = [r[0] for r in cur.fetchall()]
    tidx = {d: i for i, d in enumerate(tdays)}

    cur.execute(
        "select code, date, close, high, low, volume, trade_value from daily_bars_adjusted "
        "where date between %s::date - 20 and %s::date",
        (days[0], days[-1]),
    )
    dbar: dict[str, list] = collections.defaultdict(list)
    for code, d, cl, hi, lo, vol, tv in cur.fetchall():
        dbar[code].append(
            (d, float(cl or 0), float(hi or 0), float(lo or 0), float(vol or 0), float(tv or 0))
        )
    for v in dbar.values():
        v.sort()

    def prev_rows(code: str, day: str, n: int):
        d = dt.date.fromisoformat(day)
        seq = dbar.get(code) or []
        pr = [x for x in seq if x[0] < d]
        return pr[-n:]

    fout = (OUT / f"feats2_{period}.jsonl").open("w", encoding="utf-8")  # noqa: SIM115
    n_out = 0
    missing: collections.Counter = collections.Counter()

    for day in days:
        codes = sorted(codes_by_day[day] | {"069500"})
        cur.execute(
            "select code, ts, open, high, low, close, volume from minute_bars "
            "where ts::date = %s::date and code = any(%s) "
            "and ts::time >= '09:00' and ts::time <= '15:30' order by code, ts",
            (day, codes),
        )
        bars: dict[str, list] = collections.defaultdict(list)
        for code, ts, o, hi, lo, cl, vol in cur.fetchall():
            m = ts.hour * 60 + ts.minute - 540
            bars[code].append((m, float(o), float(hi), float(lo), float(cl), float(vol or 0)))

        # trailing same-time-of-day volume baseline: prior 5 trading days, 10-min buckets
        di = tidx.get(dt.date.fromisoformat(day))
        base: dict[tuple[str, int], float] = {}
        if di is not None and di >= 5:
            lo_d, hi_d = tdays[di - 5], tdays[di - 1]
            cur.execute(
                "select code, (((extract(hour from ts)*60+extract(minute from ts))::int-540)/10)::int b, "
                "percentile_cont(0.5) within group (order by volume) v "
                "from minute_bars where ts::date between %s and %s and code = any(%s) "
                "and ts::time >= '09:00' and ts::time <= '15:30' group by 1,2",
                (lo_d, hi_d, codes),
            )
            for code, b, v in cur.fetchall():
                base[(code, int(b))] = float(v or 0)

        mkt = bars.get("069500", [])
        mkt_open = mkt[0][1] if mkt else None

        # prior-day traded-value rank inside this day's own candidate universe
        pv_map = {}
        for code in codes_by_day[day]:
            pr = prev_rows(code, day, 1)
            pv_map[code] = pr[-1][5] if pr else None
        vals = sorted(v for v in pv_map.values() if v)
        rank_of = {}
        for code, v in pv_map.items():
            if v:
                import bisect

                rank_of[code] = bisect.bisect_left(vals, v) / max(1, len(vals) - 1)

        dow = float(dt.date.fromisoformat(day).weekday())

        for r in labels_by_day[day]:
            key = r["key"]
            code = key.split("|", 1)[1]
            act = int(r["act_min"])
            st = strata.get((day, code))
            if st is None or st.get("sigma") is None or st.get("sig_q") is None:
                missing["strata"] += 1
                continue
            pv = pv_map.get(code)
            if not pv or pv <= 0:
                missing["prev_value"] += 1
                continue
            m = master.get(key)
            if m is None:
                missing["master"] += 1
                continue

            b = bars.get(code, [])
            done = [x for x in b if x[0] < act]
            win = done[-5:]
            if win:
                bs = win[0][1] if len(win) == 1 else win[0][4]
                ret_5m = (win[-1][4] / bs - 1.0) if bs > 0 else 0.0
            else:
                ret_5m = 0.0
            if mkt_open and mkt_open > 0:
                mdone = [x for x in mkt if x[0] < act]
                mkt_move = (mdone[-1][4] / mkt_open - 1.0) if mdone else 0.0
                has_mkt = True
            else:
                mkt_move = 0.0
                has_mkt = False
            n_today, gap = burst.get(key, (0.0, SENTINEL_MIN))
            mk = market.get(code, "거래소")
            cost = round_trip(day, mk, at_open=act < 5)

            # ---- (b) volume / turnover ----
            cum_vol = sum(x[5] for x in done)
            cum_val = sum(x[4] * x[5] for x in done)
            last_vol = done[-1][5] if done else 0.0
            v5 = sum(x[5] for x in done[-5:])
            v30 = sum(x[5] for x in done[-30:])
            n5, n30 = len(done[-5:]), len(done[-30:])
            vol_accel = ((v5 / n5) / (v30 / n30)) if n5 and n30 and v30 > 0 else 1.0
            bexp = 0.0
            for bkt in range(0, max(1, (act + 9) // 10)):
                bexp += base.get((code, bkt), 0.0) * 10.0
            rvol_cum = (cum_vol / bexp) if bexp > 0 else -1.0
            bnow = base.get((code, max(0, (act - 1) // 10)), 0.0)
            rvol_last = (last_vol / bnow) if bnow > 0 else -1.0
            rvol_5 = ((v5 / n5) / bnow) if bnow and n5 and bnow > 0 else -1.0
            vol_share_5 = (v5 / cum_vol) if cum_vol > 0 else -1.0
            traded_frac = (sum(1 for x in done if x[5] > 0) / len(done)) if done else -1.0
            prevrows = prev_rows(code, day, 2)
            prev_day_vol = prevrows[-1][4] if prevrows else 0.0
            turnover_vs_prevday = (cum_vol / prev_day_vol) if prev_day_vol > 0 else -1.0

            # ---- (c) free deterministic ----
            pub = dt.datetime.fromisoformat(m["published_at"]).replace(tzinfo=None)
            pub_min = pub.hour * 60 + pub.minute - 540
            pub_to_act = (
                float(act - pub_min) if pub.date() == dt.date.fromisoformat(day) else 9999.0
            )
            day_open = b[0][1] if b else None
            pub_px = None
            if done:
                pre = [x for x in done if x[0] <= pub_min]
                pub_px = pre[-1][4] if pre else done[0][1]
            last_px = done[-1][4] if done else (day_open or 0.0)
            move_since_pub = (last_px / pub_px - 1.0) if pub_px and pub_px > 0 else 0.0
            move_before_pub = (
                (pub_px / day_open - 1.0) if pub_px and day_open and day_open > 0 else 0.0
            )
            ret_since_open = (last_px / day_open - 1.0) if day_open and day_open > 0 else 0.0
            if len(prevrows) >= 2:
                prev_abs_move = (
                    abs(prevrows[-1][1] / prevrows[-2][1] - 1.0) if prevrows[-2][1] > 0 else 0.0
                )
                prev_range = (
                    ((prevrows[-1][2] - prevrows[-1][3]) / prevrows[-1][1])
                    if prevrows[-1][1] > 0
                    else 0.0
                )
                gap_open = (
                    (day_open / prevrows[-1][1] - 1.0) if day_open and prevrows[-1][1] > 0 else 0.0
                )
            else:
                prev_abs_move = prev_range = gap_open = 0.0
            entry = float(r.get("entry") or last_px or 0.0)
            tick_bp = (tick_size(entry) / entry) if entry > 0 else 0.0
            title = m.get("title") or ""
            tag = TAG_RE.match(title)
            aid = m.get("news_id") or key.split("|", 1)[0]

            # stratum-matched null for this (day, act_min, cell)
            cells = nullcell.get((day, act), {})
            cellkey = f"{int(st['sig_q'])}|{int(st['val_q'])}"
            cm = cells.get(cellkey)

            feats = {
                # existing 10
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
                # (b) volume / turnover
                "log1p_cum_value": math.log1p(max(0.0, cum_val)),
                "rvol_cum": rvol_cum,
                "rvol_last": rvol_last,
                "rvol_5": rvol_5,
                "vol_accel": vol_accel,
                "vol_share_5": vol_share_5,
                "traded_frac": traded_frac,
                "turnover_vs_prevday": turnover_vs_prevday,
                "log1p_last_bar_value": math.log1p(max(0.0, last_px * last_vol)),
                # (c) free deterministic
                "fanout": float(art_n.get(aid, 1)),
                "pub_to_act_min": pub_to_act,
                "move_since_pub": move_since_pub,
                "move_before_pub": move_before_pub,
                "ret_since_open": ret_since_open,
                "prevday_val_rank": float(rank_of.get(code, 0.5)),
                "prevday_abs_move": prev_abs_move,
                "prevday_range": prev_range,
                "gap_open": gap_open,
                "log_entry": math.log(entry) if entry > 0 else 0.0,
                "tick_bp": tick_bp,
                "dow": dow,
                "is_kosdaq": 1.0 if mk == "코스닥" else 0.0,
                "n_bars_done": float(len(done)),
            }
            cats = {
                "sector": sector.get(code, "?"),
                "session": m.get("session") or "?",
                "title_tag": (tag.group(1) if tag else ""),
            }
            rec = {
                "key": key,
                "day": day,
                "code": code,
                "month": day[:7],
                "market": mk,
                "ret_close": r["ret_close"],
                "cost": cost,
                "reward": r["ret_close"] - cost,
                "null_cell": cm,
                "alpha": (r["ret_close"] - cm) if cm is not None else None,
                "has_mkt": has_mkt,
                "features": feats,
                "cats": cats,
            }
            v2 = m.get("v2")
            if v2:
                rec["v2"] = v2
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
        print(f"{day} n={n_out}", file=sys.stderr)
    fout.close()
    print(json.dumps({"period": period, "n": n_out, "missing": dict(missing)}))


if __name__ == "__main__":
    main()
