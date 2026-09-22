"""feats3: adds MA position (daily + intraday), market cap / size, and
per-stock (not pool-wide) normalization for every stock-specific signal.

    python data/eval/rlnews/feats3_build.py prior
writes data/eval/rlnews/feats3_prior.jsonl

Decision-time clean.  Price at the decision minute = labels_prior.entry (the
live-parity fill).  Nothing dated on or after the decision day enters, and the
shares-outstanding snapshot must also have been KNOWN before the decision day
(`knowledge_date < day`), because kr_quant publishes a snapshot >= 14 days after
its nominal `date`.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "scripts/eval")
import edge_db  # noqa: E402
from costs import round_trip  # noqa: E402

REPO = Path("/home/young/git/daytrade-it-rlnews")
OUT = REPO / "data/eval/rlnews"
SENTINEL_MIN = 10080.0
MA_PERIODS = (5, 10, 20, 50, 100, 150, 200)
VOL_WIN = 60  # trailing daily returns for the stock's own volatility
VOL_MIN = 20
NEWS_WIN = 30  # trailing calendar days for the stock's own news rate
HIST_START = "2024-06-01"  # >= 200 trading days before 2025-08-01


def jl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    period = sys.argv[1]
    labels = list(jl(OUT / f"labels_{period}.jsonl"))
    master = {r["key"]: r for r in jl(OUT / f"master_{period}.jsonl")}
    strata = {(r["day"], r["code"]): r for r in jl(OUT / f"strata_{period}.jsonl")}

    # ---- news burst + that stock's OWN trailing news baseline -------------
    by_code: dict[str, list[tuple[dt.datetime, str]]] = collections.defaultdict(list)
    for k, r in master.items():
        pub = dt.datetime.fromisoformat(r["published_at"]).replace(tzinfo=None)
        by_code[r["code"]].append((pub, k))
    burst: dict[str, tuple[float, float]] = {}
    news_base: dict[
        str, tuple[float, float]
    ] = {}  # key -> (trailing mean n/day, trailing median gap)
    for items in by_code.values():
        items.sort()
        prev_pub: dt.datetime | None = None
        same_day: list[dt.datetime] = []
        past_gaps: list[float] = []
        day_counts: dict[dt.date, int] = collections.Counter()
        for pub, k in items:
            gap = (
                SENTINEL_MIN
                if prev_pub is None
                else min(SENTINEL_MIN, (pub - prev_pub).total_seconds() / 60.0)
            )
            n_today = sum(1 for q in same_day if q.date() == pub.date() and q < pub)
            burst[k] = (float(n_today), float(gap))
            # baselines use only STRICTLY EARLIER DAYS for the rate, and only
            # strictly earlier articles for the gap.
            lo = pub.date() - dt.timedelta(days=NEWS_WIN)
            prior = [c for d, c in day_counts.items() if lo <= d < pub.date()]
            rate = (sum(prior) / NEWS_WIN) if prior else float("nan")
            med = statistics.median(past_gaps) if len(past_gaps) >= 3 else float("nan")
            news_base[k] = (rate, med)
            prev_pub = pub
            same_day.append(pub)
            day_counts[pub.date()] += 1
            if prev_pub is not None and gap < SENTINEL_MIN:
                past_gaps.append(gap)

    days = sorted({r["day"] for r in labels})
    codes_by_day: dict[str, set[str]] = collections.defaultdict(set)
    for r in labels:
        codes_by_day[r["day"]].add(r["key"].split("|", 1)[1])
    all_codes = sorted(set().union(*codes_by_day.values()))
    labels_by_day: dict[str, list[dict]] = collections.defaultdict(list)
    for r in labels:
        labels_by_day[r["day"]].append(r)

    conn = edge_db.connect()
    cur = conn.cursor()
    cur.execute("select code, market from stocks")
    market = dict(cur.fetchall())

    # ---- daily history: closes + trade value, one shot -------------------
    cur.execute(
        "select code, date, close, trade_value from daily_bars_adjusted "
        "where code = any(%s) and date between %s::date and %s::date order by code, date",
        (all_codes, HIST_START, days[-1]),
    )
    hist: dict[str, list[tuple[dt.date, float, float]]] = collections.defaultdict(list)
    for code, d, c, tv in cur.fetchall():
        hist[code].append((d, float(c or 0.0), float(tv or 0.0)))

    # ---- shares outstanding, point-in-time ------------------------------
    cur.execute(
        "select code, date, shares_outstanding, knowledge_date "
        "from shares_outstanding_history where code = any(%s) order by code, date",
        (all_codes,),
    )
    shares: dict[str, list[tuple[dt.date, float, dt.date]]] = collections.defaultdict(list)
    for code, d, so, kd in cur.fetchall():
        if so and so > 0:
            shares[code].append((d, float(so), kd or d))

    def prior_daily(code: str, day: dt.date):
        """(closes, values, dates) strictly before `day`, chronological."""
        seq = hist.get(code) or []
        cut = [x for x in seq if x[0] < day]
        return [x[1] for x in cut], [x[2] for x in cut], [x[0] for x in cut]

    def ma_block(price: float, closes: list[float], vol: float):
        n_avail, above, dists = 0, 0, []
        for k in MA_PERIODS:
            if len(closes) < k:
                continue
            ma = sum(closes[-k:]) / k
            if ma <= 0:
                continue
            n_avail += 1
            if price > ma:
                above += 1
            dists.append(price / ma - 1.0)
        if not n_avail:
            return 0, float("nan"), 0
        raw = sum(dists) / len(dists)
        return above, (raw / vol if vol and vol > 0 else float("nan")), n_avail

    out = (OUT / f"feats3_{period}.jsonl").open("w", encoding="utf-8")  # noqa: SIM115
    n_out = 0
    missing: collections.Counter = collections.Counter()
    stale: list[int] = []
    avail_by_act: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    for day in days:
        dd = dt.date.fromisoformat(day)
        codes = sorted(codes_by_day[day] | {"069500"})
        cur.execute(
            "select code, ts, open, close from minute_bars "
            "where ts::date = %s::date and code = any(%s) "
            "and ts::time >= '09:00' and ts::time <= '15:30' order by code, ts",
            (day, codes),
        )
        bars: dict[str, list[tuple[int, float, float]]] = collections.defaultdict(list)
        for code, ts, o, c in cur.fetchall():
            bars[code].append((ts.hour * 60 + ts.minute - 540, float(o), float(c)))
        mkt = bars.get("069500", [])
        mkt_open = mkt[0][1] if mkt else None

        pend = []
        dcache: dict[str, tuple] = {}
        for r in labels_by_day[day]:
            key = r["key"]
            code = key.split("|", 1)[1]
            act = int(r["act_min"])
            st = strata.get((day, code))
            if st is None or st.get("sigma") is None or st.get("sig_q") is None:
                missing["strata"] += 1
                continue
            if code not in dcache:
                dcache[code] = prior_daily(code, dd)
            closes, values, _ = dcache[code]
            if not values or values[-1] <= 0:
                missing["prev_value"] += 1
                continue
            pv = values[-1]
            prev_close = closes[-1]
            # --- that stock's OWN trailing daily volatility, strictly before
            rets = [
                math.log(closes[i] / closes[i - 1])
                for i in range(max(1, len(closes) - VOL_WIN), len(closes))
                if closes[i] > 0 and closes[i - 1] > 0
            ]
            if len(rets) < VOL_MIN:
                missing["own_vol"] += 1
                continue
            vol = statistics.pstdev(rets)
            if vol <= 0:
                missing["own_vol_zero"] += 1
                continue

            price = float(r["entry"])
            if not price or price <= 0:
                missing["entry"] += 1
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

            cnt_d, dist_d, navail_d = ma_block(price, closes, vol)
            icl = [x[2] for x in done]
            cnt_i, dist_i, navail_i = ma_block(price, icl, vol)
            avail_by_act[act][navail_i] += 1

            # --- market cap, point-in-time --------------------------------
            snaps = [s for s in shares.get(code, []) if s[0] < dd and s[2] < dd]
            if not snaps or prev_close <= 0:
                missing["mcap"] += 1
                mcap = float("nan")
            else:
                sd, so, _ = snaps[-1]
                mcap = so * prev_close
                stale.append((dd - sd).days)

            n_today, gap = burst[key]
            rate, med_gap = news_base[key]
            mk = market.get(code, "거래소")

            # --- per-stock normalization (baselines strictly before `day`) --
            trail_val = statistics.median(values[-VOL_WIN:])
            sigma_rel = float(st["sigma"]) / vol
            value_rel = math.log(pv / trail_val) if trail_val > 0 else 0.0
            # 1-min vol from the stock's own daily vol; 5 completed minutes
            ret_5m_norm = ret_5m / (vol * math.sqrt(5.0 / 390.0))
            news_rate_rel = math.log1p(n_today) - math.log1p(rate) if rate == rate else float("nan")
            gap_rel = (
                math.log1p(min(gap, SENTINEL_MIN)) - math.log1p(med_gap)
                if med_gap == med_gap
                else float("nan")
            )

            cost = round_trip(day, mk, at_open=act < 5)
            pend.append(
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
                    "own_vol": vol,
                    "mcap": mcap,
                    "features": {
                        # --- the 10 prior features, per-stock normalized ----
                        "act_min": float(act),
                        "premarket": float(bool(r["premarket"])),
                        "sigma_rel": sigma_rel,
                        "sig_q": float(st["sig_q"]),
                        "val_q": float(st["val_q"]),
                        "value_rel": value_rel,
                        "ret_5m_norm": ret_5m_norm,
                        "mkt_move": mkt_move,
                        "news_rate_rel": news_rate_rel,
                        "gap_rel": gap_rel,
                        # --- new: MA position, collapsed --------------------
                        "ma_count_daily": float(cnt_d),
                        "ma_dist_norm_daily": dist_d,
                        "ma_n_avail_daily": float(navail_d),
                        "ma_count_intraday": float(cnt_i),
                        "ma_dist_norm_intraday": dist_i,
                        "ma_n_avail_intraday": float(navail_i),
                        # --- new: size --------------------------------------
                        "log_mcap": math.log(mcap) if mcap == mcap and mcap > 0 else float("nan"),
                        "size_q": float("nan"),  # filled below, cross-sectional
                        "is_kosdaq": float(mk == "코스닥"),
                        # --- raw prior features kept for the OLD-set rerun ---
                        "sigma": float(st["sigma"]),
                        "log_prev_value": math.log(pv),
                        "ret_5m": ret_5m,
                        "n_articles_today_before": n_today,
                        "minutes_since_prev_article": gap,
                    },
                }
            )
        # --- size_q: within-day cross-sectional quartile of mcap ----------
        live = [p for p in pend if p["mcap"] == p["mcap"]]
        live.sort(key=lambda p: p["mcap"])
        n = len(live)
        for i, p in enumerate(live):
            p["features"]["size_q"] = float(min(4, int(4 * i / n) + 1)) if n else float("nan")
        for p in pend:
            out.write(json.dumps(p) + "\n")
            n_out += 1
        print(f"{day} n={n_out}", file=sys.stderr)
    out.close()

    diag = {
        "period": period,
        "n": n_out,
        "missing": dict(missing),
        "staleness_days": {
            "n": len(stale),
            "min": min(stale, default=None),
            "p25": statistics.quantiles(stale, n=4)[0] if stale else None,
            "median": statistics.median(stale) if stale else None,
            "p75": statistics.quantiles(stale, n=4)[2] if stale else None,
            "p95": sorted(stale)[int(0.95 * len(stale))] if stale else None,
            "max": max(stale, default=None),
            "mean": statistics.mean(stale) if stale else None,
        },
        "ma_n_avail_intraday_by_act_bucket": {},
    }
    buckets = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 150), (150, 200), (200, 400)]
    for lo, hi in buckets:
        agg: collections.Counter = collections.Counter()
        for act, c in avail_by_act.items():
            if lo <= act < hi:
                agg.update(c)
        if agg:
            tot = sum(agg.values())
            diag["ma_n_avail_intraday_by_act_bucket"][f"{lo}-{hi}"] = {
                "n": tot,
                "dist": {str(k): round(v / tot, 4) for k, v in sorted(agg.items())},
            }
    (OUT / f"feats3_{period}_diag.json").write_text(json.dumps(diag, indent=1, ensure_ascii=False))
    print(json.dumps(diag, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
