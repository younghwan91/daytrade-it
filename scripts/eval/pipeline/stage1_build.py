"""Stage-1 data: the pair population, decision-time features, text, and outcome labels.

    .venv/bin/python scripts/eval/pipeline/stage1_build.py            # all steps
    .venv/bin/python scripts/eval/pipeline/stage1_build.py --skip-label  # reuse labels

Writes (paths relative to the repo; all from pipeline_config.json dates):
  data/eval/pipeline/stage1/pairs.jsonl     one row per (article, code) pair
  data/eval/pipeline/stage1/texts.jsonl     unique model texts {hash, text} (plain + name-masked)
  data/eval/pipeline/stage1/labels_live_range.jsonl   prompt_labels output for days
                                            after labels_prior's range, stage1 train_range only
  data/eval/pipeline/stage1/build_stats.json

Population (identical to the exit stage): `data/eval/rlnews/master_hourly.jsonl`,
built by `news_master_prior` -- every (article, code) with code in that day's
top-300 universe (previous day's traded value), published in [08:00, 15:20) KST,
decide_at = published_at + 4 min (pre-09:00 acts at the 09:00 open). Kept days:
stage1.train_range plus the frozen holdout days. Holdout rows get features and
text only; no outcome is ever computed or loaded for them.

Labels (train_range only, guard applied on every label file right after load):
  * `data/eval/rlnews/labels_prior.jsonl` for its days (same construction);
  * `scripts/eval/prompt_labels.py` on the remaining train_range days.

Decision-time features (strictly before decide_at; computed identically for
train and holdout rows, never from a label file):
  act_min      minutes from 09:00 to the acting minute (0 premarket)
  premarket    published before 09:00 (acts at the 09:00 open)
  log_vol      log of the stock's own 60-day daily log-return std, adjusted
               closes strictly before the decision day (>= 20 returns)
  log_prev_tv  log previous trading day's traded value (daily_bars, raw)
  pre_move     entry / previous raw close - 1, entry = live SimBroker quote
               (decision minute's open, else the last completed bar's close)
  ret_5m       last completed <= 5 one-minute bars before the acting minute:
               close of the last / open of the first - 1 (missing premarket)
`own_vol` (not logged) is also kept for target normalisation. Nothing uses
069500, so its missing August-2025 bars do not matter here.

minute_bars.ts is naive KST: `ts::date` / `ts::time`, never AT TIME ZONE.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/eval"))

import config  # noqa: E402
import stage1_text as T  # noqa: E402

MASTER = REPO / "data/eval/rlnews/master_hourly.jsonl"
LABELS_PRIOR = REPO / "data/eval/rlnews/labels_prior.jsonl"
PREV_CLOSE = REPO / "data/eval/prev_close"
OUTDIR = REPO / "data/eval/pipeline/stage1"
VOL_WIN, VOL_MIN = 60, 20
OPEN = dt.time(9, 0)
LABEL_FIELDS = ("entry", "ret_close", "mfe", "mae", "act_min", "premarket", "pre_move")


def jl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def acting_minute(decide_at: str) -> tuple[dt.datetime, bool]:
    """(acting minute, premarket) -- prompt_labels.label_day's rule, decision-time only."""
    decide = dt.datetime.fromisoformat(decide_at).replace(tzinfo=None)
    minute = decide.replace(second=0, microsecond=0)
    pre = decide.time() < OPEN
    if minute.time() < OPEN:
        minute = dt.datetime.combine(minute.date(), OPEN)
    return minute, pre


def load_labels(path: Path, days: set[str]) -> dict[str, dict]:
    rows = list(jl(path))
    config.assert_no_holdout_labels({r["day"] for r in rows}, stage=1)
    return {r["key"]: {k: r.get(k) for k in LABEL_FIELDS} for r in rows if r["day"] in days}


def run_labels(master_rows: list[dict], out: Path) -> None:
    tmp = OUTDIR / "master_live_range.jsonl"
    with tmp.open("w", encoding="utf-8") as f:
        for r in master_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/eval/prompt_labels.py"),
            "--master",
            str(tmp),
            "--out",
            str(out),
            "--prev-close",
            str(PREV_CLOSE),
        ],
        check=True,
        cwd=REPO,
    )


def features(cur, rows: list[dict], stats: dict) -> dict[str, dict]:
    """Decision-time features per key. Reads nothing at or after each pair's acting minute
    except that minute's OPEN (the live entry quote)."""
    days = sorted({r["day"] for r in rows})
    codes = sorted({r["code"] for r in rows})
    first = dt.date.fromisoformat(days[0]) - dt.timedelta(days=150)
    cur.execute(
        "select code, date, close from daily_bars_adjusted "
        "where code = any(%s) and date >= %s and date < %s order by code, date",
        (codes, first, days[-1]),
    )
    adj: dict[str, list[tuple[dt.date, float]]] = collections.defaultdict(list)
    for code, d, c in cur.fetchall():
        if c and c > 0:
            adj[code].append((d, float(c)))
    cur.execute(
        "select code, date, close, trade_value from daily_bars "
        "where code = any(%s) and date >= %s and date < %s order by code, date",
        (codes, first, days[-1]),
    )
    raw: dict[str, list[tuple[dt.date, float, float]]] = collections.defaultdict(list)
    for code, d, c, tv in cur.fetchall():
        raw[code].append((d, float(c or 0), float(tv or 0)))

    def before(seq, day):
        lo, hi = 0, len(seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid][0] < day:
                lo = mid + 1
            else:
                hi = mid
        return seq[:lo]

    by_day: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    out: dict[str, dict] = {}
    miss = collections.Counter()
    for day in days:
        dd = dt.date.fromisoformat(day)
        drows = by_day[day]
        acts = {r["key"]: acting_minute(r["decide_at"]) for r in drows}
        last_minute = max(m for m, _ in acts.values())
        dcodes = sorted({r["code"] for r in drows})
        # Bars up to the latest acting minute of the day only.
        cur.execute(
            "select code, ts, open, close from minute_bars "
            "where code = any(%s) and ts::date = %s::date and ts::time >= '09:00' "
            "and ts <= %s order by code, ts",
            (dcodes, day, last_minute),
        )
        bars: dict[str, list[tuple[dt.datetime, float, float]]] = collections.defaultdict(list)
        for code, ts, o, c in cur.fetchall():
            bars[code].append((ts, float(o), float(c)))
        vcache: dict[str, tuple] = {}
        for r in drows:
            code = r["code"]
            minute, pre = acts[r["key"]]
            if minute.date() != dd:
                miss["acting_minute_other_day"] += 1
                continue
            bs = bars.get(code, [])
            # entry: SimBroker quote rule
            entry = None
            done = []
            for b in bs:
                if b[0] == minute:
                    entry = b[1]
                elif b[0] < minute:
                    done.append(b)
            if entry is None and done:
                entry = done[-1][2]
            if not entry or entry <= 0:
                miss["no_entry_quote"] += 1
                continue
            if code not in vcache:
                a = before(adj.get(code, []), dd)[-(VOL_WIN + 1) :]
                rets = [math.log(a[i][1] / a[i - 1][1]) for i in range(1, len(a))]
                vol = statistics.pstdev(rets) if len(rets) >= VOL_MIN else None
                rw = before(raw.get(code, []), dd)
                prev = rw[-1] if rw else None
                vcache[code] = (vol if vol and vol > 0 else None, prev)
            vol, prev = vcache[code]
            win = [b for b in done if b[0] >= minute - dt.timedelta(minutes=5)]
            ret_5m = (win[-1][2] / win[0][1] - 1.0) if win and win[0][1] > 0 else None
            pre_move = (entry / prev[1] - 1.0) if prev and prev[1] > 0 else None
            tv = prev[2] if prev and prev[2] > 0 else None
            if vol is None:
                miss["own_vol"] += 1
            out[r["key"]] = {
                "entry": entry,
                "act_min": int((minute - dt.datetime.combine(dd, OPEN)).total_seconds() // 60),
                "premarket": bool(pre),
                "own_vol": vol,
                "log_vol": math.log(vol) if vol else None,
                "log_prev_tv": math.log(tv) if tv else None,
                "pre_move": pre_move,
                "ret_5m": ret_5m,
            }
        print(
            f"features {day}: {sum(1 for r in drows if r['key'] in out)}/{len(drows)}", flush=True
        )
    stats["feature_missing"] = dict(miss)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--skip-label", action="store_true", help="reuse labels_live_range.jsonl")
    args = ap.parse_args()
    cfg = config.load()
    t0, t1 = cfg["stage1"]["train_range"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {"train_range": [t0, t1], "holdout_start": config.holdout_start()}

    # train_range rows are labelled; every later row (the consumed post-train
    # window, then the forward holdout) is carried unlabelled, to be scored.
    master = [r for r in jl(MASTER) if r["day"] >= t0]
    in_train = [r for r in master if r["day"] <= t1]
    stats["master_rows"] = len(master)
    stats["master_rows_train"] = len(in_train)
    stats["master_rows_post_train"] = sum(
        1 for r in master if r["day"] > t1 and not config.is_holdout(r["day"])
    )
    stats["master_rows_holdout"] = sum(1 for r in master if config.is_holdout(r["day"]))

    # ---- labels (train_range only) --------------------------------------
    train_days = {r["day"] for r in in_train}
    prior = load_labels(LABELS_PRIOR, train_days)
    prior_last = max(v for v in {r["day"] for r in master if r["key"] in prior})
    live_rows = [r for r in in_train if r["day"] > prior_last]
    config.assert_no_holdout_labels({r["day"] for r in live_rows}, stage=1)  # never label holdout
    live_path = OUTDIR / "labels_live_range.jsonl"
    if not args.skip_label:
        run_labels(live_rows, live_path)
    live = load_labels(live_path, train_days)
    labels = {**prior, **live}
    stats["labels_prior_rows"] = len(prior)
    stats["labels_prior_last_day"] = prior_last
    stats["labels_live_range_rows"] = len(live)
    stats["train_rows_unlabelled"] = stats["master_rows_train"] - len(labels)

    # ---- features ---------------------------------------------------------
    import edge_db

    conn = edge_db.connect()
    cur = conn.cursor()
    cur.execute("select code, name, market from stocks")
    stock = {c: (n, m) for c, n, m in cur.fetchall()}
    feats = features(cur, master, stats)
    conn.close()

    # ---- text -------------------------------------------------------------
    wanted: dict[str, set[str]] = collections.defaultdict(set)
    for r in master:
        wanted[r["code"]].add(r["published_at"][:10])
    texts = T.archive_texts(wanted)

    uniq: dict[str, str] = {}
    n = collections.Counter()
    pairs_path = OUTDIR / "pairs.jsonl"
    entry_check = []
    with pairs_path.open("w", encoding="utf-8") as f:
        for r in master:
            key, code = r["key"], r["code"]
            # after the train range every row is scored-only: never labelled here
            scored_only = r["day"] > t1
            lab = labels.get(key)
            if not scored_only and lab is None:
                n["drop_unlabelled"] += 1
                continue
            ft = feats.get(key)
            if ft is None:
                n["drop_no_features"] += 1
                continue
            tx = texts.get((r["news_id"], code))
            if tx is None:
                n["drop_no_text"] += 1
                continue
            name, market = stock.get(code, (None, None))
            text = T.model_text(*tx)
            masked = T.mask_text(text, name, code)
            h, hm = T.text_hash(text), T.text_hash(masked)
            uniq[h] = text
            uniq[hm] = masked
            if lab is not None:
                entry_check.append((ft["entry"], lab["entry"], ft["act_min"] == lab["act_min"]))
            f.write(
                json.dumps(
                    {
                        "key": key,
                        "news_id": r["news_id"],
                        "code": code,
                        "name": name,
                        "market": market,
                        "day": r["day"],
                        "month": r["day"][:7],
                        "published_at": r["published_at"],
                        "decide_at": r["decide_at"],
                        "holdout": config.is_holdout(r["day"]),
                        "text_hash": h,
                        "masked_hash": hm,
                        "name_in_text": hm != h,
                        "features": {k: v for k, v in ft.items() if k != "entry"},
                        "label": None
                        if scored_only
                        else {k: lab[k] for k in ("ret_close", "mfe", "mae")},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n["scored_only" if scored_only else "train"] += 1
    with (OUTDIR / "texts.jsonl").open("w", encoding="utf-8") as f:
        for h, t in uniq.items():
            f.write(json.dumps({"hash": h, "text": t}, ensure_ascii=False) + "\n")
    stats["pairs_written"] = dict(n)
    stats["unique_texts_plain_plus_masked"] = len(uniq)
    stats["entry_matches_label_entry"] = sum(1 for a, b, _ in entry_check if abs(a - b) < 1e-9)
    stats["act_min_matches_label"] = sum(1 for *_, ok in entry_check if ok)
    stats["entry_checked"] = len(entry_check)
    (OUTDIR / "build_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
