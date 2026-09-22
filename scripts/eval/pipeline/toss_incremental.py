"""Keep the Toss universe news archive growing, one trading day at a time.

The archive `data/toss_universe_42m/<code>.jsonl` is what stage 1 -- and the
forward holdout that stage 4 waits on -- is built from. It was filled by one
long backfill that stopped on 2026-09-17; this extends it.

`backfill_toss_company_news.py` cannot be pointed at the archive directly: it
REPLACES `<code>.jsonl` with only the rows since `--since`, and it skips every
code that already has a `.done`. Run against the archive it would either do
nothing or overwrite three years of history with a few days. So this fetches
into a staging directory with that script, unchanged -- same pacing, same UA,
same market-hours pause, same block detection -- and only then merges into the
archive by appending rows whose `id` is not already there.

The archive changes only when a whole run succeeds. If the fetch stops (a
suspected block, a failing endpoint) or any code is left unfinished, nothing is
merged and the run exits non-zero; the staging directory is kept, so the next
run resumes where this one stopped. The wrapper turns a non-zero exit into a
STOPPED marker that blocks further runs until a person clears it.

Which codes: for every weekday d in [since, until], that day's live universe --
top 300 by the previous trading day's traded value (`select_universe`, the same
rule as stage 1), without 069500. A holiday gets a universe too; that only
costs a few extra requests.

A day is recorded as COMPLETE in the ledger only if every code of its universe
was fetched in a run that started after 15:20 KST on that day, when the live
sweep window has closed and every article that could be a pair exists.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "src"))

import edge_db  # noqa: E402
from universe_coverage import select_universe  # noqa: E402

KST = dt.timezone(dt.timedelta(hours=9))
#: Overridable only so the merge can be tested on a copy of the archive.
ARCHIVE = Path(os.environ.get("TOSS_ARCHIVE", ROOT / "data" / "toss_universe_42m"))
WORK = Path(os.environ.get("TOSS_WORK", ROOT / "data" / "toss_universe_incr"))
LEDGER = WORK / "ledger.jsonl"
BACKFILL = ROOT / "scripts" / "eval" / "backfill_toss_company_news.py"
MARKET = "069500"
#: The live sweep window closes here; a day's pairs all exist after it.
WINDOW_CLOSE = dt.time(15, 20)


def log(msg: str) -> None:
    print(f"{dt.datetime.now(KST):%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def last_complete_day() -> str | None:
    """The latest day any successful run recorded as complete."""
    if not LEDGER.exists():
        return None
    best = None
    for line in LEDGER.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        for d, v in r["days"].items():
            if v["complete"] and (best is None or d > best):
                best = d
    return best


def universes(since: str, until: str) -> dict[str, list[str]]:
    """{weekday: that day's universe}, from the previous trading day's traded value."""
    cur = edge_db.connect().cursor()
    lo = dt.date.fromisoformat(since) - dt.timedelta(days=14)
    cur.execute(
        "SELECT date, code, trade_value FROM daily_bars_adjusted "
        "WHERE date >= %s AND date < %s AND trade_value > 0",
        (lo, dt.date.fromisoformat(until)),
    )
    by_day: dict[dt.date, list[tuple[str, float]]] = {}
    for d, code, v in cur.fetchall():
        by_day.setdefault(d, []).append((code, float(v)))
    traded = sorted(by_day)
    out: dict[str, list[str]] = {}
    d = dt.date.fromisoformat(since)
    end = dt.date.fromisoformat(until)
    while d <= end:
        if d.weekday() < 5:
            prev = [t for t in traded if t < d]
            if not prev:
                raise SystemExit(f"no daily bars before {d} to build its universe")
            out[str(d)] = [c for c in select_universe(by_day[prev[-1]]) if c != MARKET]
        d += dt.timedelta(days=1)
    return out


def merge(staging: Path, codes: list[str]) -> Counter:
    """Append staged rows whose id is new to each archive file. Returns new rows by date."""
    added: Counter = Counter()
    for code in codes:
        src = staging / f"{code}.jsonl"
        if not src.exists():
            continue
        dst = ARCHIVE / f"{code}.jsonl"
        have = set()
        old = dst.read_text() if dst.exists() else ""
        for line in old.splitlines():
            if line.strip():
                have.add(json.loads(line)["id"])
        new = []
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] not in have:
                have.add(row["id"])
                new.append(line)
                added[row["createdAt"][:10]] += 1
        if not new:
            continue
        body = old if (not old or old.endswith("\n")) else old + "\n"
        tmp = dst.with_suffix(".jsonl.tmp")
        tmp.write_text(body + "\n".join(new) + "\n")
        os.replace(tmp, dst)
    return added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", help="first day to (re)fetch; default: the last complete day")
    ap.add_argument("--until", default=str(dt.datetime.now(KST).date()))
    ap.add_argument("--sleep", type=float, default=1.2, help="passed to the backfill (min 1.0)")
    ap.add_argument("--only-codes", help="test hook: restrict every universe to these codes")
    args = ap.parse_args()

    since = args.since or last_complete_day()
    if since is None:
        raise SystemExit("no ledger yet: pass --since for the first run")
    started = dt.datetime.now(KST)
    uni = universes(since, args.until)
    if args.only_codes:
        keep = set(args.only_codes.split(","))
        uni = {d: [c for c in cs if c in keep] for d, cs in uni.items()}
    codes = sorted({c for cs in uni.values() for c in cs})
    staging = WORK / f"staging_since_{since}"
    staging.mkdir(parents=True, exist_ok=True)
    log(
        f"start: days {since}..{args.until} ({len(uni)} weekdays), {len(codes)} codes, staging {staging}"
    )

    rc = subprocess.run(
        [
            sys.executable,
            str(BACKFILL),
            "--since",
            since,
            "--codes",
            ",".join(codes),
            "--out-dir",
            str(staging),
            "--sleep",
            str(args.sleep),
        ],
        cwd=ROOT,
    ).returncode
    unfinished = [c for c in codes if not (staging / f"{c}.done").exists()]
    statuses = Counter(
        json.loads((staging / f"{c}.done").read_text())["status"]
        for c in codes
        if (staging / f"{c}.done").exists()
    )
    record = {
        "run_started": started.isoformat(timespec="seconds"),
        "run_finished": dt.datetime.now(KST).isoformat(timespec="seconds"),
        "since": since,
        "until": args.until,
        "codes": len(codes),
        "fetch_exit": rc,
        "code_status": dict(statuses),
        "unfinished": len(unfinished),
    }
    if rc != 0 or unfinished:
        record.update(status="failed", days={}, new_rows_by_created_date={})
        with LEDGER.open("a") as f:
            f.write(json.dumps(record) + "\n")
        log(
            f"ERROR: fetch exit {rc}, {len(unfinished)} code(s) unfinished "
            f"(first {unfinished[:5]}). Archive NOT modified; staging kept for resume."
        )
        return rc or 1

    added = merge(staging, codes)
    days = {}
    for d, cs in uni.items():
        closed = dt.datetime.combine(dt.date.fromisoformat(d), WINDOW_CLOSE, tzinfo=KST)
        days[d] = {"universe_codes": len(cs), "complete": started > closed}
    record.update(status="ok", days=days, new_rows_by_created_date=dict(sorted(added.items())))
    with LEDGER.open("a") as f:
        f.write(json.dumps(record) + "\n")
    for p in staging.iterdir():
        p.unlink()
    staging.rmdir()
    log(f"ok: merged {sum(added.values())} new rows; by date {dict(sorted(added.items()))}")
    log(f"complete days: {[d for d, v in days.items() if v['complete']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
