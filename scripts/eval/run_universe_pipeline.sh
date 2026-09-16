#!/usr/bin/env bash
# Judged days -> baseline -> optimizer judge -> bandit judge, at N = UNIVERSE_SIZE.
#
#   scripts/eval/run_universe_pipeline.sh --days-file data/eval/judge_days_complete.txt
#   scripts/eval/run_universe_pipeline.sh --days-json data/eval/judge_days_result.json
#
# Everything downstream of `judge_days.py`. It makes **no Anthropic call**; the
# DB is read-only; Toss and Kiwoom are never touched. Run it on simnode.
#
# What it does, in order:
#   0. resolve the day list (from judge_days' result JSON, or a plain file)
#   1. universe_readiness --event-tickers  -> keep only days whose minute bars
#      are there (a day with no bars replays as a hole, not as a zero)
#   2. build_universe_events --universe-size N --prefilter -> per-day events
#   3. stage judged AND usable days into a run-specific events dir -- every tool
#      below replays *every* day it finds, so the staging dir IS the day set
#   4. replay_baseline            (live defaults on the pool)
#   5. optimize_policy --stage stage2 --events <staged>   (LODO judge)
#   6. bandit_dataset + bandit_lodo --events <staged>     (LODO judge)
#
# Each stage writes under data/eval/<run-name>/ and the script stops at the
# first failure, so a partial run leaves the earlier stages intact.
set -euo pipefail

RUN="${RUN:-n300_$(date +%Y%m%d_%H%M)}"
UNIVERSE_SIZE_ARG="${UNIVERSE_SIZE_ARG:-300}"
TRIALS="${TRIALS:-200}"
DAYS_FILE=""
DAYS_JSON=""
SKIP_EVENTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --days-file) DAYS_FILE="$2"; shift 2 ;;
    --days-json) DAYS_JSON="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
    --universe-size) UNIVERSE_SIZE_ARG="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --skip-events) SKIP_EVENTS=1; shift ;;   # events dir already built
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/../.."
OUT="data/eval/$RUN"
EVENTS_ALL="$OUT/events_all"
EVENTS="$OUT/events"
mkdir -p "$OUT"

# ---- 0. day list --------------------------------------------------------
if [[ -n "$DAYS_JSON" ]]; then
  uv run python -c "
import json,sys
rows=json.load(open('$DAYS_JSON'))
days=[r['day'] for r in rows if r['complete']]
if not days: sys.exit('no complete day in $DAYS_JSON')
print('\n'.join(sorted(days)))
" > "$OUT/days.txt"
elif [[ -n "$DAYS_FILE" ]]; then
  sort -u "$DAYS_FILE" > "$OUT/days.txt"
else
  echo "one of --days-file / --days-json is required" >&2; exit 2
fi
mapfile -t DAYS < "$OUT/days.txt"
N_DAYS=${#DAYS[@]}
START="${DAYS[0]}"; END="${DAYS[-1]}"
echo "== $N_DAYS judged day(s): $START .. $END  (N=$UNIVERSE_SIZE_ARG, run=$RUN)"

# ---- 1./2. events at N, then readiness over those event tickers ---------
if [[ $SKIP_EVENTS -eq 0 ]]; then
  echo "== build_universe_events"
  DAYTRADE_NEWS_PREFILTER=1 uv run python scripts/eval/build_universe_events.py \
    --days "$START..$END" --allow-incomplete --prefilter \
    --universe-size "$UNIVERSE_SIZE_ARG" --out "$EVENTS_ALL" | tee "$OUT/events.log"
fi

echo "== universe_readiness (relaxed, event tickers)"
uv run python scripts/eval/universe_readiness.py \
  --start "$START" --end "$END" --universe-size "$UNIVERSE_SIZE_ARG" \
  --event-tickers "$EVENTS_ALL" --out-dir "$OUT/readiness" | tee "$OUT/readiness.log"

# ---- 3. stage judged AND usable days ------------------------------------
mkdir -p "$EVENTS"
: > "$OUT/days_used.txt"
while read -r d; do
  if grep -qx "$d" "$OUT/readiness/usable_days.txt" 2>/dev/null; then
    cp "$EVENTS_ALL/$d.jsonl" "$EVENTS/" && echo "$d" >> "$OUT/days_used.txt"
  else
    echo "skip $d: not bar-usable"
  fi
done < "$OUT/days.txt"
N_USED=$(wc -l < "$OUT/days_used.txt")
echo "== $N_USED day(s) staged in $EVENTS"
[[ "$N_USED" -gt 0 ]] || { echo "no usable judged day -- stop" >&2; exit 1; }

# Memory, not CPU, is the limit: a replay env is ~70 MB per day per worker and
# simnode has 61 GB. Over-subscribing makes the pool respawn OOM-killed workers
# forever and write nothing (A7-prep lost 90 minutes to exactly this).
JOBS=$(uv run python -c "print(max(1, min(8, int(45.0 / (0.07 * $N_USED + 0.5)))))")
echo "== jobs=$JOBS for $N_USED day(s)"

# ---- 4. baseline --------------------------------------------------------
# replay_baseline writes per-day files into --out; pass this run's own
# directory so it never merges into -- or clobbers -- the shared
# data/eval/replay_baseline pool or another run's/stage's days.
echo "== replay_baseline"
uv run python scripts/eval/replay_baseline.py \
  --events "$EVENTS" --work-root "$OUT/replay_work" \
  --out "$OUT/baseline" | tee "$OUT/baseline.log"

# ---- 5. optimizer judge -------------------------------------------------
echo "== optimize_policy stage2 (LODO)"
uv run python scripts/eval/optimize_policy.py --stage stage2 \
  --trials "$TRIALS" --jobs "$JOBS" --events "$EVENTS" \
  --out "$OUT/opt/stage2" --work-root "$OUT/opt_work" | tee "$OUT/opt.log"

# ---- 6. bandit judge ----------------------------------------------------
echo "== bandit_dataset"
uv run python scripts/eval/bandit_dataset.py --events "$EVENTS" \
  --out "$OUT/bandit/dataset.jsonl" --work-root "$OUT/bandit_work" \
  --jobs "$JOBS" | tee "$OUT/bandit_dataset.log"

echo "== bandit_lodo"
uv run python scripts/eval/bandit_lodo.py --events "$EVENTS" \
  --dataset "$OUT/bandit/dataset.jsonl" --out "$OUT/bandit/lodo" \
  --work-root "$OUT/bandit_lodo_work" --jobs "$JOBS" | tee "$OUT/bandit_lodo.log"

echo
echo "== done: $OUT"
echo "  baseline   $OUT/baseline/"
echo "  optimizer  $OUT/opt/stage2/   (adopt: see bayes_judge verdict in opt.log)"
echo "  bandit     $OUT/bandit/lodo/"
