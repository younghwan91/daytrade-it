#!/usr/bin/env bash
# Daily Toss universe news collection (see toss_incremental.py for what and why).
#
# One lock for cron and manual runs alike: the script re-executes itself under
# `flock -n`, so a second invocation while one is running exits at once.
#
# It stops, loudly, instead of carrying on: a failed run writes STOPPED, and
# every later run refuses to start until a person has read the log and removed
# the marker.  rm data/toss_universe_incr/STOPPED  to resume.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
LOGDIR=/home/young/logs/toss_universe
LOG="$LOGDIR/collect.log"
LOCK="$LOGDIR/collect.lock"
STOP="$ROOT/data/toss_universe_incr/STOPPED"
mkdir -p "$LOGDIR" "$ROOT/data/toss_universe_incr"

# Not `exec flock ...`: an exec'd shell never reaches its `||` fallback, so a
# held lock would exit 1 with nothing in the log. -E gives a lock conflict its
# own code, distinct from any exit code of the collection itself.
if [[ "${TOSS_COLLECT_LOCKED:-}" != "1" ]]; then
  rc=0
  TOSS_COLLECT_LOCKED=1 /usr/bin/flock -n -E 75 "$LOCK" "$0" "$@" || rc=$?
  if [[ $rc -eq 75 ]]; then
    echo "$(date '+%F %T') another collection holds $LOCK -- not starting" >> "$LOG"
    exit 0
  fi
  exit "$rc"
fi

stamp() { date '+%F %T'; }
if [[ -f "$STOP" ]]; then
  echo "$(stamp) REFUSING: $STOP is present ($(cat "$STOP")). Read the log, fix, then remove it." >> "$LOG"
  exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
echo "$(stamp) ---- run $* ----" >> "$LOG"
if ./.venv/bin/python "$HERE/toss_incremental.py" "$@" >> "$LOG" 2>&1; then
  echo "$(stamp) ---- done ----" >> "$LOG"
else
  rc=$?
  echo "$(stamp) failed with exit $rc (see above)" > "$STOP"
  echo "$(stamp) ERROR: exit $rc -- wrote $STOP; collection is stopped until it is removed" >> "$LOG"
  exit "$rc"
fi
