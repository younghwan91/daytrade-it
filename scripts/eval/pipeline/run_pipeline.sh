#!/usr/bin/env bash
# One command, four stages. Each stage reads pipeline_config.json and the
# previous stage's output file; none defines its own dates or holdout.
#
#   scripts/eval/pipeline/run_pipeline.sh            # stages 1-3
#   scripts/eval/pipeline/run_pipeline.sh --with-4   # also the frozen-holdout combination
#
# Stage 4 is opt-in on purpose: it is the only stage allowed to read outcomes on
# the frozen holdout, and running it is a one-way door for that holdout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
OUT="$ROOT/data/eval/pipeline"
mkdir -p "$OUT"
cd "$ROOT"

echo "== stage 1: news selection (upside / risk-reward scores, out-of-fold)"
"$PY" "$HERE/stage1_select.py" --out "$OUT/stage1_oof.jsonl" --model-dir "$OUT/stage1_model"

echo "== stage 2: entry timing within selected news"
"$PY" "$HERE/stage2_entry.py" --in "$OUT/stage1_oof.jsonl" --out "$OUT/stage2_oof.jsonl"

echo "== stage 3: exit timing"
"$PY" "$HERE/stage3_exit.py" --in "$OUT/stage2_oof.jsonl" --out "$OUT/stage3_oof.jsonl"

if [[ "${1:-}" == "--with-4" ]]; then
  echo "== stage 4: end-to-end on the frozen holdout (one-way door)"
  "$PY" "$HERE/stage4_combine.py" --in "$OUT/stage3_oof.jsonl" --out "$OUT/stage4_report.json"
else
  echo "== stage 4 skipped (pass --with-4 to spend the frozen holdout)"
fi
