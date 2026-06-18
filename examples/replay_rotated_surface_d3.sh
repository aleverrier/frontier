#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/frontier-example.XXXXXX")}"
SAMPLE_ROWS="${OUT_ROOT}/rotated_surface_d3_sample_rows.csv"
REPLAY_DIR="${OUT_ROOT}/rotated_surface_d3_replay"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="${PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

"${PYTHON_BIN}" -m tools.frontier_sample_rows \
  --out "${SAMPLE_ROWS}" \
  --backend rotated_surface_d3 \
  --p-location 0.001 \
  --shots 4 \
  --seed 20260615 \
  --progress-every-rows 0

"${PYTHON_BIN}" -m tools.frontier_sample_replay \
  --sample-rows "${SAMPLE_ROWS}" \
  --out-dir "${REPLAY_DIR}" \
  --code rotated_surface_d3 \
  --backend rotated_surface_d3 \
  --p-location 0.001 \
  --shot-start 0 \
  --shot-stop 3 \
  --K 16 \
  --Delta 100 \
  --direction-mode fwd_bwd_committee \
  --engine auto \
  --column-order deadline_reorder \
  --backward-column-order backward_deadline_reorder \
  --cpus 1 \
  --progress-every-shards 1

echo "sample_rows=${SAMPLE_ROWS}"
echo "replay_dir=${REPLAY_DIR}"
