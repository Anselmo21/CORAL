#!/usr/bin/env bash
# Launch all 12 CC/LSM runs for the Qwen3-32B datasets:
# 4 datasets × 3 budgets (4, 8, 16) in parallel across available GPUs.
#
# Each job checkpoints after every case, so kill & restart is safe.
# cc_cases.json must exist for each dataset (produced by run_qwen3_subsample.py).
#
# Usage:
#   bash scripts/shared/run_cc_qwen_parallel.sh
#
# Tunables:
#   GPU_IDS  — space-separated GPU indices (default "0 1 2 3")
#   CONDA_ENV — conda env name (default "coral")
#
# Example (2 GPUs):
#   GPU_IDS="0 1" bash scripts/shared/run_cc_qwen_parallel.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-coral}"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  if conda env list | awk '{print $1}' | grep -qx "CORAL-backend"; then
    CONDA_ENV=CORAL-backend
  else
    echo "ERROR: conda env '$CONDA_ENV' not found (set CONDA_ENV=... to override)"
    conda env list
    exit 1
  fi
fi

GPU_IDS="${GPU_IDS:-0 1 2 3}"
read -ra GPU_ARR <<< "$GPU_IDS"
export VLLM_USE_FLASHINFER_SAMPLER=0

# ---------------------------------------------------------------------------
# Verify cc_cases.json exists for all 4 datasets
# ---------------------------------------------------------------------------

DATASETS=(msmarco_criteria msmarco_pragma sonnet_criteria sonnet_pragma)
for ds in "${DATASETS[@]}"; do
  path="logged_results/qwen3_${ds}/cc_cases.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $path missing — run run_qwen3_subsample.py --dataset $ds first"
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Define 12 jobs: dataset × budget
# ---------------------------------------------------------------------------

JOBS=()
for ds in "${DATASETS[@]}"; do
  for budget in 4 8 16; do
    JOBS+=("${ds}|${budget}")
  done
done

echo "Launching ${#JOBS[@]} jobs across ${#GPU_ARR[@]} GPU(s): ${GPU_ARR[*]}"
mkdir -p logs/cc_qwen

# ---------------------------------------------------------------------------
# Dispatch: round-robin across GPUs, one job per GPU at a time
# ---------------------------------------------------------------------------

declare -A PID_BY_GPU=()
JOB_IDX=0

launch_job() {
  local gpu=$1
  local job=$2
  IFS='|' read -r ds budget <<< "$job"
  local log="logs/cc_qwen/${ds}_budget${budget}_gpu${gpu}.log"

  echo "  [GPU $gpu] dataset=$ds budget=$budget  →  $log"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python3 -m scripts.shared.run_cc_qwen \
        --dataset "$ds" --budget "$budget" --vllm \
      > "$log" 2>&1 &
  PID_BY_GPU[$gpu]=$!
}

for gpu in "${GPU_ARR[@]}"; do
  if (( JOB_IDX < ${#JOBS[@]} )); then
    launch_job "$gpu" "${JOBS[$JOB_IDX]}"
    JOB_IDX=$((JOB_IDX + 1))
  fi
done

while (( ${#PID_BY_GPU[@]} > 0 )); do
  for gpu in "${!PID_BY_GPU[@]}"; do
    pid=${PID_BY_GPU[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || echo "  [GPU $gpu] job exited non-zero"
      unset PID_BY_GPU[$gpu]
      if (( JOB_IDX < ${#JOBS[@]} )); then
        launch_job "$gpu" "${JOBS[$JOB_IDX]}"
        JOB_IDX=$((JOB_IDX + 1))
      fi
    fi
  done
  sleep 5
done

echo ""
echo "All ${#JOBS[@]} CC runs complete."
echo "Results written to logged_results/qwen3_<dataset>/cc_results_budget_<N>.json"
