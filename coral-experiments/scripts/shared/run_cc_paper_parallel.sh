#!/usr/bin/env bash
# Launch all 8 CC runs (4 datasets × 2 budget configs) across 3 GPUs.
#
# Each job pins CUDA_VISIBLE_DEVICES to one GPU and writes to a unique
# data/<dataset>/cc_results_<config>.json file. The runners checkpoint after
# every case, so killing & restarting is safe (it will resume).
#
# Each dataset must have cc_cases.json on disk first. If anything is missing,
# this script runs the --export steps up-front.
#
# Usage:
#   bash scripts/shared/run_cc_paper_parallel.sh
#
# Tunables:
#   GPU_IDS — space-separated list of GPU indices to use (default "1 2 3").
#             Example: GPU_IDS="0 2" bash scripts/shared/run_cc_paper_parallel.sh

set -euo pipefail

# Default conda env name is "coral" in docs; this machine uses CORAL-backend.
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

GPU_IDS="${GPU_IDS:-1 2 3}"
read -ra GPU_ARR <<< "$GPU_IDS"
export VLLM_USE_FLASHINFER_SAMPLER=0

# ---------------------------------------------------------------------------
# Step 1 — make sure cc_cases.json exists for every dataset
# ---------------------------------------------------------------------------

declare -A EXPORTS=(
  [data/msmarco_criteria/cc_cases.json]="(already built by run_subsample.py)"
  [data/sonnet_criteria/cc_cases.json]="(already built by run_subsample.py)"
  [data/msmarco_pragma/cc_cases.json]="python3 -m scripts.pragma.msmarco_regen.run_cc_full --export"
  [data/sonnet_pragma/cc_cases.json]="python3 -m scripts.pragma.sonnet.run_cc_ablation --export"
)

for path in "${!EXPORTS[@]}"; do
  if [[ ! -f "$path" ]]; then
    cmd="${EXPORTS[$path]}"
    if [[ "$cmd" == "("* ]]; then
      echo "ERROR: $path missing and produced by $cmd"
      echo "       run scripts.<dataset>.run_subsample first"
      exit 1
    fi
    echo "Exporting $path ..."
    conda run -n "$CONDA_ENV" $cmd
  fi
done

# ---------------------------------------------------------------------------
# Step 2 — define the 8 (dataset, budget-config) jobs
# ---------------------------------------------------------------------------

# format: "<module>|<needs --run: 0|1>|<budget flag>"
JOBS=(
  "scripts.msmarco.run_cc_full|0|--budget 16"
  "scripts.msmarco.run_cc_full|0|--pct 10"
  "scripts.sonnet.run_cc_full|0|--budget 16"
  "scripts.sonnet.run_cc_full|0|--pct 10"
  "scripts.pragma.msmarco_regen.run_cc_full|1|--budget 16"
  "scripts.pragma.msmarco_regen.run_cc_full|1|--pct 10"
  "scripts.pragma.sonnet.run_cc_full|0|--budget 16"
  "scripts.pragma.sonnet.run_cc_full|0|--pct 10"
)

# ---------------------------------------------------------------------------
# Step 3 — dispatch up to NUM_GPUS jobs in parallel, one per GPU
# ---------------------------------------------------------------------------

mkdir -p logs/cc_paper
declare -A PID_BY_GPU=()
JOB_IDX=0

launch_job() {
  local gpu=$1
  local job=$2
  IFS='|' read -r mod needs_run flag <<< "$job"
  local tag="${mod//./_}_${flag//[ -]/_}"
  local log="logs/cc_paper/${tag}_gpu${gpu}.log"
  local run_flag=()
  if [[ "$needs_run" == "1" ]]; then
    run_flag=(--run)
  fi

  echo "  [GPU $gpu] $mod ${run_flag[*]} --vllm $flag  →  $log"
  CUDA_VISIBLE_DEVICES=$gpu \
    conda run -n "$CONDA_ENV" python3 -m $mod "${run_flag[@]}" --vllm $flag \
      > "$log" 2>&1 &
  PID_BY_GPU[$gpu]=$!
}

# Initial fill: one job per GPU
for gpu in "${GPU_ARR[@]}"; do
  if (( JOB_IDX < ${#JOBS[@]} )); then
    launch_job "$gpu" "${JOBS[$JOB_IDX]}"
    JOB_IDX=$((JOB_IDX + 1))
  fi
done

# Drain: as each GPU finishes, queue the next job on it
while (( ${#PID_BY_GPU[@]} > 0 )); do
  for gpu in "${!PID_BY_GPU[@]}"; do
    pid=${PID_BY_GPU[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || echo "  [GPU $gpu] job exited with non-zero status"
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
echo "All 8 CC runs complete. Merging into paper_results.json ..."
conda run -n "$CONDA_ENV" python3 -m scripts.shared.merge_cc_paper
