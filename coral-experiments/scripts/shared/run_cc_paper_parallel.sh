#!/usr/bin/env bash
# Launch CC/LSM runs for the Llama paper datasets (4 datasets × N budgets).
#
# Default: 4 datasets × budgets 4, 8, 16 = 12 jobs, dispatched across GPUs.
# Each job pins CUDA_VISIBLE_DEVICES to one GPU and writes to
# logged_results/<dataset>/cc_results_budget_<N>.json. Runners checkpoint after
# every case, so kill & restart is safe (they resume).
#
# Model defaults to meta-llama/Llama-3.1-8B-Instruct (see load_cc_model.py).
# cc_cases.json must exist for each dataset; missing pragma exports are run
# automatically.
#
# Usage:
#   bash scripts/shared/run_cc_paper_parallel.sh
#
# Tunables:
#   GPU_IDS   — space-separated GPU indices (default "0 1 2 3")
#   BUDGETS   — space-separated fixed budgets (default "4 8 16")
#   CONDA_ENV — conda env name (default "coral")
#   MODEL     — optional HF model id (default: Llama-3.1-8B-Instruct)
#
# Examples:
#   GPU_IDS="0 1" bash scripts/shared/run_cc_paper_parallel.sh
#   BUDGETS="4 8" bash scripts/shared/run_cc_paper_parallel.sh
#   BUDGETS="16" GPU_IDS="2" bash scripts/shared/run_cc_paper_parallel.sh

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
BUDGETS="${BUDGETS:-4 8 16}"
read -ra GPU_ARR <<< "$GPU_IDS"
read -ra BUDGET_ARR <<< "$BUDGETS"
export VLLM_USE_FLASHINFER_SAMPLER=0

DATASETS=(msmarco_criteria msmarco_pragma sonnet_criteria sonnet_pragma)

# dataset → python module
declare -A MODULE=(
  [msmarco_criteria]=scripts.criteria.msmarco.run_cc_full
  [sonnet_criteria]=scripts.criteria.sonnet.run_cc_full
  [msmarco_pragma]=scripts.pragma.msmarco.run_cc_full
  [sonnet_pragma]=scripts.pragma.sonnet.run_cc_full
)

# dataset → needs --run flag (pragma msmarco only)
declare -A NEEDS_RUN=(
  [msmarco_pragma]=1
)

# dataset → export command when cc_cases.json is missing
declare -A EXPORTS=(
  [msmarco_criteria]="(already built by run_subsample.py)"
  [sonnet_criteria]="(already built by run_subsample.py)"
  [msmarco_pragma]="python3 -m scripts.pragma.msmarco.run_cc_full --export"
  [sonnet_pragma]="python3 -m scripts.pragma.sonnet.run_cc_ablation --export"
)

# ---------------------------------------------------------------------------
# Step 1 — make sure cc_cases.json exists for every dataset
# ---------------------------------------------------------------------------

for ds in "${DATASETS[@]}"; do
  path="logged_results/${ds}/cc_cases.json"
  if [[ ! -f "$path" ]]; then
    cmd="${EXPORTS[$ds]}"
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
# Step 2 — define jobs: dataset × budget
# ---------------------------------------------------------------------------

JOBS=()
for ds in "${DATASETS[@]}"; do
  for budget in "${BUDGET_ARR[@]}"; do
    JOBS+=("${ds}|${budget}")
  done
done

echo "Launching ${#JOBS[@]} jobs across ${#GPU_ARR[@]} GPU(s): ${GPU_ARR[*]}"
echo "Budgets: ${BUDGET_ARR[*]}"
mkdir -p logs/cc_paper

# ---------------------------------------------------------------------------
# Step 3 — dispatch: one job per GPU at a time, queue the rest
# ---------------------------------------------------------------------------

declare -A PID_BY_GPU=()
JOB_IDX=0

launch_job() {
  local gpu=$1
  local job=$2
  IFS='|' read -r ds budget <<< "$job"
  local mod="${MODULE[$ds]}"
  local log="logs/cc_paper/${ds}_budget${budget}_gpu${gpu}.log"
  local run_flag=()
  if [[ "${NEEDS_RUN[$ds]:-0}" == "1" ]]; then
    run_flag=(--run)
  fi
  local model_flag=()
  if [[ -n "${MODEL:-}" ]]; then
    model_flag=(--model "$MODEL")
  fi

  echo "  [GPU $gpu] $ds budget=$budget  →  $log"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python3 -m "$mod" "${run_flag[@]}" --vllm "${model_flag[@]}" \
        --budget "$budget" \
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
echo "Results written to logged_results/<dataset>/cc_results_budget_<N>.json"
