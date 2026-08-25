#!/usr/bin/env bash
# Launch CC/LSM runs for Llama paper datasets (4 datasets × N budgets).
#
# Default: Llama-3.1-8B, one GPU per job, writes to
# logged_results/llama31-8b/<dataset>/cc_results_budget_<N>.json.
# Runners checkpoint after every case, so kill & restart is safe (they resume).
#
# 70B needs tensor parallelism across all 4 L40S (weights ~140GB fp16):
#   DATASET_ROOT=llama31-70b MODEL=meta-llama/Llama-3.1-70B-Instruct \
#     TENSOR_PARALLEL_SIZE=4 BUDGETS="4" bash scripts/shared/run_cc_paper_parallel.sh
#
# Usage:
#   bash scripts/shared/run_cc_paper_parallel.sh
#
# Tunables:
#   GPU_IDS              — space-separated GPU indices (default "0 1 2 3")
#   BUDGETS              — space-separated fixed budgets (default "4 8 16")
#   CONDA_ENV            — conda env name (default "coral")
#   MODEL                — HF model id (default: Llama-3.1-8B-Instruct)
#   DATASET_ROOT         — logged_results/<root>/<dataset>/ prefix (default "llama31-8b")
#   TENSOR_PARALLEL_SIZE — GPUs per job (default 1; 70B needs 4)
#   MAX_MODEL_LEN        — vLLM context cap (default 16384; Llama-3.1 native 128k)
#
# Examples:
#   GPU_IDS="0 1" bash scripts/shared/run_cc_paper_parallel.sh
#   BUDGETS="4" bash scripts/shared/run_cc_paper_parallel.sh
#   DATASET_ROOT=llama31-70b MODEL=meta-llama/Llama-3.1-70B-Instruct \
#     TENSOR_PARALLEL_SIZE=4 BUDGETS="4" bash scripts/shared/run_cc_paper_parallel.sh

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
DATASET_ROOT="${DATASET_ROOT:-llama31-8b}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
read -ra GPU_ARR <<< "$GPU_IDS"
read -ra BUDGET_ARR <<< "$BUDGETS"
export VLLM_USE_FLASHINFER_SAMPLER=0

NUM_GPUS=${#GPU_ARR[@]}
if (( NUM_GPUS % TENSOR_PARALLEL_SIZE != 0 )); then
  echo "ERROR: ${NUM_GPUS} GPU(s) in GPU_IDS not divisible by TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
  exit 1
fi

WORKERS=()
for (( i=0; i<NUM_GPUS; i+=TENSOR_PARALLEL_SIZE )); do
  slot=""
  for (( j=0; j<TENSOR_PARALLEL_SIZE; j++ )); do
    if [[ -n "$slot" ]]; then slot+=","; fi
    slot+="${GPU_ARR[$((i + j))]}"
  done
  WORKERS+=("$slot")
done

DATASETS=(msmarco_criteria msmarco_pragma curated_criteria curated_pragma)

for ds in "${DATASETS[@]}"; do
  path="logged_results/${DATASET_ROOT}/${ds}/cc_cases.json"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $path missing"
    echo "       expected under logged_results/${DATASET_ROOT}/<dataset>/"
    exit 1
  fi
done

JOBS=()
for ds in "${DATASETS[@]}"; do
  for budget in "${BUDGET_ARR[@]}"; do
    JOBS+=("${ds}|${budget}")
  done
done

echo "Launching ${#JOBS[@]} jobs across ${#WORKERS[@]} worker slot(s): ${WORKERS[*]}"
echo "Dataset root: logged_results/${DATASET_ROOT}/  Model: ${MODEL}"
echo "Budgets: ${BUDGET_ARR[*]}  tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
mkdir -p logs/cc_paper

declare -A PID_BY_WORKER=()
JOB_IDX=0

launch_job() {
  local worker_idx=$1
  local job=$2
  IFS='|' read -r ds budget <<< "$job"
  local gpus="${WORKERS[$worker_idx]}"
  local log="logs/cc_paper/${DATASET_ROOT}_${ds}_budget${budget}_gpu${gpus//,/_}.log"

  echo "  [GPUs $gpus] $ds budget=$budget  →  $log"
  CUDA_VISIBLE_DEVICES=$gpus PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python3 -m scripts.shared.run_cc_qwen \
        --dataset "$ds" --budget "$budget" --vllm \
        --model "$MODEL" \
        --dataset-root "$DATASET_ROOT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
      > "$log" 2>&1 &
  PID_BY_WORKER[$worker_idx]=$!
}

for worker_idx in "${!WORKERS[@]}"; do
  if (( JOB_IDX < ${#JOBS[@]} )); then
    launch_job "$worker_idx" "${JOBS[$JOB_IDX]}"
    JOB_IDX=$((JOB_IDX + 1))
  fi
done

while (( ${#PID_BY_WORKER[@]} > 0 )); do
  for worker_idx in "${!PID_BY_WORKER[@]}"; do
    pid=${PID_BY_WORKER[$worker_idx]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || echo "  [worker $worker_idx] job exited non-zero"
      unset PID_BY_WORKER[$worker_idx]
      if (( JOB_IDX < ${#JOBS[@]} )); then
        launch_job "$worker_idx" "${JOBS[$JOB_IDX]}"
        JOB_IDX=$((JOB_IDX + 1))
      fi
    fi
  done
  sleep 5
done

echo ""
echo "All ${#JOBS[@]} CC runs complete."
echo "Results written to logged_results/${DATASET_ROOT}/<dataset>/cc_results_budget_<N>.json"
