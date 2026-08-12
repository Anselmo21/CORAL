#!/usr/bin/env bash
# Launch CC/LSM runs for the cf2 case pool: 4 lattice runs × 4 datasets × N budgets.
#
# Prereq (CPU, any machine — outputs must be synced here):
#   conda run -n coral python3 -m scripts.shared.export_cc_cf2_cases
#   → logged_results/<run>-cf2/<dataset>/cc_cases.json
#
# Each job checkpoints after every case, so kill & restart is safe. Results:
#   logged_results/<run>-cf2/<dataset>/cc_results_budget_<N>.json
#
# Usage:
#   bash scripts/shared/run_cc_cf2_parallel.sh
#
# Tunables:
#   GPU_IDS              — space-separated GPU indices (default "0 1 2 3")
#   BUDGETS              — space-separated budgets (default "4 8 16", matches ERA)
#   TENSOR_PARALLEL_SIZE — GPUs per job (default 2)
#   GPU_MEM_UTIL         — vLLM gpu_memory_utilization (default 0.85; lower to
#                          0.80 if prompt-logprob scoring OOMs at runtime)
#   MAX_MODEL_LEN        — cap context length (default 16384; needed for models
#                          whose native max, e.g. 262k, demands more KV cache
#                          than fits)
#   MAX_NUM_SEQS         — cap concurrent sequences (default 64; hybrid/Mamba
#                          models need one cache block per sequence and vLLM's
#                          256 default can exceed what fits)
#   CONDA_ENV            — conda env name (default "coral")
#   RUNS                 — lattice runs to process (default all four)
#   MODEL_32B            — HF id for the qwen3-32b-* runs (default Qwen/Qwen3-32B)
#   MODEL_122B           — HF id for the qwen35-122b-* runs. NO DEFAULT — set it,
#                          or restrict RUNS to the 32b runs.
#
# Examples:
#   RUNS="qwen3-32b-100 qwen3-32b-percentage" bash scripts/shared/run_cc_cf2_parallel.sh
#   MODEL_122B=Qwen/Qwen3.5-122B TENSOR_PARALLEL_SIZE=4 bash scripts/shared/run_cc_cf2_parallel.sh

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
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
RUNS="${RUNS:-qwen3-32b-100 qwen3-32b-percentage qwen35-122b-100 qwen35-122b-percentage}"
MODEL_32B="${MODEL_32B:-Qwen/Qwen3-32B}"
MODEL_122B="${MODEL_122B:-}"
read -ra GPU_ARR <<< "$GPU_IDS"
read -ra BUDGET_ARR <<< "$BUDGETS"
read -ra RUN_ARR <<< "$RUNS"
export VLLM_USE_FLASHINFER_SAMPLER=0

model_for_run() {
  case "$1" in
    qwen3-32b-*)   echo "$MODEL_32B" ;;
    qwen35-122b-*) echo "$MODEL_122B" ;;
    *)             echo "" ;;
  esac
}

for run in "${RUN_ARR[@]}"; do
  if [[ -z "$(model_for_run "$run")" ]]; then
    echo "ERROR: no model id for run '$run' (set MODEL_122B=... or restrict RUNS)"
    exit 1
  fi
done

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
for run in "${RUN_ARR[@]}"; do
  for ds in "${DATASETS[@]}"; do
    path="logged_results/${run}-cf2/${ds}/cc_cases.json"
    if [[ ! -f "$path" ]]; then
      echo "ERROR: $path missing — run scripts.shared.export_cc_cf2_cases and sync it here"
      exit 1
    fi
  done
done

# Group jobs by run so a worker slot finishes one model before another job
# forces a different (potentially much larger) model load.
JOBS=()
for run in "${RUN_ARR[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for budget in "${BUDGET_ARR[@]}"; do
      JOBS+=("${run}|${ds}|${budget}")
    done
  done
done

echo "Launching ${#JOBS[@]} jobs across ${#WORKERS[@]} worker slot(s): ${WORKERS[*]}"
echo "Runs: ${RUN_ARR[*]}  Budgets: ${BUDGET_ARR[*]}  tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
mkdir -p logs/cc_cf2

declare -A PID_BY_WORKER=()
JOB_IDX=0

launch_job() {
  local worker_idx=$1
  local job=$2
  IFS='|' read -r run ds budget <<< "$job"
  local gpus="${WORKERS[$worker_idx]}"
  local model
  model=$(model_for_run "$run")
  local log="logs/cc_cf2/${run}_${ds}_budget${budget}_gpu${gpus//,/_}.log"

  echo "  [GPUs $gpus] run=$run dataset=$ds budget=$budget model=$model  →  $log"
  CUDA_VISIBLE_DEVICES=$gpus PYTHONUNBUFFERED=1 \
    conda run --no-capture-output -n "$CONDA_ENV" \
      python3 -m scripts.shared.run_cc_qwen \
        --dataset "$ds" --budget "$budget" --vllm \
        --model "$model" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --dataset-root "${run}-cf2" \
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
echo "Results written to logged_results/<run>-cf2/<dataset>/cc_results_budget_<N>.json"
