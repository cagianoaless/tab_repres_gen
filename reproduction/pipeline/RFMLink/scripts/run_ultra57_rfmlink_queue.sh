#!/bin/sh
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
RUNNER="$ROOT/RFMLink/scripts/run_ultra57_rfmlink.py"
AUDIT="$ROOT/RFMLink/scripts/audit_ultra_graph_consistency.py"
BATCH_BENCHMARK="$ROOT/RFMLink/scripts/benchmark_tabpfn_prediction_batch.py"
BATCH_DECISION="$ROOT/results/analysis/tabpfn_prediction_batch_benchmark.json"
STAGER="$ROOT/RFMLink/scripts/stage_ultra57_datasets.py"
OLD_ROOT="$ROOT/results/ultra4g_ood_reassessment_v1"
RESULT_ROOT="$ROOT/results/ultra57_rfmlink_batch1_v1"
LOG_ROOT="$RESULT_ROOT/launcher_logs"
STATE="$LOG_ROOT/launcher_state.txt"
CONDA_ENV=${CONDA_ENV:-tfms}
GPU_WORKERS=${GPU_WORKERS:-4}
CPU_WORKERS=${CPU_WORKERS:-4}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOG_ROOT"
exec 9>"$RESULT_ROOT/.launcher.lock"
if ! flock -n 9; then
  printf '%s\n' "Another ULTRA-57 launcher is active." >&2
  exit 0
fi

printf '%s\n' "launcher_pid=$$ gpu_workers=$GPU_WORKERS cpu_workers=$CPU_WORKERS" > "$STATE"

while :
do
  old_status=$(tail -n 1 "$OLD_ROOT/launcher_logs/launcher_state.txt" 2>/dev/null || true)
  case "$old_status" in
    COMPLETE)
      printf '%s\n' "PREVIOUS_SUITE_COMPLETE" >> "$STATE"
      break
      ;;
    LAUNCHER_EXIT*|FAILED*)
      printf '%s\n' "PREVIOUS_SUITE_TERMINAL_STATE $old_status" >> "$STATE"
      break
      ;;
    *)
      printf '%s\n' "WAITING_FOR_PREVIOUS_SUITE status=$old_status" >> "$STATE"
      sleep 60
      ;;
  esac
done

if [ ! -f "$BATCH_DECISION" ]; then
  if ! conda run --no-capture-output -n "$CONDA_ENV" python "$BATCH_BENCHMARK" \
    > "$LOG_ROOT/tabpfn_prediction_batch_benchmark.log" 2>&1
  then
    printf '%s\n' "FAILED prediction_batch_benchmark" >> "$STATE"
    exit 2
  fi
fi
PREDICTION_BATCH_ROWS=$(conda run -n "$CONDA_ENV" python -c \
  "import json; print(json.load(open('$BATCH_DECISION'))['recommended_batch_rows'])")
case "$PREDICTION_BATCH_ROWS" in
  50000|500000) ;;
  *)
    printf '%s\n' "FAILED invalid_prediction_batch=$PREDICTION_BATCH_ROWS" >> "$STATE"
    exit 2
    ;;
esac
export RFMLINK_TABPFN_PREDICT_BATCH="$PREDICTION_BATCH_ROWS"
printf '%s\n' "PREDICTION_BATCH_SELECTED rows=$PREDICTION_BATCH_ROWS" >> "$STATE"

if ! conda run --no-capture-output -n "$CONDA_ENV" python "$AUDIT" --run \
  > "$LOG_ROOT/ultra_graph_consistency_audit.log" 2>&1
then
  printf '%s\n' "FAILED masking_audit_execution" >> "$STATE"
  exit 2
fi

if ! conda run --no-capture-output -n "$CONDA_ENV" python -c \
  "import json; p=json.load(open('$ROOT/results/analysis/ultra_graph_consistency_audit.json')); assert p['broad_suite_allowed'], p" \
  > "$LOG_ROOT/ultra_graph_consistency_gate.log" 2>&1
then
  printf '%s\n' "STOPPED masking_audit_gate" >> "$STATE"
  exit 3
fi
printf '%s\n' "MASKING_AUDIT_PASS" >> "$STATE"

if ! conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --validate \
  > "$LOG_ROOT/validate.log" 2>&1
then
  printf '%s\n' "FAILED suite_validation" >> "$STATE"
  exit 4
fi
printf '%s\n' "VALIDATED planned_jobs=684 estimated_hours=14-35" >> "$STATE"

run_group () {
  dataset=$1
  group=$2
  workers=$3
  pids=""
  worker=0
  while [ "$worker" -lt "$workers" ]
  do
    conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
      --run-shard --group "$group" --dataset "$dataset" \
      --shard-id "$worker" --num-shards "$workers" \
      > "$LOG_ROOT/${dataset}_${group}_${worker}.log" 2>&1 &
    pid=$!
    pids="$pids $pid"
    printf '%s\n' "dataset=$dataset group=$group worker=$worker pid=$pid" >> "$STATE"
    worker=$((worker + 1))
  done
  status=0
  for pid in $pids
  do
    if ! wait "$pid"
    then
      status=1
    fi
  done
  return "$status"
}

DATASETS=$(conda run -n "$CONDA_ENV" python "$RUNNER" --queue-list)

fail_dataset () {
  dataset=$1
  stage=$2
  conda run --no-capture-output -n "$CONDA_ENV" python "$STAGER" \
    --fail --dataset "$dataset" --message "consumer stage failed: $stage" \
    >> "$LOG_ROOT/staging_${dataset}.log" 2>&1 || true
  printf '%s\n' "DATASET_FAILED dataset=$dataset stage=$stage" >> "$STATE"
  printf '%s\n' "STOPPED_ON_FAILURE dataset=$dataset stage=$stage" >> "$STATE"
  exit 5
}

for dataset in $DATASETS
do
  printf '%s\n' "WAITING_FOR_STAGED_DATASET dataset=$dataset" >> "$STATE"
  if ! claim_output=$(conda run --no-capture-output -n "$CONDA_ENV" python "$STAGER" \
    --claim --dataset "$dataset" 2>&1)
  then
    printf '%s\n' "$claim_output" > "$LOG_ROOT/staging_${dataset}.log"
    fail_dataset "$dataset" staging_claim
  fi
  printf '%s\n' "$claim_output" > "$LOG_ROOT/staging_${dataset}.log"
  case "$claim_output" in
    *ALREADY_DONE*)
      printf '%s\n' "DATASET_ALREADY_COMPLETE dataset=$dataset" >> "$STATE"
      continue
      ;;
  esac
  printf '%s\n' "DATASET_START dataset=$dataset" >> "$STATE"
  if ! conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
    --prepare --dataset "$dataset" > "$LOG_ROOT/prepare_${dataset}.log" 2>&1
  then
    fail_dataset "$dataset" prepare
  fi
  if ! conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
    --tune --dataset "$dataset" > "$LOG_ROOT/tune_${dataset}.log" 2>&1
  then
    fail_dataset "$dataset" tune
  fi
  if ! conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" \
    --smoke --dataset "$dataset" > "$LOG_ROOT/smoke_${dataset}.log" 2>&1
  then
    fail_dataset "$dataset" smoke
  fi
  if ! run_group "$dataset" tabpfn "$GPU_WORKERS"
  then
    fail_dataset "$dataset" tabpfn
  fi
  if ! run_group "$dataset" classical "$CPU_WORKERS"
  then
    fail_dataset "$dataset" classical
  fi
  if ! conda run --no-capture-output -n "$CONDA_ENV" python "$STAGER" \
    --complete --dataset "$dataset" >> "$LOG_ROOT/staging_${dataset}.log" 2>&1
  then
    fail_dataset "$dataset" completion_validation
  fi
  conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --summarize \
    > "$LOG_ROOT/summary_partial.log" 2>&1 || true
  printf '%s\n' "DATASET_COMPLETE dataset=$dataset" >> "$STATE"
done

conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --summarize \
  > "$LOG_ROOT/summary.log" 2>&1
printf '%s\n' "COMPLETE" >> "$STATE"
