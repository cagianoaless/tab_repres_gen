#!/bin/sh
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
RUNNER="$ROOT/RFMLink/scripts/run_ultra57_rfmlink_q512.py"
AUDIT="$ROOT/RFMLink/scripts/audit_ultra_graph_consistency.py"
PARENT_ROOT="$ROOT/results/ultra57_rfmlink_batch1_v1"
RESULT_ROOT="$ROOT/results/ultra57_rfmlink_q512_v1"
PLAN="$ROOT/results/analysis/ultra57_rfmlink_q512_v1_plan.json"
LOG_ROOT="$RESULT_ROOT/launcher_logs"
STATE="$LOG_ROOT/launcher_state.txt"
FAILURE="$RESULT_ROOT/launcher_failure.json"
CONDA_ENV=${CONDA_ENV:-tfms}
GPU_WORKERS=${GPU_WORKERS:-4}
CPU_WORKERS=${CPU_WORKERS:-4}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export RFMLINK_TABPFN_PREDICT_BATCH=50000

mkdir -p "$LOG_ROOT"
exec 9>"$RESULT_ROOT/.launcher.lock"
if ! flock -n 9; then
  printf '%s\n' "Another q512 launcher is active." >&2
  exit 0
fi

printf '%s\n' "launcher_pid=$$ gpu_workers=$GPU_WORKERS cpu_workers=$CPU_WORKERS" > "$STATE"
printf '%s\n' "PLAN jobs=684 query_triples_per_split=512 max_directional_queries=1024 estimated_hours=35-75 estimated_storage_gib=90-140" >> "$STATE"

while :
do
  parent_status=$(tail -n 1 "$PARENT_ROOT/launcher_logs/launcher_state.txt" 2>/dev/null || true)
  case "$parent_status" in
    COMPLETE)
      printf '%s\n' "PARENT_SUITE_COMPLETE" >> "$STATE"
      break
      ;;
    STOPPED*|FAILED*|LAUNCHER_EXIT*)
      printf '%s\n' "STOPPED parent_suite_state=$parent_status" >> "$STATE"
      exit 2
      ;;
    *)
      printf '%s\n' "WAITING_FOR_PARENT_SUITE status=$parent_status" >> "$STATE"
      sleep 30
      ;;
  esac
done

if [ ! -f "$PLAN" ]; then
  if ! conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --register \
    > "$LOG_ROOT/register.log" 2>&1
  then
    printf '%s\n' "FAILED registration" >> "$STATE"
    exit 3
  fi
fi

if ! conda run --no-capture-output -n "$CONDA_ENV" python "$AUDIT" --run \
  > "$LOG_ROOT/ultra_graph_consistency_audit.log" 2>&1
then
  printf '%s\n' "FAILED masking_audit_execution" >> "$STATE"
  exit 4
fi
if ! conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --validate \
  > "$LOG_ROOT/validate.log" 2>&1
then
  printf '%s\n' "FAILED suite_validation" >> "$STATE"
  exit 5
fi
printf '%s\n' "VALIDATED planned_jobs=684" >> "$STATE"

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

fail_dataset () {
  dataset=$1
  stage=$2
  temporary="$FAILURE.tmp"
  printf '{"dataset":"%s","stage":"%s","message":"high-query suite stopped; no dataset skipped"}\n' \
    "$dataset" "$stage" > "$temporary"
  mv "$temporary" "$FAILURE"
  printf '%s\n' "STOPPED_ON_FAILURE dataset=$dataset stage=$stage" >> "$STATE"
  exit 6
}

DATASETS=$(conda run -n "$CONDA_ENV" python "$RUNNER" --queue-list)
for dataset in $DATASETS
do
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
  conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --summarize \
    > "$LOG_ROOT/summary_partial.log" 2>&1 || true
  printf '%s\n' "DATASET_COMPLETE dataset=$dataset" >> "$STATE"
done

conda run --no-capture-output -n "$CONDA_ENV" python "$RUNNER" --summarize \
  > "$LOG_ROOT/summary.log" 2>&1
rm -f "$FAILURE"
printf '%s\n' "COMPLETE" >> "$STATE"
