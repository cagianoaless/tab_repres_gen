#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PYTHON_BIN:-/home/alessandrocagiano/.conda/envs/tfms/bin/python}
WORKERS=${MATERIALIZATION_WORKERS:-12}
EXP="$ROOT/RFMLink/side_experiments/g2t_q512_hybrid_solver_metric_free"
RESULT="$ROOT/results/g2t_q512_hybrid_solver_metric_free_v1"
ANALYSIS="$ROOT/results/analysis/g2t_q512_hybrid_solver_metric_free_v1"
LOGROOT="$RESULT/materialization/pool${WORKERS}"
STATE="$LOGROOT/state.log"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$LOGROOT"
exec 9>"$RESULT/.materialization_pool.lock"
flock -n 9 || exit 0
printf 'START workers=%s utc=%s pid=%s\n' \
  "$WORKERS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "$STATE"

spectral="$ANALYSIS/spectral_audit.json"
status=$(
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$spectral"
)
[[ "$status" == "PASS" ]] || {
  printf 'FAIL stage=spectral_gate\n' >> "$STATE"
  exit 2
}

pids=()
for ((worker=0; worker<WORKERS; worker++)); do
  "$PY" "$EXP/materialize.py" --shard-id "$worker" --num-shards "$WORKERS" \
    > "$LOGROOT/shard${worker}.log" 2>&1 &
  pids+=("$!")
  printf 'WORKER_START shard=%s pid=%s\n' "$worker" "$!" >> "$STATE"
done

failed=0
for ((worker=0; worker<WORKERS; worker++)); do
  if wait "${pids[$worker]}"; then
    printf 'WORKER_COMPLETE shard=%s utc=%s\n' \
      "$worker" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
  else
    printf 'WORKER_FAIL shard=%s utc=%s\n' \
      "$worker" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
    failed=1
  fi
done
(( failed == 0 )) || exit 2

"$PY" "$EXP/spectral_audit.py" --finalize-materialized \
  > "$LOGROOT/finalize.log" 2>&1
final_status=$(
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$ANALYSIS/materialized_audit.json"
)
[[ "$final_status" == "PASS" ]] || exit 2
printf 'COMPLETE datasets=57 metrics_accessed=false utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
