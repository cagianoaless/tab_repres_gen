#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PYTHON_BIN:-/home/alessandrocagiano/.conda/envs/tfms/bin/python}
EXP="$ROOT/RFMLink/side_experiments/g2t_q512_hybrid_solver_metric_free"
RESULT="$ROOT/results/g2t_q512_hybrid_solver_metric_free_v1"
ANALYSIS="$ROOT/results/analysis/g2t_q512_hybrid_solver_metric_free_v1"
STATE="$RESULT/coordinator/state.log"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$RESULT/coordinator" "$RESULT/materialization/logs" "$ANALYSIS"
exec 9>"$RESULT/.metric_free_coordinator.lock"
flock -n 9 || exit 0
printf 'START utc=%s pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "$STATE"

fail() {
  printf 'FAIL stage=%s utc=%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
  exit 2
}

json_field() {
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

for shard in 0 1; do
  path="$RESULT/spectral_audit/shards/shard${shard}of2.json"
  while [[ ! -f "$path" ]]; do
    printf 'WAIT spectral_shard=%s utc=%s\n' "$shard" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
    sleep 30
  done
  [[ "$(json_field "$path" status)" == "PASS" ]] || fail "spectral_shard_${shard}"
done

"$PY" "$EXP/spectral_audit.py" --finalize-shards \
  > "$RESULT/coordinator/spectral_finalize.log" 2>&1 || fail spectral_finalize

spectral="$ANALYSIS/spectral_audit.json"
[[ "$(json_field "$spectral" status)" == "PASS" ]] || fail spectral_aggregate_status
[[ "$(json_field "$spectral" datasets_completed)" == "57" ]] || fail spectral_dataset_count
[[ "$(json_field "$spectral" graphs_completed)" == "1767" ]] || fail spectral_graph_count
printf 'SPECTRAL_PASS datasets=57 graphs=1767 utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"

"$PY" "$EXP/materialize.py" --shard-id 0 --num-shards 2 \
  > "$RESULT/materialization/logs/shard0.log" 2>&1 &
pid0=$!
"$PY" "$EXP/materialize.py" --shard-id 1 --num-shards 2 \
  > "$RESULT/materialization/logs/shard1.log" 2>&1 &
pid1=$!
printf 'MATERIALIZATION_START pid0=%s pid1=%s utc=%s\n' \
  "$pid0" "$pid1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"

wait "$pid0" || fail materialization_shard_0
wait "$pid1" || fail materialization_shard_1

for shard in 0 1; do
  path="$RESULT/materialization/shards/shard${shard}of2.json"
  [[ "$(json_field "$path" status)" == "COMPLETE" ]] \
    || fail "materialization_manifest_${shard}"
done

"$PY" "$EXP/spectral_audit.py" --finalize-materialized \
  > "$RESULT/coordinator/materialized_finalize.log" 2>&1 || fail materialized_finalize

materialized="$ANALYSIS/materialized_audit.json"
[[ "$(json_field "$materialized" status)" == "PASS" ]] || fail materialized_status
[[ "$(json_field "$materialized" datasets_completed)" == "57" ]] \
  || fail materialized_dataset_count
printf 'COMPLETE datasets=57 metrics_accessed=false utc=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE"
