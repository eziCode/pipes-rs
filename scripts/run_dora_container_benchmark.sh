#!/bin/sh
set -eu

run_id="${1:-dora-01}"
workload="${2:-full}"
case "$run_id" in *[!A-Za-z0-9._-]*|'') echo "invalid run ID" >&2; exit 2;; esac
case "$workload" in full|camera|lidar|transport) ;; *) echo "workload must be full, camera, lidar, or transport" >&2; exit 2;; esac

mkdir -p benchmark-results
export BENCHMARK_RUN_ID="$run_id"
export BENCHMARK_WORKLOAD="$workload"
export BENCHMARK_UID="$(id -u)"
export BENCHMARK_GID="$(id -g)"

docker compose build dora-benchmark
docker compose run --rm dora-benchmark

result_group=dora
[ "$workload" = full ] || result_group="dora-$workload"
result="benchmark-results/$result_group/$run_id"
docker image inspect pipes-rs-dora-benchmark:dora-1.0.1-arrow-59 > "$result/image.json"
docker compose config > "$result/compose.yaml"
echo "results: $result"
