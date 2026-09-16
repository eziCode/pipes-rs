#!/bin/sh
set -eu

run_id="${1:-dora-01}"
case "$run_id" in *[!A-Za-z0-9._-]*|'') echo "invalid run ID" >&2; exit 2;; esac

mkdir -p benchmark-results
export BENCHMARK_RUN_ID="$run_id"
export BENCHMARK_UID="$(id -u)"
export BENCHMARK_GID="$(id -g)"

docker compose build dora-benchmark
docker compose run --rm dora-benchmark

result="benchmark-results/dora/$run_id"
docker image inspect pipes-rs-dora-benchmark:dora-1.0.1-arrow-59 > "$result/image.json"
docker compose config > "$result/compose.yaml"
echo "results: $result"
