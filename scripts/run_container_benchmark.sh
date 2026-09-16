#!/bin/sh
set -eu

scenario="${1:-realtime}"
run_id="${2:-run-001}"

case "$scenario" in
  realtime|burst-drop|burst-backpressure|overloaded|perception|perception-onnx|perception-camera|perception-lidar|perception-overloaded|arrow-transport) ;;
  *)
    echo "usage: $0 {realtime|burst-drop|burst-backpressure|overloaded|perception|perception-onnx|perception-camera|perception-lidar} [run-id]" >&2
    exit 2
    ;;
esac

mkdir -p benchmark-results

export BENCHMARK_SCENARIO="$scenario"
export BENCHMARK_RUN_ID="$run_id"
export BENCHMARK_UID="$(id -u)"
export BENCHMARK_GID="$(id -g)"

docker compose build benchmark
docker compose run --rm benchmark

docker image inspect pipes-rs-benchmark:rust-1.94-arrow-59 > \
  "benchmark-results/$scenario/$run_id/image.json"
docker compose config > "benchmark-results/$scenario/$run_id/compose.yaml"

echo "results: benchmark-results/$scenario/$run_id"
