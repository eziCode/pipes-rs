#!/bin/sh
set -eu

run_id="${1:-dora-01}"
case "$run_id" in *[!A-Za-z0-9._-]*|'') echo "invalid run ID" >&2; exit 2;; esac
command -v dora >/dev/null 2>&1 || { echo "install Dora first: cargo install dora-cli --version 1.0.1 --locked" >&2; exit 2; }

root="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
result="$root/benchmark-results/dora/$run_id"
[ ! -e "$result" ] || { echo "result already exists: $result" >&2; exit 2; }
mkdir -p "$result"

export DATA_ROOT="$root/data/waymo-v2-sample"
export SPLIT="${BENCHMARK_SPLIT:-training}"
export SEGMENT="${BENCHMARK_SEGMENT:-10023947602400723454_1120_000_1140_000}"
export MODEL_PATH="$root/models/yolox_nano.onnx"
export OUTPUT_CSV="$result/perception.csv"
export CONFIDENCE="${CONFIDENCE:-0.3}"
export NMS="${NMS:-0.45}"
export FRAME_LIMIT="${FRAME_LIMIT:-0}"

cargo build --manifest-path "$root/Cargo.toml" --release -p pipes-dora-benchmark
start="$(date +%s)"
(cd "$root/benchmarks/dora" && dora run dataflow.yml) 2>&1 | tee "$result/output.log"
end="$(date +%s)"
echo "elapsed_seconds=$((end-start))" > "$result/timing.txt"
shasum -a 256 "$MODEL_PATH" > "$result/input.sha256"
echo "results: $result"
