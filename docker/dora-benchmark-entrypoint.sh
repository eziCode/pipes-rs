#!/bin/bash
set -euo pipefail

run_id="${RUN_ID:-dora-01}"
segment="${SEGMENT:-10023947602400723454_1120_000_1140_000}"
split="${SPLIT:-training}"

case "$run_id" in *[!A-Za-z0-9._-]*|'') echo "invalid RUN_ID" >&2; exit 2;; esac
case "$segment" in *[!0-9_]*|'') echo "invalid SEGMENT" >&2; exit 2;; esac
case "$split" in training|validation|testing) ;; *) echo "invalid SPLIT" >&2; exit 2;; esac

result_dir="/results/dora/$run_id"
[ ! -e "$result_dir" ] || { echo "result already exists: $result_dir" >&2; exit 2; }
mkdir -p "$result_dir"

data_root=/data/waymo-v2-sample
model=/models/yolox_nano.onnx
inputs=(
  "$data_root/$split/camera_image/$segment.parquet"
  "$data_root/$split/lidar/$segment.parquet"
  "$data_root/$split/lidar_camera_projection/$segment.parquet"
  "$data_root/$split/lidar_calibration/$segment.parquet"
  "$data_root/$split/camera_box/$segment.parquet"
  "$model"
)
for input in "${inputs[@]}"; do
  [ -r "$input" ] || { echo "missing benchmark input: $input" >&2; exit 2; }
done
sha256sum "${inputs[@]}" > "$result_dir/input.sha256"

export DATA_ROOT="$data_root" MODEL_PATH="$model"
export OUTPUT_CSV="$result_dir/perception.csv"
export SPLIT="$split" SEGMENT="$segment"
export CONFIDENCE="${CONFIDENCE:-0.3}" NMS="${NMS:-0.45}"
export FRAME_LIMIT="${FRAME_LIMIT:-0}"

{
  echo "framework=dora-1.0.1"
  echo "arrow=59"
  echo "run_id=$run_id"
  echo "segment=$segment"
  echo "split=$split"
  echo "frame_limit=$FRAME_LIMIT"
  echo "kernel=$(uname -srvm)"
  echo "architecture=$(uname -m)"
  echo "cpu_limit=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unavailable)"
  echo "memory_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unavailable)"
} > "$result_dir/environment.txt"

start_ns="$(date +%s%N)"
cp /benchmark/dataflow.yml /tmp/dataflow.yml
(cd /tmp && dora run dataflow.yml) 2>&1 | tee "$result_dir/output.log"
end_ns="$(date +%s%N)"
echo "elapsed_ns=$((end_ns-start_ns))" | tee "$result_dir/timing.txt"
