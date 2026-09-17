#!/bin/bash
set -euo pipefail

run_id="${RUN_ID:-dora-01}"
segment="${SEGMENT:-10023947602400723454_1120_000_1140_000}"
split="${SPLIT:-training}"
workload="${WORKLOAD:-full}"

case "$run_id" in *[!A-Za-z0-9._-]*|'') echo "invalid RUN_ID" >&2; exit 2;; esac
case "$segment" in *[!0-9_]*|'') echo "invalid SEGMENT" >&2; exit 2;; esac
case "$split" in training|validation|testing) ;; *) echo "invalid SPLIT" >&2; exit 2;; esac
case "$workload" in full|camera|lidar|transport) ;; *) echo "invalid WORKLOAD" >&2; exit 2;; esac

result_group="dora"
[ "$workload" = full ] || result_group="dora-$workload"
result_dir="/results/$result_group/$run_id"
[ ! -e "$result_dir" ] || { echo "result already exists: $result_dir" >&2; exit 2; }
mkdir -p "$result_dir"

data_root=/data/waymo-v2-sample
model=/models/yolox_nano.onnx
inputs=()
if [ "$workload" = full ] || [ "$workload" = camera ]; then
  inputs+=("$data_root/$split/camera_image/$segment.parquet" "$data_root/$split/camera_box/$segment.parquet" "$model")
fi
if [ "$workload" = full ] || [ "$workload" = lidar ]; then
  inputs+=("$data_root/$split/lidar/$segment.parquet" "$data_root/$split/lidar_camera_projection/$segment.parquet" "$data_root/$split/lidar_calibration/$segment.parquet")
fi
for input in "${inputs[@]}"; do
  [ -r "$input" ] || { echo "missing benchmark input: $input" >&2; exit 2; }
done
if [ "${#inputs[@]}" -gt 0 ]; then
  sha256sum "${inputs[@]}" > "$result_dir/input.sha256"
else
  : > "$result_dir/input.sha256"
fi

export DATA_ROOT="$data_root" MODEL_PATH="$model"
export OUTPUT_CSV="$result_dir/perception.csv"
export OUTPUT_DIR="$result_dir"
export SPLIT="$split" SEGMENT="$segment"
export CONFIDENCE="${CONFIDENCE:-0.3}" NMS="${NMS:-0.45}"
export FRAME_LIMIT="${FRAME_LIMIT:-0}"
export FUSION_WORK_MS="${FUSION_WORK_MS:-0}"
export TRANSPORT_FRAMES="${TRANSPORT_FRAMES:-199}"
export TRANSPORT_POINTS="${TRANSPORT_POINTS:-149796}"

{
  echo "framework=dora-1.0.1"
  echo "arrow=59"
  echo "run_id=$run_id"
  echo "segment=$segment"
  echo "split=$split"
  echo "frame_limit=$FRAME_LIMIT"
  echo "workload=$workload"
  echo "fusion_work_ms=$FUSION_WORK_MS"
  echo "kernel=$(uname -srvm)"
  echo "architecture=$(uname -m)"
  echo "cpu_limit=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unavailable)"
  echo "memory_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unavailable)"
} > "$result_dir/environment.txt"

start_ns="$(date +%s%N)"
cpu_start="$(awk '$1=="usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
descriptor=/benchmark/dataflow.yml
[ "$workload" = camera ] && descriptor=/benchmark/camera-only.yml
[ "$workload" = lidar ] && descriptor=/benchmark/lidar-only.yml
[ "$workload" = transport ] && descriptor=/benchmark/transport.yml
cp "$descriptor" /tmp/dataflow.yml
(cd /tmp && dora run dataflow.yml) 2>&1 | tee "$result_dir/output.log"
end_ns="$(date +%s%N)"
echo "elapsed_ns=$((end_ns-start_ns))" | tee "$result_dir/timing.txt"
cpu_end="$(awk '$1=="usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
{
  echo "cpu_usage_usec=$((cpu_end-cpu_start))"
  echo "memory_peak_bytes=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
} > "$result_dir/resources.txt"
{
  echo "framework=dora"
  echo "framework_version=1.0.1"
  echo "arrow_version=59.3.0"
  echo "rust_builder_version=1.95.0"
  if [ -r "$model" ]; then echo "model_sha256=$(sha256sum "$model" | awk '{print $1}')"; else echo "model_sha256=not-used"; fi
  echo "queue_size=4"
  echo "queue_policy=backpressure"
  echo "confidence=$CONFIDENCE"
  echo "nms=$NMS"
  echo "workload=$workload"
} > "$result_dir/manifest.txt"
