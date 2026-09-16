#!/bin/bash
set -euo pipefail

scenario="${SCENARIO:-realtime}"
run_id="${RUN_ID:-run-001}"
segment="${SEGMENT:-10023947602400723454_1120_000_1140_000}"
split="${SPLIT:-training}"
frame_limit="${FRAME_LIMIT:-0}"

case "$scenario" in
  realtime|burst-drop|burst-backpressure|overloaded|perception|perception-onnx|perception-camera|perception-lidar|perception-overloaded|arrow-transport) ;;
  *) echo "unknown scenario: $scenario" >&2; exit 2 ;;
esac

case "$run_id" in
  *[!A-Za-z0-9._-]*|'') echo "RUN_ID may contain only letters, digits, dot, underscore, and dash" >&2; exit 2 ;;
esac

case "$segment" in
  *[!0-9_]*|'') echo "SEGMENT has an invalid format" >&2; exit 2 ;;
esac

case "$split" in
  training|validation|testing) ;;
  *) echo "SPLIT must be training, validation, or testing" >&2; exit 2 ;;
esac

result_dir="/results/$scenario/$run_id"
if [ -e "$result_dir" ]; then
  echo "result directory already exists: $result_dir" >&2
  echo "choose another run ID so an earlier benchmark is never overwritten" >&2
  exit 2
fi
mkdir -p "$result_dir"

camera_file="/data/waymo-v2-sample/$split/camera_image/$segment.parquet"
lidar_file="/data/waymo-v2-sample/$split/lidar/$segment.parquet"
inputs="$camera_file $lidar_file"
if [ "$scenario" = perception ] || [ "$scenario" = perception-onnx ] || [ "$scenario" = perception-overloaded ]; then
  inputs="$inputs /data/waymo-v2-sample/$split/lidar_camera_projection/$segment.parquet"
  inputs="$inputs /data/waymo-v2-sample/$split/lidar_calibration/$segment.parquet"
  inputs="$inputs /data/waymo-v2-sample/$split/camera_box/$segment.parquet"
fi
if [ "$scenario" = perception-camera ]; then
  inputs="$camera_file /data/waymo-v2-sample/$split/camera_box/$segment.parquet"
fi
if [ "$scenario" = perception-lidar ]; then
  inputs="$lidar_file /data/waymo-v2-sample/$split/lidar_camera_projection/$segment.parquet"
  inputs="$inputs /data/waymo-v2-sample/$split/lidar_calibration/$segment.parquet"
fi
if [ "$scenario" = arrow-transport ]; then inputs=""; fi
if [ "$scenario" = perception-onnx ] || [ "$scenario" = perception-camera ] || [ "$scenario" = perception-overloaded ]; then
  inputs="$inputs /models/yolox_nano.onnx"
fi
for input in $inputs; do
  if [ ! -r "$input" ]; then
    echo "missing benchmark input: $input" >&2
    exit 2
  fi
done

# Hashing verifies the exact input and intentionally warms both files before
# every timed run, avoiding a cold-cache/warm-cache difference between repeats.
if [ -n "$inputs" ]; then sha256sum $inputs > "$result_dir/input.sha256"; else : > "$result_dir/input.sha256"; fi

if [ "$scenario" = arrow-transport ]; then
  set -- --transport-frames "${TRANSPORT_FRAMES:-199}" --transport-points "${TRANSPORT_POINTS:-149796}" --stage-queue-size 4 --transport-csv "$result_dir/perception.csv"
elif [ "$scenario" = perception ]; then
  set -- --frame-limit "$frame_limit" --stage-queue-size 4 --perception-csv "$result_dir/perception.csv"
elif [ "$scenario" = perception-onnx ]; then
  set -- --frame-limit "$frame_limit" --stage-queue-size 4 --onnx-model /models/yolox_nano.onnx --perception-csv "$result_dir/perception.csv"
elif [ "$scenario" = perception-camera ]; then
  set -- --frame-limit "$frame_limit" --stage-queue-size 4 --onnx-model /models/yolox_nano.onnx --perception-component camera --perception-csv "$result_dir/perception.csv"
elif [ "$scenario" = perception-lidar ]; then
  set -- --frame-limit "$frame_limit" --stage-queue-size 4 --perception-component lidar --perception-csv "$result_dir/perception.csv"
elif [ "$scenario" = perception-overloaded ]; then
  set -- --frame-limit "$frame_limit" --stage-queue-size 4 --onnx-model /models/yolox_nano.onnx --fusion-work-ms 125 --perception-csv "$result_dir/perception.csv"
else
  case "$scenario" in
    realtime)
      set -- --speed 1 --queue-size 8 --queue-policy drop-oldest
      ;;
    burst-drop)
      set -- --speed 0 --queue-size 8 --queue-policy drop-oldest
      ;;
    burst-backpressure)
      set -- --speed 0 --queue-size 8 --queue-policy backpressure
      ;;
    overloaded)
      set -- --speed 1 --queue-size 8 --queue-policy drop-oldest --lidar-work-ms 125
      ;;
  esac
  set -- "$@" --csv "$result_dir/measurements.csv"
fi

{
  echo "scenario=$scenario"
  echo "run_id=$run_id"
  echo "segment=$segment"
  echo "split=$split"
  echo "command=pipes-rs --segment $segment --split $split $*"
  echo "kernel=$(uname -srvm)"
  echo "architecture=$(uname -m)"
  echo "cpu_limit=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unavailable)"
  echo "memory_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unavailable)"
} > "$result_dir/environment.txt"

start_ns="$(date +%s%N)"
cpu_start="$(awk '$1=="usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
pipes-rs \
  --data-root /data/waymo-v2-sample \
  --segment "$segment" \
  --split "$split" \
  --camera front \
  --lidar top \
  "$@" \
  2>&1 | tee "$result_dir/output.log"
end_ns="$(date +%s%N)"
elapsed_ns=$((end_ns - start_ns))
echo "elapsed_ns=$elapsed_ns" | tee "$result_dir/timing.txt"
cpu_end="$(awk '$1=="usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)"
{
  echo "cpu_usage_usec=$((cpu_end-cpu_start))"
  echo "memory_peak_bytes=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)"
} > "$result_dir/resources.txt"
{
  echo "framework=pipes-rs"
  echo "framework_version=0.1.0"
  echo "arrow_version=59.3.0"
  echo "rust_builder_version=1.94.0"
  if [ -r /models/yolox_nano.onnx ]; then
    echo "model_sha256=$(sha256sum /models/yolox_nano.onnx | awk '{print $1}')"
  else
    echo "model_sha256=not-used"
  fi
  echo "queue_size=4"
  echo "queue_policy=backpressure"
  echo "confidence=0.3"
  echo "nms=0.45"
} > "$result_dir/manifest.txt"
