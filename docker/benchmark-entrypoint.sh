#!/bin/bash
set -euo pipefail

scenario="${SCENARIO:-realtime}"
run_id="${RUN_ID:-run-001}"
segment="${SEGMENT:-10023947602400723454_1120_000_1140_000}"
split="${SPLIT:-training}"

case "$scenario" in
  realtime|burst-drop|burst-backpressure|overloaded) ;;
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
for input in "$camera_file" "$lidar_file"; do
  if [ ! -r "$input" ]; then
    echo "missing benchmark input: $input" >&2
    exit 2
  fi
done

# Hashing verifies the exact input and intentionally warms both files before
# every timed run, avoiding a cold-cache/warm-cache difference between repeats.
sha256sum "$camera_file" "$lidar_file" > "$result_dir/input.sha256"

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
pipes-rs \
  --data-root /data/waymo-v2-sample \
  --segment "$segment" \
  --split "$split" \
  --camera front \
  --lidar top \
  "$@" \
  --csv "$result_dir/measurements.csv" \
  2>&1 | tee "$result_dir/output.log"
end_ns="$(date +%s%N)"
elapsed_ns=$((end_ns - start_ns))
echo "elapsed_ns=$elapsed_ns" | tee "$result_dir/timing.txt"
