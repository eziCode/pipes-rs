#!/bin/sh
set -eu

suite="${1:-baseline-$(date +%Y%m%d-%H%M%S)}"
repeats="${2:-5}"
case "$suite" in *[!A-Za-z0-9._-]*|'') echo "invalid suite ID" >&2; exit 2;; esac
case "$repeats" in *[!0-9]*|'') echo "repeat count must be positive" >&2; exit 2;; esac
[ "$repeats" -gt 0 ] || exit 2

run_pair() {
  workload="$1"; number="$2"; prefix="$suite-$workload-$(printf '%02d' "$number")"
  case "$workload" in
    full) pipes_scenario=perception-onnx; dora_workload=full; fusion_ms=0; pipes_group=perception-onnx; dora_group=dora ;;
    camera) pipes_scenario=perception-camera; dora_workload=camera; fusion_ms=0; pipes_group=perception-camera; dora_group=dora-camera ;;
    lidar) pipes_scenario=perception-lidar; dora_workload=lidar; fusion_ms=0; pipes_group=perception-lidar; dora_group=dora-lidar ;;
    overloaded) pipes_scenario=perception-overloaded; dora_workload=full; fusion_ms=125; pipes_group=perception-overloaded; dora_group=dora ;;
    transport) pipes_scenario=arrow-transport; dora_workload=transport; fusion_ms=0; pipes_group=arrow-transport; dora_group=dora-transport ;;
  esac
  run_pipes() {
    result="benchmark-results/$pipes_group/$prefix-pipes"
    if [ -f "$result/timing.txt" ]; then echo "already complete: $result"; else ./scripts/run_container_benchmark.sh "$pipes_scenario" "$prefix-pipes"; fi
  }
  run_dora() {
    result="benchmark-results/$dora_group/$prefix-dora"
    if [ -f "$result/timing.txt" ]; then echo "already complete: $result"; else FUSION_WORK_MS="$fusion_ms" ./scripts/run_dora_container_benchmark.sh "$prefix-dora" "$dora_workload"; fi
  }
  if [ $((number % 2)) -eq 1 ]; then run_pipes; run_dora; else run_dora; run_pipes; fi
}

for workload in full camera lidar overloaded transport; do
  i=1
  while [ "$i" -le "$repeats" ]; do run_pair "$workload" "$i"; i=$((i+1)); done
  case "$workload" in
    full) pipes_group=perception-onnx; dora_group=dora ;;
    camera) pipes_group=perception-camera; dora_group=dora-camera ;;
    lidar) pipes_group=perception-lidar; dora_group=dora-lidar ;;
    overloaded) pipes_group=perception-overloaded; dora_group=dora ;;
    transport) pipes_group=arrow-transport; dora_group=dora-transport ;;
  esac
  python3 scripts/build_baseline_dashboard.py --suite "$suite-$workload" \
    --pipes-group "$pipes_group" --dora-group "$dora_group"
done

python3 scripts/build_baseline_index.py --suite "$suite"
echo "open benchmark-results/baseline/$suite/index.html"
