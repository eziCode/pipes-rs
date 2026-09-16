#!/bin/sh
set -eu

suite="${1:-baseline-$(date +%Y%m%d-%H%M%S)}"
repeats="${2:-5}"
case "$suite" in *[!A-Za-z0-9._-]*|'') echo "invalid suite ID" >&2; exit 2;; esac
case "$repeats" in *[!0-9]*|'') echo "repeat count must be a positive integer" >&2; exit 2;; esac
[ "$repeats" -gt 0 ] || { echo "repeat count must be positive" >&2; exit 2; }

run_one() {
  framework="$1"
  run_id="$2"
  if [ "$framework" = pipes ]; then
    ./scripts/run_container_benchmark.sh perception-onnx "$run_id"
  else
    ./scripts/run_dora_container_benchmark.sh "$run_id"
  fi
}

i=1
while [ "$i" -le "$repeats" ]; do
  ordinal="$(printf '%02d' "$i")"
  if [ $((i % 2)) -eq 1 ]; then
    run_one pipes "$suite-$ordinal-pipes"
    run_one dora "$suite-$ordinal-dora"
  else
    run_one dora "$suite-$ordinal-dora"
    run_one pipes "$suite-$ordinal-pipes"
  fi
  i=$((i + 1))
done

python3 scripts/build_baseline_dashboard.py --suite "$suite"
echo "open benchmark-results/baseline/$suite/dashboard.html"
