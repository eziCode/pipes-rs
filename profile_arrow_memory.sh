#!/bin/bash
#SBATCH --job-name=ArrowMemory
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=24G
#SBATCH --qos=embers
#SBATCH --time=02:00:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/ArrowMemory-%j.out
set -euo pipefail
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
scratch_dir="$HOME/scratch/pipes-rs"
frames="${ARROW_FRAMES:-20}"
repeats="${ARROW_REPEATS:-3}"
warmup="${ARROW_WARMUP:-2}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch profile_arrow_memory.sh'; exit 2; }
cd "$project_dir"
job_root="$scratch_dir/arrow-memory-results/job-$SLURM_JOB_ID"
mkdir -p "$scratch_dir/arrow-memory-results"
mkdir "$job_root"
mkdir "$job_root/source"
tar --exclude=target --exclude=__pycache__ -cf - Cargo.toml Cargo.lock src assets scripts profile_arrow_memory.sh docs/arrow-memory-profile.md | tar -xf - -C "$job_root/source"
test -s "$job_root/source/assets/fusion-demo.html"
git rev-parse HEAD > "$job_root/git-head.txt"
git diff > "$job_root/source.diff"
cd "$job_root/source"
export CARGO_HOME="$scratch_dir/tooling/cargo"
export RUSTUP_HOME="$scratch_dir/tooling/rustup"
export CARGO_TARGET_DIR="$scratch_dir/tooling/target"
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export RUSTUP_TOOLCHAIN=1.95.0
export PATH="$CARGO_HOME/bin:$PATH"
export ORT_LIB_LOCATION="$scratch_dir/tooling/onnxruntime-1.27.0"
export ORT_PREFER_DYNAMIC_LINK=1 ORT_SKIP_DOWNLOAD=1
export LD_LIBRARY_PATH="$ORT_LIB_LOCATION${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$scratch_dir/tooling/build.lock"
flock 9
python3 scripts/prepare_pace_onnxruntime.py "$ORT_LIB_LOCATION"
cargo test --locked --release --lib
cargo test --locked --release --bin arrow-memory-profile
cargo build --locked --release --bin arrow-memory-profile
cp "$CARGO_TARGET_DIR/release/arrow-memory-profile" "$job_root/arrow-memory-profile"
flock -u 9
{
    rustc --version
    uname -a
    lscpu
    echo "frames=$frames repeats=$repeats warmup=$warmup"
    echo 'CPU-only kernels on GPU allocation; no inference or Dora'
    env | sort | grep '^SLURM_' || true
} > "$job_root/environment.txt"
sha256sum "$job_root/arrow-memory-profile" > "$job_root/binary.sha256"
if ! cat /proc/sys/kernel/perf_event_paranoid > "$job_root/perf_event_paranoid.txt" 2> "$job_root/perf-policy-read-error.txt"; then
    echo 'Policy diagnostic unavailable; the profiler will probe actual counter access.' >&2
fi
if ! ls /sys/bus/event_source/devices > "$job_root/pmu-devices.txt" 2> "$job_root/pmu-list-error.txt"; then
    echo 'PMU listing unavailable; continuing to counter probes.' >&2
fi
if command -v objdump >/dev/null; then
    objdump -d -C "$job_root/arrow-memory-profile" > "$job_root/disassembly.txt" 2> "$job_root/disassembly-error.txt" || echo 'Optional disassembly failed.' >&2
fi
profile_failed=0
for item in training:10023947602400723454_1120_000_1140_000 validation:1024360143612057520_3580_000_3600_000; do
    split="${item%%:*}"
    segment="${item#*:}"
    # One bound CPU for all variants; four CPUs were allocated for compilation.
    if ! srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$job_root/arrow-memory-profile" \
        --data-root "$scratch_dir/data/waymo-v2-sample" --split "$split" --segment "$segment" \
        --frames "$frames" --repeats "$repeats" --warmup "$warmup" --output "$job_root/$split"; then
        profile_failed=1
    fi
done
python3 scripts/summarize_arrow_memory.py "$job_root"
exit "$profile_failed"
