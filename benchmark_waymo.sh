#!/bin/bash
#SBATCH --job-name=PipesDoraWaymo
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=24G
#SBATCH --qos=embers
#SBATCH --time=08:00:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/PipesDoraWaymo-%j.out

# Usage: sbatch benchmark_waymo.sh [--repeats 5] [--frame-limit 0]
# Preview inputs without building/submitting: bash benchmark_waymo.sh --dry-run
set -euo pipefail
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
scratch_dir="$HOME/scratch/pipes-rs"
cd "$project_dir"
for arg in "$@"; do
    if [[ "$arg" == --dry-run || "$arg" == --help ]]; then
        exec python3 scripts/run_pace_benchmark.py "$@"
    fi
done
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch to build and benchmark on a compute node.' >&2; exit 2; }
# Dora CLI's vendored OpenSSL needs core Perl modules missing from /usr/bin/perl.
module load perl/5.36.0
perl -MFindBin -MIPC::Cmd -e 'print "OpenSSL core Perl prerequisites OK ($^V)\n"'
python3 scripts/run_pace_benchmark.py --dry-run "$@"

# Preserve the exact source used by this job, including uncommitted changes.
job_root="$scratch_dir/benchmark-results/job-$SLURM_JOB_ID"
mkdir -p "$scratch_dir/benchmark-results"
mkdir "$job_root"
mkdir "$job_root/source"
tar --exclude=target --exclude=__pycache__ -cf - Cargo.toml Cargo.lock src assets scripts benchmarks | tar -xf - -C "$job_root/source"
git rev-parse HEAD > "$job_root/git-head.txt"
git diff > "$job_root/source.diff"
cd "$job_root/source"

# Keep compiler, downloaded crates, and build products off home storage.
export CARGO_HOME="$scratch_dir/tooling/cargo"
export RUSTUP_HOME="$scratch_dir/tooling/rustup"
export CARGO_TARGET_DIR="$scratch_dir/tooling/target"
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export PATH="$CARGO_HOME/bin:$PATH"
export RUSTUP_TOOLCHAIN=1.95.0
mkdir -p "$scratch_dir/tooling"
# Serialize shared build-cache/toolchain changes across submitted jobs.
exec 9>"$scratch_dir/tooling/build.lock"
flock 9
if [[ ! -x "$CARGO_HOME/bin/rustup" ]]; then
    curl --fail --location --retry 3 https://sh.rustup.rs -o "$job_root/rustup-init.sh"
    bash "$job_root/rustup-init.sh" -y --no-modify-path --profile minimal --default-toolchain "$RUSTUP_TOOLCHAIN"
fi
rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal
# The default ort static archive requires a newer glibc than PACE provides.
# Use the same pinned, host-compatible CPU shared library for both frameworks.
export ORT_LIB_LOCATION="$scratch_dir/tooling/onnxruntime-1.27.0"
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_SKIP_DOWNLOAD=1
python3 scripts/prepare_pace_onnxruntime.py "$ORT_LIB_LOCATION"
export LD_LIBRARY_PATH="$ORT_LIB_LOCATION${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cp "$ORT_LIB_LOCATION/runtime.sha256" "$job_root/onnxruntime.sha256"
cargo build --locked --release
cargo test --locked --release --lib
cargo build --locked --release --manifest-path benchmarks/dora/Cargo.toml
python3 -m unittest discover -s scripts -p 'test_pace_benchmark.py'
cargo install dora-cli --version 1.0.1 --locked
mkdir "$job_root/bin"
cp "$CARGO_TARGET_DIR/release/pipes-rs" "$CARGO_TARGET_DIR/release/pipes-dora-benchmark" "$CARGO_HOME/bin/dora" "$job_root/bin/"
flock -u 9
export PATH="$job_root/bin:$PATH"
{
    rustc --version
    cargo --version
    dora --version
    echo "onnxruntime=1.27.0"
    echo "onnxruntime_library=$ORT_LIB_LOCATION"
    uname -a
    lscpu
    env | sort | grep '^SLURM_' || true
} > "$job_root/environment.txt"
sha256sum "$job_root/bin/"* > "$job_root/binaries.sha256"
srun --cpu-bind=cores python3 scripts/run_pace_benchmark.py --results-root "$job_root/runs" "$@"
