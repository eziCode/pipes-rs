#!/bin/bash
#SBATCH --job-name=PipesPythonArrow
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=24G
#SBATCH --qos=embers
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/PipesPythonArrow-%j.out
# CPU-only representation experiment. GPU allocation matches the existing
# Embers job configuration; this job does not claim GPU transport measurements.
set -euo pipefail
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
scratch_dir="$HOME/scratch/pipes-rs"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch benchmark_python_boundary.sh'; exit 2; }
cd "$project_dir"
job_root="$scratch_dir/python-boundary-results/job-$SLURM_JOB_ID"
mkdir -p "$scratch_dir/python-boundary-results" "$scratch_dir/tooling"
mkdir "$job_root"
mkdir "$job_root/source"
tar --exclude=target --exclude=__pycache__ -cf - Cargo.toml Cargo.lock src assets scripts experiments/python-boundary benchmark_python_boundary.sh | tar -xf - -C "$job_root/source"
git rev-parse HEAD > "$job_root/git-head.txt"
git diff > "$job_root/source.diff"
cd "$job_root/source"
export CARGO_HOME="$scratch_dir/tooling/cargo" RUSTUP_HOME="$scratch_dir/tooling/rustup"
export CARGO_TARGET_DIR="$scratch_dir/tooling/target" RUSTUP_TOOLCHAIN=1.95.0
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-4}"
export PATH="$CARGO_HOME/bin:$PATH"
export ORT_LIB_LOCATION="$scratch_dir/tooling/onnxruntime-1.27.0"
export ORT_PREFER_DYNAMIC_LINK=1 ORT_SKIP_DOWNLOAD=1
export LD_LIBRARY_PATH="$ORT_LIB_LOCATION${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec 9>"$scratch_dir/tooling/build.lock"
flock 9
python3 scripts/prepare_pace_onnxruntime.py "$ORT_LIB_LOCATION"
cargo test --locked --release --lib
cargo build --locked --release --lib
cp "$CARGO_TARGET_DIR/release/libpipes_rs.so" "$job_root/libpipes_rs.so"
flock -u 9
# A per-job environment records exact resolved packages and avoids concurrent
# package installation into a shared environment.
python3 -m venv "$job_root/venv"
"$job_root/venv/bin/python" -m pip install -r experiments/python-boundary/requirements.txt
"$job_root/venv/bin/python" -m pip freeze > "$job_root/python-packages.txt"
{
    rustc --version
    uname -a
    lscpu
    env | sort | grep '^SLURM_' || true
} > "$job_root/environment.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores "$job_root/venv/bin/python" scripts/run_python_boundary.py \
    --library "$job_root/libpipes_rs.so" --data-root "$scratch_dir/data/waymo-v2-sample" \
    --model "$scratch_dir/models/yolox_nano.onnx" --output "$job_root/results" "$@"
