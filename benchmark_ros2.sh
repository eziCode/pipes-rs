#!/bin/bash
#SBATCH --job-name=PipesROS2
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=24G
#SBATCH --qos=embers
#SBATCH --time=04:00:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/PipesROS2-%j.out
set -euo pipefail
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
scratch_dir="$HOME/scratch/pipes-rs"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch benchmark_ros2.sh'; exit 2; }
cd "$project_dir"
job_root="$scratch_dir/ros2-results/job-$SLURM_JOB_ID"
mkdir -p "$scratch_dir/ros2-results" "$scratch_dir/tooling"
mkdir "$job_root"
mkdir "$job_root/source"
tar --exclude=target --exclude=__pycache__ -cf - Cargo.toml Cargo.lock src assets scripts benchmarks/ros2 benchmark_ros2.sh docs/ros2-execution-experiment.md | tar -xf - -C "$job_root/source"
test -s "$job_root/source/assets/fusion-demo.html"
git rev-parse HEAD > "$job_root/git-head.txt"
git diff > "$job_root/source.diff"
cd "$job_root/source"
# Use the already bootstrapped compiler/runtime, shared with the Arrow pilot.
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
mkdir "$job_root/lib"
cp "$CARGO_TARGET_DIR/release/libpipes_rs.so" "$job_root/lib/"
flock -u 9
# An isolated RoboStack installation; never alter the user's active conda env.
export MAMBA_ROOT_PREFIX="$scratch_dir/tooling/ros2-mamba"
export MAMBA_NO_BANNER=1
export TMPDIR="$job_root/tmp"
mkdir -p "$TMPDIR"
exec 8>"$scratch_dir/tooling/ros2-install.lock"
flock 8
mamba="$scratch_dir/tooling/micromamba/bin/micromamba"
if [[ ! -x "$mamba" ]]; then
    mkdir -p "$scratch_dir/tooling/micromamba"
    curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$job_root/micromamba.tar.bz2"
    sha256sum "$job_root/micromamba.tar.bz2" > "$job_root/micromamba-download.sha256"
    tar -xjf "$job_root/micromamba.tar.bz2" -C "$scratch_dir/tooling/micromamba" bin/micromamba
fi
ros_prefix="$MAMBA_ROOT_PREFIX/envs/pipes-ros2-jazzy"
if [[ ! -f "$ros_prefix/.pipes-ready" ]]; then
    "$mamba" create --yes --prefix "$ros_prefix" --file benchmarks/ros2/environment.yml --channel-priority strict
    touch "$ros_prefix/.pipes-ready"
fi
"$mamba" list --prefix "$ros_prefix" --explicit > "$job_root/ros2-packages.txt"
"$mamba" --version > "$job_root/micromamba-version.txt"
sha256sum "$mamba" > "$job_root/micromamba-binary.sha256"
flock -u 8
# Remove inherited Python overrides before activating the isolated ROS environment.
unset PYTHONPATH PYTHONHOME
# PACE's GCC_ROOT redirects the conda compiler to the cluster GCC's libexec
# and sysroot. Keep this cleanup local to the batch job, after the Rust build.
unset GCC_ROOT GCCROOT GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH
# Activation enables conda compiler and ament hooks without touching shell config.
eval "$("$mamba" shell hook --shell bash)"
set +u
micromamba activate "$ros_prefix"
set -u
# Do not inherit the cluster module's older C++ compiler for RoboStack packages.
export CC="$ros_prefix/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ros_prefix/bin/x86_64-conda-linux-gnu-c++"
test -x "$CC" && test -x "$CXX"
printf 'int main(){return 0;}\n' > "$job_root/compiler-check.cpp"
"$CXX" "$job_root/compiler-check.cpp" -o "$job_root/compiler-check"
"$job_root/compiler-check"
export LD_LIBRARY_PATH="$job_root/lib:$ORT_LIB_LOCATION:$ros_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S benchmarks/ros2 -B "$job_root/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPIPES_LIBRARY="$job_root/lib/libpipes_rs.so" -DCMAKE_PREFIX_PATH="$ros_prefix"
cmake --build "$job_root/build" --parallel "${SLURM_CPUS_PER_TASK:-4}"
{
    rustc --version
    "$CXX" --version
    lscpu
    env | sort | grep '^SLURM_' || true
} > "$job_root/environment.txt"
sha256sum "$job_root/build/execution-pilot" "$job_root/lib/libpipes_rs.so" > "$job_root/binaries.sha256"
python3 -m unittest discover -s scripts -p 'test_execution_pilot.py'
srun --ntasks=1 --cpus-per-task=2 --cpu-bind=cores python3 scripts/run_execution_pilot.py \
    --binary "$job_root/build/execution-pilot" --data-root "$scratch_dir/data/waymo-v2-sample" \
    --output "$job_root/runs" "${@}"
