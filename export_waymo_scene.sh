#!/bin/bash
#SBATCH --job-name=WaymoPlayer
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem-per-gpu=24G
#SBATCH --qos=embers
#SBATCH --time=02:00:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/WaymoPlayer-%j.out
set -euo pipefail
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
scratch_dir="$HOME/scratch/pipes-rs"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit with sbatch export_waymo_scene.sh'; exit 2; }
cd "$project_dir"
job_root="$scratch_dir/scene-exports/job-$SLURM_JOB_ID"
mkdir -p "$scratch_dir/scene-exports"
mkdir "$job_root"
mkdir "$job_root/source"
tar --exclude=target --exclude=__pycache__ -cf - Cargo.toml Cargo.lock src assets scripts export_waymo_scene.sh docs/scene-player.md | tar -xf - -C "$job_root/source"
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
cargo build --locked --release --bin export-scene
cp "$CARGO_TARGET_DIR/release/export-scene" "$job_root/export-scene"
flock -u 9
{
    rustc --version
    uname -a
    lscpu
    echo "entire scene; offline YOLOX inference"
    echo 'CPU-only kernels on GPU allocation; no inference or Dora'
    env | sort | grep '^SLURM_' || true
} > "$job_root/environment.txt"
sha256sum "$job_root/export-scene" > "$job_root/binary.sha256"
model="$scratch_dir/models/yolox_nano.onnx"
[[ -f "$model" ]] || model="$scratch_dir/yolox_nano.onnx"
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$job_root/export-scene" \
    --data-root "$scratch_dir/data/waymo-v2-sample" --model "$model" \
    --output "$job_root/waymo-scene" "$@"
python3 - "$job_root" <<'PYZIP'
import json,sys,zipfile
from pathlib import Path
root=Path(sys.argv[1]);scene=root/'waymo-scene'
manifest=json.loads((scene/'manifest.json').read_text())
assert len(manifest['frames'])>1
for i,f in enumerate(manifest['frames']):
    path=scene/'frames'/f'{i:04}.js'
    text=path.read_text()
    data=json.loads(text[len('window.receiveSceneFrame('):-3])
    assert data['index']==i and data['timestamp_micros']==f['timestamp_micros']
    assert data['jpeg'] and data['total_points']>0
with zipfile.ZipFile(root/'waymo-scene.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(scene.rglob('*')):
        if path.is_file():z.write(path,path.relative_to(root))
print('Download:',root/'waymo-scene.zip')
print('Frames:',len(manifest['frames']))
PYZIP
