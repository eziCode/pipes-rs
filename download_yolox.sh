#!/bin/bash
#SBATCH --job-name=DownloadYOLOX
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=12G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/DownloadYOLOX-%j.out

# Usage: sbatch download_yolox.sh [output-model-path]
# Also works directly: bash download_yolox.sh
set -euo pipefail

# Slurm executes a spooled copy, so do not derive the root from $0.
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
cd "$project_dir"
if (( $# > 1 )); then
    echo "Usage: sbatch download_yolox.sh [output-model-path]" >&2
    exit 2
fi
for tool in curl; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "$tool is required to download and verify YOLOX." >&2
        exit 1
    }
done
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    echo "sha256sum or shasum is required to verify YOLOX." >&2
    exit 1
fi

echo "Downloading and verifying YOLOX-Nano."
bash scripts/download_yolox_model.sh "${1:-$HOME/scratch/pipes-rs/models/yolox_nano.onnx}"
echo "YOLOX model is ready."
