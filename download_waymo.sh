#!/bin/bash
#SBATCH --job-name=DownloadWaymo
#SBATCH --account=gts-agarg35
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --qos=embers
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=12G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/home/hcoda1/7/eakresh3/slurm_out/DownloadWaymo-%j.out

# Usage: sbatch download_waymo.sh [downloader options]
# Preview: bash download_waymo.sh --dry-run
# Authenticate with gcloud auth login before submitting this job, using the
# Google account that accepted the Waymo dataset terms.
set -euo pipefail

# Slurm executes a spooled copy, so do not derive the root from $0.
project_dir="${PIPES_ROOT:-/storage/home/hcoda1/7/eakresh3/Documents/pipes-rs}"
cd "$project_dir"
export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"

command -v python3 >/dev/null 2>&1 || {
    echo "python3 is required to run the Waymo downloader." >&2
    exit 1
}

echo "Downloading Waymo camera/LiDAR data and labels (two sample segments by default)."
# Later command-line options override these defaults, e.g. --count 1 or
# --output /path/to/data. Keep the large dataset on scratch storage.
python3 scripts/download_waymo_sample.py \
    --count 2 --with-labels \
    --output "$HOME/scratch/pipes-rs/data/waymo-v2-sample" "$@"
echo "Waymo download command completed."
