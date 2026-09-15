#!/usr/bin/env python3
"""Download one or two small Waymo Open Dataset v2 perception segments.

The selected segments are examples used by the official Waymo v2 tutorial and
validation tooling. Each segment contains about 20 seconds of camera and LiDAR
data. Waymo stores every component as a separate Parquet object, so this script
downloads only the sensor payloads and metadata needed to decode them.

Before downloading:
  1. Accept the Waymo Open Dataset terms at https://waymo.com/open/terms/.
  2. Install the Google Cloud CLI and run `gcloud auth login`.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


BUCKET = "waymo_open_dataset_v_2_0_1"


@dataclass(frozen=True)
class Segment:
    context_name: str
    split: str
    description: str


SEGMENTS = (
    Segment(
        context_name="10023947602400723454_1120_000_1140_000",
        split="training",
        description="official Waymo v2 tutorial segment with camera and box examples",
    ),
    Segment(
        context_name="1024360143612057520_3580_000_3600_000",
        split="validation",
        description="official validation example used by Waymo camera tooling",
    ),
)

# Raw payloads plus the minimum metadata needed to decode, calibrate, and
# motion-compensate camera and LiDAR observations.
CORE_COMPONENTS = (
    "camera_image",
    "camera_calibration",
    "lidar",
    "lidar_camera_projection",
    "lidar_calibration",
    "lidar_pose",
    "vehicle_pose",
    "stats",
)

LABEL_COMPONENTS = (
    "camera_box",
    "lidar_box",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of 20-second segments to download (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/waymo-v2-sample"),
        help="download directory (default: data/waymo-v2-sample)",
    )
    parser.add_argument(
        "--billing-project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud billing project; defaults to GOOGLE_CLOUD_PROJECT",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="also download 2D camera and 3D LiDAR boxes for evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print files and commands without downloading",
    )
    parser.add_argument(
        "--transfer-tool",
        choices=("gcloud", "gsutil"),
        default="gcloud",
        help="Cloud Storage transfer command to use (default: gcloud)",
    )
    return parser.parse_args()


def source_uri(segment: Segment, component: str) -> str:
    return (
        f"gs://{BUCKET}/{segment.split}/{component}/"
        f"{segment.context_name}.parquet"
    )


def command_for(
    uri: str,
    destination: Path,
    billing_project: str | None,
    transfer_tool: str,
) -> list[str]:
    if transfer_tool == "gsutil":
        command = ["gsutil"]
        if billing_project:
            command.extend(("-u", billing_project))
        command.extend(("-m", "cp", "-n", uri, str(destination)))
        return command

    command = [
        "gcloud",
        "storage",
        "cp",
        "--no-clobber",
        uri,
        str(destination),
    ]
    if billing_project:
        command.insert(4, f"--billing-project={billing_project}")
    return command


def require_prerequisites(transfer_tool: str) -> None:
    if shutil.which(transfer_tool) is None:
        raise RuntimeError(
            f"{transfer_tool} was not found. Install the Google Cloud CLI, "
            "then run `gcloud auth login`."
        )


def main() -> int:
    args = parse_args()
    selected = SEGMENTS[: args.count]
    components = CORE_COMPONENTS + (LABEL_COMPONENTS if args.with_labels else ())

    if args.dry_run:
        billing_project = args.billing_project
    else:
        try:
            require_prerequisites(args.transfer_tool)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        billing_project = args.billing_project

    manifest = {
        "dataset": "Waymo Open Dataset Perception v2.0.1",
        "bucket": BUCKET,
        "components": list(components),
        "segments": [asdict(segment) for segment in selected],
    }

    print(f"Selected {len(selected)} segment(s), {len(components)} components each:")
    for segment in selected:
        print(f"  {segment.context_name} ({segment.split})")
        print(f"    {segment.description}")

    for segment in selected:
        for component in components:
            # Preserve Waymo's split/component/context layout so its v2 tools
            # can read this directory without a custom reshuffle step.
            destination_dir = args.output / segment.split / component
            destination = destination_dir / f"{segment.context_name}.parquet"
            uri = source_uri(segment, component)
            command = command_for(
                uri, destination, billing_project, args.transfer_tool
            )

            if args.dry_run:
                print("DRY RUN:", shlex.join(command))
                continue

            destination_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {segment.context_name}/{component}...")
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as error:
                print(
                    f"error: download failed for {uri} (exit {error.returncode})\n"
                    "Confirm that you accepted the Waymo terms, authenticated "
                    "gcloud, and have access with the same Google account.",
                    file=sys.stderr,
                )
                return error.returncode or 1

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Download complete. Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
