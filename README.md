# pipes-rs

Pipes is a Rust project for running sensor processing and fusion
in front of ROS 2.

## The architectural bet

Pipes owns the path from sensor drivers through processing and fusion. Sensor
drivers write directly into Pipes-owned Apache Arrow buffers. Pipeline stages
share those buffers without repeatedly copying or serializing the sensor data.
After fusion, Pipes converts the much smaller results into standard ROS 2
messages.

```text
camera / lidar / radar / IMU
              |
              v
     Rust sensor drivers
              |
              v
 Pipes metadata + Arrow buffers
              |
              v
 processing, synchronization, and fusion
              |
              v
 detections / tracks / state / health
              |
              v
       standard ROS 2 topics
```

Significant performance gains are expected because the largest data stays inside
one controlled pipeline:

- a sensor can write into its pipeline buffer once;
- processing, recording, visualization, and fusion can share that data;
- large images and point clouds do not need to be serialized and reconstructed
  between every stage; and
- ROS conversion happens once, after the data has been reduced to the result the
  rest of the robot needs.

Pipes should interoperate cleanly with ROS rather than replace it. ROS can keep
handling control, planning, mission logic, user interfaces, and communication
across the robot. Existing ROS nodes consume normal topics and do not need to
know that Pipes is running the sensor pipeline.

The same pipeline can run from live sensors, simulation, or recorded MCAP data.
Pipes records sensor timing, input order, synchronization decisions, delays, and
drops. This lets developers isolate the sensor and fusion subsystem, replay the
same workload, tune it quickly, and determine whether a bad result came from the
fusion algorithm or from data arriving late or going missing.

## Fusion in Motion

[Fusion in Motion](https://github.com/EthanMBoos/fusion-in-motion) is a separate
simulation playground. It can generate repeatable sensor workloads for Pipes,
but neither project depends on the other.

## Download a small Waymo sample

The sample downloader fetches the camera and LiDAR components for one or two
20-second Waymo Open Dataset v2 segments. Accept the
[Waymo Open Dataset terms](https://waymo.com/open/terms/), authenticate with
`gcloud auth login`, and use the same Google account for both steps.

Preview the download without transferring data:

```bash
python3 scripts/download_waymo_sample.py --dry-run
```

Download one segment, including camera and LiDAR boxes for evaluation:

```bash
python3 scripts/download_waymo_sample.py \
  --with-labels
```

Pass `--count 2` to download both sample segments. Data is written under
`data/waymo-v2-sample/`, which is ignored by Git. If your Cloud Storage setup
requires a quota project, pass `--billing-project YOUR_GCP_PROJECT`. If
`gcloud storage` reports a broken local CRC32C checksum, retry with
`--transfer-tool gsutil`.

## Run the Arrow pipeline MVP

The MVP replays the front camera and top LiDAR from one downloaded segment into
separate bounded queues. Each measurement remains a one-row Arrow record batch.
Consumers encode the batch as Arrow IPC, checksum it, and report queue wait,
measurement age, delivery rate, drops, sequence gaps, bytes, and queue depth.

```bash
cargo run --release -- \
  --segment 10023947602400723454_1120_000_1140_000 \
  --camera front \
  --lidar top \
  --queue-size 8 \
  --queue-policy drop-oldest \
  --speed 1.0 \
  --csv pipeline-metrics.csv
```

Set `--speed 0` to run without real-time pacing. Use `--camera-work-ms` or
`--lidar-work-ms` to simulate slow processing and exercise queue drops, or use
`--queue-policy backpressure` to make the producer wait instead. The optional
CSV contains one row per delivered or dropped measurement.

## Build the camera + LiDAR fusion demo

Milestone 1 converts one top-LiDAR range image into Cartesian `x/y/z` Arrow
columns, associates projected LiDAR ranges with front-camera boxes, and writes
a standalone inspection page alongside the Arrow IPC point cloud:

```bash
cargo run --release -- \
  --segment 10023947602400723454_1120_000_1140_000 \
  --frame-index 0 \
  --demo-output demo-output/frame-000.html
```

Open `demo-output/frame-000.html` in a browser. Hover over a camera box to see
its median LiDAR depth and highlight the associated points in the bird's-eye
view. The boxes are Waymo ground truth in this milestone; this gives the future
camera model a deterministic reference. The Arrow file contains the processed
point cloud in the vehicle coordinate frame, including range, intensity,
elongation, and front-camera projection columns.

The first run may take a little while while Cargo builds the release binary.
Subsequent frames can be selected with `--frame-index`. Per-pixel LiDAR motion
compensation and model-produced camera detections are intentionally deferred to
the next milestone.

## Run the staged perception workload

The multi-frame workload runs camera JPEG decoding and top-LiDAR conversion on
separate worker threads connected to bounded channels. A synchronization stage
matches timestamps, fuses projected LiDAR depth into detections, and records one
CSV row per frame:

```bash
cargo run --release -- \
  --segment 10023947602400723454_1120_000_1140_000 \
  --stage-queue-size 4 \
  --perception-csv demo-output/perception.csv
```

Use `--frame-limit 10` for a short smoke test; the default processes the entire
segment. Metrics include camera decode time, LiDAR-to-Arrow conversion time,
both queue waits, timestamp skew, fusion time, end-to-end time, point count,
detection count, and depth-match count. Camera pixels are genuinely JPEG
decoded and checksummed. Without `--onnx-model`, detections come from Waymo
ground truth to provide a deterministic performance and correctness baseline.

Download the official 416×416 YOLOX-Nano ONNX model and run real inference:

```bash
./scripts/download_yolox_model.sh

cargo run --release -- \
  --segment 10023947602400723454_1120_000_1140_000 \
  --onnx-model models/yolox_nano.onnx \
  --confidence-threshold 0.3 \
  --nms-threshold 0.45 \
  --perception-csv demo-output/perception-onnx.csv
```

YOLOX predictions are emitted as Arrow detection batches, fused with LiDAR,
and evaluated against Waymo boxes at IoU 0.5. The CSV includes inference time,
true positives, false positives, false negatives, and the existing pipeline
timings. COCO `person`, `bicycle`/`motorcycle`, `car`/`bus`/`truck`, and
`stop sign` predictions map to the corresponding Waymo categories.

Run the same full workload inside the constrained benchmark container with:

```bash
./scripts/run_container_benchmark.sh perception perception-01
./scripts/run_container_benchmark.sh perception-onnx perception-onnx-01
```

Its frame-level output is written to
`benchmark-results/<scenario>/<run-id>/perception.csv`.

## Reproducible container benchmarks

The benchmark container bakes in the release binary and runs without network
access. Waymo inputs are mounted read-only, results are isolated by scenario and
run ID, and Compose limits each run to 2 CPUs, 4 GB of memory, and 128 processes.
The Rust and Debian base images are pinned by digest and Cargo uses the committed
lockfile.

Docker Desktop must be running. Execute the real-time baseline with:

```bash
./scripts/run_container_benchmark.sh realtime run-001
```

Available scenarios use fixed parameters:

| Scenario | Replay and queue behavior |
| --- | --- |
| `realtime` | 1× replay, queue 8, drop-oldest |
| `burst-drop` | Unpaced replay, queue 8, drop-oldest |
| `burst-backpressure` | Unpaced replay, queue 8, producer backpressure |
| `overloaded` | 1× replay with 125 ms of LiDAR work per sample |

Use a new run ID for every repeat; existing results are never overwritten:

```bash
./scripts/run_container_benchmark.sh realtime run-002
```

Each run writes `measurements.csv`, `output.log`, `timing.txt`,
`environment.txt`, `input.sha256`, `image.json`, and the resolved `compose.yaml`
beneath
`benchmark-results/<scenario>/<run-id>/`. Input hashing happens before timing
and warms the same camera and LiDAR files for every run.

The default platform is `linux/arm64`, matching Apple Silicon. A different host
can set `BENCHMARK_PLATFORM`, but performance results should only be compared
when the platform, Docker resource allocation, and host machine are identical.
