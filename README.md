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

On PACE, the project-root Bash scripts request one RTX 6000 GPU with the
`embers` QoS and account `gts-agarg35`, with logs in
`/storage/home/hcoda1/7/eakresh3/slurm_out`:

```bash
cd /storage/home/hcoda1/7/eakresh3/Documents/pipes-rs
mkdir -p /storage/home/hcoda1/7/eakresh3/slurm_out
bash download_waymo.sh --dry-run
sbatch download_waymo.sh
sbatch download_yolox.sh
```

Before submitting the Waymo job, install the Google Cloud CLI and authenticate
with `gcloud auth login` using the account that accepted the Waymo terms. The
wrapper also searches `~/google-cloud-sdk/bin` and `~/.local/bin` for the CLI.
The jobs require outbound network access from their assigned compute nodes.
Neither download requires Rust. The GPU request places these jobs on GPU
nodes; the downloads themselves use CPU and network resources.

`download_waymo.sh` defaults to both sample segments with labels in
`~/scratch/pipes-rs/data/waymo-v2-sample/`; pass `--count 1` for just the first segment, or any
other Python downloader options. `download_yolox.sh` saves the checksum-verified
model to `~/scratch/pipes-rs/models/yolox_nano.onnx`. Both wrappers can also run with `bash`.
They use an explicit project path because Slurm spools submitted scripts;
set `PIPES_ROOT` if the checkout moves. Re-running skips existing Waymo objects
and verifies an existing YOLOX model before reusing it.

For native perception runs with this scratch download, pass
`--data-root "$HOME/scratch/pipes-rs/data/waymo-v2-sample"` and, for inference,
`--onnx-model "$HOME/scratch/pipes-rs/models/yolox_nano.onnx"`.
The container benchmark scripts still mount `data/waymo-v2-sample` and `models`
inside the checkout; point those paths at the scratch directories before
using those scripts.

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

## Equivalent Dora benchmark

The Dora comparison runs the same YOLOX camera inference, Waymo top-LiDAR
conversion, timestamp matching, depth fusion, and IoU evaluation as three
separate Rust nodes. Camera results and the full point cloud cross node
boundaries as Apache Arrow arrays. Both Dora inputs use a bounded queue of four
messages with backpressure.

Run the reproducible container workload with:

```bash
./scripts/run_dora_container_benchmark.sh dora-01
```

For a quick one-frame validation before a full segment:

```bash
FRAME_LIMIT=1 ./scripts/run_dora_container_benchmark.sh dora-smoke
```

Results are written to `benchmark-results/dora/<run-id>/`. The
`perception.csv` file records camera, inference, LiDAR, Dora fusion, and
end-to-end timing alongside point counts and detection quality. The Dora image
uses the same pinned OS image and the same 2 CPU, 4 GB memory, and read-only
input mounts as the pipes-rs image. Its builder uses pinned Rust 1.95, the
minimum version supported by Dora 1.0.1 (the pipes-rs builder uses Rust 1.94).

To run without Docker, install Dora CLI 1.0.1 and use:

```bash
cargo install dora-cli --version 1.0.1 --locked
./scripts/run_dora_benchmark.sh dora-local-01
```

## Normalized pipes-rs × Dora baseline

### PACE: both complete Waymo segments

The native Slurm runner uses the downloaded dataset and model under
`~/scratch/pipes-rs`, without Docker:

```bash
bash benchmark_waymo.sh --dry-run
sbatch benchmark_waymo.sh
```

It requests an RTX 6000 node with `embers`, four CPU cores, and 24 GB memory.
Both implementations currently use CPU ONNX inference, even on a GPU node.
The first job installs Rust 1.95.0 and Dora CLI 1.0.1 into scratch storage and
builds both implementations with their lockfiles. It needs outbound network
access for that setup. Subsequent jobs reuse compiler and crate caches.
Both link against the same checksum-pinned ONNX Runtime 1.27.0 CPU library
extracted from Microsoft's manylinux wheel. The runner checks that the host
can load API 27 before building; this avoids the newer glibc requirement of
the default static runtime archive. No Python environment is changed.

By default, every segment in the download manifest runs in full five times
per framework: two downloaded segments produce 20 measured runs. The runner
first runs a separate detailed validation pass over the same frames for each
framework, alternates order between timed repeats, and hashes the same inputs
before each run. Validation compares detection values and point-field
fingerprints as well as timestamps, RGB checksums, and counts. Timed runs
check event ordering, delivery accounting, and bounded processing concurrency.
A mismatch or failed process stops the job. This increases
frame coverage to two full segments; repeats measure variance, not additional
unique data. To choose a different size, use:

```bash
sbatch benchmark_waymo.sh --repeats 3 --frame-limit 100
```

For the initial pilot, use `--repeats 2 --frame-limit 20 --warmup-frames 5`.
Both frameworks use a same-host monotonic clock and at most four outstanding
frames per source. Each timed process excludes its first five source frames
from steady-state summaries; complete raw rows and whole-process timings are
retained. See [the metric contract](docs/pilot-metric-contract.md) for exact
timer boundaries, validation tolerances, and interpretation limits.

Logs go to `~/slurm_out/PipesDoraWaymo-JOBID.out`. Sources, executable hashes,
environment information, raw CSVs, and a report covering both segments
go to `~/scratch/pipes-rs/benchmark-results/job-JOBID/`. See
`runs/comparison.json` for frame counts and correctness checks, and
`runs/pilot-report.html` for the report. Compilation and detailed validation
passes are excluded from measured runs.
Native timings include process startup/shutdown. Memory is GNU time maximum
RSS, not the aggregate simultaneous memory of all Dora processes. These
results should be compared within this job, not against earlier Docker runs
with different resource limits. This runner covers the full perception
workload; the Docker matrix below also isolates camera, LiDAR, and transport.

### Docker matrix

The normalized benchmark makes both implementations decode the same camera
JPEGs, convert the image to RGB, calculate the same CRC32 checksum, run the
same YOLOX model, and transport the same nine LiDAR point fields. Full and
component dataflows use queue capacity four with backpressure. Normalized run
artifacts use one shared metric contract covering setup, compute,
queue/transport, fusion, and source-to-fusion age measurements.

Run the complete matrix five times per framework with alternating execution
order:

```bash
./scripts/run_baseline_matrix.sh baseline-01 5
```

This runs five isolated scenarios:

| Scenario | What it isolates |
| --- | --- |
| `full` | Parquet input, camera, LiDAR, Arrow transport, synchronization and fusion |
| `camera` | JPEG decode, RGB checksum, YOLOX and detection transport |
| `lidar` | range-image conversion and the full nine-field point-cloud transport |
| `overloaded` | full pipeline with 125 ms of fixed fusion work and bounded backpressure |
| `transport` | 199 in-memory Arrow messages of 149,796 points without Waymo decoding or inference |

The compute-only view is the stage-timing panel in the `full` dashboard. Its
timers begin after each Parquet row has been materialized, so it excludes
dataset iteration and framework startup while retaining the real decode,
inference, conversion and fusion work.

Open the cross-scenario dashboard after the matrix completes:

```bash
open benchmark-results/baseline/baseline-01/index.html
```

Each scenario also has a detailed dashboard with per-run runtime, medians,
p50/p95 latency, standard deviation, throughput, CPU time, peak memory,
correctness and dropped frames. Raw and aggregate results are retained in
`runs.csv` and `summary.json`.

Every container run writes `manifest.txt`, `environment.txt`, `input.sha256`,
`image.json`, and the resolved `compose.yaml`. Together these freeze model and
dataset hashes, queue settings, confidence/NMS thresholds, framework/Arrow/
compiler versions, container image, platform, CPU and memory limits. Existing
run directories are never overwritten.

For a quick wiring check instead of a performance result:

```bash
FRAME_LIMIT=1 TRANSPORT_FRAMES=3 TRANSPORT_POINTS=1000 \
  ./scripts/run_baseline_matrix.sh smoke-01 1
```

## Play and scrub a full Waymo scene

Export every front-camera / top-LiDAR frame through the staged perception
pipeline, including actual YOLOX inference and depth fusion:

```bash
cargo run --release -- \
  --segment 10023947602400723454_1120_000_1140_000 \
  --onnx-model models/yolox_nano.onnx \
  --scene-output demo-output/scene
open demo-output/scene/index.html
```

The viewer works directly from disk with no server or external dependencies.
Keep its directory together: the page loads per-frame images and results on
demand, with a bounded browser cache. Drag or scroll the timeline, use the
arrow keys to step, or press Space to play. Playback uses source timestamps,
with speed and loop controls. Camera detections, projected LiDAR, the bird's-eye
point cloud, object depths, and measured stage timings stay on the same frame.
Select an object below the pipeline panel to highlight its associated points.

The default exports the entire segment; `--frame-limit 10` makes a short preview.
Omit `--onnx-model` for an explicitly labeled Waymo ground-truth baseline.
`perception.csv` in the export directory contains the full run metrics; use
`--perception-csv PATH` to select another location. The viewer replays computed
results; it does not run inference in JavaScript. Displayed clouds are sampled
to at most 18,000 points while fusion uses all points. This is the full temporal
segment for the front camera and top LiDAR, not a multi-camera surround view.
The existing vehicle-relative conversion does not apply per-pixel motion
compensation or object tracking.

## Arrow representation experiment

Run `sbatch benchmark_arrow.sh` for a separate CPU depth-association pilot on a GPU allocation. It compares native point objects, native columns, and direct Arrow access, including conversion costs. See [the experiment contract](docs/arrow-experiment.md) for timing boundaries and limitations. Results go to `~/scratch/pipes-rs/arrow-results/job-JOBID/`.

## Composed ROS 2 execution pilot

Run `sbatch benchmark_ros2.sh` to bootstrap ROS 2 Jazzy in scratch and compare native Rust channels with a composed in-process ROS graph calling the same Rust fusion kernel. This initial reference-handoff experiment uses prepared Waymo inputs, not the full perception pipeline. See [scope and measurement contract](docs/ros2-execution-experiment.md).

## Arrow hardware-counter profile

Run `sbatch profile_arrow_memory.sh` to measure supported per-thread hardware counters around prebuilt fusion. Results go to `~/scratch/pipes-rs/arrow-memory-results/job-JOBID/`. Cache-access counts are not DRAM-byte measurements; see [measurement scope](docs/arrow-memory-profile.md).

## Downloadable Waymo scene replay

Run `sbatch export_waymo_scene.sh` to export an entire downloaded segment with offline YOLOX predictions, LiDAR depth association, and a synchronized HTML viewer. Download `~/scratch/pipes-rs/scene-exports/job-JOBID/waymo-scene.zip`, extract it, and open `waymo-scene/index.html`. It supports playback, a draggable timeline, wheel scrubbing, overlays, and detection selection without a server or internet access. See [export details](docs/scene-player.md).
