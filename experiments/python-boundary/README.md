# Arrow versus native buffers across Rust → Python

This experiment changes the representation/interface while holding execution
and Python algorithms fixed. It is **separate from Pipes versus ROS 2**.
Rust reads Waymo, decodes JPEGs, converts calibrated range images into columns,
and owns the sensor buffers. Python invokes YOLOX Nano using ONNX Runtime and
runs NumPy depth fusion. The reduced results return to Rust for validation and
checksumming. Each arm uses an in-process Rust/Python boundary via a narrow ctypes C ABI.
Sustained arms run independently in separate containers; paired mode cycles
through representations in one process.
This isolates the language boundary; it does not benchmark process transport.

## Existing data, additional sensors

The manifest names only the two existing sample segments. Defaults select all
five cameras, all five LiDARs, and both returns. The reader retains both camera
projection triples for each point. Fusion associates each return at most once
with a given camera, including cameras in the second projection slot.

Every selected camera/LiDAR/projection must exist at the same timestamp. Missing
components, duplicate sensors, unsorted timestamps, missing requested frames,
and incompatible range/projection shapes fail rather than silently narrowing
the workload. Camera images keep their original resolution at the boundary;
resizing to the model's 416×416 input happens in Python in all variants.
Calibration transforms each LiDAR into vehicle coordinates. Per-pixel motion
compensation is not applied, matching the existing geometry approximation.

In paired mode, only a timestamp bundle and bounded Parquet reader batches are kept in memory,
not all decoded frames in a segment. Native sensor buffers remain resident
while the representations are tested in paired order on that frame.

## Matched producer-copy experiment

This control tests whether the earlier conversion-inclusive Arrow advantage
came from copying/prefetching on the producer rather than Arrow itself.
All three arms use `conversion_inclusive`, so producer allocation/copy work is
inside the timed run and arrival-to-result latency. Every arm runs identical
Python preprocessing, inference and fusion.

| Variant | Producer | Consumer boundary |
|---|---|---|
| `native_shared` | Queue mapped input references | Native zero-copy NumPy views |
| `native_producer_copy` | Copy every payload into Rust-owned allocations, then enqueue | Native zero-copy NumPy views of the copies |
| `arrow` | Copy every payload into Arrow buffers, then enqueue | Arrow C Data import and zero-copy NumPy views |

The new native allocations use the same system allocator, alignment and
64-byte capacity rounding as the installed Arrow buffer allocator. Each copy
variant allocates once per buffer per bundle, copies the same bytes in the same
order after scheduled arrival, and retains its original mapping until processing
finishes. Native copies contain no Arrow arrays and do not use Arrow exports.
Both copies can overlap processing of earlier bundles. The existing `native_copy`
option copies at the consumer and is **not** this control.

Rebuild and prepare a **new** full cache with the updated library:

```bash
docker compose -f compose.python-boundary.yaml build native
CACHE_ID=producer-copy docker compose -f compose.python-boundary.yaml run --rm prepare
```

The old `full` cache records the previous library hash and will fail the runtime
check after rebuilding. The new cache uses the same existing Waymo segments,
takes roughly 17 GB, and leaves earlier caches and results intact.

Run five independent trials per variant, in seeded randomized order:

```bash
python3 scripts/run_producer_copy_experiment.py --name producer-copy-001
```

This runs 15 fresh containers sequentially, with the same image, 2-CPU quota,
4-GiB memory limit, one model thread, queue capacity 8, five warmup bundles and
three complete passes (1,194 measured bundles per trial). The runs are independent;
there are no per-frame barriers. Image, input, runtime, harness and cgroup limits
are checked for consistency. A failed trial stops the suite and prevents its
completion marker. Use a fresh `--name` for another suite.

For an arrival-rate sweep, choose rates around the new unpaced result:

```bash
python3 scripts/run_producer_copy_experiment.py --name producer-copy-rates-001 \
  --rates 6 8 10 --repeats 5
```

The flags `--cpus`, `--memory`, `--threads`, `--queue-size`, `--queue-policy`,
`--passes` and `--warmup` apply equally to all arms. If changing model threads,
prepare a cache with the matching `MODEL_THREADS`. `--dry-run` prints the planned
order without starting containers or writing outputs. The suite uses explicit
settings rather than inheriting benchmark settings from shell environment variables.

Results are in `benchmark-results/python-sustained/NAME/`:

- `plan.json`: trial order, seed, parameters and Docker image ID.
- `trials.csv`: one row per independent run, including throughput, runtime,
  median/p95 latency, CPU, peak RSS, producer-copy time and drops.
- `summary.md`, `summary.csv`, `summary.json`: per-variant/run-rate medians and
  observed min/max values across runs. Ranges are not confidence intervals;
  frame samples or repeated scene passes are not independent trials.
- Each trial also retains the original detailed frame/queue CSVs and summaries.
  `producer_copy_ns` and `producer_copy_bytes` record both copy variants' work;
  `conversion_ns` is retained as an alias of producer materialization time.
- `completed.json`: only written after every trial succeeds and validates.

Compare **native_producer_copy versus Arrow** to assess the representation and
handoff under matched producer copying. Compare both against **native_shared**
to assess the effect of doing those copies. Consider CPU/memory cost alongside
throughput and latency, and include drops when interpreting overloaded runs.

For a quick single native-producer-copy versus Arrow run using the same two
services as before:

```bash
CACHE_ID=producer-copy RUN_ID=copy-pair-001 CONDITION=conversion_inclusive \
  NATIVE_VARIANT=native_producer_copy \
  docker compose -f compose.python-boundary.yaml up native arrow
```

A smoke test of the complete randomized runner:

```bash
CACHE_ID=producer-copy-smoke FRAME_LIMIT=3 \
  docker compose -f compose.python-boundary.yaml run --rm prepare
python3 scripts/run_producer_copy_experiment.py --name producer-copy-smoke-001 \
  --cache-id producer-copy-smoke --frames 3 --passes 2 --warmup 1 --repeats 1
```

## Independent sustained throughput in Docker

The `native` and `arrow` services use the **same image**, CPU quota (default 2),
memory limit (default 4 GiB, no extra swap allowance), model thread count, data,
arrival rate and queue capacity. Each has its own producer thread, waiting queue
and Python inference/fusion worker. There are no cross-container barriers.
Finishing one run does not stop the other. This queue harness is held constant;
it is **not the Pipes-versus-ROS execution experiment**.

First build the image and prepare the common inputs once:

```bash
docker compose -f compose.python-boundary.yaml build native
docker compose -f compose.python-boundary.yaml run --rm prepare
```

Preparation reads all sensors from both existing complete segments and runs
reference inference. It writes roughly **16.9 GB for 398 timestamp bundles** to
`data/python-boundary-cache/full/`. Each bundle contains about 42 MB of decoded
images and LiDAR columns. Both benchmark containers mount this cache read-only.
Preparation is outside measured execution, so Waymo decoding cannot throttle the
producer and hide differences at the language boundary. Inputs are memory mapped
by Rust, not all copied into RAM. Hash checks before timing verify identical data.
The cache records library/model hashes, package versions, architecture and thread
count; regenerate it under a new `CACHE_ID` if those change. Do not edit or truncate
cache files while a run is using them.

Run the two approaches independently at maximum speed, excluding conversion:

```bash
RUN_ID=throughput-001 CONDITION=prepared RATE=0 PASSES=3 \
  docker compose -f compose.python-boundary.yaml up native arrow
```

Measure backlog at a fixed input rate (10 complete multisensor bundles/second):

```bash
RUN_ID=arrival-001 RATE=10 QUEUE_SIZE=8 PASSES=3 \
  docker compose -f compose.python-boundary.yaml up native arrow
```

Measure overload with a policy that discards the oldest waiting input:

```bash
RUN_ID=overload-001 RATE=30 QUEUE_SIZE=8 QUEUE_POLICY=drop-oldest PASSES=3 \
  docker compose -f compose.python-boundary.yaml up native arrow
```

Include Arrow materialization in the producer:

```bash
RUN_ID=inclusive-001 CONDITION=conversion_inclusive RATE=0 PASSES=3 \
  docker compose -f compose.python-boundary.yaml up native arrow
```

Use a fresh `RUN_ID` for every repeat. Services can also run separately, e.g.
`docker compose -f compose.python-boundary.yaml run --rm native` and then `arrow`,
with the same environment settings. Never use `up --abort-on-container-exit`: it
would stop the slower arm when the faster arm completes. CPU/memory settings are
`BENCHMARK_CPUS` and `BENCHMARK_MEMORY`; `MODEL_THREADS` controls ONNX threads.
`NATIVE_VARIANT=native_copy` selects the copied baseline; the default
`native_shared` already crosses without payload copies. Arrow is not guaranteed
to beat this baseline. Both still use the same model preprocessing allocations.

A small end-to-end container smoke test uses a separate cache:

```bash
CACHE_ID=smoke FRAME_LIMIT=3 \
  docker compose -f compose.python-boundary.yaml run --rm prepare
CACHE_ID=smoke FRAME_LIMIT=3 RUN_ID=smoke-001 PASSES=2 WARMUP=1 \
  docker compose -f compose.python-boundary.yaml up native arrow
```

### Arrival, conversion and memory semantics

- `RATE=0` feeds as fast as the producer can prepare/admit bundles. Use it for
  completed bundles/s. With backpressure a fast producer naturally fills the
  queue; queue size alone does not identify the faster approach.
- Positive `RATE` schedules uniformly spaced arrivals on each run’s own clock.
  It does not wait for the other container or reset its schedule when blocked.
  This is a controlled replay rate, not original inter-frame timestamp timing.
- `backpressure` bounds the waiting queue and loses no input. Once full, the
  producer blocks. `admission_lag_ns` captures growing lateness relative to the
  input schedule even when visible queue depth has reached its limit.
- `drop-oldest` admits new inputs by evicting the oldest waiting item. Every
  sequence is accounted for as delivered or dropped. Producer preparation can
  still fall behind the requested rate; `source_lateness_ns` exposes that.
- `prepared` wraps mapped buffers as Arrow arrays before timing; native inputs
  reference the same mapped bytes. Per-item Arrow export/import is still timed.
  `conversion_inclusive` copies native payloads into Arrow buffers on the
  producer after arrival, including that work in latency and resource usage.
  Native-copy allocates/copies at the Python boundary in both conditions.
- One item being processed and one being prepared are additional to the waiting
  queue. Queued bytes represent **logical backlog**, not extra allocated RAM:
  prepared handles share mapped data. Page faults and memory pressure remain
  possible, and are counted. Repeating passes reuses the same scenes and does
  not increase unique data coverage.

Results go to `benchmark-results/python-sustained/RUN_ID/CONDITION/VARIANT/`:

- `frames.csv`: every sequence, outcome, due/admitted/received/finished times,
  conversion, boundary, model/fusion, queue wait, backpressure and total latency.
- `queue.csv`: timestamped enqueue/dequeue/drop events, waiting depth and logical
  payload bytes. Times use the process monotonic clock; subtract the run’s
  `start_ns` for relative time, not another container’s timestamp.
- `summary.json` / `summary.md`: completed bundles/s and logical GB/s, queue peak,
  time-weighted mean depth, fraction full, drops, drain time, latency percentiles,
  process CPU and page-fault deltas. Peak RSS includes setup and warmup.
- `environment.json`: input manifest hash, runtime identity and all parameters.
- `completed.json`: written only after every delivered result matches its
  offline reference CRC and every offered sequence is accounted for.

Run several repeats and a rate sweep around the measured throughput. Compare
throughput at rate 0; at the same positive rate compare latency, admission lag,
queue occupancy and drops. These measurements do not count hardware memory
reads or DRAM bandwidth. Output checks are small and timed equally in each arm.

The equivalent local commands are:

```bash
.venv/bin/python scripts/run_python_boundary.py --mode prepare \
  --output data/python-boundary-cache/local
.venv/bin/python scripts/run_python_boundary.py --mode sustained \
  --input-cache data/python-boundary-cache/local --variant arrow \
  --condition prepared --rate 0 --queue-size 8 --passes 3 \
  --output benchmark-results/python-boundary/arrow-sustained-001
```

Build local caches with the local runtime and container caches with the container
runtime. This avoids treating different inference versions as the same reference.

## Original paired setup and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/python-boundary/requirements.txt
cargo build --release --locked --lib

# Check both existing segments, sensor coverage, and file hashes.
.venv/bin/python scripts/run_python_boundary.py \
  --inventory-only --output benchmark-results/python-boundary/inventory-001

# Actual Python inference and fusion: two timestamp bundles per segment.
.venv/bin/python scripts/run_python_boundary.py \
  --frames 2 --repeats 1 --warmup 0 \
  --output benchmark-results/python-boundary/smoke-001

# Both complete segments, all sensors, three measured repetitions after warmup.
.venv/bin/python scripts/run_python_boundary.py \
  --output benchmark-results/python-boundary/full-001
```

The default model is `models/yolox_nano.onnx` from the existing model downloader.
Use `--data-root`, `--model`, or `--library` for other locations. `--frames 0`
means the complete segment. Output directories must be new. A failure never
writes `completed.json`; partial CSVs must not be interpreted as complete runs.
`--segment ID` restricts the existing manifest without downloading a new scene.
Use `--cameras 1 --lidars 1 --returns 1` for the old sensor configuration.

The validation scene needs its `lidar_camera_projection` component. If absent:

```bash
python3 scripts/download_waymo_sample.py --count 2 --with-labels --transfer-tool gsutil
```

That downloader skips existing files and fetches missing components of the same
two scenes. It does not add further segments. The existing Waymo terms and
Google Cloud authentication are required for downloads.

## Three interface variants

| Variant | Rust → Python path | Expected sensor-payload copies at this boundary |
|---|---|---|
| `native_shared` | Rust native columns/RGB buffers → read-only NumPy views | None |
| `native_copy` | Same native input → independent NumPy arrays | One |
| `arrow` | Rust Arrow primitive arrays → Arrow C Data Interface → PyArrow → NumPy views | None after Arrow construction |

Native columns, rather than Python objects per point, are the strong baseline.
All columns and RGB images cross the boundary, including fields not used by the
current fusion filter. Arrow images are flat UInt8 arrays plus shared shape
metadata; this experiment does not claim that Arrow provides a model-specific
image layout. PyArrow import and `to_numpy(zero_copy_only=True)` are timed and
verify pointer identity. Native views also verify pointer identity, retain the
Rust frame owner, and are read-only. Arrow imports own their buffers even after
the source frame is freed. The low-level PyArrow C import is confined to the
bridge and covered by integration tests.

Python preprocessing, inference, postprocessing, and fusion are exactly the
same functions in every variant. CPUExecutionProvider and fixed thread counts
avoid silently selecting different inference backends. Model input preparation
still allocates/resizes/reorders pixels; those costs are recorded separately.
The output crosses back through the same small native ABI in all arms.

## Two independent timing conditions

- `conversion_inclusive`: start with Rust-native decoded buffers, construct
  Arrow columns if needed, import into Python, run the model/fusion, and return
  results to Rust. Arrow construction clones every native payload column.
- `prepared`: construct the Arrow representation **before** starting the timer;
  time its import, model/fusion, and return. Native buffers already exist. The
  copied-native path still copies during handoff; copying is its interface cost.

Both exclude common Parquet input, JPEG decoding, and LiDAR coordinate conversion.
Their duration is recorded once per bundle as `source_preparation_ns`; it is
not a per-variant measurement and must not be summed across repeated rows.
Neither condition is claimed to be full sensor-to-result system latency.
Conversion-free timing is measured explicitly, not inferred by subtraction.
Destruction and correctness checks occur after the total timer.

Arrow materialization uses extra memory alongside the native source. Only the
currently tested Arrow representation is retained; it is cleared between trials.
Prepared trials remain cache-warm because all variants operate on the same frame.
Longer segments increase distinct input coverage, not instantaneous concurrency.
These statements describe the original `--mode paired` runner. Sustained mode
below adds independent execution and an explicit arrival/queue model.

## Recorded evidence

- `inventory.json`: exact files, sizes, SHA-256 hashes, selected sensors, frames.
- `environment.json`: Python/package versions, platform, arguments, model and
  Rust-library hashes, inference provider and process layout.
- `frames.jsonl`: timestamp, image shapes, per-LiDAR/return point counts.
- `raw.csv`: conversion, boundary, preprocessing, inference, postprocessing,
  fusion, result-return and total times; actual payload bytes; known copy bytes;
  input fingerprints and returned-result checksums.
- `summary.json` / `summary.md`: per-frame medians across repeats, then medians
  across frames. Ratios are paired by frame against native shared arrays.
- `completed.json`: written only after every output and input comparison passes.

Known copy bytes cover explicit representation materialization and boundary
copies, not model internals, allocators, cache traffic or DRAM bandwidth. In the
prepared condition the reported Arrow materialization bytes occur outside the
timer. Process CPU time and peak RSS cover all arms, loading, warmups and checks;
they are not evidence of one representation's independent CPU/RSS usage.
No winner or statistical significance is inferred from smoke runs.

## PACE

```bash
sbatch benchmark_python_boundary.sh --frames 2 --repeats 1 --warmup 0
sbatch benchmark_python_boundary.sh
```

The wrapper follows the existing scratch/toolchain convention, snapshots source,
builds the Rust library, creates a per-job Python environment, records resolved
versions, and runs both segments. It expects the existing scratch dataset and
`scratch/pipes-rs/models/yolox_nano.onnx`. The Embers GPU allocation matches the
other job scripts; this experiment currently uses CPU inference only.

## Verification

```bash
cargo test --locked --lib
.venv/bin/python -m unittest discover -s scripts -p 'test_python*.py' -v
.venv/bin/python -m unittest discover -s scripts -p test_producer_copy_experiment.py -v
```

Rust tests check shape validation, calibration geometry, both projections and
sensor/return IDs. Python tests cover projected-depth semantics, paired
summaries, actual all-sensor ingestion, all three representations, and ownership
when Rust frame handles are freed. Data-dependent tests explicitly skip if the
library/sample is absent. Actual model outputs are compared on every warmup and
measured trial; full input fingerprints are compared outside the timed region.
