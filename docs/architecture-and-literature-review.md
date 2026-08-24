# Why Pipes might work

_A short architecture and literature review. Written August 2026, before an
implementation or independent Pipes benchmark exists._

## The bet

Pipes keeps the heavy sensor path in one Rust pipeline. A camera, LiDAR, radar,
or IMU driver writes into an Apache Arrow representation. Processing, models,
recording, and fusion share that data. Once the pipeline has turned the raw
inputs into detections, tracks, state, or health, it publishes those smaller
results as normal ROS 2 messages.

The bet is not that Rust is automatically faster than C++, or that Arrow is
automatically faster than ROS. It is this:

> If raw sensor data enters one controlled memory and timing system, stays there
> while it is processed and fused, and crosses into ROS only after it has become
> a smaller result, we can avoid a lot of copying, serialization, queueing, and
> timing uncertainty.

That should make the sensor path faster and easier to understand. It also keeps
the adoption risk small. A robot can continue using ROS for control, planning,
launch, lifecycle, visualization, and everything else. Existing ROS nodes only
see ordinary topics. They do not need to link against Pipes or trust it with the
whole robot.

There is another part of the bet that matters just as much as speed: the same
pipeline should run from live sensors or a recording. Every frame should retain
its sensor time, arrival order, calibration, and the decisions that caused it to
be admitted, delayed, joined, or dropped. That makes it possible to isolate the
perception and fusion system, replay the exact workload, and tune it without
recreating the physical run.

## Why it could be substantially faster

Images and point clouds are large. Moving them through a graph of processes can
cost more than the useful work being done on them, especially when each step
allocates a new message, serializes it, copies it, and reconstructs it at the
other end. Fan-out makes the problem worse: the processor, recorder, viewer,
and fusion node may each receive another copy.

Pipes is designed around a simpler path:

1. The driver writes into a Pipes-owned Arrow buffer when the device API allows
   it.
2. Pipeline stages share that buffer or create another Arrow result when they
   actually transform the data.
3. Recording, visualization, and models receive branches from the same stream
   instead of opening the sensor again.
4. The final, smaller result is converted to a ROS message once.

This should reduce allocations, serialization work, memory bandwidth, and
queueing between stages. Rust helps because ownership and lifetimes are
especially important around drivers, shared buffers, fan-out, and shutdown.
Rust's type system can prevent many shared-state and lifetime errors, although
vendor SDKs and GPU libraries still introduce unsafe foreign interfaces
([Rust concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html),
[Rust FFI](https://doc.rust-lang.org/nomicon/ffi.html)).

dora-rs gives real evidence that this pattern can matter. Its runtime uses
Arrow and shared memory for large local messages. In dora's 2026 paper, a local
32 MB payload sent at 50 Hz took about 2.78 ms through dora versus 87 ms through
the ROS 2 Python path used in that test
([dora evaluation](https://arxiv.org/html/2602.13252#S6)). That is a large and
relevant result. It supports the decision to write into one shared
representation and let consumers read it without rebuilding the payload.

It does not prove that Pipes will be thirty times faster than a well-built ROS
system. The comparison was against one ROS configuration and language path, not
every ROS 2 transport. The producer still had to put the data into Arrow.
Remote transport, recording, format conversion, and CPU-to-GPU transfer still
move bytes. A camera API that owns its output buffer may force one copy before
the data enters Pipes.

The realistic claim is still strong: Pipes should remove many repeated copies
and conversions after acquisition, particularly when large data fans out to
several consumers. The size of the gain will depend on the sensor, driver API,
pipeline, hardware, and ROS baseline.

## Arrow is the internal data rule

Arrow should be the standard format between Pipes components. This is not a
choice the runtime makes based on current load or message size.

The path is simple: the sensor driver produces timing metadata plus an Arrow
payload; processing, model, and fusion stages consume and produce the same
combination; and only the final result is converted to ROS.

A component may use OpenCV, PCL, nalgebra, GTSAM, ONNX Runtime, TensorRT, or a
native tensor type while it does its work. That is private to the component.
What it sends to the next Pipes component is Arrow again.

Arrow provides a stable layout and cross-language interfaces
([Arrow format](https://arrow.apache.org/docs/format/Intro.html),
[C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)).
It does not provide a scheduler, sensor timestamps, bounded queues, overload
policy, or replay. Pipes has to provide those.

This also gives large video or behavior models a natural place in the graph.
The model reads a branch of the same camera stream, builds clips or batches,
runs inference, and returns a smaller timestamped result. It should not open
the camera a second time. Because a large model may run slowly or unevenly, its
branch needs a bounded queue and a clear policy: process every frame, sample at
a lower rate, keep only the newest frame, or drop old work when it falls behind.
The model must not stall capture or fast fusion.

Arrow can often expose CPU memory to Python and tensor libraries without an
extra serialization step. Moving that data to a GPU may still require a copy.
Arrow's device-memory interface may help later, but it should extend the same
data model rather than create a second one
([Arrow device interface](https://arrow.apache.org/docs/format/CDeviceDataInterface.html)).

## Sensor time matters as much as transport speed

Fusion does not merely need measurements to arrive. It needs to know when they
were true.

A camera timestamp might mean exposure start, exposure midpoint, driver
receipt, or publication time. A LiDAR scan covers an interval rather than one
instant. Different devices have different clocks, drift, buffering, and
transfer delays. Research on joint temporal and spatial calibration treats
accurate time alignment as necessary for good state estimation
([Furgale, Rehder, and Siegwart, IROS 2013](https://doi.org/10.1109/IROS.2013.6696514)).

ROS correctly distinguishes system, steady, and simulated time, including the
fact that simulated time may pause or move backward
([ROS clock design](https://design.ros2.org/articles/clock_and_time.html)).
ROS message filters also support exact and approximate timestamp matching and
warn against treating arrival time as measurement time
([message_filters](https://docs.ros.org/en/ros2_packages/rolling/api/message_filters/message_filters.html)).

Pipes can go further because it owns the whole pre-ROS path. For each sample it
can retain:

- the device timestamp and what that timestamp means;
- the clock conversion and its uncertainty;
- when the host first saw the sample;
- its source sequence and arrival order;
- the calibration and model versions used; and
- where it waited, joined another stream, or was dropped.

This is useful during normal operation, but it becomes especially valuable
during replay. An engineer should be able to answer, “Did the model make a bad
decision, or did it receive stale, misaligned, or incomplete inputs?” A bag of
payloads alone often cannot answer that.

Replay has three different meanings:

| Promise | What it means |
|---|---|
| Same inputs | Run with the same sensor data, time mapping, order, and configuration |
| Same pipeline decisions | Reproduce which samples were accepted, dropped, synchronized, or expired |
| Same numbers | Produce bit-for-bit identical model or estimator output |

The first two should be Pipes goals. The third is not always possible across
different GPUs, libraries, thread schedules, or floating-point reductions.

MCAP is a good recording boundary because it already stores timestamped,
schema-described data and is supported by ROS tooling
([MCAP specification](https://mcap.dev/spec),
[ROS 2 guide](https://mcap.dev/guides/getting-started/ros-2)). Pipes can record
raw inputs and useful intermediates alongside small evidence records describing
timing, drops, joins, calibration, models, and the build that produced the run.
The same recording can then be used for debugging, offline tuning, model
comparison, regression testing, or faster-than-real-time iteration.

## ROS already solves a lot

Pipes should not justify itself by comparing against a careless ROS graph.

ROS 2 already has sensor-oriented QoS. Its sensor profile favors fresh data over
retransmitting stale readings, and DDS settings expose history, depth,
reliability, deadlines, lifespan, and liveliness
([ROS QoS](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html)).
ROS can compose nodes into one process, transfer ownership through
intra-process communication, and use middleware-loaned messages where the
client library, RMW implementation, and message type support them
([intra-process communication](https://docs.ros.org/en/dashing/Tutorials/Intra-Process-Communication.html),
[loaned-message design](https://design.ros2.org/articles/zero_copy.html)).
Type adaptation and negotiation can also keep native representations inside a
compatible graph
([REP-2007](https://ros.org/reps/rep-2007.html),
[REP-2009](https://ros.org/reps/rep-2009.html)).

ROS also has rosbag2/MCAP recording and low-overhead tracing through
ros2_tracing
([ros2_tracing](https://github.com/ros2/ros2_tracing/blob/rolling/README.md)).
These tools are useful and Pipes should work with them.

NVIDIA NITROS is the strongest example of the same broad idea inside ROS. It
uses type adaptation and negotiation to keep data in GPU-friendly
representations between compatible nodes, then interoperates with ordinary ROS
at the boundary
([NITROS](https://nvidia-isaac-ros.github.io/concepts/nitros/index.html)).
It validates the architecture rather than making it unnecessary. NITROS is
tied to NVIDIA's acceleration stack; Pipes is aiming at a broader sensor path
with direct Rust drivers, CPU and mixed native libraries, explicit replay, and
fusion timing. On NVIDIA hardware, Pipes should interoperate with or benchmark
against NITROS rather than rebuild working CUDA plumbing for ideological
reasons.

ROS executors do have timing limitations under load. The default executor
reports which entities are ready rather than exposing full queue state, and it
does not promise simple FIFO callback execution. ROS documentation discusses
round-robin scheduling, priority inversion, and limited control over callback
order
([ROS executors](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Executors.html)).
That gives Pipes room to provide simpler, explicit scheduling for a fixed
sensor graph. It does not mean that any Rust async runtime will automatically
be more predictable.

The fair comparison is therefore Pipes against:

- a conventional multi-process ROS 2 graph;
- an optimized composed ROS 2 graph;
- ROS shared-memory or loaned-message paths where they actually apply; and
- NITROS on supported NVIDIA hardware.

If optimized ROS performs just as well and the missing replay evidence can be
added cleanly, a separate runtime may not be worth maintaining.

## What to take from the reference projects

### dora-rs

dora shows that an explicit dataflow graph, Arrow payloads, shared memory, and
bounded queues can work together in a robotics runtime
([dora architecture](https://dora-rs.ai/dora/concepts/architecture)).
It also makes dropped data visible instead of pretending every consumer can
always keep up.

Pipes should take the data layout, graph, buffer-sharing, and replay lessons.
It does not need to take dora's full distributed runtime, Zenoh orchestration,
custom recording format, or ROS wire implementation. The intended user already
trusts ROS as the system boundary, so normal ROS messages are the safer
integration story.

### Copper

Copper is the closest architectural neighbor. It uses a statically described
Rust task graph, generates a purpose-built runtime, separates critical work
from setup and cleanup, and integrates execution with logging and replay
([runtime overview](https://copper-project.github.io/copper-rs/Copper-Runtime-Overview/),
[task lifecycle](https://copper-project.github.io/copper-rs-book/task-lifecycle.html),
[logging and replay](https://copper-project.github.io/copper-rs-book/logging-replay.html)).
Its message model can describe a time interval, which is a better fit than one
timestamp for scans, rolling shutters, and accumulated windows.

Pipes should take Copper's execution discipline and its view that replay belongs
in the runtime from the beginning. The distinction is scope: Copper is moving
toward a broader robotics runtime, while Pipes is specifically the large-data
sensor and fusion front end for a ROS robot. Copper is close enough that Pipes
should keep asking whether reuse or collaboration is better than maintaining a
second scheduler.

### Rerun

Rerun is the model for inspection, not for critical-path scheduling. It stores
multimodal data in Arrow chunks and gives it explicit entities, components, and
timelines
([Rerun architecture](https://github.com/rerun-io/rerun/blob/main/ARCHITECTURE.md),
[timelines](https://rerun.io/docs/concepts/logging-and-ingestion/timelines)).
It already knows how to display images, point clouds, transforms, boxes, and
other data that Pipes users will need to inspect.

A Rerun output should be a bounded side branch. If the viewer is slow or
disconnected, fusion must continue. MCAP remains the durable run record, and
Rerun becomes the best way to explore it. Rerun can already open MCAP and
understand many ROS and Foxglove schemas
([Rerun MCAP support](https://rerun.io/docs/howto/logging-and-ingestion/mcap)).
That gives Pipes good visualization without building a custom GUI.

Together, the three projects point toward a coherent design:

| Reference | Main lesson for Pipes |
|---|---|
| dora-rs | Share large Arrow data through an explicit, bounded graph |
| Copper | Make execution, timing, logging, and replay one design |
| Rerun | Treat observability as a rich but non-blocking branch |
| ROS 2 | Keep the public interface conventional and useful |

## What Pipes has to prove

The architecture should produce impressive results if the implementation
preserves the intended memory path. But a fast channel benchmark will not prove
the project.

The benchmark needs a real graph: camera and LiDAR payloads, high-rate IMU,
one-to-many fan-out, synchronization, a model or preprocessing stage,
recording, fusion, and compact ROS output. The same algorithms and data layouts
should be used across implementations wherever possible.

Measure:

- end-to-end measurement age, including p95, p99, and worst cases;
- delivered rate and every drop, not latency alone;
- allocations and bytes copied;
- CPU time, memory bandwidth, and host-to-device transfers;
- queue depth and how long samples wait;
- completed, degraded, and expired fusion sets;
- the effect of recording and visualization; and
- whether replay reproduces the same admission, synchronization, and drop
  decisions.

Overload tests matter as much as the happy path. Slow a model down, stall the
recorder, disconnect the viewer, reorder samples, reset a device clock, remove
a sensor, and fill the GPU. Pipes should either keep operating according to a
declared policy or fail visibly. Silent stale-data reuse would undermine the
whole project.

The bet is supported if Pipes shows a repeatable advantage over optimized ROS
on at least one realistic sensor graph:

- fewer copies and less memory traffic;
- lower tail latency or lower measurement age;
- fewer stale or incomplete fusion inputs under load;
- substantially faster replay and tuning;
- clearer explanations of when and why data was dropped; or
- easier integration of Rust drivers, native libraries, and large models.

The bet should be reconsidered if optimized ROS or NITROS meets the same
workload just as well, direct sensor access is rarely available, or Pipes grows
into another general robotics middleware. The point is to make the high-rate
sensor and fusion path better, not to replace working parts of ROS.

## Conclusion

Pipes makes sense because it chooses a narrow place to be opinionated. Raw
sensor data is large, time-sensitive, and awkward to move. Rust is useful at
the driver and buffer-ownership boundary. Arrow gives every component one
shared data language. A controlled graph makes queueing and synchronization
visible. Recording those decisions makes fusion and model development far
easier to repeat.

ROS remains the trusted perimeter. Pipes does the heavy sensor work and reports
the result.

The expected performance gains are plausible and could be large. dora provides
strong supporting evidence for the Arrow/shared-memory part of the design.
Modern ROS and NITROS provide strong counterexamples that prevent an automatic
victory claim. The next useful evidence is therefore not another architectural
argument. It is a complete, reproducible sensor-to-fusion benchmark.

## Selected sources

- Apache Arrow, [format](https://arrow.apache.org/docs/format/Intro.html),
  [C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html),
  and [device interface](https://arrow.apache.org/docs/format/CDeviceDataInterface.html).
- Copper, [runtime overview](https://copper-project.github.io/copper-rs/Copper-Runtime-Overview/),
  [task lifecycle](https://copper-project.github.io/copper-rs-book/task-lifecycle.html),
  and [logging/replay](https://copper-project.github.io/copper-rs-book/logging-replay.html).
- dora-rs, [architecture](https://dora-rs.ai/dora/concepts/architecture) and
  [2026 paper](https://arxiv.org/abs/2602.13252).
- Rerun, [architecture](https://github.com/rerun-io/rerun/blob/main/ARCHITECTURE.md),
  [timelines](https://rerun.io/docs/concepts/logging-and-ingestion/timelines), and
  [MCAP support](https://rerun.io/docs/howto/logging-and-ingestion/mcap).
- ROS 2, [QoS](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html),
  [executors](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Executors.html),
  [clock design](https://design.ros2.org/articles/clock_and_time.html), and
  [loaned-message design](https://design.ros2.org/articles/zero_copy.html).
- ROS, [REP-2007 type adaptation](https://ros.org/reps/rep-2007.html) and
  [REP-2009 type negotiation](https://ros.org/reps/rep-2009.html).
- NVIDIA Isaac ROS,
  [NITROS](https://nvidia-isaac-ros.github.io/concepts/nitros/index.html).
- MCAP, [specification](https://mcap.dev/spec) and
  [ROS 2 guide](https://mcap.dev/guides/getting-started/ros-2).
- P. Furgale, J. Rehder, and R. Siegwart,
  [“Unified Temporal and Spatial Calibration for Multi-Sensor Systems”](https://doi.org/10.1109/IROS.2013.6696514),
  IROS 2013.
