"""Independent producer/consumer runs. One variant and condition per process.

The same module runs inside either Docker service. There is no cross-run barrier
or scheduler. Prepared cache construction and reference inference are separate.
"""

import collections
import csv
import importlib.metadata
import json
import platform
import resource
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from python_boundary import Frame, Library, Source
from python_perception import Yolox, process

PACKAGES = ("numpy", "pyarrow", "Pillow", "onnxruntime")


def runtime_identity(a):
    from run_python_boundary import digest

    return {
        "library_sha256": digest(a.library),
        "model_sha256": digest(a.model),
        "perception_sha256": digest(Path(__file__).with_name("python_perception.py")),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "machine": platform.machine(),
        "system": platform.system(),
        "threads": a.threads,
    }


def prepare(a, segments, inventory):
    """Write aligned native columns once; both services mount this cache read-only."""
    from run_python_boundary import digest

    lib = Library(a.library.resolve())
    model = Yolox(a.model, a.threads)
    entries = []
    for segment in segments:
        config = dict(
            data_root=str(a.data_root),
            **segment,
            cameras=a.cameras,
            lidars=a.lidars,
            returns=a.returns,
            frame_limit=a.frames,
        )
        with Source(lib, config) as source:
            while (frame := source.next()) is not None:
                meta = frame.metadata
                arrays = frame.arrays("native_shared")
                name = f"bundle-{len(entries):06d}.bin"
                descriptors = []
                with (a.output / name).open("xb") as output:
                    for descriptor in meta["buffers"]:
                        padding = (-output.tell()) % 64
                        output.write(b"\0" * padding)
                        descriptors.append(dict(descriptor, offset=output.tell()))
                        arrays[descriptor["name"]].tofile(output)
                result, _ = process(arrays, a.cameras, model)
                entries.append(
                    {
                        "segment": segment["segment"],
                        "split": segment["split"],
                        "file": name,
                        "metadata": dict(meta, buffers=descriptors),
                        "payload_bytes": sum(v.nbytes for v in arrays.values()),
                        "sha256": digest(a.output / name),
                        "expected_crc32": lib.consume(result),
                        "detections": len(result),
                    }
                )
                print(
                    f"cached {segment['split']} frame {meta['index']}: {entries[-1]['payload_bytes'] / 1e6:.1f} MB",
                    flush=True,
                )
                del arrays, result, frame
    if len(entries) != sum(s["frames"] for s in inventory):
        raise ValueError("incomplete prepared inputs")
    manifest = {
        "contract": 1,
        "runtime": runtime_identity(a),
        "inventory": inventory,
        "cameras": a.cameras,
        "lidars": a.lidars,
        "returns": a.returns,
        "entries": entries,
    }
    (a.output / "cache.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (a.output / "completed.json").write_text(
        json.dumps({"prepared": True, "frames": len(entries)}) + "\n"
    )
    print(
        f"Prepared {len(entries)} bundles, {sum(e['payload_bytes'] for e in entries) / 1e9:.2f} GB: {a.output}",
        flush=True,
    )


@dataclass
class Work:
    sequence: int
    entry: dict
    frame: object
    due: int
    ready: int
    conversion_ns: int = 0
    source_ns: int = 0
    admitted: int = 0
    requested: int = 0


class WorkQueue:
    """Bounded waiting queue; processing and one producer item are outside it."""

    def __init__(self, capacity, policy, clock=time.perf_counter_ns):
        if capacity < 1 or policy not in ("backpressure", "drop-oldest"):
            raise ValueError("invalid queue configuration")
        self.capacity, self.policy, self.clock = capacity, policy, clock
        self.condition = threading.Condition()
        self.items = collections.deque()
        self.closed = False
        self.events = []
        self.bytes = 0
        self.peak = 0

    def event(self, action, sequence):
        self.events.append(
            {
                "time_ns": self.clock(),
                "event": action,
                "sequence": sequence,
                "depth": len(self.items),
                "queued_payload_bytes": self.bytes,
            }
        )
        self.peak = max(self.peak, len(self.items))

    def put(self, item):
        with self.condition:
            item.requested = self.clock()
            while (
                len(self.items) == self.capacity
                and self.policy == "backpressure"
                and not self.closed
            ):
                self.condition.wait()
            if self.closed:
                raise RuntimeError("queue closed before producer completed")
            dropped = None
            if len(self.items) == self.capacity:
                dropped = self.items.popleft()
                self.bytes -= dropped.entry["payload_bytes"]
                self.event("drop", dropped.sequence)
            item.admitted = self.clock()
            self.items.append(item)
            self.bytes += item.entry["payload_bytes"]
            self.event("enqueue", item.sequence)
            self.condition.notify_all()
            return dropped

    def get(self):
        with self.condition:
            while not self.items and not self.closed:
                self.condition.wait()
            if not self.items:
                return None
            item = self.items.popleft()
            self.bytes -= item.entry["payload_bytes"]
            received = self.clock()
            self.event("dequeue", item.sequence)
            self.condition.notify_all()
            return item, received

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def queue_statistics(events, start, end, capacity):
    """Integrate the piecewise-constant waiting depth, including idle intervals."""
    last, depth, size = start, 0, 0
    area = byte_area = full = 0
    for event in [*events, {"time_ns": end, "depth": 0, "queued_payload_bytes": 0}]:
        now = min(end, max(start, event["time_ns"]))
        dt = max(0, now - last)
        area += depth * dt
        byte_area += size * dt
        full += dt if depth == capacity else 0
        depth, size, last = event["depth"], event["queued_payload_bytes"], now
    duration = max(1, end - start)
    return {
        "mean_queue_depth": area / duration,
        "mean_queued_payload_bytes": byte_area / duration,
        "fraction_queue_full": full / duration,
        "peak_queue_depth": max((e["depth"] for e in events), default=0),
        "peak_queued_payload_bytes": max(
            (e["queued_payload_bytes"] for e in events), default=0
        ),
    }


def timing_row(item, status, finished):
    return {
        "sequence": item.sequence,
        "segment": item.entry["segment"],
        "split": item.entry["split"],
        "timestamp": item.entry["metadata"]["timestamp"],
        "status": status,
        "payload_bytes": item.entry["payload_bytes"],
        "due_ns": item.due,
        "ready_ns": item.ready,
        "requested_ns": item.requested,
        "admitted_ns": item.admitted,
        "finished_ns": finished,
        "source_ns": item.source_ns,
        "conversion_ns": item.conversion_ns,
        "producer_copy_ns": item.conversion_ns,
        "producer_copy_bytes": item.entry["payload_bytes"] if item.conversion_ns else 0,
        "source_lateness_ns": max(0, item.requested - item.due),
        "backpressure_ns": max(0, item.admitted - item.requested),
        "admission_lag_ns": max(0, item.admitted - item.due),
        "source_to_result_ns": finished - item.due if status == "delivered" else None,
    }


def prepare_producer(frame, a):
    """Both copies run on the caller thread, after arrival and before enqueue."""
    if a.condition != "conversion_inclusive":
        return False
    if a.variant == "arrow":
        frame.prepare_arrow()
    elif a.variant == "native_producer_copy":
        frame.prepare_native_copy()
    else:
        return False
    return True


def measure(entries, bases, lib, model, a, processor=process):
    """Timed section, shared by deterministic queue tests and actual runs."""
    queue = WorkQueue(a.queue_size, a.queue_policy)
    cancelled = threading.Event()
    failures, rows = [], []
    start = time.perf_counter_ns()
    producer_finished = [start]
    total = len(entries) * a.passes

    def producer():
        try:
            for sequence in range(total):
                if cancelled.is_set():
                    break
                idx = sequence % len(entries)
                source_start = time.perf_counter_ns()
                due = start + round(sequence / a.rate * 1e9) if a.rate else source_start
                frame = bases[idx].clone()
                source_ns = time.perf_counter_ns() - source_start
                # Do not copy either representation before its scheduled arrival.
                remaining = (due - time.perf_counter_ns()) / 1e9
                if remaining > 0 and cancelled.wait(remaining):
                    break
                conversion_start = time.perf_counter_ns()
                copied = prepare_producer(frame, a)
                ready = time.perf_counter_ns()
                item = Work(
                    sequence,
                    entries[idx],
                    frame,
                    due,
                    ready,
                    ready - conversion_start if copied else 0,
                    source_ns,
                )
                dropped = queue.put(item)
                if dropped:
                    rows.append(timing_row(dropped, "dropped", time.perf_counter_ns()))
                    dropped.frame = None
                del frame, item, dropped
        except Exception as error:  # noqa: BLE001 -- propagate producer failures to caller
            failures.append(error)
            cancelled.set()
        finally:
            producer_finished[0] = time.perf_counter_ns()
            queue.close()

    worker = threading.Thread(target=producer, name="sensor-producer")
    worker.start()
    completed = 0
    try:
        while (received := queue.get()) is not None:
            item, received_ns = received
            if cancelled.is_set():
                break
            arrays = item.frame.arrays(a.variant)
            boundary_end = time.perf_counter_ns()
            result, stages = processor(arrays, a.cameras, model)
            compute_end = time.perf_counter_ns()
            checksum = lib.consume(result)
            return_end = time.perf_counter_ns()
            if checksum != item.entry["expected_crc32"]:
                raise AssertionError(f"result mismatch at sequence {item.sequence}")
            detections = len(result)
            del arrays, result
            item.frame = None  # release per-item representation before next frame
            finished = time.perf_counter_ns()
            row = timing_row(item, "delivered", finished)
            row.update(
                received_ns=received_ns,
                queue_wait_ns=received_ns - item.admitted,
                boundary_ns=boundary_end - received_ns,
                **stages,
                return_ns=return_end - compute_end,
                cleanup_ns=finished - return_end,
                service_ns=finished - received_ns,
                result_crc32=checksum,
                detections=detections,
            )
            rows.append(row)
            completed += 1
            if completed % 50 == 0:
                print(
                    f"{a.variant}: completed sequence {item.sequence}/{total - 1}, queue peak {queue.peak}",
                    flush=True,
                )
    finally:
        cancelled.set()
        queue.close()
        worker.join()
    if failures:
        raise RuntimeError("producer failed") from failures[0]
    if sorted(r["sequence"] for r in rows) != list(range(total)):
        raise AssertionError("missing or duplicated outcomes")
    end = time.perf_counter_ns()
    return rows, queue.events, start, end, producer_finished[0]


def run(a):
    from run_python_boundary import digest

    if not a.input_cache:
        raise ValueError("--input-cache is required; build it once with --mode prepare")
    if a.output.exists():
        raise ValueError("output already exists; use a new run directory")
    manifest = json.loads((a.input_cache / "cache.json").read_text())
    if manifest.get("contract") != 1 or not (a.input_cache / "completed.json").exists():
        raise ValueError("incomplete or incompatible prepared cache")
    identity = runtime_identity(a)
    if manifest["runtime"] != identity:
        raise ValueError(
            "Cache runtime/model/library differs. Prepare the cache with the same Docker image/runtime as the runs."
        )
    for name in ("cameras", "lidars", "returns"):
        if manifest[name] != getattr(a, name):
            raise ValueError(f"{name} differ from prepared cache")
    entries = [
        e
        for e in manifest["entries"]
        if (not a.segment or e["segment"] in a.segment)
        and (not a.frames or e["metadata"]["index"] < a.frames)
    ]
    if not entries:
        raise ValueError("empty selected input")
    if a.segment and set(a.segment) - {e["segment"] for e in entries}:
        raise ValueError("requested segment missing from cache")
    if a.frames:
        counts = collections.Counter(e["segment"] for e in entries)
        if any(n != a.frames for n in counts.values()):
            raise ValueError("requested frame count unavailable in cache")
    lib, model = Library(a.library.resolve()), Yolox(a.model, a.threads)
    bases = []
    # Scan all selected cache files identically before either timed run. The OS
    # may evict pages later; page faults are reported rather than assumed absent.
    for entry in entries:
        path = a.input_cache / entry["file"]
        if (
            path.parent.resolve() != a.input_cache.resolve()
            or digest(path) != entry["sha256"]
        ):
            raise ValueError("invalid or corrupted cached bundle")
        frame = Frame.cached(lib, path, entry["metadata"])
        if a.variant == "arrow" and a.condition == "prepared":
            frame.prepare_arrow_shared()
        bases.append(frame)
    for i in range(a.warmup):
        frame = bases[i % len(bases)].clone()
        prepare_producer(frame, a)
        arrays = frame.arrays(a.variant)
        result, _ = process(arrays, a.cameras, model)
        if lib.consume(result) != entries[i % len(entries)]["expected_crc32"]:
            raise AssertionError("warmup mismatch")
        del frame, arrays, result
    a.output.mkdir(parents=True)
    (a.output / "environment.json").write_text(
        json.dumps(
            {
                "runtime": identity,
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                },
                "cache_manifest_sha256": digest(a.input_cache / "cache.json"),
                "providers": model.session.get_providers(),
                "harness_sha256": {
                    name: digest(Path(__file__).with_name(name))
                    for name in (
                        "python_sustained.py",
                        "python_boundary.py",
                        "run_python_boundary.py",
                    )
                },
                "cgroup": {
                    name: Path("/sys/fs/cgroup", name).read_text().strip()
                    for name in ("cpu.max", "memory.max", "memory.swap.max")
                    if Path("/sys/fs/cgroup", name).is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{a.variant}/{a.condition}: {len(entries) * a.passes} bundles, rate={a.rate or 'unpaced'}, queue={a.queue_size}/{a.queue_policy}",
        flush=True,
    )
    cpu_start = time.process_time()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    rows, events, start, end, producer_end = measure(entries, bases, lib, model, a)
    cpu = time.process_time() - cpu_start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    delivered = [r for r in rows if r["status"] == "delivered"]
    duration = (end - start) / 1e9

    def percentiles(key):
        values = [r[key] / 1e6 for r in delivered]
        return (
            dict(
                zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99]).tolist())
            )
            if values
            else None
        )

    summary = dict(
        start_ns=start,
        end_ns=end,
        variant=a.variant,
        condition=a.condition,
        rate_bundles_s=a.rate,
        queue_size=a.queue_size,
        queue_policy=a.queue_policy,
        passes=a.passes,
        offered=len(rows),
        delivered=len(delivered),
        dropped=len(rows) - len(delivered),
        active_wall_s=duration,
        completed_bundles_s=len(delivered) / duration,
        completed_payload_gb_s=sum(r["payload_bytes"] for r in delivered)
        / duration
        / 1e9,
        producer_finished_s=(producer_end - start) / 1e9,
        drain_s=max(0, end - producer_end) / 1e9,
        latency_ms={
            key: percentiles(key)
            for key in (
                "source_to_result_ns",
                "admission_lag_ns",
                "source_lateness_ns",
                "backpressure_ns",
                "queue_wait_ns",
                "service_ns",
            )
        },
        producer_copy_s=sum(r["producer_copy_ns"] for r in rows) / 1e9,
        producer_copy_bytes=sum(r["producer_copy_bytes"] for r in rows),
        total_backpressure_s=sum(r["backpressure_ns"] for r in rows) / 1e9,
        cpu_s=cpu,
        cpu_core_equivalents=cpu / duration,
        process_peak_rss_bytes=usage.ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
        minor_page_faults=usage.ru_minflt - usage_start.ru_minflt,
        major_page_faults=usage.ru_majflt - usage_start.ru_majflt,
        **queue_statistics(events, start, end, a.queue_size),
        limitations=[
            "One producer and one Python worker; each variant runs independently.",
            "Prepared cache excludes Waymo decoding/geometry and reference inference; no per-frame paired barriers.",
            "Prepared Arrow wraps cached buffers without copying; conversion-inclusive Arrow materializes buffers on the producer.",
            "Native-copy copies at the Python boundary in both conditions; native_producer_copy copies on the producer with matched Arrow allocation alignment and capacity rounding.",
            "Queue depth counts waiting items; one worker item and one producer item are additional.",
            "Queued payload bytes measure logical backlog, not incremental allocated memory; mapped inputs are shared.",
            "Memory-mapped page faults and memory limits can affect throughput; cache is not guaranteed RAM resident.",
            "Peak RSS includes pre-run mapping/warmup; CPU/page-fault deltas cover active execution.",
            "Fixed-rate latency begins at scheduled arrival, including backpressure; rate 0 begins at producer preparation.",
        ],
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (a.output / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["sequence"]))
    with (a.output / "queue.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_ns",
                "event",
                "sequence",
                "depth",
                "queued_payload_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(events)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (a.output / "summary.md").write_text(
        f"# {a.variant} — sustained {a.condition}\n\n"
        f"Completed **{summary['completed_bundles_s']:.2f} bundles/s**; {len(delivered)} delivered, {summary['dropped']} dropped.\n\n"
        f"Queue peak **{summary['peak_queue_depth']} / {a.queue_size}**; mean {summary['mean_queue_depth']:.2f}.\n\n"
        f"Source-to-result latency (ms): {summary['latency_ms']['source_to_result_ns']}\n\n"
        + "\n".join("- " + x for x in summary["limitations"])
        + "\n"
    )
    (a.output / "completed.json").write_text(
        json.dumps(
            {
                "validated": True,
                "offered": len(rows),
                "delivered": len(delivered),
                "dropped": summary["dropped"],
            }
        )
        + "\n"
    )
    print(
        f"Completed {summary['completed_bundles_s']:.2f} bundles/s; {summary['dropped']} drops. Results: {a.output}",
        flush=True,
    )
