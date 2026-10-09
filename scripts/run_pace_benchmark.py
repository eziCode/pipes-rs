#!/usr/bin/env python3
"""Native, sequential full-perception comparison on both downloaded Waymo segments."""
import argparse
import csv
import hashlib
import html
import math
import statistics
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path.home() / "scratch/pipes-rs"
MODEL_SHA = "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"
COMPONENTS = ("camera_image", "camera_box", "lidar", "lidar_camera_projection", "lidar_calibration")
FIELDS = ("timestamp_micros", "status", "camera_width", "camera_height", "camera_checksum",
          "points", "detections", "ground_truth", "true_positives", "false_positives",
          "false_negatives", "depth_matched")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_metrics(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty metrics: {path}")
    timestamps = [r["timestamp_micros"] for r in rows]
    if len(timestamps) != len(set(timestamps)):
        raise RuntimeError(f"duplicate timestamps: {path}")
    if any(r["status"] != "delivered" for r in rows):
        raise RuntimeError(f"undelivered frames: {path}")
    for row in rows:
        ready, finished = int(row["pair_ready_ns"]), int(row["fusion_finished_ns"])
        for sensor in ("camera", "lidar"):
            start, send, receive = [int(row[f"{sensor}_{event}_ns"]) for event in ("started", "sent", "received")]
            if not 0 < start <= send <= receive <= ready <= finished:
                raise RuntimeError(f"invalid {sensor} timing order: {path}")
            if int(row[f"{sensor}_queue_wait_ns"]) != receive - send:
                raise RuntimeError(f"handoff timing mismatch: {path}")
            if int(row[f"{sensor}_pair_wait_ns"]) != ready - receive:
                raise RuntimeError(f"pair-wait timing mismatch: {path}")
        if int(row["end_to_end_ns"]) != finished - min(int(row["camera_started_ns"]), int(row["lidar_started_ns"])):
            raise RuntimeError(f"frame age mismatch: {path}")
    for sensor in ("camera", "lidar"):
        events = [(int(r[f"{sensor}_started_ns"]), 1) for r in rows]
        events += [(int(r["fusion_finished_ns"]), -1) for r in rows]
        outstanding = 0
        for _, change in sorted(events):
            outstanding += change
            if not 0 <= outstanding <= 4:
                raise RuntimeError(f"{sensor} processing concurrency exceeds credit limit: {path}")
    return rows


def compare(pipes, dora):
    a, b = read_metrics(pipes), read_metrics(dora)
    if sorted(tuple(r[f] for f in FIELDS) for r in a) != sorted(tuple(r[f] for f in FIELDS) for r in b):
        raise RuntimeError(f"Frame correctness mismatch: {pipes} versus {dora}")
    return len(a)


def compare_values(pipes, dora):
    def load(path):
        with path.with_suffix(".validation.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle]
        values = {r["timestamp_micros"]: r for r in rows}
        expected = {int(r["timestamp_micros"]) for r in read_metrics(path)}
        if len(values) != len(rows) or set(values) != expected:
            raise RuntimeError("validation timestamps do not match delivered frames")
        return values
    left, right = load(pipes), load(dora)
    if left.keys() != right.keys():
        raise RuntimeError("validation frame mismatch")
    for timestamp in left:
        a, b = left[timestamp], right[timestamp]
        if (a["points"], a["points_crc32"]) != (b["points"], b["points_crc32"]):
            raise RuntimeError(f"point-field fingerprint mismatch at {timestamp}")
        aa = sorted(a["detections"], key=lambda d: d["id"])
        bb = sorted(b["detections"], key=lambda d: d["id"])
        if len(aa) != len(bb) or len({d["id"] for d in aa}) != len(aa) or len({d["id"] for d in bb}) != len(bb):
            raise RuntimeError(f"detection count/ID mismatch at {timestamp}")
        for x, y in zip(aa, bb):
            for field in ("id", "kind", "depth_points"):
                if x[field] != y[field]:
                    raise RuntimeError(f"detection {field} mismatch at {timestamp}")
            for field in ("x", "y", "width", "height", "score", "depth_m"):
                u, v = x[field], y[field]
                equal = u is None and v is None
                if u is not None and v is not None:
                    equal = math.isfinite(u) and math.isfinite(v) and math.isclose(u, v, abs_tol=1e-5, rel_tol=1e-6)
                if not equal:
                    raise RuntimeError(f"detection {field} mismatch at {timestamp}: {u} != {v}")
    return len(left)


def validate_accounting(directory, rows):
    values = {}
    for name in ("perception.setup.txt", "submitted.txt", "delivered.txt", "dropped.txt"):
        p = directory / name
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    values[k] = int(v)
    if any(k not in values for k in ("submitted", "delivered", "dropped")):
        raise RuntimeError(f"missing completion accounting: {directory}")
    if values["submitted"] != len(rows) or values["delivered"] != len(rows) or values["dropped"] != 0:
        raise RuntimeError(f"incomplete delivery: {directory}")


def summarize(directory, framework, rows, elapsed, launch, warmup):
    # Exclude both sensors' initial frames, within this measured process.
    camera_order = sorted(rows, key=lambda r: int(r["camera_started_ns"]))
    lidar_order = sorted(rows, key=lambda r: int(r["lidar_started_ns"]))
    excluded = {r["timestamp_micros"] for r in camera_order[:warmup] + lidar_order[:warmup]}
    steady = [r for r in rows if r["timestamp_micros"] not in excluded]
    if not steady:
        raise RuntimeError("no frames left after warmup; increase frame-limit or reduce --warmup-frames")
    def pct(key, q):
        data = sorted(int(r[key]) / 1e6 for r in steady)
        return data[max(0, math.ceil(q * len(data)) - 1)]
    first = min(min(int(r["camera_started_ns"]), int(r["lidar_started_ns"])) for r in steady)
    last = max(int(r["fusion_finished_ns"]) for r in steady)
    summary = dict(framework=framework, run_id=directory.name, frames=len(rows),
                   warmup_frames_excluded=len(excluded), steady_frames=len(steady),
                   wall_seconds=elapsed / 1e9,
                   launch_to_first_result_seconds=(min(int(r["fusion_finished_ns"]) for r in rows) - launch) / 1e9,
                   post_warmup_active_seconds=(last - first) / 1e9,
                   post_warmup_fps=len(steady) * 1e9 / (last - first))
    for key in ("end_to_end_ns", "camera_queue_wait_ns", "lidar_queue_wait_ns", "camera_pair_wait_ns", "lidar_pair_wait_ns"):
        for label, q in (("p50", .5), ("p95", .95), ("p99", .99)):
            summary[f"{key[:-3]}_{label}_ms"] = pct(key, q)
    for key in ("camera_stage_ns", "lidar_stage_ns", "inference_ns", "fusion_ns"):
        summary[f"{key[:-3]}_mean_ms"] = statistics.mean(int(r[key]) / 1e6 for r in steady)
    (directory / "pilot-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_report(results, reports):
    columns = ("run_id", "frames", "steady_frames", "wall_seconds", "launch_to_first_result_seconds",
               "post_warmup_fps", "end_to_end_p50_ms", "end_to_end_p95_ms", "camera_pair_wait_p95_ms")
    parts = ["<!doctype html><meta charset='utf-8'><title>Pipes–Dora pilot</title>",
             "<style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>",
             "<h1>Pipes–Dora validated pilot</h1><p>Contract v2. CPU inference; four outstanding frames per source. "
             "Detailed validation is separate from timed runs. Warmup occurs inside every timed process. "
             "Handoff includes adapters and backpressure; pair wait is separate. Small pilots do not establish tail-latency guarantees.</p>"]
    for report in reports:
        parts.append(f"<h2>{html.escape(report['segment']['split'])}: {html.escape(report['segment']['context_name'])}</h2>")
        parts.append(f"<p>Validated {report['validation_frames']} frames per framework; point fingerprints and detection values matched.</p>")
        parts.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>")
        for pair in report["pairs"]:
            for record in pair["runs"]:
                parts.append("<tr>" + "".join(f"<td>{html.escape(str(round(record[c], 3) if isinstance(record[c], float) else record[c]))}</td>" for c in columns) + "</tr>")
        parts.append("</table>")
    (results / "pilot-report.html").write_text("\n".join(parts))


def run(framework, directory, args, segment, binaries, limit, validation=False):
    directory.mkdir(parents=True)
    metrics = directory / "perception.csv"
    inputs = [args.data_root / segment["split"] / c / (segment["context_name"] + ".parquet")
              for c in COMPONENTS] + [args.model]
    # Same input hashing/warm-cache treatment before every measured process.
    (directory / "input.sha256").write_text("".join(f"{digest(p)}  {p}\n" for p in inputs))
    env = os.environ.copy()
    env.update(DATA_ROOT=str(args.data_root), SPLIT=segment["split"], SEGMENT=segment["context_name"],
               MODEL_PATH=str(args.model), OUTPUT_CSV=str(metrics), OUTPUT_DIR=str(directory),
               FRAME_LIMIT=str(limit), CONFIDENCE="0.3", NMS="0.45", FUSION_WORK_MS="0",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PILOT_VALIDATE="1" if validation else "0")
    if framework == "pipes-rs":
        command = [binaries[framework], "--data-root", str(args.data_root), "--split", segment["split"],
                   "--segment", segment["context_name"], "--onnx-model", str(args.model),
                   "--frame-limit", str(limit), "--stage-queue-size", "4", "--confidence-threshold", "0.3",
                   "--nms-threshold", "0.45", "--perception-csv", str(metrics)]
    else:
        descriptor = (ROOT / "benchmarks/dora/dataflow.container.yml").read_text()
        descriptor = descriptor.replace("/usr/local/bin/pipes-dora-benchmark", json.dumps(binaries["node"]))
        (directory / "dataflow.yml").write_text(descriptor)
        command = [binaries[framework], "run", "dataflow.yml"]
    (directory / "manifest.json").write_text(json.dumps(dict(
        framework=framework, segment=segment, frame_limit=limit, command=command,
        contract_version=2, max_outstanding_per_source=4, clock="CLOCK_MONOTONIC",
        validation=validation, warmup_frames=args.warmup_frames,
        queue_size=4, queue_policy="backpressure", confidence=0.3, nms=0.45,
        inference="CPU", execution="native Slurm", cpu_affinity=sorted(os.sched_getaffinity(0)),
        memory_metric="GNU time maximum RSS; not aggregate simultaneous process-tree memory"), indent=2))
    print(f"Running {framework}: {directory.name}, frame_limit={limit}", flush=True)
    start = time.monotonic_ns()
    with (directory / "output.log").open("w") as log:
        result = subprocess.run(["timeout", "--kill-after=15s", "900s", "/usr/bin/time", "-f", "%U %S %M", "-o", str(directory / "resource-usage.txt"),
                                 *command], cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.monotonic_ns() - start
    if result.returncode:
        raise RuntimeError(f"{framework} failed ({result.returncode}); see {directory / 'output.log'}")
    user, system, rss = (directory / "resource-usage.txt").read_text().split()
    (directory / "resources.txt").write_text(
        f"cpu_usage_usec={round((float(user) + float(system)) * 1e6)}\nmemory_peak_bytes={int(rss) * 1024}\n")
    rows = read_metrics(metrics)
    if any(min(int(r["camera_started_ns"]), int(r["lidar_started_ns"])) < start
           or int(r["fusion_finished_ns"]) > start + elapsed for r in rows):
        raise RuntimeError(f"events are outside the runner's monotonic process interval: {directory}")
    validate_accounting(directory, rows)
    if limit and len(rows) != limit:
        raise RuntimeError(f"expected {limit} frames, got {len(rows)}: {directory}")
    if not validation:
        summarize(directory, framework, rows, elapsed, start, args.warmup_frames)
    (directory / "timing.txt").write_text(f"elapsed_ns={elapsed}\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=SCRATCH / "data/waymo-v2-sample")
    parser.add_argument("--model", type=Path, default=SCRATCH / "models/yolox_nano.onnx")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.frame_limit < 0 or args.warmup_frames < 0:
        parser.error("repeats must be positive; frame-limit and warmup-frames must be nonnegative")
    if args.frame_limit and args.frame_limit <= args.warmup_frames:
        parser.error("frame-limit must exceed warmup-frames")
    if not args.model.exists() and args.model == SCRATCH / "models/yolox_nano.onnx":
        args.model = SCRATCH / "yolox_nano.onnx"
    args.model, args.data_root = args.model.resolve(), args.data_root.resolve()
    if digest(args.model) != MODEL_SHA:
        raise RuntimeError(f"YOLOX checksum mismatch: {args.model}")
    segments = json.loads((args.data_root / "manifest.json").read_text())["segments"]
    if not segments:
        raise RuntimeError("dataset manifest contains no segments")
    for s in segments:
        for c in COMPONENTS:
            path = args.data_root / s["split"] / c / (s["context_name"] + ".parquet")
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"Missing or empty input: {path}")
    print(json.dumps(dict(segments=segments, model=str(args.model), data_root=str(args.data_root),
                          repeats=args.repeats, frame_limit=args.frame_limit,
                          measured_runs=len(segments) * args.repeats * 2), indent=2), flush=True)
    if args.dry_run:
        return
    if not args.results_root:
        parser.error("--results-root is required for an actual run")
    binaries = {name: shutil.which(exe) for name, exe in
                (("pipes-rs", "pipes-rs"), ("dora", "dora"), ("node", "pipes-dora-benchmark"))}
    if not all(binaries.values()):
        raise RuntimeError(f"Missing binaries: {binaries}")
    results = args.results_root.resolve()
    results.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, segment in enumerate(segments):
        prefix = f"segment-{index + 1}-{segment['split']}"
        # Same input coverage as the timed workload, in separate validation processes.
        validated = [run(f, results / "validation" / f"{prefix}-{f}", args, segment, binaries,
                         args.frame_limit, validation=True) for f in ("pipes-rs", "dora")]
        compare(*validated)
        validation_frames = compare_values(*validated)
        pairs = []
        expected_frames = None
        for repeat in range(1, args.repeats + 1):
            outputs = {}
            order = ("pipes-rs", "dora") if repeat % 2 else ("dora", "pipes-rs")
            for framework in order:
                group = "perception-onnx" if framework == "pipes-rs" else "dora"
                outputs[framework] = run(framework, results / group / f"{prefix}-{repeat:02d}-{framework}",
                                         args, segment, binaries, args.frame_limit)
            frames = compare(outputs["pipes-rs"], outputs["dora"])
            compare(validated[0], outputs["pipes-rs"])
            compare(validated[1], outputs["dora"])
            if expected_frames is not None and frames != expected_frames:
                raise RuntimeError("Frame counts changed between repeats")
            expected_frames = frames
            pairs.append(dict(repeat=repeat, frames=frames, correctness="matched", runs=[
                json.loads(outputs[f].with_name("pilot-summary.json").read_text()) for f in ("pipes-rs", "dora")]))
        reports.append(dict(segment=segment, validation_frames=validation_frames, pairs=pairs))
        write_report(results, reports)
        (results / "comparison.json").write_text(json.dumps(reports, indent=2))
    print(f"Completed. Per-segment comparisons: {results / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
