#!/usr/bin/env python3
"""Paired Arrow/native Rust -> Python experiment on all existing Waymo sensors."""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from python_boundary import Library, Source
from python_perception import Yolox, process

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("native_shared", "native_copy", "arrow")
ORDERS = list(itertools.permutations(VARIANTS))
REQUIRED = ("camera_image", "lidar", "lidar_camera_projection", "lidar_calibration")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root, segments, cameras, lidars, returns, frames):
    result = []
    for segment in segments:
        paths = {
            c: root / segment["split"] / c / (segment["segment"] + ".parquet")
            for c in REQUIRED
        }
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise ValueError(
                "Missing required components (no sensors will be silently skipped):\n"
                + "\n".join(missing)
            )
        identities = {}
        for component, key, selected in [
            ("camera_image", "key.camera_name", cameras),
            ("lidar", "key.laser_name", lidars),
            ("lidar_camera_projection", "key.laser_name", lidars),
        ]:
            table = pq.read_table(
                paths[component], columns=["key.frame_timestamp_micros", key]
            ).to_pydict()
            groups = {}
            for ts, sensor in zip(table["key.frame_timestamp_micros"], table[key]):
                if sensor in selected:
                    if sensor in groups.setdefault(ts, set()):
                        raise ValueError(f"duplicate {component} row")
                    groups[ts].add(sensor)
            timestamps = sorted(groups)
            if frames and len(timestamps) < frames:
                raise ValueError(
                    f"{component}: only {len(timestamps)} timestamps, requested {frames}"
                )
            for ts in timestamps[: frames or None]:
                if groups[ts] != set(selected):
                    raise ValueError(f"{component}: missing requested sensors at {ts}")
            identities[component] = timestamps[: frames or None]
            if component != "camera_image":
                schema = pq.read_schema(paths[component])
                prefix = (
                    "LiDARComponent"
                    if component == "lidar"
                    else "LiDARCameraProjectionComponent"
                )
                for ret in returns:
                    for field in ("values", "shape"):
                        if (
                            f"[{prefix}].range_image_return{ret}.{field}"
                            not in schema.names
                        ):
                            raise ValueError(f"{component} missing return {ret}")
        if not identities["camera_image"] or any(
            v != identities["camera_image"] for v in identities.values()
        ):
            raise ValueError("Sensor timestamp sets do not match")
        result.append(
            dict(
                **segment,
                frames=len(identities["camera_image"]),
                cameras=cameras,
                lidars=lidars,
                returns=returns,
                files={
                    c: {"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
                    for c, p in paths.items()
                },
            )
        )
    return result


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(
            (row["segment"], row["timestamp"], row["condition"], row["variant"]), []
        ).append(row)
    medians = {
        key: {
            metric: statistics.median(r[metric] for r in values)
            for metric in (
                "conversion_ns",
                "boundary_ns",
                "preprocess_ns",
                "inference_ns",
                "postprocess_ns",
                "fusion_ns",
                "return_ns",
                "total_ns",
            )
        }
        for key, values in grouped.items()
    }
    output = []
    for condition in ("conversion_inclusive", "prepared"):
        for variant in VARIANTS:
            selected = [
                (k, v) for k, v in medians.items() if k[2:] == (condition, variant)
            ]
            if not selected:
                continue
            ratios = [
                v["total_ns"]
                / medians[(k[0], k[1], condition, "native_shared")]["total_ns"]
                for k, v in selected
            ]
            output.append(
                {
                    "condition": condition,
                    "variant": variant,
                    "frames": len(selected),
                    "median_ms": {
                        m: statistics.median(v[m] for _, v in selected) / 1e6
                        for m in selected[0][1]
                    },
                    "paired_total_over_native_shared": statistics.median(ratios),
                }
            )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / "data/waymo-v2-sample")
    p.add_argument(
        "--segments",
        type=Path,
        default=ROOT / "experiments/python-boundary/segments.json",
    )
    p.add_argument(
        "--segment",
        action="append",
        help="Restrict manifest to this segment ID; repeatable",
    )
    p.add_argument(
        "--library",
        type=Path,
        default=ROOT
        / "target/release"
        / ("libpipes_rs.dylib" if sys.platform == "darwin" else "libpipes_rs.so"),
    )
    p.add_argument("--model", type=Path, default=ROOT / "models/yolox_nano.onnx")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Timestamp bundles per segment; 0 means complete segments",
    )
    p.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--lidars", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--returns", type=int, nargs="+", default=[1, 2])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Paired: iterations per arm/frame; sustained: warmup bundles before timing",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ONNX intra-op CPU threads; fixed across arms",
    )
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument(
        "--mode", choices=["paired", "prepare", "sustained"], default="paired"
    )
    p.add_argument(
        "--input-cache", type=Path, help="Shared prepared cache for sustained mode"
    )
    p.add_argument(
        "--variant",
        choices=(*VARIANTS, "native_producer_copy"),
        default="native_shared",
    )
    p.add_argument(
        "--condition", choices=["conversion_inclusive", "prepared"], default="prepared"
    )
    p.add_argument(
        "--rate", type=float, default=0, help="Sustained bundles/second; 0 runs unpaced"
    )
    p.add_argument("--queue-size", type=int, default=8)
    p.add_argument(
        "--queue-policy",
        choices=["backpressure", "drop-oldest"],
        default="backpressure",
    )
    p.add_argument(
        "--passes", type=int, default=1, help="Complete scene passes in sustained mode"
    )
    a = p.parse_args()
    if a.frames < 0 or a.repeats < 1 or a.warmup < 0 or a.threads < 1:
        p.error("invalid frame/repeat/warmup/thread count")
    for name, valid in [
        ("cameras", set(range(1, 6))),
        ("lidars", set(range(1, 6))),
        ("returns", {1, 2}),
    ]:
        ids = getattr(a, name)
        if not set(ids) <= valid or len(set(ids)) != len(ids):
            p.error(f"invalid or duplicate {name}")
    import math

    if not math.isfinite(a.rate) or a.rate < 0 or a.queue_size < 1 or a.passes < 1:
        p.error("invalid rate/queue-size/passes")
    if a.variant == "native_producer_copy" and (
        a.mode != "sustained" or a.condition != "conversion_inclusive"
    ):
        p.error(
            "native_producer_copy requires --mode sustained --condition conversion_inclusive"
        )
    if a.mode == "sustained":
        if a.inventory_only:
            p.error("--inventory-only applies to source data, not sustained runs")
        from python_sustained import run

        run(a)
        return
    segments = json.loads(a.segments.read_text())["segments"]
    if a.segment:
        if not set(a.segment) <= {s["segment"] for s in segments}:
            p.error("segment not found in manifest")
        segments = [s for s in segments if s["segment"] in a.segment]
    if not segments:
        p.error("empty segment manifest")
    if len({(s["split"], s["segment"]) for s in segments}) != len(segments):
        p.error("duplicate segments")
    if a.output.exists():
        p.error("output already exists; use a new run directory")
    # Validate all components before running any condition or creating output.
    data = inventory(a.data_root, segments, a.cameras, a.lidars, a.returns, a.frames)
    a.output.mkdir(parents=True)
    (a.output / "inventory.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "segment": d["segment"],
                    "frames": d["frames"],
                    "cameras": d["cameras"],
                    "lidars": d["lidars"],
                    "returns": d["returns"],
                }
                for d in data
            ],
            indent=2,
        ),
        flush=True,
    )
    if a.inventory_only:
        return
    if a.mode == "prepare":
        from python_sustained import prepare

        prepare(a, segments, data)
        return
    library = Library(a.library.resolve())
    model = Yolox(a.model, a.threads)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pyarrow", "Pillow", "onnxruntime")
    }
    (a.output / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": versions,
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                },
                "model_sha256": digest(a.model),
                "library_sha256": digest(a.library),
                "providers": model.session.get_providers(),
                "process_layout": "in-process ctypes; Rust ingestion/ownership, Python CPU inference and fusion",
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with (
        (a.output / "raw.csv").open("w", newline="") as file,
        (a.output / "frames.jsonl").open("w") as frame_file,
    ):
        writer = None
        for segment in segments:
            config = dict(
                data_root=str(a.data_root),
                **segment,
                cameras=a.cameras,
                lidars=a.lidars,
                returns=a.returns,
                frame_limit=a.frames,
            )
            with Source(library, config) as source:
                while True:
                    load_start = time.perf_counter_ns()
                    frame = source.next()
                    source_ns = time.perf_counter_ns() - load_start
                    if frame is None:
                        break
                    meta = frame.metadata
                    frame_file.write(
                        json.dumps(
                            dict(
                                segment=segment["segment"],
                                split=segment["split"],
                                **meta,
                            )
                        )
                        + "\n"
                    )
                    frame_file.flush()
                    baseline = frame.arrays("native_shared")
                    payload = sum(x.nbytes for x in baseline.values())
                    expected, _ = process(baseline, a.cameras, model)
                    expected_checksum = library.consume(expected)
                    input_checksum = hashlib.sha256(
                        b"".join(
                            hashlib.sha256(v.tobytes()).digest()
                            for v in baseline.values()
                        )
                    ).hexdigest()
                    del baseline
                    for iteration in range(a.warmup + a.repeats):
                        conditions = (
                            ("conversion_inclusive", "prepared")
                            if iteration % 2 == 0
                            else ("prepared", "conversion_inclusive")
                        )
                        for condition in conditions:
                            for position, variant in enumerate(
                                ORDERS[(iteration + meta["index"]) % len(ORDERS)]
                            ):
                                if condition == "prepared" and variant == "arrow":
                                    frame.prepare_arrow()
                                started = time.perf_counter_ns()
                                if (
                                    condition == "conversion_inclusive"
                                    and variant == "arrow"
                                ):
                                    frame.prepare_arrow()
                                ready = time.perf_counter_ns()
                                arrays = frame.arrays(variant)
                                received = time.perf_counter_ns()
                                result, timings = process(arrays, a.cameras, model)
                                computed = time.perf_counter_ns()
                                checksum = library.consume(result)
                                finished = time.perf_counter_ns()
                                # Checks are outside timers, including the integrity of every crossed buffer.
                                np.testing.assert_array_equal(result, expected)
                                if checksum != expected_checksum:
                                    raise AssertionError("Rust result checksum differs")
                                crossed = hashlib.sha256(
                                    b"".join(
                                        hashlib.sha256(v.tobytes()).digest()
                                        for v in arrays.values()
                                    )
                                ).hexdigest()
                                if crossed != input_checksum:
                                    raise AssertionError(
                                        "input buffers changed across boundary"
                                    )
                                row = dict(
                                    segment=segment["segment"],
                                    split=segment["split"],
                                    frame=meta["index"],
                                    timestamp=meta["timestamp"],
                                    repeat=iteration - a.warmup,
                                    position=position,
                                    condition=condition,
                                    variant=variant,
                                    cameras=len(a.cameras),
                                    lidars=len(a.lidars),
                                    returns=len(a.returns),
                                    points=arrays["range"].size,
                                    payload_bytes=payload,
                                    known_representation_copy_bytes=payload
                                    if variant == "arrow"
                                    else 0,
                                    known_boundary_copy_bytes=payload
                                    if variant == "native_copy"
                                    else 0,
                                    source_preparation_ns=source_ns,
                                    conversion_ns=ready - started
                                    if condition == "conversion_inclusive"
                                    and variant == "arrow"
                                    else 0,
                                    boundary_ns=received - ready,
                                    **timings,
                                    return_ns=finished - computed,
                                    total_ns=finished - started,
                                    detections=len(result),
                                    depth_matched=int(np.isfinite(result[:, 7]).sum()),
                                    result_crc32=checksum,
                                    input_sha256=input_checksum,
                                )
                                if iteration >= a.warmup:
                                    if writer is None:
                                        writer = csv.DictWriter(file, fieldnames=row)
                                        writer.writeheader()
                                    writer.writerow(row)
                                    rows.append(row)
                                del arrays, result
                                if variant == "arrow":
                                    frame.clear_arrow()
                    print(
                        f"{segment['split']} frame {meta['index']}: {payload / 1e6:.1f} MB, {sum(s['points'] for s in meta['sensors']):,} points, {len(expected)} detections",
                        flush=True,
                    )
                    file.flush()
                    del frame, expected
    expected_rows = sum(d["frames"] for d in data) * a.repeats * len(VARIANTS) * 2
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} measured rows, got {len(rows)}")
    elapsed = time.perf_counter() - wall_start
    report = {
        "contract": 1,
        "validated": True,
        "rows": len(rows),
        "summaries": summarize(rows),
        "process_wall_s": elapsed,
        "process_cpu_s": time.process_time() - cpu_start,
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "limitations": [
            "In-process, single sequential worker; no ROS comparison or IPC.",
            "Source preparation (Parquet/JPEG/range conversion) excluded from both timed conditions.",
            "Prepared excludes Arrow materialization; native_copy still copies on every boundary crossing.",
            "Model preprocessing/layout copies remain in all conditions; CPU inference only.",
            "Process CPU/RSS cover the entire paired run, not individual variants.",
            "Both returns retained; vehicle-relative points have no per-pixel motion compensation.",
            "Per-frame paired repeats are cache-warm; this is not an overload or sustained-throughput test.",
        ],
    }
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Rust–Python representation experiment",
        "",
        "Same Python YOLOX model and depth fusion, all selected sensors. Ratios >1 mean slower than native shared buffers.",
        "",
        "| Condition | Representation | Conversion ms | Boundary ms | Inference ms | Fusion ms | Total ms | Total / native shared |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report["summaries"]:
        m = s["median_ms"]
        lines.append(
            f"| {s['condition']} | {s['variant']} | {m['conversion_ns']:.3f} | {m['boundary_ns']:.3f} | {m['inference_ns']:.3f} | {m['fusion_ns']:.3f} | {m['total_ns']:.3f} | {s['paired_total_over_native_shared']:.3f} |"
        )
    lines += [
        "",
        "Medians across per-frame medians; each ratio is paired by frame. No significance claim.",
        "",
    ] + ["- " + x for x in report["limitations"]]
    (a.output / "summary.md").write_text("\n".join(lines) + "\n")
    (a.output / "completed.json").write_text(
        json.dumps({"validated": True, "rows": len(rows)}) + "\n"
    )
    print("Results:", a.output / "summary.md")


if __name__ == "__main__":
    main()
