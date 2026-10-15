#!/usr/bin/env python3
"""Run independent Docker trials of the matched producer-copy control."""

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark-results/python-sustained"
VARIANTS = ("native_shared", "native_producer_copy", "arrow")
METRICS = (
    "throughput_bundles_s",
    "runtime_s",
    "latency_p50_ms",
    "latency_p95_ms",
    "cpu_s",
    "peak_rss_bytes",
    "dropped",
    "producer_copy_s",
)


def schedule(repeats, rates, seed):
    rng = random.Random(seed)
    jobs = []
    for rate_index, rate in enumerate(rates):
        for repeat in range(1, repeats + 1):
            variants = list(VARIANTS)
            rng.shuffle(variants)
            for variant in variants:
                jobs.append(
                    {
                        "rate_index": rate_index,
                        "rate": rate,
                        "repeat": repeat,
                        "variant": variant,
                    }
                )
    return jobs


def aggregate(rows):
    result = []
    for rate in sorted({r["rate"] for r in rows}):
        for variant in VARIANTS:
            runs = [r for r in rows if r["rate"] == rate and r["variant"] == variant]
            if not runs:
                continue
            item = {"rate": rate, "variant": variant, "runs": len(runs)}
            for metric in METRICS:
                values = [r[metric] for r in runs]
                item.update(
                    {
                        f"{metric}_median": statistics.median(values),
                        f"{metric}_min": min(values),
                        f"{metric}_max": max(values),
                    }
                )
            result.append(item)
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="New experiment directory name")
    p.add_argument(
        "--cache-id",
        default="producer-copy",
        help="Cache prepared with the updated image",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent runs of EACH variant per rate",
    )
    p.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[0],
        help="0: unpaced; positive: bundles/s",
    )
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--queue-size", type=int, default=8)
    p.add_argument(
        "--queue-policy",
        choices=["backpressure", "drop-oldest"],
        default="backpressure",
    )
    p.add_argument("--cpus", type=float, default=2)
    p.add_argument("--memory", default="4g")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial order without running Docker or writing results",
    )
    a = p.parse_args()
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) for x in (a.name, a.cache_id)
    ):
        p.error("name and cache-id must be simple directory names")
    if (
        min(a.repeats, a.passes, a.queue_size, a.threads) < 1
        or min(a.frames, a.warmup) < 0
    ):
        p.error("invalid count")
    if (
        not math.isfinite(a.cpus)
        or a.cpus <= 0
        or any(not math.isfinite(r) or r < 0 for r in a.rates)
    ):
        p.error("invalid CPU count or rate")
    if len(set(a.rates)) != len(a.rates):
        p.error("duplicate rates")
    jobs = schedule(a.repeats, a.rates, a.seed)
    if a.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    output = RESULTS / a.name
    if output.exists():
        p.error("experiment already exists; choose a new --name")
    cache = ROOT / "data/python-boundary-cache" / a.cache_id
    if not (cache / "completed.json").is_file():
        p.error(
            "prepare the cache with the updated Docker image first; see experiment README"
        )

    compose = ["docker", "compose", "-f", str(ROOT / "compose.python-boundary.yaml")]
    env = dict(
        os.environ,
        CACHE_ID=a.cache_id,
        CONDITION="conversion_inclusive",
        FRAME_LIMIT=str(a.frames),
        PASSES=str(a.passes),
        WARMUP=str(a.warmup),
        QUEUE_SIZE=str(a.queue_size),
        QUEUE_POLICY=a.queue_policy,
        BENCHMARK_CPUS=str(a.cpus),
        BENCHMARK_MEMORY=a.memory,
        MODEL_THREADS=str(a.threads),
    )

    def image_id():
        return subprocess.check_output(
            [
                "docker",
                "image",
                "inspect",
                "pipes-python-boundary:local",
                "--format",
                "{{.Id}}",
            ],
            text=True,
            cwd=ROOT,
        ).strip()

    expected_image = image_id()
    output.mkdir(parents=True)
    (output / "plan.json").write_text(
        json.dumps(
            {"arguments": vars(a), "image": expected_image, "order": jobs}, indent=2
        )
        + "\n"
    )
    rows, expected_identity = [], None
    for index, job in enumerate(jobs):
        if image_id() != expected_image:
            raise RuntimeError("benchmark image changed during the experiment")
        run_id = f"{a.name}/rate-{job['rate_index']}/repeat-{job['repeat']:02d}"
        trial_env = dict(
            env, RUN_ID=run_id, RATE=str(job["rate"]), NATIVE_VARIANT=job["variant"]
        )
        service = "arrow" if job["variant"] == "arrow" else "native"
        print(f"Trial {index + 1}/{len(jobs)}: {job}", flush=True)
        # A fresh container per trial; no simultaneous competitors on the host.
        # Docker failures propagate, unlike an unchecked `compose up` exit status.
        subprocess.run(
            [*compose, "run", "--rm", "--no-deps", service],
            env=trial_env,
            cwd=ROOT,
            check=True,
        )
        trial = RESULTS / run_id / "conversion_inclusive" / job["variant"]
        completed = json.loads((trial / "completed.json").read_text())
        s = json.loads((trial / "summary.json").read_text())
        provenance = json.loads((trial / "environment.json").read_text())
        identity = {
            k: provenance[k]
            for k in (
                "runtime",
                "cache_manifest_sha256",
                "providers",
                "harness_sha256",
                "cgroup",
            )
        }
        if expected_identity is None:
            expected_identity = identity
        if identity != expected_identity:
            raise RuntimeError(
                "runtime, inputs, harness or resource limits differ between trials"
            )
        if (
            not completed["validated"]
            or completed["offered"] != completed["delivered"] + completed["dropped"]
        ):
            raise RuntimeError("incomplete trial")
        if s["variant"] != job["variant"] or s["condition"] != "conversion_inclusive":
            raise RuntimeError("unexpected variant/condition")
        latency = s["latency_ms"]["source_to_result_ns"]
        rows.append(
            dict(
                job,
                throughput_bundles_s=s["completed_bundles_s"],
                runtime_s=s["active_wall_s"],
                latency_p50_ms=latency["p50"],
                latency_p95_ms=latency["p95"],
                cpu_s=s["cpu_s"],
                peak_rss_bytes=s["process_peak_rss_bytes"],
                dropped=s["dropped"],
                producer_copy_s=s["producer_copy_s"],
                result_directory=str(trial.relative_to(ROOT)),
            )
        )
        write_csv(output / "trials.csv", rows)
    summary = aggregate(rows)
    write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Producer-copy comparison",
        "",
        "Medians across independent runs; ranges are observed min–max, not confidence intervals.",
        "",
        "| Arrival rate | Variant | Runs | Throughput bundles/s (range) | Runtime s | Median latency ms | p95 latency ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        lines.append(
            f"| {s['rate'] or 'unpaced'} | {s['variant']} | {s['runs']} | "
            f"{s['throughput_bundles_s_median']:.3f} ({s['throughput_bundles_s_min']:.3f}–{s['throughput_bundles_s_max']:.3f}) | "
            f"{s['runtime_s_median']:.2f} | {s['latency_p50_ms_median']:.2f} | {s['latency_p95_ms_median']:.2f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "completed.json").write_text(
        json.dumps({"validated": True, "trials": len(rows)}) + "\n"
    )
    print(f"Results: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
