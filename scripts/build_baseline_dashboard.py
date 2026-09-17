#!/usr/bin/env python3
"""Aggregate normalized pipes-rs/Dora perception runs into a standalone dashboard."""

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


def kv(path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return values


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    return values[round((len(values) - 1) * fraction)]


def discover(root, suite, pipes_group, dora_group):
    candidates = []
    for framework, folder in (("pipes-rs", pipes_group), ("dora", dora_group)):
        base = root / folder
        if not base.exists():
            continue
        for run in sorted(base.glob(f"{suite}-*")):
            if (run / "perception.csv").exists() and (run / "timing.txt").exists():
                candidates.append((framework, run))
    return candidates


def summarize(framework, run):
    rows = list(csv.DictReader((run / "perception.csv").open()))
    if not rows:
        raise ValueError(f"no measurements in {run}")
    timing = kv(run / "timing.txt")
    resources = kv(run / "resources.txt")
    setup = kv(run / "perception.setup.txt")
    setup.update(kv(run / "camera_setup_ns.txt"))
    setup.update(kv(run / "lidar_setup_ns.txt"))
    setup.update(kv(run / "submitted.txt"))
    setup.update(kv(run / "delivered.txt"))
    setup.update(kv(run / "dropped.txt"))

    def numbers(column):
        return [int(row.get(column, 0) or 0) for row in rows]

    def ms_mean(column):
        return statistics.mean(numbers(column)) / 1e6

    elapsed = int(timing["elapsed_ns"]) / 1e9
    tp = sum(numbers("true_positives"))
    fp = sum(numbers("false_positives"))
    fn = sum(numbers("false_negatives"))
    queue_column = "camera_queue_wait_ns" if "camera_queue_wait_ns" in rows[0] else "queue_wait_ns"
    delivered = sum(1 for row in rows if row.get("status", "delivered") == "delivered")
    explicit_dropped = sum(1 for row in rows if row.get("status", "delivered") == "dropped")
    accounting_available = all(key in setup for key in ("submitted", "delivered", "dropped"))
    submitted = int(setup.get("submitted", len(rows)))
    reported_delivered = int(setup.get("delivered", delivered))
    reported_dropped = int(setup.get("dropped", explicit_dropped))
    accounting_valid = (
        submitted == reported_delivered + reported_dropped
        and delivered == reported_delivered
    )
    return {
        "framework": framework,
        "run_id": run.name,
        "frames": len(rows),
        "elapsed_s": elapsed,
        "fps": len(rows) / elapsed,
        "camera_ms": ms_mean("camera_stage_ns"),
        "inference_ms": ms_mean("inference_ns"),
        "lidar_ms": ms_mean("lidar_stage_ns"),
        "fusion_ms": ms_mean("fusion_ns"),
        "queue_p50_ms": percentile(numbers(queue_column), 0.5) / 1e6,
        "queue_p95_ms": percentile(numbers(queue_column), 0.95) / 1e6,
        "age_p50_ms": percentile(numbers("end_to_end_ns"), 0.5) / 1e6,
        "age_p95_ms": percentile(numbers("end_to_end_ns"), 0.95) / 1e6,
        "camera_setup_ms": int(setup.get("camera_setup_ns", 0)) / 1e6,
        "lidar_setup_ms": int(setup.get("lidar_setup_ns", 0)) / 1e6,
        "cpu_s": int(resources.get("cpu_usage_usec", 0)) / 1e6,
        "memory_peak_mb": int(resources.get("memory_peak_bytes", 0)) / 1048576,
        "detections": sum(numbers("detections")),
        "points": sum(numbers("points")),
        "submitted": submitted,
        "delivered": reported_delivered,
        "dropped": reported_dropped,
        "accounting_available": int(accounting_available),
        "accounting_valid": int(accounting_valid),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
    }


def aggregate(runs):
    result = {}
    numeric = [key for key, value in runs[0].items() if isinstance(value, (int, float))]
    for framework in ("pipes-rs", "dora"):
        selected = [run for run in runs if run["framework"] == framework]
        if not selected:
            continue
        result[framework] = {"runs": len(selected)}
        for key in numeric:
            values = [run[key] for run in selected]
            result[framework][key] = min(values) if key in ("accounting_available", "accounting_valid") else statistics.median(values)
            result[framework][f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
    return result


def write_csv(path, runs):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=runs[0].keys())
        writer.writeheader()
        writer.writerows(runs)


def dashboard(suite, runs, aggregate_data):
    payload = json.dumps({"suite": suite, "runs": runs, "aggregate": aggregate_data})
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sensor Pipeline Baseline · {html.escape(suite)}</title>
<style>
:root{{--ink:#e8edf2;--muted:#8d9aa7;--panel:#121820;--line:#29333d;--pipes:#54d6a1;--dora:#ffb454;--bg:#090d12}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 85% 0,#152431 0,transparent 38%),var(--bg);color:var(--ink);font:14px ui-monospace,SFMono-Regular,Menlo,monospace}}
main{{max-width:1180px;margin:auto;padding:42px 24px 72px}} header{{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:1px solid var(--line);padding-bottom:24px}}
h1{{font:600 clamp(28px,5vw,54px) system-ui,sans-serif;letter-spacing:-.045em;margin:0}} .eyebrow,.muted{{color:var(--muted)}} .stamp{{text-align:right}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:20px}} .panel{{background:linear-gradient(145deg,#151c24,#10161d);border:1px solid var(--line);border-radius:8px;padding:20px;box-shadow:0 16px 50px #0005}}
.hero{{grid-column:span 4}} .wide{{grid-column:span 8}} .half{{grid-column:span 6}} .full{{grid-column:1/-1}} h2{{font:600 13px system-ui,sans-serif;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 18px}}
.value{{font:600 34px system-ui,sans-serif;letter-spacing:-.04em}} .delta{{margin-top:8px}} .good{{color:var(--pipes)}}
.bars{{display:grid;gap:13px}} .bar-row{{display:grid;grid-template-columns:100px 1fr 85px;align-items:center;gap:12px}} .track{{height:10px;background:#080b0f;border-radius:2px;overflow:hidden}} .fill{{height:100%;border-radius:2px}} .pipes{{background:var(--pipes)}} .dora{{background:var(--dora)}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:11px 8px;border-bottom:1px solid var(--line);text-align:right}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-weight:500}} .dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px}}
.info{{display:inline-grid;place-items:center;width:16px;height:16px;border:1px solid var(--muted);border-radius:50%;font-size:10px;cursor:help}}
@media(max-width:800px){{.hero,.wide,.half{{grid-column:1/-1}}header{{display:block}}.stamp{{text-align:left;margin-top:12px}}}}
</style></head><body><main>
<header><div><div class="eyebrow">REPRODUCIBLE SENSOR PIPELINE LAB</div><h1>pipes-rs × Dora</h1></div><div class="stamp">suite / {html.escape(suite)}<br><span class="muted">median values · lower latency is better</span></div></header>
<section class="grid">
<article class="panel hero"><h2>Fastest median runtime <span class="info" title="Wall-clock container time including framework startup and shutdown">i</span></h2><div id="winner" class="value"></div><div id="runtimeDelta" class="delta"></div></article>
<article class="panel hero"><h2>Throughput <span class="info" title="Delivered frames divided by total wall-clock time">i</span></h2><div id="fps" class="value"></div><div class="muted">frames / second</div></article>
<article class="panel hero"><h2>Correctness lock <span class="info" title="Prediction counts and IoU evaluation should match between frameworks">i</span></h2><div id="correctness" class="value"></div><div class="muted">matching detection output</div></article>
<article class="panel wide"><h2>Median runtime by run <span class="info" title="Alternating execution order reduces systematic warm-cache and thermal bias">i</span></h2><div id="runtimeBars" class="bars"></div></article>
<article class="panel hero"><h2>Run variance <span class="info" title="Sample standard deviation of total runtime; repeat count appears below">i</span></h2><div id="variance"></div></article>
<article class="panel half"><h2>Mean compute stages <span class="info" title="Per-frame stage duration; camera includes decode, checksum, inference and Arrow construction">i</span></h2><div id="stageBars" class="bars"></div></article>
<article class="panel half"><h2>Latency and transport <span class="info" title="Frame age is source-stage start to fusion completion; queue is message send to fusion receipt">i</span></h2><div id="latencyBars" class="bars"></div></article>
<article class="panel full"><h2>Baseline ledger</h2><table id="ledger"></table></article>
</section></main><script>
const data={payload}, A=data.aggregate, names=['pipes-rs','dora'], colors={{'pipes-rs':'pipes','dora':'dora'}};
const fmt=(n,d=2)=>Number(n).toFixed(d), lower=(a,b)=>a<=b?'pipes-rs':'dora';
const p=A['pipes-rs'],d=A.dora,sameWork=p.submitted===d.submitted&&p.delivered===d.delivered&&p.dropped===d.dropped,
valid=p.accounting_valid===1&&d.accounting_valid===1&&sameWork,w=lower(p.elapsed_s,d.elapsed_s),other=w==='pipes-rs'?'dora':'pipes-rs';
document.querySelector('#winner').textContent=valid?w:'INVALID'; document.querySelector('#runtimeDelta').textContent=valid?(fmt((1-A[w].elapsed_s/A[other].elapsed_s)*100,1)+'% less wall time'):'work/accounting mismatch';
document.querySelector('#fps').textContent=valid?fmt(Math.max(p.fps,d.fps)):'—'; document.querySelector('#correctness').textContent=(valid&&p.detections===d.detections&&p.points===d.points)?'MATCH':'CHECK';
function bars(id,items){{const max=Math.max(...items.map(x=>x[2]));document.querySelector(id).innerHTML=items.map(([label,n,v])=>`<div class="bar-row"><span>${{label}}</span><div class="track"><div class="fill ${{colors[n]}}" style="width:${{100*v/max}}%"></div></div><b>${{fmt(v)}}</b></div>`).join('')}}
bars('#runtimeBars',data.runs.map(r=>[r.run_id,r.framework,r.elapsed_s]));
document.querySelector('#variance').innerHTML=names.map(n=>`<p><span class="dot ${{colors[n]}}"></span>${{n}} · ${{fmt(A[n].elapsed_s_stdev)}} s<br><span class="muted">${{A[n].runs}} runs</span></p>`).join('');
bars('#stageBars',names.flatMap(n=>[['camera '+n,n,A[n].camera_ms],['LiDAR '+n,n,A[n].lidar_ms],['fusion '+n,n,A[n].fusion_ms]]));
bars('#latencyBars',names.flatMap(n=>[['age p95 '+n,n,A[n].age_p95_ms],['queue p95 '+n,n,A[n].queue_p95_ms]]));
const rows=[['Runtime (s)','elapsed_s'],['Throughput (FPS)','fps'],['Submitted','submitted'],['Delivered','delivered'],['Dropped','dropped'],['Accounting valid','accounting_valid'],['Camera setup (ms)','camera_setup_ms'],['LiDAR setup (ms)','lidar_setup_ms'],['Camera compute (ms)','camera_ms'],['Inference (ms)','inference_ms'],['LiDAR compute (ms)','lidar_ms'],['Fusion (ms)','fusion_ms'],['Queue p50 (ms)','queue_p50_ms'],['Queue p95 (ms)','queue_p95_ms'],['Frame age p50 (ms)','age_p50_ms'],['Frame age p95 (ms)','age_p95_ms'],['CPU time (s)','cpu_s'],['Peak memory (MiB)','memory_peak_mb'],['Precision','precision'],['Recall','recall']];
document.querySelector('#ledger').innerHTML='<thead><tr><th>Metric</th>'+names.map(n=>`<th><span class="dot ${{colors[n]}}"></span>${{n}}</th>`).join('')+'</tr></thead><tbody>'+rows.map(([label,key])=>`<tr><td>${{label}}</td>${{names.map(n=>`<td>${{fmt(A[n][key],key==='precision'||key==='recall'?3:2)}}</td>`).join('')}}</tr>`).join('')+'</tbody>';
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--pipes-run")
    parser.add_argument("--dora-run")
    parser.add_argument("--pipes-group", default="perception-onnx")
    parser.add_argument("--dora-group", default="dora")
    args = parser.parse_args()
    if bool(args.pipes_run) != bool(args.dora_run):
        parser.error("--pipes-run and --dora-run must be used together")
    candidates = (
        [
            ("pipes-rs", args.results_root / "perception-onnx" / args.pipes_run),
            ("dora", args.results_root / "dora" / args.dora_run),
        ]
        if args.pipes_run
        else discover(args.results_root, args.suite, args.pipes_group, args.dora_group)
    )
    runs = [summarize(framework, path) for framework, path in candidates]
    if not runs:
        raise SystemExit(f"no runs found for suite prefix {args.suite!r}")
    missing = {"pipes-rs", "dora"} - {run["framework"] for run in runs}
    if missing:
        raise SystemExit(f"missing framework runs: {', '.join(sorted(missing))}")
    output = args.results_root / "baseline" / args.suite
    output.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate(runs)
    write_csv(output / "runs.csv", runs)
    (output / "summary.json").write_text(json.dumps(aggregates, indent=2) + "\n")
    (output / "dashboard.html").write_text(dashboard(args.suite, runs, aggregates))
    print(f"dashboard: {output / 'dashboard.html'}")


if __name__ == "__main__":
    main()
