#!/usr/bin/env python3
"""Build a compact cross-scenario index for a baseline matrix."""
import argparse, html, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--suite", required=True)
parser.add_argument("--results-root", type=Path, default=Path("benchmark-results"))
args = parser.parse_args()
base = args.results_root / "baseline"
scenarios = []
for path in sorted(base.glob(f"{args.suite}-*")):
    summary = path / "summary.json"
    if not summary.exists(): continue
    data = json.loads(summary.read_text())
    if "pipes-rs" not in data or "dora" not in data: continue
    name = path.name[len(args.suite)+1:]
    p, d = data["pipes-rs"], data["dora"]
    winner = "pipes-rs" if p["elapsed_s"] <= d["elapsed_s"] else "dora"
    faster = max(p["elapsed_s"], d["elapsed_s"]) / max(.000001, min(p["elapsed_s"], d["elapsed_s"])) - 1
    scenarios.append((name, p, d, winner, faster, path.name))
if not scenarios: raise SystemExit("no scenario summaries found")
rows = "".join(f'''<a class="card" href="../{folder}/dashboard.html"><div class="tag">{html.escape(name)}</div><strong>{winner}</strong><span>{faster*100:.1f}% less wall time</span><dl><dt>pipes-rs</dt><dd>{p['elapsed_s']:.2f}s · {p['fps']:.2f} FPS</dd><dt>Dora</dt><dd>{d['elapsed_s']:.2f}s · {d['fps']:.2f} FPS</dd></dl></a>''' for name,p,d,winner,faster,folder in scenarios)
out = base / args.suite
out.mkdir(parents=True, exist_ok=True)
(out / "index.html").write_text(f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Baseline matrix · {html.escape(args.suite)}</title><style>
:root{{--bg:#090d12;--panel:#121820;--line:#29333d;--ink:#e8edf2;--muted:#8d9aa7;--accent:#54d6a1}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px ui-monospace,monospace}}main{{max-width:1100px;margin:auto;padding:48px 24px}}h1{{font:600 48px system-ui;margin:8px 0 12px;letter-spacing:-.045em}}p,.tag,dt{{color:var(--muted)}}section{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:32px}}.card{{color:inherit;text-decoration:none;background:linear-gradient(145deg,#151c24,#10161d);border:1px solid var(--line);border-radius:8px;padding:22px;transition:.15s}}.card:hover{{transform:translateY(-2px);border-color:var(--accent)}}strong{{display:block;font:600 28px system-ui;margin:16px 0 4px}}span{{color:var(--accent)}}dl{{display:grid;grid-template-columns:80px 1fr;gap:9px;margin:22px 0 0}}dd{{margin:0;text-align:right}}</style></head><body><main><p>REPRODUCIBLE SENSOR PIPELINE LAB</p><h1>Baseline matrix</h1><p>{html.escape(args.suite)} · median of alternating-order runs · select a scenario for details</p><section>{rows}</section></main></body></html>''')
print(f"matrix dashboard: {out / 'index.html'}")
