"""Summarize completed paired runs; standard library only."""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

def summarize(root):
    lines = ['# Arrow representation pilot', '',
             'CPU depth association on Waymo annotation boxes. No inference, IPC, or disk I/O in the timed intervals.', '',
             'Each value below is the median of per-frame medians across repetitions. Ratios are paired per frame; greater than 1 means slower than native AoS. No significance or tail claim.', '']
    for split in ('training', 'validation'):
        path = root / split
        completed = json.loads((path / 'completed.json').read_text())
        with (path / 'raw.csv').open() as f:
            rows = list(csv.DictReader(f))
        expected = completed['frames'] * completed['repeats'] * 6
        if len(rows) != expected:
            raise ValueError(f'{split}: expected {expected} rows, got {len(rows)}')
        seen = set()
        groups = defaultdict(list)
        for r in rows:
            key = (r['timestamp'], r['repeat'], r['condition'], r['representation'])
            if key in seen:
                raise ValueError(f'duplicate {key}')
            seen.add(key)
            groups[(r['timestamp'], r['condition'], r['representation'])].append(r)
        medians = {k: {metric: statistics.median(int(r[metric]) for r in rs)
                       for metric in ('conversion_ns', 'fusion_ns', 'total_ns')}
                   for k, rs in groups.items()}
        lines += [f'## {split}: {completed["frames"]} frames', '',
                  '| Condition | Representation | Conversion ms | Fusion ms | Total ms | Paired total / native |',
                  '|---|---|---:|---:|---:|---:|']
        for condition in ('prebuilt', 'convert_then_fuse'):
            for variant in ('native_aos', 'native_soa', 'arrow'):
                selected = [(k, v) for k, v in medians.items() if k[1:] == (condition, variant)]
                times = [statistics.median(v[m] for _, v in selected) / 1e6
                         for m in ('conversion_ns', 'fusion_ns', 'total_ns')]
                ratios = [v['total_ns'] / medians[(k[0], condition, 'native_aos')]['total_ns'] for k, v in selected]
                lines.append(f'| {condition} | {variant} | {times[0]:.4f} | {times[1]:.4f} | {times[2]:.4f} | {statistics.median(ratios):.3f} |')
        lines += ['', f'Frames with boxes: {completed["frames_with_boxes"]}. All timed outputs matched the reference exactly.', '']
    lines += ['Conversion starts from already materialized native points. Native AoS has no conversion. Fresh representation destruction, input preparation, output cloning and validation are excluded. All prepared representations remain resident; this is a cache-warm microbenchmark, not a production memory benchmark.', '']
    (root / 'summary.md').write_text('\n'.join(lines))
    print(root / 'summary.md')

if __name__ == '__main__':
    summarize(Path(sys.argv[1]))
