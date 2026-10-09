"""Summarize scoped hardware events without interpreting counts as DRAM bytes."""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

def summarize(root):
    lines=['# Arrow memory-access profile', '',
           'Per-thread user-space counters around prebuilt fusion only. These are event counts, NOT DRAM bytes. Hardware cache-event semantics depend on the CPU/kernel.', '',
           'Counts include filtering, range-vector allocation, sorting, and a small counter/timer wrapper. Empty counter-bracket counts are retained in the availability JSON; no subtraction is applied.', '']
    for split in ('training','validation'):
        path=root/split
        lines += [f'## {split}', '']
        availability=path/'counter-availability.json'
        if availability.exists():
            for event in json.loads(availability.read_text()):
                if not event['available']:
                    lines += [f'- Unavailable `{event["event"]}`: {event["error"]}']
        if not (path/'completed.json').exists():
            lines += ['', 'INCOMPLETE: no validated profile. See the Slurm log and counter availability file.', '']
            continue
        meta=json.loads((path/'completed.json').read_text())
        with (path/'counters.csv').open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=meta['frames']*meta['repeats']*meta['available_events']*3:
            raise ValueError('unexpected counter row count')
        groups=defaultdict(list);seen=set();excluded=0
        for r in rows:
            key=(r['timestamp'],r['event'],r['repeat'],r['representation'])
            if key in seen:raise ValueError('duplicate profile sample')
            seen.add(key)
            enabled=int(r['time_enabled_ns']);running=int(r['time_running_ns'])
            if not 0<running<=enabled:raise ValueError('invalid counter scheduling time')
            if running/enabled<0.99 or int(r['point_examinations'])==0:
                excluded+=1;continue
            groups[(r['timestamp'],r['event'],r['representation'])].append(r)
        med={k:statistics.median(int(r['count'])/int(r['point_examinations']) for r in rs) for k,rs in groups.items()}
        lines += ['',f'Validated {meta["frames"]} frames; excluded {excluded} rows with <99% running coverage or zero point examinations. No multiplex scaling is applied.', '',
                  '| Event | Representation | Median events / point examination | Paired count / native | Paired frames |',
                  '|---|---|---:|---:|---:|']
        events=sorted({r['event'] for r in rows})
        for event in events:
            for variant in ('native_aos','native_soa','arrow'):
                values=[(k,v) for k,v in med.items() if k[1:]==(event,variant)]
                ratios=[v/med[(k[0],event,'native_aos')] for k,v in values if med.get((k[0],event,'native_aos'),0)>0]
                count=f'{statistics.median(v for _,v in values):.6f}' if values else 'unavailable'
                ratio=f'{statistics.median(ratios):.3f}' if ratios else 'unavailable'
                lines.append(f'| {event} | {variant} | {count} | {ratio} | {len(ratios)} |')
        lines+=['','Each frame is summarized over repetitions before combining frames. Events were collected in separate executions; ratios between different event types are not simultaneous measurements.','']
    lines += ['Interpretation: more instructions with similar cache behavior would support an instruction-overhead explanation. Fewer cache misses alone would not prove lower DRAM traffic. No causal conclusion is generated automatically.','']
    (root/'summary.md').write_text('\n'.join(lines))
    print(root/'summary.md')

if __name__=='__main__':summarize(Path(sys.argv[1]))
