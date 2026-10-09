"""Run and validate paired native-channel / composed-ROS executions."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

SEGMENTS = [('training','10023947602400723454_1120_000_1140_000'),
            ('validation','1024360143612057520_3580_000_3600_000')]

def inspect(path, frames, capacity):
    with path.open() as f:
        rows = [{k:int(v) for k,v in r.items()} for r in csv.DictReader(f)]
    if len(rows)!=frames or [r['index'] for r in rows]!=list(range(frames)):
        raise ValueError('incomplete or reordered frames')
    if len({r['timestamp'] for r in rows}) != frames:
        raise ValueError('duplicate timestamps')
    events=[]
    for r in rows:
        a,b,c=(r[k] for k in ('started_ns','received_ns','finished_ns'))
        if not a<=b<=c or (b-a,c-b,c-a)!=(r['handoff_ns'],r['fusion_ns'],r['result_ns']):
            raise ValueError('invalid clock ordering or intervals')
        events.extend([(a,1),(c,-1)])
    active=0
    for _,delta in sorted(events):
        active+=delta
        if not 0<=active<=capacity:
            raise ValueError('outstanding work exceeded capacity')
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=20)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=4)
    a=p.parse_args()
    if not 0<=a.warmup<a.frames or a.repeats<1: p.error('invalid frame/warmup/repeat count')
    a.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    # Capacity 1 exposes single-result latency; capacity 4 allows bounded overlap.
    for split,segment in SEGMENTS:
        for capacity in (1,4):
            reference=None
            for repeat in range(a.repeats):
                order=('native','ros2') if repeat%2==0 else ('ros2','native')
                for mode in order:
                    dest=a.output/f'{split}-q{capacity}-r{repeat}-{mode}'
                    dest.mkdir()
                    env=os.environ.copy()
                    env['RMW_IMPLEMENTATION']='rmw_cyclonedds_cpp'
                    env['ROS_AUTOMATIC_DISCOVERY_RANGE']='LOCALHOST'
                    env['ROS_DOMAIN_ID']=str(20+(os.getpid()%180))
                    command=['timeout','--kill-after=10s','300s','/usr/bin/time','-f','%U %S %M','-o',str(dest/'resources.txt'),
                             str(a.binary),mode,str(a.data_root),split,segment,str(a.frames),str(capacity),str(dest/'raw.csv'),str(dest/'run.json')]
                    started=time.monotonic()
                    with (dest/'process.log').open('w') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    wall=time.monotonic()-started
                    rows=inspect(dest/'raw.csv',a.frames,capacity)
                    identity=[(r['timestamp'],r['points'],r['boxes']) for r in rows]
                    if reference is None:reference=identity
                    if reference!=identity:raise ValueError('input mismatch across configurations')
                    retained=rows[a.warmup:]
                    duration=(max(r['finished_ns'] for r in retained)-min(r['started_ns'] for r in retained))/1e9
                    cpu_user,cpu_system,max_rss=map(float,(dest/'resources.txt').read_text().split())
                    summaries.append(dict(split=split,capacity=capacity,repeat=repeat,mode=mode,retained_frames=len(retained),
                        active_fps=len(retained)/duration,median_result_ms=statistics.median(r['result_ns'] for r in retained)/1e6,
                        median_handoff_ms=statistics.median(r['handoff_ns'] for r in retained)/1e6,
                        median_fusion_ms=statistics.median(r['fusion_ns'] for r in retained)/1e6,
                        process_wall_s=wall,process_cpu_s=cpu_user+cpu_system,process_max_rss_kib=max_rss))
                    print(dest,flush=True)
    (a.output/'comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
    lines=['# Native channels versus composed ROS 2 pilot','',
           'Prepared inputs and one shared Rust kernel. This is not the full perception graph.', '',
           'CPU seconds and wall time cover each whole process, including input loading and validation. They do not isolate executor CPU use.', '',
           '| Split | Capacity | Run | Mode | Active FPS | Median result ms | Median handoff ms | CPU seconds |',
           '|---|---:|---:|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {s["split"]} | {s["capacity"]} | {s["repeat"]} | {s["mode"]} | {s["active_fps"]:.2f} | {s["median_result_ms"]:.4f} | {s["median_handoff_ms"]:.4f} | {s["process_cpu_s"]:.2f} |')
    lines+=['','First five frames excluded by default within every process. Warmup sufficiency and shared-host noise remain unestablished. No winner or statistical significance is inferred automatically.','']
    (a.output/'summary.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
