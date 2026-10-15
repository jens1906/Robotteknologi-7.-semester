#!/usr/bin/env python3
"""Run the detector over MOT17 sequences through ROS and tabulate the ground-truth scores.

Each run is the normal `ros2 run vision detector ... -p use_dataset:=true` (real time, frames are
dropped like on the robot), so it takes about the length of the clip. Needs ROS and the workspace
sourced; `scripts/vision.sh score ...` does that.

Usage: scripts/score_tracking.py [--seqs 02 10 13] [--runs 3] [--variant "NAME=k:=v k2:=v2" ...]
  --seqs     MOT17 sequences in data/ (default: every data/mot17_* with a gt/gt.txt)
  --runs     repetitions per sequence and variant, results are averaged (default 3)
  --variant  a named set of detector parameters. Default: bytetrack (the old tracker) and current
             (the packaged config). Example:
             --variant "keep10=occlusion_keep_s:=10" --variant "osnet=reid_model:=osnet"
             Use NAME alone (no =) for "defaults".
"""
import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
import time

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = re.compile(r'IDF1 ([\d.]+)% \| ID switches (\d+) \| IDs per person ([\d.]+) \| '
                    r'fragmentations (\d+) \| mostly tracked (\d+)%')
FPS = re.compile(r'processed ([\d.]+) fps')
FIELDS = ('idf1', 'switches', 'ids_per_person', 'fragmentations', 'mostly_tracked', 'fps')


def parse_variant(text):
    name, _, params = text.partition('=')
    return name.strip(), params.split()


def default_seqs():
    found = glob.glob(os.path.join(WS, 'data', 'mot17_??', 'gt', 'gt.txt'))
    return sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))[-2:] for p in found)


def run_once(seq, params, timeout):
    cmd = ['ros2', 'run', 'vision', 'detector', '--ros-args',
           '-p', 'use_dataset:=true', '-p', f'video_path:=data/mot17_{seq}', '-p', 'show:=false']
    for p in params:
        cmd += ['-p', p]
    try:
        out = subprocess.run(cmd, cwd=WS, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return None
    m, f = RESULT.search(out), FPS.search(out)
    if not m or not f:
        return None
    idf1, sw, per, frag, mt = m.groups()
    return dict(idf1=float(idf1), switches=int(sw), ids_per_person=float(per),
                fragmentations=int(frag), mostly_tracked=float(mt), fps=float(f.group(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seqs', nargs='*', help='sequence numbers, e.g. 02 10 13')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--variant', action='append', default=[])
    ap.add_argument('--timeout', type=float, default=300.0, help='seconds per run')
    args = ap.parse_args()

    seqs = args.seqs or default_seqs()
    variants = [parse_variant(v) for v in args.variant] or [
        ('bytetrack', ['tracker_cfg:=bytetrack.yaml']), ('current', [])]
    if not seqs:
        sys.exit('no data/mot17_*/gt/gt.txt found, run scripts/fetch_mot17.py first')

    results = {}                          # (variant, seq) -> list of run dicts
    total = len(seqs) * len(variants) * args.runs
    done = 0
    for seq in seqs:
        for name, params in variants:
            for run in range(args.runs):
                done += 1
                print(f'[{done}/{total}] MOT17-{seq} {name} run {run + 1}', flush=True)
                r = run_once(seq, params, args.timeout)
                if r is None:
                    print('   no result (run failed or no ground truth)', flush=True)
                    continue
                results.setdefault((name, seq), []).append(r)
                print(f'   IDF1 {r["idf1"]:.1f}%  switches {r["switches"]}  '
                      f'IDs/person {r["ids_per_person"]:.2f}  {r["fps"]:.1f} fps', flush=True)

    def mean(rs, key):
        return statistics.fmean(r[key] for r in rs)

    lines = [f'{"variant":<14}{"seq":>5}{"IDF1 %":>9}{"ID sw":>8}{"IDs/person":>12}'
             f'{"frag":>7}{"MT %":>6}{"fps":>7}{"runs":>6}']
    per_variant = {}
    for name, _ in variants:
        for seq in seqs:
            rs = results.get((name, seq))
            if not rs:
                continue
            per_variant.setdefault(name, []).append({k: mean(rs, k) for k in FIELDS})
            lines.append(f'{name:<14}{seq:>5}{mean(rs, "idf1"):>9.1f}{mean(rs, "switches"):>8.0f}'
                         f'{mean(rs, "ids_per_person"):>12.2f}{mean(rs, "fragmentations"):>7.0f}'
                         f'{mean(rs, "mostly_tracked"):>6.0f}{mean(rs, "fps"):>7.1f}{len(rs):>6}')
    lines.append('')
    lines.append('mean over sequences')
    for name, rows in per_variant.items():
        lines.append(f'{name:<14}{"all":>5}{mean(rows, "idf1"):>9.1f}{mean(rows, "switches"):>8.0f}'
                     f'{mean(rows, "ids_per_person"):>12.2f}{mean(rows, "fragmentations"):>7.0f}'
                     f'{mean(rows, "mostly_tracked"):>6.0f}{mean(rows, "fps"):>7.1f}{len(rows):>6}')
    text = '\n'.join(lines)
    print('\n' + text)
    out = os.path.expanduser(time.strftime('~/.ros/follow_anywhere/scores_%Y%m%d_%H%M%S.txt'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write(text + '\n')
    print(f'\nsaved to {out}')


if __name__ == '__main__':
    main()
