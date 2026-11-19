"""Runs the lost-leader search scenarios many times and prints the pass rate per scene.

Each run is `ros2 launch go2_sim2d search_eval.launch.py world:=<scene> seed:=<n>` with its
own ROS_DOMAIN_ID, so --jobs runs go in parallel without seeing each other. The sim runs in
real time: one run takes about a minute.

  ros2 run go2_sim2d search_batch                       # 5 scenes x 50 seeds, 4 jobs
  ros2 run go2_sim2d search_batch --scenes search_t_junction --runs 5 --jobs 1
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from go2_sim2d.search_eval import SCENARIOS

SPEC = {
    'search_corridor': 'reacquired',
    'search_side_door': 'target at door before the robot passes it',
    'search_t_junction': '<= 2 target switches',
    'search_corner_stop': 'reacquired',
    'search_walk_out': 'GIVE_UP within t_max, no crash',
}


def run_one(scene, seed, slot, out_dir, timeout):
    result = os.path.join(out_dir, f'{scene}_{seed}.jsonl')
    env = dict(os.environ, ROS_DOMAIN_ID=str(100 + slot))
    cmd = ['ros2', 'launch', 'go2_sim2d', 'search_eval.launch.py', f'world:={scene}',
           f'seed:={seed}', f'result_file:={result}']
    try:
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        pass
    try:
        with open(result) as f:
            return json.loads(f.readline())
    except (OSError, ValueError):
        return {'world': scene, 'seed': seed, 'passed': False, 'crashed': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenes', nargs='+', default=list(SCENARIOS))
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--out', default=tempfile.mkdtemp(prefix='search_batch_'))
    args = ap.parse_args()
    jobs = [(s, seed) for s in args.scenes for seed in range(args.runs)]
    print(f'{len(jobs)} runs, {args.jobs} at a time, results in {args.out}', flush=True)
    results = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_one, s, seed, i % args.jobs, args.out,
                               SCENARIOS[s]['duration'] + 60)
                   for i, (s, seed) in enumerate(jobs)]
        for i, f in enumerate(futures):
            r = f.result()
            results.append(r)
            verdict = {True: 'pass', False: 'FAIL', None: 'not lost'}[r.get('passed', False)]
            print(f"  [{i + 1}/{len(jobs)}] {r['world']} seed {r['seed']}: {verdict}", flush=True)

    print()
    print(f"{'scene':<20} {'criterion':<44} {'pass':>9} {'need':>6}  {'lost':>5} {'crash':>5} "
          f"{'hits/run (searching)':>21}")
    ok_all = True
    for s in args.scenes:
        rs = [r for r in results if r['world'] == s and r.get('passed', False) is not None]
        n = len(rs)  # runs where the leader was lost
        passed = sum(1 for r in rs if r.get('passed'))
        lost = n
        n_all = sum(1 for r in results if r['world'] == s)
        crashed = sum(1 for r in rs if r.get('crashed'))
        hits = sum(r.get('collisions', 0) for r in rs) / max(n, 1)
        hits_s = sum(r.get('collisions_searching', 0) for r in rs) / max(n, 1)
        need = SCENARIOS[s]['need']
        ok = n > 0 and passed / n >= need
        ok_all &= ok
        print(f'{s:<20} {SPEC[s]:<44} {passed:>3}/{n:<3} {100 * passed / max(n, 1):>3.0f}% '
              f'{100 * need:>4.0f}%  {lost:>2}/{n_all:<2} {crashed:>5} {hits:>9.1f} ({hits_s:.1f})  '
              f'{"ok" if ok else "BELOW"}')
    sys.exit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
