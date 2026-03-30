#!/usr/bin/env python3
"""
Run eval on multiple GPUs in parallel until the desired number of games is reached.

Usage:
  python eval.py --gpus 0,1,2,3 --cmd "CMD" --num_games 400
  python eval.py --cmd "CMD" --num_games 400   # uses all detected GPUs
"""

import argparse
import math
import os
import re
import subprocess
import sys
from queue import Queue, Empty
from threading import Lock, Thread

print_lock = Lock()
results_lock = Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def detect_gpus() -> list[int]:
    """Return list of available GPU IDs via nvidia-smi, falling back to [0]."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        ids = [int(x.strip()) for x in out.strip().splitlines() if x.strip().isdigit()]
        return ids if ids else [0]
    except Exception:
        return [0]


def parse_games_per_cmd(cmd: str) -> int:
    m = re.search(r'--num_games[= ](\d+)', cmd)
    if not m:
        raise ValueError("Could not find --num_games in command")
    return int(m.group(1))


def make_unique_cmd(cmd: str, gpu_id: int, inv_id: int) -> str:
    """Rewrite --res_write_path to be unique per invocation to avoid file conflicts."""
    def replace_path(m):
        path = m.group(1)
        # Strip any existing _gpu_inv suffix and add a new unique one
        path = re.sub(r'_gpu\d+_inv\d+$', '', path)
        return f'--res_write_path={path}_gpu{gpu_id}_inv{inv_id}'

    return re.sub(r'--res_write_path=(\S+)', replace_path, cmd)


def parse_output(output: str) -> dict | None:
    won_match = re.search(r'Cand won (\d+) games of (\d+)', output)
    wr_match = re.search(r'Win Rate \(p95\):\s*([-\d.]+)\s*\+-\s*([\d.]+)', output)
    elo_match = re.search(r'Relative Elo \(p95\):\s*([-\d.]+)\s*\+-\s*([\d.]+)', output)

    if not won_match:
        return None

    return {
        'won': int(won_match.group(1)),
        'played': int(won_match.group(2)),
        'winrate': float(wr_match.group(1)) if wr_match else None,
        'winrate_ci': float(wr_match.group(2)) if wr_match else None,
        'elo': float(elo_match.group(1)) if elo_match else None,
        'elo_ci': float(elo_match.group(2)) if elo_match else None,
    }


def run_invocation(cmd: str, gpu_id: int, inv_id: int) -> dict | None:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if 'LD_PRELOAD' not in env:
        mimalloc = '/usr/local/lib/libmimalloc.so'
        if os.path.exists(mimalloc):
            env['LD_PRELOAD'] = mimalloc

    unique_cmd = make_unique_cmd(cmd, gpu_id, inv_id)
    log(f"[GPU {gpu_id}] Starting invocation {inv_id}")

    try:
        proc = subprocess.Popen(
            unique_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        lines = []
        for line in proc.stdout:
            line = line.rstrip('\n')
            lines.append(line)
            log(f"[GPU {gpu_id}] {line}")
        proc.wait()

        output = '\n'.join(lines)
        result = parse_output(output)
        if result is None:
            log(f"[GPU {gpu_id}] WARNING: Could not parse stats from invocation {inv_id} (exit {proc.returncode})")
        return result

    except Exception as e:
        log(f"[GPU {gpu_id}] ERROR in invocation {inv_id}: {e}")
        return None


def compute_final_stats(results: list) -> dict:
    total_won = sum(r['won'] for r in results)
    total_played = sum(r['played'] for r in results)

    if total_played == 0:
        return {}

    p = total_won / total_played
    # Normal approximation 95% CI for win rate
    se = math.sqrt(p * (1 - p) / total_played) if 0 < p < 1 else 0.0
    wr_ci = 1.96 * se

    # Relative Elo and CI via delta method
    if 0 < p < 1:
        elo = -400 * math.log10(1 / p - 1)
        d_elo_d_p = 400 / (math.log(10) * p * (1 - p))
        elo_ci = 1.96 * abs(d_elo_d_p) * se
    elif p == 1.0:
        elo, elo_ci = float('inf'), float('inf')
    else:
        elo, elo_ci = float('-inf'), float('inf')

    return {
        'won': total_won,
        'played': total_played,
        'winrate': p,
        'winrate_ci': wr_ci,
        'elo': elo,
        'elo_ci': elo_ci,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run eval across multiple GPUs until target game count is reached.'
    )
    parser.add_argument(
        '--gpus',
        default=None,
        help='Comma-separated GPU IDs (e.g. 0,1,2,3). Defaults to all detected GPUs.',
    )
    parser.add_argument('--cmd', required=True, help='Eval command to run (must include --num_games)')
    parser.add_argument('--num_games', type=int, required=True, help='Total target number of games')
    args = parser.parse_args()

    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        gpu_ids = detect_gpus()

    games_per_cmd = parse_games_per_cmd(args.cmd)
    total_invocations = math.ceil(args.num_games / games_per_cmd)

    print(f"GPUs:                 {gpu_ids}")
    print(f"Games per invocation: {games_per_cmd}")
    print(f"Total invocations:    {total_invocations}  ({total_invocations * games_per_cmd} games)")
    print(f"Target games:         {args.num_games}")
    print()

    invocation_queue: Queue = Queue()
    for i in range(total_invocations):
        invocation_queue.put(i)

    results = []

    def gpu_worker(gpu_id: int):
        while True:
            try:
                inv_id = invocation_queue.get_nowait()
            except Empty:
                break
            result = run_invocation(args.cmd, gpu_id, inv_id)
            if result is not None:
                with results_lock:
                    results.append(result)
                    running = compute_final_stats(results)
                log(
                    f"[GPU {gpu_id}] Invocation {inv_id} complete: "
                    f"{result['won']}/{result['played']} wins  |  "
                    f"Running: {running['won']}/{running['played']} wins  "
                    f"WR={running['winrate']:.3f}+-{running['winrate_ci']:.3f}  "
                    f"Elo={running['elo']:.1f}+-{running['elo_ci']:.1f}"
                )

    threads = [Thread(target=gpu_worker, args=(gpu_id,), daemon=True) for gpu_id in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print()
    print("=" * 50)
    print("FINAL STATS")
    print("=" * 50)

    if not results:
        print("No results collected.")
        sys.exit(1)

    stats = compute_final_stats(results)
    print(f"Cand won {stats['won']} games of {stats['played']}")
    print(f"Win Rate (p95): {stats['winrate']:.3f} +- {stats['winrate_ci']:.3f}")
    print(f"Relative Elo (p95): {stats['elo']:.3f} +- {stats['elo_ci']:.3f}")


if __name__ == '__main__':
    main()
