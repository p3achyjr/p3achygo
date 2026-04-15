#!/usr/bin/env python3
"""
Run a command once per GPU in parallel, setting CUDA_VISIBLE_DEVICES for each.

Usage:
  python run_per_gpu.py --gpus 0,1,2 --cmd "python train.py"
  python run_per_gpu.py --gpus=0,1,2 --cmd "python train.py"
"""

import argparse
import os
import subprocess
from threading import Lock, Thread

print_lock = Lock()


def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def run_on_gpu(cmd: str, gpu_id: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log(f"[GPU {gpu_id}] Starting: {cmd}")
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    for line in proc.stdout:
        log(f"[GPU {gpu_id}] {line.rstrip()}")
    proc.wait()
    log(f"[GPU {gpu_id}] Exited with code {proc.returncode}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a command once per GPU in parallel."
    )
    parser.add_argument(
        "--gpus", required=True, help="Comma-separated GPU IDs (e.g. 0,1,2)"
    )
    parser.add_argument("--cmd", required=True, help="Command to run on each GPU")
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]

    threads = [Thread(target=run_on_gpu, args=(args.cmd, gpu_id)) for gpu_id in gpu_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
