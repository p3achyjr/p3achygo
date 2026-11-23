"""
Builds temporary goldens for a sliding-window replay of an existing run's
self-play data (KataGo-style train bucket: --reuse training samples per new
data row, drawn uniformly from the training window as it stood at that point).

Step t ends its window at generation E_t = start_gen + t * gens_per_step.
  N_t = rows produced through generation E_t (from the raw chunk filenames)
  W_t = training_window_size(N_t)            (the source run's window formula)
  S_t = reuse * (N_t - N_{t-1})              (step 0: --initial_samples)
S_t is split over m = ceil(S_t / max_golden_rows) one-shot shuffler runs, each
with --max_gen=E_t --train_window_size=W_t --p=S_t / (m * rows in the window).

Raw chunks are downloaded from R2 into --work_dir/chunks just ahead of the
window and removed once the window has passed them; goldens are written by the
shuffler to --work_dir/goldens with a manifest, and removed once the trainer
(`replay_train_loop.py`) marks them `.trained`. R2 is only read.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time

from absl import app, flags, logging
from pathlib import Path

from rl_loop.shuffle_metadata import training_window_size

FLAGS = flags.FLAGS

flags.DEFINE_string("bin_path", "", "Path to the shuffler binary.")
flags.DEFINE_string("source", "r2:p3achygo/v4/chunks", "rclone path of raw chunks.")
flags.DEFINE_string("work_dir", "", "Local dir; uses <work_dir>/chunks and /goldens.")
flags.DEFINE_integer("start_gen", 200, "Window end of step 0.")
flags.DEFINE_integer("gens_per_step", 5, "Generations the window end moves per step.")
flags.DEFINE_float("reuse", 2.0, "Training samples per new data row.")
flags.DEFINE_integer(
    "max_end_gen",
    -1,
    "Stop (write DONE) before any step whose window end exceeds this; -1 = off.",
)
flags.DEFINE_integer("initial_samples", 1_000_000, "Samples for step 0.")
flags.DEFINE_integer(
    "max_golden_rows", 2_000_000, "Rows per golden (shuffler cap 2.048M)."
)
flags.DEFINE_integer(
    "max_pending", 3, "Untrained goldens to keep ahead of the trainer."
)

SP_CHUNK_RE = re.compile(r"gen(\d+)_b(\d+)_g(\d+)_n(\d+)_t(\d+)_(.*)\.tfrecord\.zz")


def list_chunks() -> list[tuple[int, int, int, str]]:
    """(gen, timestamp, num_rows, name) of every raw chunk at the source."""
    out = subprocess.run(
        ["rclone", "lsf", FLAGS.source, "--files-only", "--include", "*.tfrecord.zz"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.split()
    chunks = []
    for name in out:
        m = SP_CHUNK_RE.fullmatch(name)
        if m:
            chunks.append((int(m[1]), int(m[5]), int(m[4]), name))
    return chunks


def window_files(chunks, end_gen: int, window: int):
    """Mirrors TfRecordWatcher::PopulateInitialTrainingWindow: newest files
    (gen, then timestamp) up to `end_gen` until `window` rows are covered."""
    files, rows = [], 0
    for c in sorted((c for c in chunks if c[0] <= end_gen), reverse=True):
        if rows >= window:
            break
        files.append(c)
        rows += c[2]
    return files, rows


def run_shuffler(chunk_dir: Path, idx: int, end_gen: int, window: int, p: float):
    cmd = [
        FLAGS.bin_path,
        f"--data_path={chunk_dir}",
        f"--gen={idx}",
        "--games_this_gen=1",
        f"--train_window_size={window}",
        f"--p={p}",
        "--run_continuously=false",
        "--is_local=false",
        f"--max_gen={end_gen}",
    ]
    logging.info("Running " + " ".join(cmd))
    # stdin must stay open: any line (or EOF) on stdin tells the shuffler to stop.
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"shuffler exited {rc}")


def main(_):
    work = Path(FLAGS.work_dir)
    chunk_dir, golden_dir = work / "chunks", work / "goldens"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    state_path = work / "builder_state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"step": 0, "golden_in_step": 0, "next_golden": 0}
    )

    chunks = list_chunks()
    rows_through = lambda g: sum(c[2] for c in chunks if c[0] <= g)
    while True:
        t = state["step"]
        end_gen = FLAGS.start_gen + t * FLAGS.gens_per_step
        if FLAGS.max_end_gen >= 0 and end_gen > FLAGS.max_end_gen:
            logging.info(
                f"Window end {end_gen} > --max_end_gen {FLAGS.max_end_gen}; done."
            )
            (golden_dir / "DONE").touch()
            return
        # The newest generation at the source may still be filling.
        if end_gen > max(c[0] for c in chunks) - 1:
            chunks = list_chunks()
            if end_gen > max(c[0] for c in chunks) - 1:
                logging.info(f"Window end {end_gen} is past the source tip; done.")
                (golden_dir / "DONE").touch()
                return
        n_t = rows_through(end_gen)
        new_rows = n_t - rows_through(end_gen - FLAGS.gens_per_step)
        samples = FLAGS.initial_samples if t == 0 else int(FLAGS.reuse * new_rows)
        window = training_window_size(n_t)
        files, window_rows = window_files(chunks, end_gen, window)
        m = max(1, math.ceil(samples / FLAGS.max_golden_rows))
        p = min(1.0, samples / (m * window_rows))
        min_gen = min(c[0] for c in files)
        logging.info(
            f"step {t}: end_gen {end_gen} N {n_t} new_rows {new_rows} window {window} "
            f"(rows in window files {window_rows}, gens [{min_gen}, {end_gen}]) "
            f"samples {samples} goldens {m} p {p:.5f}"
        )

        # Raw chunks: drop files the window has passed, fetch the ones it needs.
        for f in chunk_dir.glob("*.tfrecord.zz"):
            mm = SP_CHUNK_RE.fullmatch(f.name)
            if mm and int(mm[1]) < min_gen:
                f.unlink()
        need = [c[3] for c in files if not (chunk_dir / c[3]).exists()]
        if need:
            lst = work / "fetch_list.txt"
            lst.write_text("\n".join(need) + "\n")
            subprocess.run(
                [
                    "rclone",
                    "copy",
                    FLAGS.source,
                    str(chunk_dir),
                    "--files-from",
                    str(lst),
                    "--transfers",
                    "64",
                    "--checkers",
                    "64",
                ],
                check=True,
            )

        for j in range(state["golden_in_step"], m):
            # Stay only a few goldens ahead of the trainer; drop trained ones.
            while True:
                for marker in golden_dir.glob("chunk_*.trained"):
                    for ext in (".tfrecord.zz", ".size", ".json", ".trained"):
                        (golden_dir / (marker.stem + ext)).unlink(missing_ok=True)
                if len(list(golden_dir.glob("chunk_*.json"))) < FLAGS.max_pending:
                    break
                time.sleep(30)
            idx = state["next_golden"]
            run_shuffler(chunk_dir, idx, end_gen, window, p)
            manifest = dict(
                step=t,
                end_gen=end_gen,
                golden_in_step=j,
                goldens_in_step=m,
                last_of_step=(j == m - 1),
                samples_target=samples // m,
                p=p,
                window=window,
                window_rows=window_rows,
                min_gen=min_gen,
            )
            tmp = golden_dir / f"chunk_{idx:04d}.json.tmp"
            tmp.write_text(json.dumps(manifest))
            tmp.replace(golden_dir / f"chunk_{idx:04d}.json")
            state["next_golden"] = idx + 1
            state["golden_in_step"] = j + 1
            state_path.write_text(json.dumps(state))
        state["step"] = t + 1
        state["golden_in_step"] = 0
        state_path.write_text(json.dumps(state))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    app.run(main)
