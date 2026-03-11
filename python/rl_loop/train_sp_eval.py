"""
Starts Self-Play, then Trains when a chunk is ready, then runs Eval.
"""

from __future__ import annotations

import gcs_utils as gcs
import math
import os, sys, time
import rl_loop.config as config
import rl_loop.sp_loop as sp
import rl_loop.fs_utils as fs
import rl_loop.model_utils as model_utils
import numpy as np
import proc
import tensorflow as tf

from absl import app, flags, logging
from constants import *
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from rl_loop.constants import SELFPLAY_BATCH_SIZE
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 60
EVAL_CACHE_SIZE = 32768
MAX_EVAL_GAMES_PER_WORKER = 100
TARGET_EVAL_GAMES = 200

# Pool of GPU IDs used for both self-play and eval workers. Set in main().
# Each worker is assigned one GPU from this pool.
GPU_IDS: list[int] = []

flags.DEFINE_string(
    "from_existing_run",
    "",
    "Existing run from which to use SP chunks to train a new model from.",
)
flags.DEFINE_string("bin_dir", "", "Local path to bazel-bin dir.")
flags.DEFINE_string("run_id", "", "ID corresponding to the current run.")
flags.DEFINE_string(
    "local_run_dir", "/tmp/p3achygo", "Local path for temporary storage"
)
flags.DEFINE_bool("local_only", False, "Whether to run RL loop locally.")
flags.DEFINE_string("chunk_dir", "", "Local directory to read training chunks from. Defaults to local_run_dir.")
flags.DEFINE_string(
    "gpu_ids",
    "",
    "Comma-separated GPU IDs for self-play and eval (e.g. '1,2,3'). Defaults to all detected GPUs.",
)


@dataclass
class EvalResult(object):
    CUR = "cur"
    CAND = "cand"

    winner: str
    rel_elo: float


def eval(
    run_id: str,
    eval_bin_path: str,
    eval_res_path: str,
    cur_model_path: str,
    cand_model_path: str,
    local_run_dir: str,
    k: int,
    n: int,
) -> EvalResult:
    """`cur_model_path` and `cand_model_path` are the _base_ paths of the models."""
    cur_p = Path(cur_model_path)
    cand_p = Path(cand_model_path)
    cur_model_path_trt = str(cur_p.parent / "_onnx" / (cur_p.stem + ".trt"))
    cand_model_path_trt = str(cand_p.parent / "_onnx" / (cand_p.stem + ".trt"))

    num_workers = len(GPU_IDS)
    games_per_worker = min(
        math.ceil(TARGET_EVAL_GAMES / num_workers), MAX_EVAL_GAMES_PER_WORKER
    )
    worker_res_paths = [f"{eval_res_path}_{i}" for i in range(num_workers)]

    def run_worker(i: int):
        worker_env = os.environ.copy()
        worker_env["LD_PRELOAD"] = "/usr/local/lib/libmimalloc.so"
        worker_env["CUDA_VISIBLE_DEVICES"] = str(GPU_IDS[i])
        cmd = (
            f"{eval_bin_path} --cur_model_path={cur_model_path_trt}"
            + f" --cand_model_path={cand_model_path_trt}"
            + f" --res_write_path={worker_res_paths[i]}"
            + f" --recorder_path={local_run_dir}"
            + f" --cache_size={EVAL_CACHE_SIZE}"
            + f" --num_games={games_per_worker}"
            + f" --cur_n={n} --cur_use_puct=1 --cur_use_lcb=1"
            + f" --cand_n={n} --cand_use_puct=1 --cand_use_lcb=1"
        )
        logging.info(f"Running Eval Worker {i} (GPU {GPU_IDS[i]}):\n'{cmd}'")
        exit_code = proc.run_proc(cmd, env=worker_env)
        logging.info(f"Eval Worker {i} Exited with Status {exit_code}")

    threads = [Thread(target=run_worker, args=(i,)) for i in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Upload Eval SGFs. This is safe because all processes have terminated.
    _, _, _, _, local_sgf_dir = fs.ensure_local_dirs(local_run_dir)
    eval_sgfs = local_sgf_dir.glob("*EVAL*.sgf")
    for sgf in eval_sgfs:
        fs.upload_sgf(run_id, sgf)

    elos = []
    for res_path in worker_res_paths:
        with open(res_path) as f:
            elos.append(float(f.read()))
    cand_rel_elo = sum(elos) / len(elos)
    winner = EvalResult.CUR if cand_rel_elo < 30 else EvalResult.CAND
    logging.info(f"Winner: {winner}, Cand Elo: {cand_rel_elo} (per-worker: {elos})")
    return EvalResult(winner, cand_rel_elo)


def loop(
    run_id: str,
    config: config.RunConfig,
    sp_bin_path: str,
    eval_bin_path: str,
    val_ds_path: str,
    build_trt_engine_path: str,
    local_run_dir: str,
):
    """
    Does the following:

    1. Spawns Self-Play.
    2. In a separate thread, continually polls for a new training chunk. When a new
       chunk is available, shuts down self play.
    3. Trains for one epoch on the newly available chunk.
    4. Runs evaluation on the new model, and uploads the new model if it is better.
    5. Repeat from (1).
    """

    def train(
        run_id: str,
        model_gen: int,
        next_model_gen: int,
        local_model_cands_dir: str,
        chunk_path: str,
        chunk_size_path: str,
        batch_num_path: str,
        save_trt=True,
    ):
        with open(chunk_size_path, "r") as f:
            logging.info(f"Training model {next_model_gen}...")
            chunk_size = int(f.read())
            cmd = (
                f"python -m python.rl_loop.train_one_gen"
                + f" --run_id={run_id}"
                + f" --models_dir={local_model_cands_dir}"
                + f" --gen={model_gen}"
                + f" --next_gen={next_model_gen}"
                + f" --chunk_path={chunk_path}"
                + f" --chunk_size={chunk_size}"
                + f" --val_ds_path={val_ds_path}"
                + f" --batch_num_path={batch_num_path}"
                + f" --save_trt={save_trt}"
                + f" --trt_convert_path={build_trt_engine_path}"
            )
            exit_code = proc.run_proc(cmd)
            logging.info(f"Training Exited with Status {exit_code}")

    def eval_new_model(run_id: str, next_model_gen: int, eval_res_path: str):
        # Play against current _best_ model.
        current_golden_gen = fs.get_most_recent_model(run_id)
        cur_model_path = str(
            Path(local_model_cands_dir, gcs.MODEL_FORMAT.format(current_golden_gen))
        )
        cand_model_path = str(
            Path(local_model_cands_dir, gcs.MODEL_FORMAT.format(next_model_gen))
        )

        # Upload as new model candidate, in case we are pre-empted.
        logging.info(f"Uploading model candidate {cand_model_path}.")
        fs.upload_model_cand(run_id, local_model_cands_dir, next_model_gen)

        # Run eval.
        eval_result = eval(
            run_id,
            eval_bin_path,
            eval_res_path,
            cur_model_path,
            cand_model_path,
            local_run_dir,
            config.eval_k,
            config.eval_n,
        )
        if eval_result.winner == EvalResult.CAND:
            # The cand model is stronger. Upload it as new golden.
            logging.info(f"Uploading model {cand_model_path} as new golden")
            fs.upload_model(run_id, local_models_dir, next_model_gen)

        with open(eval_history_path, "a") as f:
            f.write(
                f"Elo: {eval_result.rel_elo}"
                + f" Cur: {current_golden_gen}, Cand: {next_model_gen}\n"
            )

    def train_from_existing_run(
        run_id,
        existing_run_id: str,
        local_model_cands_dir: str,
        local_golden_chunk_dir: str,
    ):
        logging.info(f"Training from existing run {existing_run_id} for run {run_id}")
        model_gen = fs.get_most_recent_model_cand(run_id)
        while True:
            latest_chunk_gen = fs.get_most_recent_chunk(existing_run_id)
            next_model_gen = model_gen + 1
            if next_model_gen > latest_chunk_gen:
                break

            chunk_path = fs.download_golden_chunk(
                existing_run_id, local_golden_chunk_dir, next_model_gen
            )
            chunk_size_path = fs.download_golden_chunk_size(
                existing_run_id, local_golden_chunk_dir, next_model_gen
            )
            train(
                run_id,
                model_gen,
                next_model_gen,
                local_model_cands_dir,
                chunk_path,
                chunk_size_path,
                batch_num_path,
                save_trt=False,
            )
            fs.upload_model_cand(run_id, local_model_cands_dir, next_model_gen)
            fs.remove_local_chunk(local_golden_chunk_dir, next_model_gen)
            model_gen = next_model_gen

    # populate local dirs
    (local_models_dir, local_model_cands_dir, local_golden_chunk_dir, _, _) = (
        fs.ensure_local_dirs(local_run_dir)
    )

    eval_history_path = Path(local_run_dir, "elo_history.txt")
    batch_num_path = str(Path(local_run_dir, "batch_num.txt"))
    if not os.path.exists(batch_num_path):
        with open(batch_num_path, "w") as f:
            f.write("0")

    # fetch or create first model
    model_gen = fs.get_most_recent_model_cand(run_id)
    if model_gen < 0:
        # make new model.
        logging.info("No existing model candidate found. Creating initial model...")
        model_gen = 0
        model_path = str(Path(local_model_cands_dir, gcs.MODEL_FORMAT.format(0)))
        checkpoint_path = str(Path(local_model_cands_dir, "live_model.keras"))
        with tf.device("/cpu:0"):
            batch_size = SELFPLAY_BATCH_SIZE
            model = model_utils.new_model(
                name=f"p3achygo",
                model_config=config.model_config,
                optimizer=config.optimizer,
            )
            model(
                tf.convert_to_tensor(
                    np.random.random([batch_size] + model.input_planes_shape()),
                    dtype=tf.float32,
                ),
                tf.convert_to_tensor(
                    np.random.random([batch_size] + model.input_features_shape()),
                    dtype=tf.float32,
                ),
            )
            model.summary()
            model.save(checkpoint_path)
            model.save(model_path)

            # convert to TRT.
            model_utils.save_onnx_trt(
                model,
                val_ds_path,
                local_model_cands_dir,
                model_gen,
                batch_size=SELFPLAY_BATCH_SIZE,
                trt_convert_path=build_trt_engine_path,
            )

        # upload to GCS.
        fs.upload_model_cand(run_id, local_model_cands_dir, model_gen)
        fs.upload_model(run_id, local_model_cands_dir, model_gen)
    else:
        fs.download_model_cand(run_id, local_model_cands_dir, model_gen)

    if config.from_existing_run:
        train_from_existing_run(
            run_id,
            config.from_existing_run,
            local_model_cands_dir,
            local_golden_chunk_dir,
        )
        return

    logging.info(f"Starting {len(GPU_IDS)} self-play worker(s) on GPUs {GPU_IDS}.")

    def start_sp():
        queues = [Queue() for _ in range(len(GPU_IDS))]
        threads = [
            Thread(
                target=sp.loop,
                args=(
                    sp_bin_path,
                    run_id,
                    local_run_dir,
                    SELFPLAY_BATCH_SIZE,
                    queues[i],
                ),
                kwargs={"gpu_device": GPU_IDS[i]},
            )
            for i in range(len(GPU_IDS))
        ]
        for t in threads:
            t.start()
        return queues, threads

    def stop_sp(queues, threads):
        for q in queues:
            q.put(())
        for t in threads:
            t.join()

    eval_res_path = str(Path(local_run_dir, "eval_res.txt"))
    while model_gen <= config.num_generations:
        # Start self-play.
        logging.info(f"Model Generation: {model_gen}")
        sp_queues, sp_threads = start_sp()

        # Poll GCS to check for the availability of a new golden chunk.
        latest_chunk_gen = fs.get_most_recent_chunk(run_id)
        while latest_chunk_gen <= model_gen:
            time.sleep(POLL_INTERVAL_S)
            latest_chunk_gen = fs.get_most_recent_chunk(run_id)

        # Found new chunk.
        logging.info(
            f"Found training chunk {latest_chunk_gen}."
            + f" Current generation is {model_gen}."
        )
        stop_sp(sp_queues, sp_threads)

        next_model_gen = model_gen + 1
        chunk_path = fs.download_golden_chunk(
            run_id, local_golden_chunk_dir, next_model_gen
        )
        chunk_size_path = fs.download_golden_chunk_size(
            run_id, local_golden_chunk_dir, next_model_gen
        )
        train(
            run_id,
            model_gen,
            next_model_gen,
            local_model_cands_dir,
            chunk_path,
            chunk_size_path,
            batch_num_path,
        )
        eval_new_model(run_id, next_model_gen, eval_res_path)
        fs.remove_local_chunk(local_golden_chunk_dir, next_model_gen)
        model_gen = next_model_gen
        logging.info("Eval finished. Restarting self-play -> train -> eval loop.")

    logging.info(
        "Reached number of generations. " + "Continuing training past end of self-play."
    )

    # We have completed all self-play. Continue to train on the tail of self-play
    # data. Shuffler is responsible for notifying when there are no more chunks.
    while model_gen <= config.num_generations + config.extra_train_gens:
        # Wait for chunk.
        latest_chunk_gen = fs.get_most_recent_chunk(run_id)
        while latest_chunk_gen <= model_gen:
            time.sleep(POLL_INTERVAL_S)
            latest_chunk_gen = fs.get_most_recent_chunk(run_id)

        # Found new chunk.
        logging.info(
            f"Found training chunk {latest_chunk_gen}."
            + f" Current generation is {model_gen}."
        )
        next_model_gen = model_gen + 1
        chunk_path = fs.download_golden_chunk(
            run_id, local_golden_chunk_dir, next_model_gen
        )
        chunk_size_path = fs.download_golden_chunk_size(
            run_id, local_golden_chunk_dir, next_model_gen
        )
        train(
            run_id,
            model_gen,
            next_model_gen,
            local_model_cands_dir,
            chunk_path,
            chunk_size_path,
            batch_num_path,
        )
        eval_new_model(run_id, next_model_gen, eval_res_path)

        fs.remove_local_chunk(local_golden_chunk_dir, next_model_gen)

        model_gen = next_model_gen
        logging.info("Eval finished. Waiting for next chunk...")

    logging.info("Run is finished. Shutting down...")
    fs.signal_done(run_id, local_run_dir)


def main(_):
    if FLAGS.run_id == "":
        logging.error("No --run_id specified.")
        return
    if FLAGS.bin_dir == "":
        logging.error("No --bin_dir specified.")
        return

    global GPU_IDS
    if FLAGS.gpu_ids:
        GPU_IDS = [int(x.strip()) for x in FLAGS.gpu_ids.split(",")]
    else:
        GPU_IDS = list(range(len(tf.config.list_physical_devices("GPU")))) or [0]
    logging.info(f"Using GPU pool: {GPU_IDS}")

    fs_mode = "local" if FLAGS.local_only else "gcs"
    chunk_source_path = FLAGS.chunk_dir if FLAGS.chunk_dir else FLAGS.local_run_dir
    fs.configure_fs(
        mode=fs_mode,
        local_path=FLAGS.local_run_dir,
        chunk_source_path=chunk_source_path,
    )

    sp_bin_path = Path(FLAGS.bin_dir, "selfplay")
    eval_bin_path = Path(FLAGS.bin_dir, "eval")
    build_trt_engine_path = Path(FLAGS.bin_dir, "build_and_run_trt_engine")

    val_ds_path = fs.download_val_ds(FLAGS.local_run_dir)
    run_config = config.parse(FLAGS.run_id)
    run_id = FLAGS.run_id

    loop(
        run_id,
        run_config,
        sp_bin_path,
        eval_bin_path,
        val_ds_path,
        build_trt_engine_path,
        FLAGS.local_run_dir,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    app.run(main)
