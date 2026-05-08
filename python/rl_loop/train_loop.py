"""
Long-running training process that loops over available chunks without restarting.

Compared to train_one_gen.py (which is spawned as a subprocess per generation),
this script loads the models and traces the TF graph once and then loops
indefinitely — avoiding the grappler re-tracing cost on every generation.
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys

import gcs_utils as gcs
from dataset import ChunkDataset
from backend_shim import (
    configure_gpu,
    load_model,
    load_with_optimizer,
    save_model,
    summary,
    compile_for_training,
    LIVE_MODEL_NAME,
    MODEL_EXT,
)
import rl_loop.model_utils as model_utils
import rl_loop.train
import rl_loop.config

from absl import app, flags, logging
from pathlib import Path
from rl_loop.constants import SELFPLAY_BATCH_SIZE

FLAGS = flags.FLAGS

flags.DEFINE_string("run_id", "", "ID corresponding to the current run.")
flags.DEFINE_string("models_dir", "", "Directory containing model checkpoints.")
flags.DEFINE_string("chunk_dir", "", "Directory containing training chunks.")
flags.DEFINE_string("val_ds_path", "", "Path to validation dataset.")
flags.DEFINE_string("batch_num_path", "", "File storing batch counter.")
flags.DEFINE_boolean("save_trt", False, "Whether to save TRT model after each gen.")
flags.DEFINE_string("trt_convert_path", "", "Path to TRT convert binary.")
flags.DEFINE_integer("max_gens", 0, "Number of generations to train (0 = run forever).")
flags.DEFINE_string(
    "source_run_id", "", "Run ID to fetch golden chunks from (defaults to run_id)."
)
S3_BUCKET = "p3achygo"


def _get_starting_gen(models_dir: str, start_gen: int) -> tuple[str, int]:
    model_paths = [
        f for f in Path(models_dir).glob("**/*") if gcs.MODEL_RE.fullmatch(f.name)
    ]
    model_paths = sorted(
        model_paths, key=lambda f: int(gcs.MODEL_RE.fullmatch(f.name).group(1))
    )
    if not model_paths:
        return str(Path(models_dir, gcs.MODEL_FORMAT.format(start_gen))), start_gen
    latest = model_paths[-1]
    return str(latest), int(gcs.MODEL_RE.fullmatch(latest.name).group(1))


def _fetch_chunk(chunk_dir: str, run_id: str, gen: int) -> str:
    """Download chunk_{gen:04d}.tfrecord.zz from S3 and return the local path."""
    chunk_name = gcs.GOLDEN_CHUNK_FORMAT.format(gen)
    local_path = Path(chunk_dir, chunk_name)
    s3_uri = f"s3://{S3_BUCKET}/{run_id}/goldens/{chunk_name}"
    if not local_path.exists():
        logging.info(f"Fetching {s3_uri} -> {local_path}")
        subprocess.run(["s5cmd", "cp", s3_uri, str(local_path)], check=True)
    else:
        logging.info(f"Chunk already exists at {local_path}, skipping fetch")
    return str(local_path)


def _train_loop():
    config = rl_loop.config.parse(FLAGS.run_id)

    Path(FLAGS.models_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.chunk_dir).mkdir(parents=True, exist_ok=True)

    swa_model_path, model_gen = _get_starting_gen(FLAGS.models_dir, config.start_gen)
    live_model_path = str(Path(FLAGS.models_dir, LIVE_MODEL_NAME))

    logging.info(f"Starting from generation {model_gen}")
    logging.info(f"Live model: {live_model_path}")
    logging.info(f"SWA model:  {swa_model_path}")

    val_workers = min(8, max(1, multiprocessing.cpu_count() // 2))
    val_ds = ChunkDataset(FLAGS.val_ds_path, config.batch_size, num_workers=val_workers)

    if not Path(FLAGS.batch_num_path).exists():
        with open(FLAGS.batch_num_path, "w") as f:
            f.write("0")
    with open(FLAGS.batch_num_path, "r") as f:
        batch_num = int(f.read())

    if not Path(live_model_path).exists():
        logging.info(
            f"No live model found at {live_model_path}. Creating new model from config '{config.model_config}'."
        )
        live_model = model_utils.new_model(
            "p3achygo", config.model_config, config.optimizer
        )
        save_model(live_model, live_model_path)

    if not Path(swa_model_path).exists():
        swa_model_path = live_model_path

    live_model, optimizer = load_with_optimizer(live_model_path)
    if optimizer is None:
        logging.info(
            "No optimizer state found in live model. "
            "This should only happen for model_0000."
        )
    swa_model = load_model(swa_model_path)

    summary(live_model)

    # Apply backend-specific training-time transforms (fp16 + channels_last
    # + torch.compile reduce-overhead for torch; no-op for TF).
    live_model = compile_for_training(live_model)

    max_gens = FLAGS.max_gens
    gens_trained = 0

    while max_gens == 0 or gens_trained < max_gens:
        config = rl_loop.config.parse(FLAGS.run_id)
        next_gen = model_gen + 1
        source_run_id = FLAGS.source_run_id or FLAGS.run_id
        chunk_path = _fetch_chunk(FLAGS.chunk_dir, source_run_id, next_gen)
        logging.info(f"Training generation {next_gen} on {chunk_path}")

        batch_num, live_model, swa_model, optimizer = rl_loop.train.train_one_gen(
            live_model,
            swa_model,
            optimizer,
            model_gen,
            chunk_path,
            val_ds,
            config=config,
            is_gpu=True,
            batch_num=batch_num,
        )

        logging.info(f"Deleting local chunk {chunk_path}")
        Path(chunk_path).unlink(missing_ok=True)

        # Save live model checkpoint, bundling optimizer state.
        save_model(live_model, live_model_path, optimizer=optimizer)

        # Save restart checkpoint.
        live_ckpt_dir = Path(FLAGS.models_dir, "_live")
        live_ckpt_dir.mkdir(exist_ok=True)
        save_model(
            live_model,
            str(live_ckpt_dir / f"live_{next_gen:04d}{MODEL_EXT}"),
            optimizer=optimizer,
        )

        # Save SWA model for selfplay.
        if FLAGS.save_trt:
            model_utils.save_onnx_trt(
                swa_model,
                FLAGS.val_ds_path,
                FLAGS.models_dir,
                next_gen,
                batch_size=SELFPLAY_BATCH_SIZE,
                trt_convert_path=FLAGS.trt_convert_path,
            )
        else:
            model_utils.save(swa_model, FLAGS.models_dir, next_gen)

        with open(FLAGS.batch_num_path, "w") as f:
            f.write(str(batch_num))

        model_gen = next_gen
        gens_trained += 1
        logging.info(
            f"Generation {next_gen} complete. Total trained this session: {gens_trained}."
        )


def main(_):
    for flag_name, val in [
        ("run_id", FLAGS.run_id),
        ("models_dir", FLAGS.models_dir),
        ("chunk_dir", FLAGS.chunk_dir),
        ("val_ds_path", FLAGS.val_ds_path),
        ("batch_num_path", FLAGS.batch_num_path),
    ]:
        if not val:
            logging.error(f"No --{flag_name} specified.")
            return
    if FLAGS.save_trt and not FLAGS.trt_convert_path:
        logging.error("No --trt_convert_path specified.")
        return

    configure_gpu()

    _train_loop()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
    app.run(main)
