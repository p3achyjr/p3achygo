"""
Routines for supervised learning.

We will train our model on samples generated from professional games.
"""

from __future__ import annotations

import sys
import types
import train
import backend_shim
from dataset import ChunkDataset
from backend_shim import configure_gpu

from absl import app, flags, logging
from constants import *
from lr_schedule import ConstantLRSchedule
from model import P3achyGoModel
from model_config import ModelConfig, CONFIG_OPTIONS
from pathlib import Path

from loss_coeffs import LossCoeffs

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS

# Flags for GCS
flags.DEFINE_boolean("upload_to_gcs", False, "Whether to upload models to GCS.")

# Flags for local storage
flags.DEFINE_string("model_save_path", "", "Folder under which to save models.")

# Flags for training configuration
flags.DEFINE_integer("batch_size", 32, "Mini-batch size")
flags.DEFINE_integer("epochs", 1, "Number of Epochs")
flags.DEFINE_float("learning_rate", 1e-3, "Initial Learning Rate")
flags.DEFINE_float("momentum", 0.9, "SGD Momentum")
flags.DEFINE_integer("shuf_buf_size", 100000, "Shuffle Buffer Size")
flags.DEFINE_integer(
    "log_interval",
    100,
    "Interval at which to log training information (in mini-batches)",
)
flags.DEFINE_integer(
    "model_save_interval",
    5000,
    "Interval at which to save a new model/model checkpoint",
)
flags.DEFINE_string("dataset_dir", "", "Directory to datasets.")
flags.DEFINE_string("tensorboard_logdir", "/tmp/logs", "Tensorboard log directory.")
flags.DEFINE_enum("model_config", "b10c128btl3", CONFIG_OPTIONS, "Model Config/Size.")
flags.DEFINE_string("from_checkpoint", "", "Path to checkpoint to load weights from.")
flags.DEFINE_enum("optimizer", "sgd", ["sgd", "muon"], "Optimizer to use.")


def main(_):
    if FLAGS.dataset_dir == "":
        logging.warning("Please provide --dataset_dir where dataset lives.")
        return

    if FLAGS.model_save_path == "":
        logging.warning("Please provide --model_save_path.")
        return

    batch_size = FLAGS.batch_size
    train_shards = [
        str(path) for path in Path(FLAGS.dataset_dir).glob("shard*.tfrecord.zz")
    ]
    val_shard = str(Path(FLAGS.dataset_dir, "val.tfrecord.zz"))
    with open(Path(FLAGS.dataset_dir, "LENGTH.txt")) as f:
        ds_len = int(f.read()) // batch_size

    tensorboard_log_dir = FLAGS.tensorboard_logdir

    lr, momentum, epochs = FLAGS.learning_rate, FLAGS.momentum, FLAGS.epochs
    config = ModelConfig.from_str(FLAGS.model_config)
    model = P3achyGoModel.create(
        config=config,
        board_len=BOARD_LEN,
        num_input_planes=num_input_planes(),
        num_input_features=num_input_features(),
        name="p3achygo_sl",
    )
    optimizer = None
    if FLAGS.from_checkpoint:
        model, optimizer = backend_shim.load_with_optimizer(FLAGS.from_checkpoint)

    # setup train ds — sequential pass through each shard, front-to-back.
    class _SequentialShards:
        def __init__(self, paths, batch_size):
            self._paths = paths
            self._batch_size = batch_size

        def __iter__(self):
            for path in self._paths:
                yield from ChunkDataset(path, self._batch_size)

    train_ds = _SequentialShards(train_shards, batch_size)

    # setup validation dataset
    val_ds = ChunkDataset(val_shard, batch_size)
    lr_schedule = ConstantLRSchedule(lr)
    print(lr_schedule.info())
    backend_shim.summary(model)

    configure_gpu()
    is_gpu = True

    # Apply backend-specific training-time transforms (no-op on TF).
    model = backend_shim.compile_for_training(model)

    # SL uses defaults for everything ConvMuon-related; build a stand-in
    # config struct that satisfies `backend_shim.make_optimizer`.
    sl_opt_config = types.SimpleNamespace(
        optimizer=FLAGS.optimizer,
        muon_wd=0.1,
        adam_wd=0.1,
        adam_lr_ratio=1.0,
        wd_lr_exponent=None,
        wd_lr_max=None,
        global_clipnorm=float("inf"),
    )
    optimizer = backend_shim.make_optimizer(
        model, sl_opt_config, lr_schedule, is_gpu, loaded_state=optimizer
    )

    logging.info(f"Running initial validation...")
    train.val(model, mode=train.Mode.SL, val_ds=val_ds, batch_num=0)

    logging.info(f"Starting Training...")
    _, optimizer = train.train(
        model,
        train_ds,
        epochs,
        momentum,
        log_interval=FLAGS.log_interval,
        mode=train.Mode.SL,
        coeffs=LossCoeffs.SLCoeffs(),
        optimizer=optimizer,
        save_interval=FLAGS.model_save_interval,
        save_path=FLAGS.model_save_path,
        tensorboard_log_dir=tensorboard_log_dir,
        lr_schedule=lr_schedule,
        is_gpu=is_gpu,
        val_ds=val_ds,
        num_val_batches=10,
    )

    logging.info(f"Running final validation...")
    train.val(model, mode=train.Mode.SL, val_ds=val_ds, batch_num=1)

    model_path = str(
        Path(FLAGS.model_save_path, f"p3achygo_sl{backend_shim.MODEL_EXT}")
    )
    backend_shim.save_model(model, model_path, optimizer=optimizer)


if __name__ == "__main__":
    app.run(main)
