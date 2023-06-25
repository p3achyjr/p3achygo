'''
Starts Self-Play, then Trains when a chunk is ready, then runs Eval.
'''

from __future__ import annotations

import gcs_utils as gcs
import sys, time
import tensorflow as tf
import transforms
import train
import trt_convert

from absl import app, flags, logging
from constants import *
from pathlib import Path
from model import P3achyGoModel
from model_config import ModelConfig
import rl_loop.config as config

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS

flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path for temporary storage')


def main(_):
  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  run_id = FLAGS.run_id
  local_run_dir = Path(FLAGS.local_run_dir, run_id)
  local_run_dir.mkdir(parents=True, exist_ok=True)
  val_ds_path = gcs.download_val_ds(local_run_dir)

  run_config = config.parse(FLAGS.run_id)


if __name__ == '__main__':
  app.run(main)
