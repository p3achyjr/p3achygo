'''
Self-Play wrapper.

Invokes C++ selfplay binary until a new model is detected. Then downloads the
new model and restarts the C++ binary.
'''

from __future__ import annotations

import gcs_utils as gcs
import sys, time

from absl import app, flags, logging
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10
NUM_SELFPLAY_THREADS = 48
GAMES_PER_GENERATION = 5000

flags.DEFINE_integer('num_generations', 10,
                     'Number of generations of self-play.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '',
                    'Local path to store local models and data.')


def main(_):
  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  if FLAGS.local_run_dir == '':
    logging.error('No --local_run_dir specified.')
    return

  local_data_dir = Path(FLAGS.local_run_dir, 'data')
  local_model_dir = Path(FLAGS.local_run_dir, 'models')

  local_data_dir.mkdir(exist_ok=True)
  local_model_dir.mkdir(exist_ok=True)

  model_gen = gcs.get_most_recent_model(FLAGS.run_id)
  while model_gen < 0:
    logging.warning(f'No model found. Sleeping for {POLL_INTERVAL_S}s')
    time.sleep(POLL_INTERVAL_S)
    model_gen = gcs.get_most_recent_model(FLAGS.run_id)

  while model_gen < FLAGS.num_generations:
    pass


if __name__ == '__main__':
  app.run(main)
