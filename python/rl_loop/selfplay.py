'''
Self-Play wrapper.

Invokes C++ selfplay binary until a new model is detected. Then downloads the
new model and restarts the C++ binary.
'''

from __future__ import annotations

import gcs_utils as gcs
import shlex, sys, time

from absl import app, flags, logging
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10
NUM_SELFPLAY_THREADS = 48
GAMES_PER_GENERATION = 5000

flags.DEFINE_integer('num_generations', 10,
                     'Number of generations of self-play.')
flags.DEFINE_string('bin_path', '', 'Local path to self-play binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '',
                    'Local path to store local models and data.')


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No binary path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  if FLAGS.local_run_dir == '':
    logging.error('No --local_run_dir specified.')
    return

  run_id = FLAGS.run_id
  local_data_dir = Path(FLAGS.local_run_dir, 'data')
  local_model_dir = Path(FLAGS.local_run_dir, 'models')

  local_data_dir.mkdir(exist_ok=True)
  local_model_dir.mkdir(exist_ok=True)

  model_gen = gcs.get_most_recent_model(run_id)
  while model_gen < 0:
    logging.warning(f'No model found. Sleeping for {POLL_INTERVAL_S}s')
    time.sleep(POLL_INTERVAL_S)
    model_gen = gcs.get_most_recent_model(run_id)

  model_path = gcs.download_model(run_id, str(local_model_dir), model_gen)
  trt_model_path = str(Path(model_path, '_trt'))

  cmd = shlex.split(f'{FLAGS.bin_path} --num_threads={NUM_SELFPLAY_THREADS}' +
                    f' --model_path={str(trt_model_path)}' +
                    f' --recorder_path={str(local_data_dir)}' +
                    f' --flush_interval={NUM_SELFPLAY_THREADS}')
  selfplay_proc = Popen(cmd,
                        stdin=PIPE,
                        stdout=PIPE,
                        stderr=STDOUT,
                        universal_newlines=True)
  t = Thread(target=print_stdout, args=(selfplay_proc.stdout,), daemon=True)
  t.start()

  while True:
    # should watch local filesystem here.
    time.sleep(5)


if __name__ == '__main__':
  app.run(main)
