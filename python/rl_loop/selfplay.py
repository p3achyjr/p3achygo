'''
Self-Play wrapper.

Invokes C++ selfplay binary until a new model is detected. Then downloads the
new model and restarts the C++ binary.
'''

from __future__ import annotations

import gcs_utils as gcs
import shlex, sys, time
import rl_loop.config as config

from absl import app, flags, logging
from pathlib import Path
from queue import Queue
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10

flags.DEFINE_integer('num_generations', 10,
                     'Number of generations of self-play.')
flags.DEFINE_string('bin_path', '', 'Local path to self-play binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path to store local models and data.')


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def loop(bin_path: str,
         run_id: str,
         local_run_dir: str,
         shared_run_dir: str,
         num_threads: int,
         queue: Queue | None = None):
  '''
  Starts self-play binary and runs it until told to stop.
  '''

  # wait for first model.
  model_gen = gcs.get_most_recent_model(run_id)
  while model_gen < 0:
    logging.warning(f'No model found. Sleeping for {POLL_INTERVAL_S}s')
    time.sleep(POLL_INTERVAL_S)
    model_gen = gcs.get_most_recent_model(run_id)

  # first model is now uploaded. Get it and start run.
  model_path = gcs.download_model(run_id, str(local_run_dir), model_gen)
  model_path = str(Path(model_path, '_trt'))
  while True:
    cmd = shlex.split(f'{bin_path} --num_threads={num_threads}' +
                      f' --model_path={str(model_path)}' +
                      f' --recorder_path={shared_run_dir}' +
                      f' --flush_interval={num_threads}')
    selfplay_proc = Popen(cmd,
                          stdin=PIPE,
                          stdout=PIPE,
                          stderr=STDOUT,
                          universal_newlines=True)
    t = Thread(target=print_stdout, args=(selfplay_proc.stdout,), daemon=True)
    t.start()

    is_done = False
    while True:
      time.sleep(POLL_INTERVAL_S)

      # Shut down loop if run is finished.
      if gcs.is_done(run_id):
        logging.info('Run is done. Shutting down...')
        is_done = True
        break

      # If new golden model, download and restart process.
      next_model_gen = gcs.get_most_recent_model(run_id)
      if next_model_gen > model_gen:
        logging.info('Found new model. Fetching and reloading self-play.')
        model_gen = next_model_gen
        model_path = gcs.download_model(run_id, str(local_run_dir), model_gen)
        model_path = str(Path(model_path, '_trt'))
        break

      if queue and not queue.empty():
        logging.info('Received shutdown signal. Shutting down...')
        _ = queue.item()
        is_done = True

    selfplay_proc.communicate('\n')

    if is_done:
      break

  if queue:
    queue.task_done()


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No binary path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  run_config = config.parse(FLAGS.run_id)
  loop(FLAGS.bin_path, FLAGS.run_id, FLAGS.local_run_dir,
       run_config.shared_volume_path, run_config.num_sp_threads)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
