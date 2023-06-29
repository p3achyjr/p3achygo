from __future__ import annotations

import gcs_utils as gcs
import shlex, time
import rl_loop.config as config

from absl import logging
from pathlib import Path
from queue import Queue
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

POLL_INTERVAL_S = 10


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

  # most recent chunk tells us which generation we are making self-play data for.
  gen = gcs.get_most_recent_chunk(run_id)
  while True:
    cmd = shlex.split(
        f'{bin_path} --num_threads={num_threads}' +
        f' --model_path={str(model_path)}' +
        f' --recorder_path={shared_run_dir}' +
        f' --flush_interval={num_threads} --gen={gen}' +
        f' --gumbel_n=36 --gumbel_k=4')
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
        _ = queue.get_nowait()
        is_done = True
        queue.task_done()
        break

    selfplay_proc.communicate('\n')

    if is_done:
      break
