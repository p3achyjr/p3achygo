from __future__ import annotations

import gcs_utils as gcs
import multiprocessing, os, secrets, shlex, time
import rl_loop.fs_utils as fs_utils

from absl import logging
from pathlib import Path
from queue import Queue
from subprocess import Popen, PIPE, STDOUT
from threading import Thread
from typing import Tuple

POLL_INTERVAL_S = 10
THREAD_CPU_RATIO = 6
TFREC_GLOB = '*.tfrecord.zz'
SGF_GLOB = '*.sgfs'
DONE_GLOB = '*.done'


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def update_and_upload_sp_files(run_id: str, sp_chunk_dir: str, sgf_dir: str,
                               sp_chunks: set[str],
                               sgfs: set[str]) -> Tuple[set[str], set[str]]:
  sp_chunks, new_sp_chunks = fs_utils.file_diff(sp_chunk_dir, sp_chunks,
                                                TFREC_GLOB)
  sgfs, new_sgfs = fs_utils.file_diff(sgf_dir, sgfs, SGF_GLOB)
  for sp_chunk in new_sp_chunks:
    # wait for `done` file.
    while sp_chunk.name.split('.')[0] not in map(lambda x: x.name.split('.')[0],
                                                 sp_chunk_dir.glob(DONE_GLOB)):
      time.sleep(.5)
    gcs.upload_sp_chunk(run_id, sp_chunk)

  for sgf in new_sgfs:
    # wait for `done` file.
    while sgf.name.split('.')[0] not in map(lambda x: x.name.split('.')[0],
                                            sgf_dir.glob(DONE_GLOB)):
      time.sleep(.5)
    gcs.upload_sgf(run_id, sgf)

  return sp_chunks, sgfs


def loop(bin_path: str,
         run_id: str,
         local_run_dir: str,
         queue: Queue | None = None):
  '''
  Starts self-play binary and runs until told to stop.
  '''
  (local_models_dir, _, local_sp_chunk_dir,
   local_sgf_dir) = fs_utils.ensure_local_dirs(local_run_dir)

  worker_id = secrets.token_hex(5)

  # keep track of self-play chunks in order to upload.
  sp_chunks = set(local_sp_chunk_dir.glob(TFREC_GLOB))
  sgfs = set(local_sgf_dir.glob(SGF_GLOB))

  # wait for first model.
  model_gen = gcs.get_most_recent_model(run_id)
  while model_gen < 0:
    logging.warning(f'No model found. Sleeping for {POLL_INTERVAL_S}s')
    time.sleep(POLL_INTERVAL_S)
    model_gen = gcs.get_most_recent_model(run_id)

  # first model is now uploaded. Get it and start run.
  model_path = gcs.download_model(run_id, str(local_models_dir), model_gen)
  model_path = str(Path(model_path, '_trt'))

  # calculate number of threads based on number of vCPUs.
  num_threads = multiprocessing.cpu_count() * THREAD_CPU_RATIO

  # most recent chunk tells us which generation we are making self-play data for.
  gen = gcs.get_most_recent_chunk(run_id)
  while True:
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/usr/local/lib/libmimalloc.so'
    cmd = shlex.split(f'{bin_path} --num_threads={num_threads}' +
                      f' --model_path={str(model_path)}' +
                      f' --recorder_path={local_run_dir}' +
                      f' --flush_interval={num_threads} --gen={gen}' +
                      f' --id={worker_id.upper()}')
    selfplay_proc = Popen(cmd,
                          stdin=PIPE,
                          stdout=PIPE,
                          stderr=STDOUT,
                          universal_newlines=True,
                          env=env)
    t = Thread(target=print_stdout, args=(selfplay_proc.stdout,), daemon=True)
    t.start()

    is_done = False
    while True:
      time.sleep(POLL_INTERVAL_S)

      # Check for new chunks on local disk, and upload them.
      sp_chunks, sgfs = update_and_upload_sp_files(run_id, local_sp_chunk_dir,
                                                   local_sgf_dir, sp_chunks,
                                                   sgfs)

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
    selfplay_proc.wait()

    # Upload any files that were flush on shutdown.
    _, _ = update_and_upload_sp_files(run_id, local_sp_chunk_dir, local_sgf_dir,
                                      sp_chunks, sgfs)

    if is_done:
      break
