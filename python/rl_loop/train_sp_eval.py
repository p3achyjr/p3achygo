'''
Starts Self-Play, then Trains when a chunk is ready, then runs Eval.
'''

from __future__ import annotations

import gcs_utils as gcs
import os, shlex, sys, time
import rl_loop.config as config
import rl_loop.model_utils as model_utils
import rl_loop.sp_loop as sp
import rl_loop.fs_utils as fs_utils

from absl import app, flags, logging
from constants import *
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from rl_loop.trt_batch_size import trt_batch_size
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10
EVAL_CACHE_SIZE = 32768
NUM_EVAL_GAMES = 100

flags.DEFINE_string(
    'from_existing_run', '',
    'Existing run from which to use SP chunks to train a new model from.')
flags.DEFINE_enum('model_config', 'small', ['small', 'b24c192', 'b32c256'])
flags.DEFINE_string('sp_bin_path', '', 'Local path to self-play binary.')
flags.DEFINE_string('eval_bin_path', '', 'Local path to eval binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path for temporary storage')


@dataclass
class EvalResult(object):
  CUR = 'cur'
  CAND = 'cand'

  winner: str
  rel_elo: float


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def run_proc(cmd: str, env=None):
  if not env:
    env = os.environ()

  cmd = shlex.split(cmd)
  proc = Popen(cmd,
               stdin=PIPE,
               stdout=PIPE,
               stderr=STDOUT,
               universal_newlines=True,
               env=env)
  t = Thread(target=print_stdout, args=(proc.stdout,), daemon=True)
  t.start()
  proc.wait()


def eval(run_id: str, eval_bin_path: str, eval_res_path: str,
         cur_model_path: str, cand_model_path: str, local_run_dir: str, k: int,
         n: int) -> EvalResult:
  '''`cur_model_path` and `cand_model_path` are the _base_ paths of the models.'''
  cur_model_path_trt = str(Path(cur_model_path, '_trt'))
  cand_model_path_trt = str(Path(cand_model_path, '_trt'))

  env = os.environ.copy()
  env['LD_PRELOAD'] = '/usr/local/lib/libmimalloc.so'
  cmd = (f'{eval_bin_path} --cur_model_path={cur_model_path_trt}' +
         f' --cand_model_path={cand_model_path_trt}' +
         f' --res_write_path={eval_res_path}' +
         f' --recorder_path={local_run_dir}' +
         f' --cache_size={EVAL_CACHE_SIZE}' + f' --num_games={NUM_EVAL_GAMES}' +
         f' --cur_n={n} --cur_k={k} --cand_n={n} --cand_k={k}')
  run_proc(cmd, env=env)

  # Upload Eval SGFs. This is safe because the process has terminated.
  _, _, _, local_sgf_dir = fs_utils.ensure_local_dirs(local_run_dir)
  eval_sgfs = local_sgf_dir.glob("*EVAL*.sgf")
  for sgf in eval_sgfs:
    gcs.upload_sgf(run_id, sgf)

  with open(eval_res_path) as f:
    cand_rel_elo = float(f.read())
    winner = EvalResult.CUR if cand_rel_elo < 0 else EvalResult.CAND
    logging.info(f'Winner: {winner}, Cand Elo: {cand_rel_elo}')

    return EvalResult(winner, cand_rel_elo)


def loop(run_id: str, config: config.RunConfig, sp_bin_path: str,
         eval_bin_path: str, val_ds_path: str, local_run_dir: str):
  '''
  Does the following:

  1. Spawns Self-Play.
  2. In a separate thread, continually polls for a new training chunk. When a new
     chunk is available, shuts down self play.
  3. Trains for one epoch on the newly available chunk.
  4. Runs evaluation on the new model, and uploads the new model if it is better.
  5. Repeat from (1).
  '''

  def train(model_gen: int, next_model_gen: int, local_models_dir: str,
            chunk_path: str, chunk_size_path: str, batch_num_path: str):
    with open(chunk_size_path, 'r') as f:
      logging.info(f'Training model {next_model_gen}...')
      chunk_size = int(f.read())
      cmd = (f'python -m python.rl_loop.train_one_gen' + f' --run_id={run_id}' +
             f' --models_dir={local_models_dir}' + f' --gen={model_gen}' +
             f' --next_gen={next_model_gen}' + f' --chunk_path={chunk_path}' +
             f' --chunk_size={chunk_size}' + f' --val_ds_path={val_ds_path}' +
             f' --batch_num_path={batch_num_path}')
      run_proc(cmd)

  def eval_new_model(next_model_gen: int, eval_res_path: str):
    # Play against current _best_ model.
    current_golden_gen = gcs.get_most_recent_model(run_id)
    cur_model_path = str(
        Path(local_models_dir, gcs.MODEL_FORMAT.format(current_golden_gen)))
    cand_model_path = str(
        Path(local_models_dir, gcs.MODEL_FORMAT.format(next_model_gen)))

    # Upload as new model candidate, in case we are pre-empted.
    logging.info(f'Uploading model candidate {cand_model_path}.')
    gcs.upload_model_cand(run_id, local_models_dir, next_model_gen)

    # Run eval.
    eval_result = eval(run_id, eval_bin_path, eval_res_path, cur_model_path,
                       cand_model_path, local_run_dir, config.eval_k,
                       config.eval_n)
    if eval_result.winner == EvalResult.CAND:
      # The cand model is stronger. Upload it as new golden.
      logging.info(f'Uploading model {cand_model_path} as new golden')
      gcs.upload_model(run_id, local_models_dir, next_model_gen)

    with open(eval_history_path, 'a') as f:
      f.write(f'Elo: {eval_result.rel_elo}' +
              f' Cur: {current_golden_gen}, Cand: {next_model_gen}\n')

  # populate local dirs
  (local_models_dir, local_golden_chunk_dir, _,
   _) = fs_utils.ensure_local_dirs(local_run_dir)

  eval_history_path = Path(local_run_dir, 'elo_history.txt')
  batch_num_path = str(Path(local_run_dir, 'batch_num.txt'))
  if not os.path.exists(batch_num_path):
    with open(batch_num_path, 'w') as f:
      f.write('0')

  # fetch or create first model
  model_gen = gcs.get_most_recent_model_cand(FLAGS.run_id)
  if model_gen < 0:
    # make new model.
    model_gen = 0
    model_path = str(Path(local_models_dir), 'model_0000')
    cmd = (f'python -m python.rl_loop.make_new_model' +
           f' --model_path={model_path}' +
           f' --model_config={config.model_config}')
    run_proc(cmd)

    # convert to TRT.
    cmd = (f'python -m python.scripts.convert_to_trt' +
           f' --model_path={model_path}' + f' --calib_ds={val_ds_path}' +
           f' --batch_size={trt_batch_size()}')
    run_proc(cmd)

    # upload to GCS.
    gcs.upload_model_cand(FLAGS.run_id, local_models_dir, model_gen)
    gcs.upload_model(FLAGS.run_id, local_models_dir, model_gen)
  else:
    gcs.download_model_cand(FLAGS.run_id, local_models_dir, model_gen)

  eval_res_path = str(Path(local_run_dir, 'eval_res.txt'))
  while model_gen < config.num_generations:
    # Start self-play.
    logging.info(f'Model Generation: {model_gen}')
    if not config.from_existing_run:
      sp_queue = Queue()
      sp_thread = Thread(target=sp.loop,
                         args=(sp_bin_path, run_id, local_run_dir,
                               trt_batch_size(), sp_queue))
      sp_thread.start()

      # Poll GCS to check for the availability of a new golden chunk.
      latest_chunk_gen = gcs.get_most_recent_chunk(run_id)
      while latest_chunk_gen <= model_gen:
        time.sleep(POLL_INTERVAL_S)
        latest_chunk_gen = gcs.get_most_recent_chunk(run_id)

      # Found new chunk.
      logging.info(f'Found training chunk {latest_chunk_gen}.' +
                   f' Current generation is {model_gen}.')
      sp_queue.put(())  # Send any message to trigger shutdown.
      sp_thread.join()

    next_model_gen = model_gen + 1
    chunk_path = gcs.download_golden_chunk(run_id, local_golden_chunk_dir,
                                           next_model_gen)
    chunk_size_path = gcs.download_golden_chunk_size(run_id,
                                                     local_golden_chunk_dir,
                                                     next_model_gen)
    train(model_gen, next_model_gen, local_models_dir, chunk_path,
          chunk_size_path, batch_num_path)
    eval_new_model(next_model_gen, eval_res_path)
    model_gen = next_model_gen
    logging.info('Eval finished. Restarting self-play -> train -> eval loop.')

  logging.info('Reached number of generations. ' +
               'Continuing training past end of self-play.')

  # We have completed all self-play. Continue to train on the tail of self-play
  # data. Shuffler is responsible for notifying when there are no more chunks.
  while model_gen < config.num_generations + config.extra_train_gens:
    # Wait for chunk.
    latest_chunk_gen = gcs.get_most_recent_chunk(run_id)
    while latest_chunk_gen <= model_gen:
      time.sleep(POLL_INTERVAL_S)
      latest_chunk_gen = gcs.get_most_recent_chunk(run_id)

    # Found new chunk.
    logging.info(f'Found training chunk {latest_chunk_gen}.' +
                 f' Current generation is {model_gen}.')
    next_model_gen = model_gen + 1
    chunk_path = gcs.download_golden_chunk(run_id, local_golden_chunk_dir,
                                           next_model_gen)
    chunk_size_path = gcs.download_golden_chunk_size(run_id,
                                                     local_golden_chunk_dir,
                                                     next_model_gen)
    train(model_gen, next_model_gen, local_models_dir, chunk_path,
          chunk_size_path, batch_num_path)
    eval_new_model(next_model_gen, eval_res_path)

    model_gen = next_model_gen
    logging.info('Eval finished. Waiting for next chunk...')

  logging.info('Run is finished. Shutting down...')
  gcs.signal_done(run_id)


def main(_):
  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return
  if FLAGS.sp_bin_path == '':
    logging.error('No --sp_bin_path specified.')
    return
  if FLAGS.eval_bin_path == '':
    logging.error('No --eval_bin_path specified.')
    return

  val_ds_path = gcs.download_val_ds(FLAGS.local_run_dir)
  run_config = config.parse(FLAGS.run_id)
  run_id = FLAGS.run_id
  if run_config.from_existing_run:
    run_id = run_config.from_existing_run

  loop(run_id, run_config, FLAGS.sp_bin_path, FLAGS.eval_bin_path, val_ds_path,
       FLAGS.local_run_dir)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
