'''
Starts Self-Play, then Trains when a chunk is ready, then runs Eval.
'''

from __future__ import annotations

import gcs_utils as gcs
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
from rl_loop.trt_batch_size import trt_batch_size
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 60
EVAL_CACHE_SIZE = 32768
NUM_EVAL_GAMES = 100

flags.DEFINE_string(
    'from_existing_run', '',
    'Existing run from which to use SP chunks to train a new model from.')
flags.DEFINE_string('bin_dir', '', 'Local path to bazel-bin dir.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path for temporary storage')
flags.DEFINE_bool('local_only', False, 'Whether to run RL loop locally.')


@dataclass
class EvalResult(object):
  CUR = 'cur'
  CAND = 'cand'

  winner: str
  rel_elo: float


def eval(run_id: str, eval_bin_path: str, eval_res_path: str,
         cur_model_path: str, cand_model_path: str, local_run_dir: str, k: int,
         n: int) -> EvalResult:
  '''`cur_model_path` and `cand_model_path` are the _base_ paths of the models.'''
  cur_model_path_trt = str(Path(cur_model_path, '_onnx', 'engine.trt'))
  cand_model_path_trt = str(Path(cand_model_path, '_onnx', 'engine.trt'))

  env = os.environ.copy()
  env['LD_PRELOAD'] = '/usr/local/lib/libmimalloc.so'
  cmd = (f'{eval_bin_path} --cur_model_path={cur_model_path_trt}' +
         f' --cand_model_path={cand_model_path_trt}' +
         f' --res_write_path={eval_res_path}' +
         f' --recorder_path={local_run_dir}' +
         f' --cache_size={EVAL_CACHE_SIZE}' +
         f' --num_games={min(NUM_EVAL_GAMES, trt_batch_size())}' +
         f' --cur_n={n} --cur_use_puct=1 --cur_use_lcb=1' +
         f' --cand_n={n} --cand_use_puct=1 --cand_use_lcb=1')
  logging.info(f'Running Eval Command:\n\'{cmd}\'')
  exit_code = proc.run_proc(cmd, env=env)
  logging.info(f'Eval Exited with Status {exit_code}')

  # Upload Eval SGFs. This is safe because the process has terminated.
  _, _, _, _, local_sgf_dir = fs.ensure_local_dirs(local_run_dir)
  eval_sgfs = local_sgf_dir.glob("*EVAL*.sgf")
  for sgf in eval_sgfs:
    fs.upload_sgf(run_id, sgf)

  with open(eval_res_path) as f:
    cand_rel_elo = float(f.read())
    winner = EvalResult.CUR if cand_rel_elo < 0 else EvalResult.CAND
    logging.info(f'Winner: {winner}, Cand Elo: {cand_rel_elo}')

    return EvalResult(winner, cand_rel_elo)


def loop(run_id: str, config: config.RunConfig, sp_bin_path: str,
         eval_bin_path: str, val_ds_path: str, build_trt_engine_path: str,
         local_run_dir: str):
  '''
  Does the following:

  1. Spawns Self-Play.
  2. In a separate thread, continually polls for a new training chunk. When a new
     chunk is available, shuts down self play.
  3. Trains for one epoch on the newly available chunk.
  4. Runs evaluation on the new model, and uploads the new model if it is better.
  5. Repeat from (1).
  '''

  def train(run_id: str,
            model_gen: int,
            next_model_gen: int,
            local_model_cands_dir: str,
            chunk_path: str,
            chunk_size_path: str,
            batch_num_path: str,
            save_trt=True):
    with open(chunk_size_path, 'r') as f:
      logging.info(f'Training model {next_model_gen}...')
      chunk_size = int(f.read())
      cmd = (f'python -m python.rl_loop.train_one_gen' + f' --run_id={run_id}' +
             f' --models_dir={local_model_cands_dir}' + f' --gen={model_gen}' +
             f' --next_gen={next_model_gen}' + f' --chunk_path={chunk_path}' +
             f' --chunk_size={chunk_size}' + f' --val_ds_path={val_ds_path}' +
             f' --batch_num_path={batch_num_path}' + f' --save_trt={save_trt}' +
             f' --trt_convert_path={build_trt_engine_path}')
      exit_code = proc.run_proc(cmd)
      logging.info(f'Training Exited with Status {exit_code}')

  def eval_new_model(run_id: str, next_model_gen: int, eval_res_path: str):
    # Play against current _best_ model.
    current_golden_gen = fs.get_most_recent_model(run_id)
    cur_model_path = str(
        Path(local_model_cands_dir,
             gcs.MODEL_FORMAT.format(current_golden_gen)))
    cand_model_path = str(
        Path(local_model_cands_dir, gcs.MODEL_FORMAT.format(next_model_gen)))

    # Upload as new model candidate, in case we are pre-empted.
    logging.info(f'Uploading model candidate {cand_model_path}.')
    fs.upload_model_cand(run_id, local_model_cands_dir, next_model_gen)

    # Run eval.
    eval_result = eval(run_id, eval_bin_path, eval_res_path, cur_model_path,
                       cand_model_path, local_run_dir, config.eval_k,
                       config.eval_n)
    if eval_result.winner == EvalResult.CAND:
      # The cand model is stronger. Upload it as new golden.
      logging.info(f'Uploading model {cand_model_path} as new golden')
      fs.upload_model(run_id, local_models_dir, next_model_gen)

    with open(eval_history_path, 'a') as f:
      f.write(f'Elo: {eval_result.rel_elo}' +
              f' Cur: {current_golden_gen}, Cand: {next_model_gen}\n')

  def train_from_existing_run(run_id, existing_run_id: str,
                              local_model_cands_dir: str,
                              local_golden_chunk_dir: str):
    logging.info(
        f'Training from existing run {existing_run_id} for run {run_id}')
    model_gen = fs.get_most_recent_model_cand(run_id)
    while True:
      latest_chunk_gen = fs.get_most_recent_chunk(existing_run_id)
      next_model_gen = model_gen + 1
      if next_model_gen > latest_chunk_gen:
        break

      chunk_path = fs.download_golden_chunk(existing_run_id,
                                            local_golden_chunk_dir,
                                            next_model_gen)
      chunk_size_path = fs.download_golden_chunk_size(existing_run_id,
                                                      local_golden_chunk_dir,
                                                      next_model_gen)
      train(run_id,
            model_gen,
            next_model_gen,
            local_model_cands_dir,
            chunk_path,
            chunk_size_path,
            batch_num_path,
            save_trt=False)
      fs.upload_model_cand(run_id, local_model_cands_dir, next_model_gen)
      fs.remove_local_chunk(local_golden_chunk_dir, next_model_gen)
      model_gen = next_model_gen

  # populate local dirs
  (local_models_dir, local_model_cands_dir, local_golden_chunk_dir, _,
   _) = fs.ensure_local_dirs(local_run_dir)

  eval_history_path = Path(local_run_dir, 'elo_history.txt')
  batch_num_path = str(Path(local_run_dir, 'batch_num.txt'))
  if not os.path.exists(batch_num_path):
    with open(batch_num_path, 'w') as f:
      f.write('0')

  # fetch or create first model
  model_gen = fs.get_most_recent_model_cand(run_id)
  if model_gen < 0:
    # make new model.
    model_gen = 0
    model_path = str(Path(local_model_cands_dir, 'model_0000'))
    with tf.device('/cpu:0'):
      batch_size = trt_batch_size()
      model = model_utils.new_model(name=f'p3achygo',
                                    model_config=config.model_config)
      model(
          tf.convert_to_tensor(np.random.random([batch_size] +
                                                model.input_planes_shape()),
                               dtype=tf.float32),
          tf.convert_to_tensor(np.random.random([batch_size] +
                                                model.input_features_shape()),
                               dtype=tf.float32))
      model.summary()
      model.save(model_path)

      # convert to TRT.
      model_utils.save_onnx_trt(model,
                                val_ds_path,
                                local_model_cands_dir,
                                model_gen,
                                batch_size=trt_batch_size(),
                                trt_convert_path=build_trt_engine_path)

    # upload to GCS.
    fs.upload_model_cand(run_id, local_model_cands_dir, model_gen)
    fs.upload_model(run_id, local_model_cands_dir, model_gen)
  else:
    fs.download_model_cand(run_id, local_model_cands_dir, model_gen)

  if config.from_existing_run:
    train_from_existing_run(run_id, config.from_existing_run,
                            local_model_cands_dir, local_golden_chunk_dir)
    return

  eval_res_path = str(Path(local_run_dir, 'eval_res.txt'))
  while model_gen < config.num_generations:
    # Start self-play.
    logging.info(f'Model Generation: {model_gen}')
    sp_queue = Queue()
    sp_thread = Thread(target=sp.loop,
                       args=(sp_bin_path, run_id, local_run_dir,
                             trt_batch_size(), sp_queue))
    sp_thread.start()

    # Poll GCS to check for the availability of a new golden chunk.
    latest_chunk_gen = fs.get_most_recent_chunk(run_id)
    while latest_chunk_gen <= model_gen:
      time.sleep(POLL_INTERVAL_S)
      latest_chunk_gen = fs.get_most_recent_chunk(run_id)

    # Found new chunk.
    logging.info(f'Found training chunk {latest_chunk_gen}.' +
                 f' Current generation is {model_gen}.')
    sp_queue.put(())  # Send any message to trigger shutdown.
    sp_thread.join()

    next_model_gen = model_gen + 1
    chunk_path = fs.download_golden_chunk(run_id, local_golden_chunk_dir,
                                          next_model_gen)
    chunk_size_path = fs.download_golden_chunk_size(run_id,
                                                    local_golden_chunk_dir,
                                                    next_model_gen)
    train(run_id, model_gen, next_model_gen, local_model_cands_dir, chunk_path,
          chunk_size_path, batch_num_path)
    eval_new_model(run_id, next_model_gen, eval_res_path)
    fs.remove_local_chunk(local_golden_chunk_dir, next_model_gen)
    model_gen = next_model_gen
    logging.info('Eval finished. Restarting self-play -> train -> eval loop.')

  logging.info('Reached number of generations. ' +
               'Continuing training past end of self-play.')

  # We have completed all self-play. Continue to train on the tail of self-play
  # data. Shuffler is responsible for notifying when there are no more chunks.
  while model_gen < config.num_generations + config.extra_train_gens:
    # Wait for chunk.
    latest_chunk_gen = fs.get_most_recent_chunk(run_id)
    while latest_chunk_gen <= model_gen:
      time.sleep(POLL_INTERVAL_S)
      latest_chunk_gen = fs.get_most_recent_chunk(run_id)

    # Found new chunk.
    logging.info(f'Found training chunk {latest_chunk_gen}.' +
                 f' Current generation is {model_gen}.')
    next_model_gen = model_gen + 1
    chunk_path = fs.download_golden_chunk(run_id, local_golden_chunk_dir,
                                          next_model_gen)
    chunk_size_path = fs.download_golden_chunk_size(run_id,
                                                    local_golden_chunk_dir,
                                                    next_model_gen)
    train(run_id, model_gen, next_model_gen, local_model_cands_dir, chunk_path,
          chunk_size_path, batch_num_path)
    eval_new_model(next_model_gen, eval_res_path)

    fs.remove_local_chunk(local_golden_chunk_dir, next_model_gen)

    model_gen = next_model_gen
    logging.info('Eval finished. Waiting for next chunk...')

  logging.info('Run is finished. Shutting down...')
  fs.signal_done(run_id, local_run_dir)


def main(_):
  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return
  if FLAGS.bin_dir == '':
    logging.error('No --bin_dir specified.')
    return

  sp_bin_path = Path(FLAGS.bin_dir, 'cc', 'selfplay', 'main')
  eval_bin_path = Path(FLAGS.bin_dir, 'cc', 'eval', 'main')
  build_trt_engine_path = Path(FLAGS.bin_dir, 'cc', 'nn', 'engine', 'scripts',
                               'build_and_run_trt_engine')

  val_ds_path = fs.download_val_ds(FLAGS.local_run_dir)
  run_config = config.parse(FLAGS.run_id)
  run_id = FLAGS.run_id

  fs_mode = 'local' if FLAGS.local_only else 'gcs'
  fs.configure_fs(mode=fs_mode, local_path=FLAGS.local_run_dir)
  loop(run_id, run_config, sp_bin_path, eval_bin_path, val_ds_path,
       build_trt_engine_path, FLAGS.local_run_dir)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
