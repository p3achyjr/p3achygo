'''
Starts Self-Play, then Trains when a chunk is ready, then runs Eval.
'''

from __future__ import annotations

import gcs_utils as gcs
import shlex, sys, time
import train
import tensorflow as tf
import transforms
import rl_loop.config as config
import rl_loop.model_utils as model_utils
import rl_loop.sp_loop as sp
import rl_loop.train

from absl import app, flags, logging
from constants import *
from dataclasses import dataclass
from pathlib import Path
from model import P3achyGoModel
from multiprocessing import Process
from queue import Queue
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10
BATCH_SIZE = 256
EVAL_CACHE_SIZE = 32768

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


def eval(eval_bin_path: str, eval_res_path: str, cur_model_path: str,
         cand_model_path: str) -> EvalResult:
  '''`cur_model_path` and `cand_model_path` are the _base_ paths of the models.'''
  cur_model_path_trt = str(Path(cur_model_path, '_trt'))
  cand_model_path_trt = str(Path(cand_model_path, '_trt'))

  cmd = shlex.split(f'{eval_bin_path}' +
                    f' --cur_model_path={cur_model_path_trt}' +
                    f' --cand_model_path={cand_model_path_trt}' +
                    f' --res_write_path={eval_res_path}' +
                    f' --cache_size={EVAL_CACHE_SIZE}')
  eval_proc = Popen(cmd,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=STDOUT,
                    universal_newlines=True)
  t = Thread(target=print_stdout, args=(eval_proc.stdout,), daemon=True)
  t.start()
  eval_proc.wait()

  with open(eval_res_path) as f:
    cand_rel_elo = float(f.read())
    winner = EvalResult.CUR if cand_rel_elo < 0 else EvalResult.CAND
    logging.info(f'Winner: {winner}, Cand Elo: {cand_rel_elo}')

    return EvalResult(winner, cand_rel_elo)


def loop(run_id: str, config: config.RunConfig, models_dir: str, model_gen: int,
         sp_bin_path: str, eval_bin_path: str, val_ds_path: str,
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
  eval_res_path = str(Path(local_run_dir, 'eval_res.txt'))
  while model_gen < config.num_generations:
    # Start self-play.
    logging.info(f'Model Generation: {model_gen}')
    sp_queue = Queue()
    sp_thread = Thread(target=sp.loop,
                       args=(sp_bin_path, run_id, local_run_dir,
                             config.shared_volume_path, config.num_sp_threads,
                             sp_queue))
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

    chunk_path = gcs.download_golden_chunk(run_id, local_run_dir,
                                           latest_chunk_gen)

    # Train for one generation.
    logging.info(f'Training model {model_gen}...')
    cmd = shlex.split(f'python -m python.rl_loop.train_one_gen' +
                      f' --models_dir={models_dir}' + f' --gen={model_gen}' +
                      f' --next_gen={latest_chunk_gen}' +
                      f' --chunk_path={chunk_path}' +
                      f' --val_ds_path={val_ds_path}')
    train_proc = Popen(cmd,
                       stdin=PIPE,
                       stdout=PIPE,
                       stderr=STDOUT,
                       universal_newlines=True)
    t = Thread(target=print_stdout, args=(train_proc.stdout,), daemon=True)
    t.start()
    train_proc.wait()

    cur_model_path = str(Path(models_dir, gcs.MODEL_FORMAT.format(model_gen)))
    cand_model_path = str(
        Path(models_dir, gcs.MODEL_FORMAT.format(latest_chunk_gen)))

    # Run eval.
    eval_result = eval(eval_bin_path, eval_res_path, cur_model_path,
                       cand_model_path)
    if eval_result.winner == EvalResult.CAND:
      # The cand model is stronger. Upload it as new golden.
      logging.info(f'Uploading model {cand_model_path} as new golden')
      gcs.upload_model(run_id, models_dir, latest_chunk_gen)
    model_gen = latest_chunk_gen

    logging.info('Eval finished. Restarting self-play -> train -> eval loop.')

  logging.info('Reached number of generations. Signaling for shutdown...')
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

  models_dir = str(Path(run_config.shared_volume_path, gcs.MODELS_DIR))
  model_gen = gcs.get_most_recent_model(FLAGS.run_id)
  if model_gen < 0:
    model_gen = 0
    model = model_utils.new_model(name=f'p3achygo_{FLAGS.run_id}')
    # upload for self-play to pick up.
    model_utils.save_trt_and_upload(model,
                                    FLAGS.val_ds_path,
                                    models_dir,
                                    gen=0,
                                    run_id=FLAGS.run_id)
  else:
    gcs.download_model(FLAGS.run_id, models_dir, model_gen)

  loop(FLAGS.run_id, run_config, models_dir, model_gen, FLAGS.sp_bin_path,
       FLAGS.eval_bin_path, val_ds_path, FLAGS.local_run_dir)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
