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
from queue import Queue
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS

POLL_INTERVAL_S = 10
BATCH_SIZE = 256

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


def eval(eval_bin_path: str, eval_write_path: str, cur_model_path: str,
         cand_model_path: str) -> EvalResult:
  '''`cur_model_path` and `cand_model_path` are the _base_ paths of the models.'''
  cur_model_path_trt = str(Path(cur_model_path, '_trt'))
  cand_model_path_trt = str(Path(cand_model_path, '_trt'))

  cmd = shlex.split(f'{eval_bin_path}' +
                    f' --cur_model_path={cur_model_path_trt}' +
                    f' --cand_model_path={cand_model_path_trt}' +
                    f' --res_write_path={eval_write_path}')
  eval_proc = Popen(cmd,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=STDOUT,
                    universal_newlines=True)
  t = Thread(target=print_stdout, args=(eval_proc.stdout,), daemon=True)
  t.start()
  eval_proc.wait()

  with open(eval_write_path) as f:
    cand_rel_elo = float(f.read())
    winner = EvalResult.CUR if cand_rel_elo < 0 else EvalResult.CAND

    return EvalResult(winner, cand_rel_elo)


def loop(run_id: str, config: config.RunConfig, model: P3achyGoModel,
         model_path: str, model_gen: int, sp_bin_path: str, eval_bin_path: str,
         val_ds_path: str, local_run_dir: str, is_gpu: bool):
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
  val_ds = tf.data.TFRecordDataset(val_ds_path, compression_type='ZLIB')
  val_ds = val_ds.map(transforms.expand_rl, num_parallel_calls=tf.data.AUTOTUNE)
  val_ds = val_ds.batch(BATCH_SIZE)
  val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

  if model_gen == 0:
    logging.info(f'Collecting initial validation stats...')
    train.val(model, mode=train.Mode.RL, val_ds=val_ds)

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
      logging.info(f'No new training chunk. Sleeping {POLL_INTERVAL_S}s...')
      time.sleep(POLL_INTERVAL_S)
      latest_chunk_gen = gcs.get_most_recent_chunk(run_id)

    # Found new chunk.
    logging.info(f'Found training chunk {latest_chunk_gen}.' +
                 f' Current generation is {model_gen}')
    model_gen = latest_chunk_gen
    sp_queue.put(())  # Send any message to trigger shutdown.
    sp_thread.join()

    chunk_path = gcs.download_golden_chunk(run_id, local_run_dir, model_gen)
    rl_loop.train.train_one_gen(model, model_gen, chunk_path, val_ds)

    cand_model_path = model_utils.save_trt(model, val_ds_path, local_run_dir,
                                           model_gen)

    # Run eval.
    eval_result = eval(eval_bin_path, eval_res_path, model_path,
                       cand_model_path)
    if eval_result.winner == EvalResult.CAND:
      # The cand model is stronger. Upload it as new golden.
      logging.info(f'Uploading model {cand_model_path} as new golden')
      gcs.upload(run_id, cand_model_path)

    logging.info('Eval finished. Restarting self-play -> train -> eval loop.')


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

  is_gpu = False
  if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    is_gpu = True
  else:
    logging.warning('No GPU detected.')

  val_ds_path = gcs.download_val_ds(FLAGS.local_run_dir)
  run_config = config.parse(FLAGS.run_id)

  model_gen, model = gcs.get_most_recent_model(FLAGS.run_id), None
  if model_gen < 0:
    model = model_utils.new_model(name=f'p3achygo_{FLAGS.run_id}')
    # upload for self-play to pick up.
    model_path = model_utils.save_trt_and_upload(model,
                                                 FLAGS.val_ds_path,
                                                 FLAGS.local_run_dir,
                                                 0,
                                                 run_id=FLAGS.run_id)
  else:
    model_path = gcs.download_model(FLAGS.run_id, FLAGS.local_run_dir,
                                    model_gen)
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects())

  loop(FLAGS.run_id, run_config, model, model_path, model_gen,
       FLAGS.sp_bin_path, FLAGS.eval_bin_path, val_ds_path, FLAGS.local_run_dir,
       is_gpu)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
