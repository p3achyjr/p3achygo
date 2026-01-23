'''
Self-Play wrapper.

Invokes C++ selfplay binary until a new model is detected. Then downloads the
new model and restarts the C++ binary.

Supports multiple GPUs by spawning a separate process per GPU.
'''

from __future__ import annotations

import multiprocessing
import sys
import rl_loop.sp_loop as sp

from absl import app, flags, logging
from rl_loop.constants import SELFPLAY_BATCH_SIZE
from rl_loop.gpu_utils import get_gpu_count

FLAGS = flags.FLAGS

flags.DEFINE_string('bin_path', '', 'Local path to self-play binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path to store local models and data.')
flags.DEFINE_integer('num_gpus', 0,
                     'Number of GPUs to use. 0 means auto-detect.')


def run_on_gpu(gpu_device: int):
  '''Runs self-play loop on a specific GPU.'''
  sp.loop(FLAGS.bin_path,
          FLAGS.run_id,
          FLAGS.local_run_dir,
          num_threads=SELFPLAY_BATCH_SIZE,
          gpu_device=gpu_device)


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No --bin_path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  num_gpus = FLAGS.num_gpus if FLAGS.num_gpus > 0 else get_gpu_count()
  logging.info(f'Starting self-play on {num_gpus} GPU(s)')

  if num_gpus == 1:
    # Single GPU: run directly without spawning extra processes
    run_on_gpu(gpu_device=0)
  else:
    # Multi-GPU: spawn a process per GPU
    processes = []
    for gpu_id in range(num_gpus):
      p = multiprocessing.Process(target=run_on_gpu, args=(gpu_id,))
      p.start()
      processes.append(p)
      logging.info(f'Started self-play process on GPU {gpu_id}')

    for p in processes:
      p.join()


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
