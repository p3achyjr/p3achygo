'''
Self-Play wrapper.

Invokes C++ selfplay binary until a new model is detected. Then downloads the
new model and restarts the C++ binary.
'''

from __future__ import annotations

import sys
import rl_loop.sp_loop as sp

from absl import app, flags, logging
from rl_loop.trt_batch_size import trt_batch_size

FLAGS = flags.FLAGS

flags.DEFINE_string('bin_path', '', 'Local path to self-play binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')
flags.DEFINE_string('local_run_dir', '/tmp/p3achygo',
                    'Local path to store local models and data.')


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No --bin_path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  sp.loop(FLAGS.bin_path,
          FLAGS.run_id,
          FLAGS.local_run_dir,
          num_threads=trt_batch_size())


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
