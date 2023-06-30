'''
Self-Play wrapper.

Invokes C++ selfplay binary until a new model is detected. Then downloads the
new model and restarts the C++ binary.
'''

from __future__ import annotations

import sys
import rl_loop.sp_loop as sp
import rl_loop.config as config

from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_generations', 10,
                     'Number of generations of self-play.')
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

  run_config = config.parse(FLAGS.run_id)
  sp.loop(FLAGS.bin_path, FLAGS.run_id, FLAGS.local_run_dir,
          run_config.num_sp_threads)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
