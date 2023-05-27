'''
Shuffler wrapper.

Invokes a C++ binary and waits until a new model is uploaded, before stopping.
'''

from __future__ import annotations

import shlex, sys, time
import gcs_utils as gcs

from absl import app, flags, logging
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error

FLAGS = flags.FLAGS
POLL_INTERVAL_S = 10

flags.DEFINE_string('data_path', '', 'Local path to self-play data.')
flags.DEFINE_string('bin_path', '', 'Local path to shuffler binary.')
flags.DEFINE_integer('model_gen', None, 'Model generation to build chunk for.')
flags.DEFINE_string('exclude_gens', '',
                    'Comma-separated list of data generations to exclude.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No binary path specified.')
    return

  if FLAGS.data_path == '':
    logging.error('No data path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No run ID specified.')
    return

  if FLAGS.model_gen is None:
    logging.error('No model generation specified.')
    return

  cmd = shlex.split(f'{FLAGS.bin_path} --data_path={FLAGS.data_path}' +
                    f' --model_gen={FLAGS.model_gen}' +
                    f' --exclude_gens={FLAGS.exclude_gens}')
  shuf_proc = Popen(cmd,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=STDOUT,
                    universal_newlines=True)
  t = Thread(target=print_stdout, args=(shuf_proc.stdout,), daemon=True)
  t.start()

  # run cc shuffler in background. Meanwhile, continually poll for a new model,
  # and signal cc process to stop when one is found.
  last_most_recent = gcs.get_most_recent_model(FLAGS.run_id)
  print('Current model iteration: ', last_most_recent)
  while True:
    time.sleep(POLL_INTERVAL_S)
    most_recent = gcs.get_most_recent_model(FLAGS.run_id)
    if most_recent > last_most_recent:
      print('Received new model iteration: ', most_recent)
      break

  shuf_proc.communicate('\n')
  t.join()

  gcs.upload_chunk(FLAGS.run_id, gcs.local_chunk_dir(FLAGS.data_path),
                   FLAGS.model_gen)
  print(f'Uploaded chunk gen {FLAGS.model_gen} to gs://p3achygo/{FLAGS.run_id}')


if __name__ == '__main__':
  app.run(main)
