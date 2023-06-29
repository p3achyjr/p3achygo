'''
Shuffler wrapper.

Invokes a C++ binary and waits until a new model is uploaded, before stopping.
'''

from __future__ import annotations

import shlex, signal, sys, time
import gcs_utils as gcs
import rl_loop.config as config

from absl import app, flags, logging
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

FLAGS = flags.FLAGS
POLL_INTERVAL_S = 10

running = True

flags.DEFINE_string('bin_path', '', 'Local path to shuffler binary.')
flags.DEFINE_string('run_id', '', 'ID corresponding to the current run.')


def handle_shutdown(signum, _):
  logging.info(f'Received Shutdown Signal {signum}')
  global running
  running = False


def print_stdout(out: Popen.stdout):  # pytype : disable=unbound-type-param
  for line in out:
    print(line.rstrip())


def loop(bin_path: str, run_id: str, config: config.RunConfig):
  '''
  Continually produces new training chunks until reaching a specified number of
  generations.
  '''
  chunk_gen = gcs.get_most_recent_chunk(run_id) + 1
  data_dir = str(Path(config.shared_volume_path, 'chunks'))
  logging.info(f'Using {data_dir} as shared data dir.')
  while chunk_gen <= config.num_generations:
    cmd = shlex.split(f'{bin_path} --data_path={data_dir}' +
                      f' --gen={chunk_gen}' +
                      f' --games_per_gen={config.games_per_gen}')
    shuf_proc = Popen(cmd,
                      stdin=PIPE,
                      stdout=PIPE,
                      stderr=STDOUT,
                      universal_newlines=True)
    t = Thread(target=print_stdout, args=(shuf_proc.stdout,), daemon=True)
    t.start()

    while running and shuf_proc.poll() is None:
      time.sleep(POLL_INTERVAL_S)

    if shuf_proc.poll() is None:
      shuf_proc.communicate('\n')  # force a flush just to be safe.

    logging.info(f'Shuffler exited with status {shuf_proc.poll()}')

    # Upload chunk.
    gcs.upload_chunk(run_id, gcs.local_chunk_dir(data_dir), chunk_gen)
    logging.info(f'Uploaded chunk gen {chunk_gen} to gs://p3achygo/{run_id}')
    chunk_gen += 1

  logging.info(f'Chunk gen: {chunk_gen}. Shutting down.')


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No --bin_path specified.')
    return

  if FLAGS.run_id == '':
    logging.error('No --run_id specified.')
    return

  run_config = config.parse(FLAGS.run_id)
  loop(FLAGS.bin_path, FLAGS.run_id, run_config)


if __name__ == '__main__':
  signal.signal(signal.SIGINT, handle_shutdown)
  signal.signal(signal.SIGTERM, handle_shutdown)
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
