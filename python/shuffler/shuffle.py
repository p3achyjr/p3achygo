'''
Shuffler wrapper.

Invokes a C++ binary and waits until a new model is uploaded, before stopping.
'''

from __future__ import annotations

import shlex, sys, time

from absl import app, flags, logging
from google.cloud import storage
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from threading import Thread

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

FLAGS = flags.FLAGS

# keep in sync with cc/shuffler/chunk_manager.cc
GOLDEN_CHUNK_DIRNAME = 'goldens'
GCS_BUCKET = 'p3achygo'

flags.DEFINE_string('data_path', '', 'Local path to self-play data.')
flags.DEFINE_string('bin_path', '', 'Local path to shuffler binary.')
flags.DEFINE_integer('poll_model_s', 5,
                     'Interval with which to poll for a new model.')
flags.DEFINE_integer('model_gen', None, 'Model generation to build chunk for.')
flags.DEFINE_string('exclude_gens', '',
                    'Comma-separated list of data generations to exclude.')
flags.DEFINE_string(
    'gcs_run_path', '', 'GCS remote directory mapping to the current run. ' +
    'Assumes gs://p3achygo/<gcs_run_dir>/(sgf|tf|models) format.')


def make_models_prefix(gcs_run_path: str):
  return str(Path(gcs_run_path, 'models'))


def print_stdout(out: Popen.stdout):
  for line in out:
    print(line.rstrip())


def get_model_num(blob_path: str):
  p, model_name = Path(blob_path), None
  for part in p.parts:
    if part != 'models' and part.startswith('model'):
      model_name = part

  if model_name is None:
    return -1

  n = ''
  for c in reversed(model_name):
    if not c.isdigit():
      break

    n += c

  return int(n[::-1])


def get_most_recent_model(gcs_client: storage.Client, models_dir: str) -> int:
  model_blobs = gcs_client.list_blobs(GCS_BUCKET, prefix=models_dir)
  most_recent = -1
  for blob in model_blobs:
    most_recent = max(get_model_num(blob.name), most_recent)

  return most_recent


def upload_to_gcs(gcs_client: storage.Client, local_path: str, gcs_path: str):
  bucket = gcs_client.bucket(GCS_BUCKET)
  blob = bucket.blob(gcs_path)
  blob.upload_from_filename(local_path)


def main(_):
  if FLAGS.bin_path == '':
    logging.error('No binary path specified.')
    return

  if FLAGS.data_path == '':
    logging.error('No data path specified.')
    return

  if FLAGS.gcs_run_path == '':
    logging.error('No GCS run path specified.')
    return

  if FLAGS.model_gen is None:
    logging.error('No model generation specified.')
    return

  cmd = shlex.split(
      f'{FLAGS.bin_path} --data_path={FLAGS.data_path} --model_gen={FLAGS.model_gen}'
      + f' --exclude_gens={FLAGS.exclude_gens}')
  shuf_proc = Popen(cmd,
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=STDOUT,
                    universal_newlines=True)
  t = Thread(target=print_stdout, args=(shuf_proc.stdout,), daemon=True)
  t.start()

  # run cc shuffler in background. Meanwhile, continually poll for a new model,
  # and signal cc process to stop when one is found.
  gcs_client = storage.Client()
  last_most_recent = get_most_recent_model(
      gcs_client, make_models_prefix(FLAGS.gcs_run_path))
  print('Current model iteration: ', last_most_recent)
  while True:
    time.sleep(FLAGS.poll_model_s)
    most_recent = get_most_recent_model(gcs_client,
                                        make_models_prefix(FLAGS.gcs_run_path))
    if most_recent > last_most_recent:
      print('Received new model iteration: ', most_recent)
      break

  shuf_proc.communicate('\n')
  t.join()

  # upload golden chunk to GCS. shuf_proc must have exited at this point.
  local_chunk_path = Path(FLAGS.data_path, GOLDEN_CHUNK_DIRNAME,
                          f'chunk_{FLAGS.model_gen}.tfrecord.zz')
  if not local_chunk_path.exists():
    raise Exception(f'Golden Chunk not found: {str(local_chunk_path)}')

  gcs_chunk_path = Path(FLAGS.gcs_run_path, GOLDEN_CHUNK_DIRNAME,
                        f'chunk_{FLAGS.model_gen}.tfrecord.zz')

  upload_to_gcs(gcs_client, str(local_chunk_path), str(gcs_chunk_path))
  print(f'Uploaded {local_chunk_path} to gs://p3achygo/{gcs_chunk_path}')


if __name__ == '__main__':
  app.run(main)
