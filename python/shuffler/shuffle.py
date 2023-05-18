from __future__ import annotations

import tensorflow as tf
from absl import app, flags, logging
import os, time

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', '',
                    'Folder under which self-play data is stored. Assumes that ' +
                    'sgfs are stored in <data_path>/sgf, and tfrecords in ' +
                    '<data_path>/tf.')
flags.DEFINE_integer('poll_interval_s', 300,
                     'Interval in seconds between polling for new data ' + 
                     '(default 5 minutes).')


def find_num(s: str) -> int:
  """
  Finds the first number in a string.
  """
  num = ''
  for c in s:
    if c.isdigit():
      num += c

  if num:
    return int(num)
  
  return -1

def poll_loop(data_path : str, poll_interval_s : int):
  """
  Polls the data_path for new data and shuffles it.
  Assumes that sgfs are named as `game_{i}.sgf` and tfrecords as
  `batch_{i}.tfrecord.zz`.
  """
  sgf_path = os.path.join(data_path, 'sgf')
  tf_path = os.path.join(data_path, 'tf')
  last_sgf_num, last_tf_num = -1, -1
  while True:
    time.sleep(poll_interval_s)
    logging.info('Polling for new data...')

    # Get the list of files in the data_path.
    sgfs = os.listdir(sgf_path)
    tfs = os.listdir(tf_path)
    if not sgfs or not tfs:
      logging.info('No games.')
      continue

    # Get the most recent file.
    sgf_num, tf_num = (max([find_num(sgf) for sgf in sgfs]),
                       max([find_num(tfrec) for tfrec in tfs]))
    if sgf_num == last_sgf_num and tf_num == last_tf_num:
      logging.info('No new data.')
      continue
    
    # load tfrecords into memory. Assume tfrecords are shuffled.
    


def main(argv):
  if FLAGS.data_path == '':
    logging.error('Please specify a data path.')
    return

  poll_loop(FLAGS.data_path, FLAGS.poll_interval_s)


if __name__ == '__main__':
  app.run(main)
