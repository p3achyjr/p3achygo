import tensorflow as tf
import tensorflow_datasets as tfds

from absl import app, flags, logging
from constants import *
from pathlib import Path

FLAGS = flags.FLAGS
DATASET = 'badukmovies_all'

flags.DEFINE_string('save_dir', '', 'Where to save dataset.')


# Copied from https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value: tf.Tensor, dtype):
  """Returns a bytes_list from a string / byte."""
  value = tf.cast(value, dtype)

  return tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[value.numpy().tobytes()]))


def _float_feature(value: tf.Tensor):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Keep in sync with //cc/recorder/tf_recorder.cc:MakeTfExample.
def serialize(board, komi, _, score, last_moves, policy):
  assert (last_moves.shape == (5, 2))
  own = tf.zeros((BOARD_LEN, BOARD_LEN), dtype=tf.int8)
  last_moves = tf.cast(last_moves, dtype=tf.int16)
  last_moves = tf.map_fn(lambda move: move[0] * BOARD_LEN + move[1], last_moves)
  policy = tf.cast(policy, dtype=tf.uint16)
  policy = policy[0] * BOARD_LEN + policy[1]
  score = tf.cast(score, tf.float32) + .5
  color = tf.cast(BLACK_RL,
                  tf.int8)  # every position in the SL dataset is black to move.

  feature = {
      "bsize": _bytes_feature(BOARD_LEN, tf.uint8),
      "board": _bytes_feature(board, tf.int8),
      "last_moves": _bytes_feature(last_moves, tf.int16),
      "color": _bytes_feature(color, tf.int8),
      "komi": _float_feature(komi),
      "own": _bytes_feature(own, tf.int8),
      "pi": _bytes_feature(policy, tf.uint16),
      "result": _float_feature(score),
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def tf_serialize(ex):
  board, komi, color, score, last_moves, policy = (ex['board'], ex['komi'],
                                                   ex['color'], ex['result'],
                                                   ex['last_moves'],
                                                   ex['policy'])
  tf_string = tf.py_function(
      serialize,
      (board, komi, color, score, last_moves,
       policy),  # Pass these args to the above function.
      tf.string)  # The return type is `tf.string`.
  return tf_string


def main(_):
  if not FLAGS.save_dir:
    logging.warning('Please provide --save_dir.')
    return

  tf.config.run_functions_eagerly(True)
  save_filename = Path(FLAGS.save_dir, 'calib.tfrecord.zz')

  chunk_len = 25600
  ds = tfds.load(
      DATASET,
      split=['train[80000000:80100000]'],  # a guess.
      shuffle_files=True)[0]
  ds = ds.shuffle(100000)
  ds = ds.take(chunk_len)
  ds = ds.map(tf_serialize, num_parallel_calls=8)
  ds = ds.ignore_errors()
  writer = tf.data.experimental.TFRecordWriter(str(save_filename),
                                               compression_type='ZLIB')
  writer.write(ds)


if __name__ == '__main__':
  app.run(main)