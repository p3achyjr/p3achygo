import tensorflow as tf
import tensorflow_datasets as tfds

from absl import app, flags, logging
from constants import *
from pathlib import Path

import transforms

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
  black, white = transforms.get_black(board), transforms.get_white(board)
  board = black * BLACK_RL + white * WHITE_RL

  own = tf.zeros((BOARD_LEN, BOARD_LEN), dtype=tf.int8)
  # tf.map_fn error spams.
  last_move_0 = transforms.as_index(tf.cast(last_moves[0], dtype=tf.int32))
  last_move_1 = transforms.as_index(tf.cast(last_moves[1], dtype=tf.int32))
  last_move_2 = transforms.as_index(tf.cast(last_moves[2], dtype=tf.int32))
  last_move_3 = transforms.as_index(tf.cast(last_moves[3], dtype=tf.int32))
  last_move_4 = transforms.as_index(tf.cast(last_moves[4], dtype=tf.int32))
  last_moves = tf.convert_to_tensor(
      [last_move_0, last_move_1, last_move_2, last_move_3, last_move_4])
  policy = transforms.as_pi_vec(tf.cast(policy, dtype=tf.int32))
  score = tf.cast(score, tf.float32) + .5
  color = tf.cast(BLACK_RL,
                  tf.int8)  # every position in the SL dataset is black to move.

  feature = {
      "bsize": _bytes_feature(tf.constant(BOARD_LEN), tf.uint8),
      "board": _bytes_feature(board, tf.int8),
      "last_moves": _bytes_feature(last_moves, tf.int16),
      "color": _bytes_feature(color, tf.int8),
      "komi": _float_feature(komi),
      "own": _bytes_feature(own, tf.int8),
      "pi": _bytes_feature(policy, tf.float32),
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

  # tf.config.run_functions_eagerly(True)
  save_filename = Path(FLAGS.save_dir, 'val.tfrecord.zz')
  chunk_len = 25600

  # overdraw to prevent large numbers of examples from a single game.
  ds = tfds.load(
      DATASET,
      split=[f'train[80000000:80500000]'],  # a guess.
      shuffle_files=True)[0]
  ds = ds.shuffle(500000)
  ds = ds.take(chunk_len)
  ds = ds.map(tf_serialize)
  ds = ds.ignore_errors()
  writer = tf.data.experimental.TFRecordWriter(str(save_filename),
                                               compression_type='ZLIB')
  writer.write(ds)


if __name__ == '__main__':
  app.run(main)
