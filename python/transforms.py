from __future__ import annotations

import functools
import symmetry as sym
import tensorflow as tf
from constants import *

RL_DESC = {
    'bsize': tf.io.FixedLenFeature([], tf.string),
    'board': tf.io.FixedLenFeature([], tf.string),
    'last_moves': tf.io.FixedLenFeature([], tf.string),
    'color': tf.io.FixedLenFeature([], tf.string),
    'komi': tf.io.FixedLenFeature([], tf.float32),
    'own': tf.io.FixedLenFeature([], tf.string),
    'pi': tf.io.FixedLenFeature([], tf.string),
    'result': tf.io.FixedLenFeature([], tf.float32),
}


def get_black(board: tf.Tensor) -> tf.Tensor:
  """Return black stones as 2D tensor."""
  return tf.cast(
      tf.where(tf.math.equal(board, BLACK), board, tf.zeros_like(board)) /
      BLACK,
      dtype=tf.float32)


def get_white(board: tf.Tensor) -> tf.Tensor:
  """Return white stones as 2D tensor."""
  return tf.cast(
      tf.where(tf.math.equal(board, WHITE), board, tf.zeros_like(board)) /
      WHITE,
      dtype=tf.float32)


def get_color(board: tf.Tensor, color) -> tf.Tensor:
  """Return `color` locs as 2D tensor."""
  return tf.cast(
      tf.where(tf.math.equal(board, color), board, tf.zeros_like(board)) /
      color,
      dtype=tf.float32)


def as_pi_vec(move: tf.Tensor, bsize=BOARD_LEN) -> tf.Tensor:
  """Broadcast move tuple to 1D one-hot tensor."""
  non_move = tf.constant(NON_MOVE, dtype=tf.int32)
  pass_move = tf.constant(PASS_MOVE, dtype=tf.int32)
  pass_move_rl = tf.constant(PASS_MOVE_RL, dtype=tf.int32)
  shape = (bsize * bsize + 1,)
  if tf.reduce_all(move == non_move):
    return tf.zeros(shape, dtype=tf.float32)

  is_pass = tf.reduce_all(move == pass_move) or tf.reduce_all(
      move == pass_move_rl)
  index = bsize * bsize + 1 if is_pass else move[0] * bsize + move[1]
  return tf.cast(tf.scatter_nd(indices=[[index]],
                               updates=tf.constant([1.0]),
                               shape=shape),
                 dtype=tf.float32)


def as_one_hot(move: tf.Tensor, bsize=BOARD_LEN) -> tf.Tensor:
  """Broadcast a move tuple to a one-hot 2D tensor."""
  non_move = tf.constant(NON_MOVE, dtype=tf.int32)
  pass_move = tf.constant(PASS_MOVE, dtype=tf.int32)
  pass_move_rl = tf.constant(PASS_MOVE_RL, dtype=tf.int32)
  if (tf.reduce_all(move == non_move) or tf.reduce_all(move == pass_move) or
      tf.reduce_all(move == pass_move_rl)):
    return tf.zeros((bsize, bsize), dtype=tf.float32)

  return tf.cast(tf.scatter_nd(indices=[move],
                               updates=tf.constant([1]),
                               shape=(bsize, bsize)),
                 dtype=tf.float32)


def as_index(move: tf.Tensor, bsize=BOARD_LEN) -> tf.Tensor:
  return tf.cast(move[0] * bsize + move[1], dtype=tf.int32)


def as_loc(mv_index: tf.Tensor, bsize=BOARD_LEN) -> tf.Tensor:
  loc = tf.convert_to_tensor(
      [tf.abs(mv_index) // bsize,
       tf.abs(mv_index) % bsize])

  if mv_index < 0:
    return -loc

  return loc


def apply_loc_symmetry(symmetry: tf.Tensor, loc: tf.Tensor,
                       grid_len: int) -> tf.Tensor:
  if (tf.reduce_all(loc == NON_MOVE) or tf.reduce_all(loc == PASS_MOVE) or
      tf.reduce_all(loc == PASS_MOVE_RL)):
    return loc

  return sym.apply_loc_symmetry(symmetry, loc, grid_len)


def filter_pass(input, komi, score, score_one_hot, policy, own):
  return policy != 361


def expand_sl(ex):
  """Expands a single training example from a supervised learning dataset."""
  board, komi, color, score, last_moves, policy = (ex['board'], ex['komi'],
                                                   ex['color'], ex['result'],
                                                   ex['last_moves'],
                                                   ex['policy'])
  assert (last_moves.shape == (5, 2))

  black_stones = get_black(board)
  white_stones = get_white(board)
  fifth_move_before = as_one_hot(tf.cast(last_moves[0], dtype=tf.int32))
  fourth_move_before = as_one_hot(tf.cast(last_moves[1], dtype=tf.int32))
  third_move_before = as_one_hot(tf.cast(last_moves[2], dtype=tf.int32))
  second_move_before = as_one_hot(tf.cast(last_moves[3], dtype=tf.int32))
  first_move_before = as_one_hot(tf.cast(last_moves[4], dtype=tf.int32))

  score_index = tf.cast([[score + SCORE_RANGE_MIDPOINT]], dtype=tf.int32)
  score_one_hot = tf.cast(tf.scatter_nd(score_index, [1.0],
                                        shape=(SCORE_RANGE,)),
                          dtype=tf.float32)
  policy = as_index(tf.cast(policy, dtype=tf.int32))

  input = tf.convert_to_tensor([
      black_stones, white_stones, fifth_move_before, fourth_move_before,
      third_move_before, second_move_before, first_move_before
  ],
                               dtype=tf.float32)

  input = tf.transpose(input, perm=(1, 2, 0))  # CHW -> HWC
  own = tf.zeros(shape=(BOARD_LEN, BOARD_LEN))

  return input, [komi], score, score_one_hot, policy, own


def expand_rl(tf_example):
  """Expands a tfrecord from cc/recorder/tf_recorder.cc"""
  ex = tf.io.parse_single_example(tf_example, RL_DESC)

  # keep these in sync with cc/recorder/tf_recorder.cc
  bsize = tf.cast(tf.squeeze(
      tf.reshape(tf.io.decode_raw(ex['bsize'], tf.uint8), shape=(1,)),),
                  dtype=tf.int32)
  board = tf.reshape(tf.io.decode_raw(ex['board'], tf.int8),
                     shape=(bsize * bsize,))
  last_moves = tf.reshape(tf.io.decode_raw(ex['last_moves'], tf.int16),
                          shape=(5,))
  color = tf.squeeze(
      tf.reshape(tf.io.decode_raw(ex['color'], tf.int8), shape=(1,)))
  komi = ex['komi']
  own = tf.reshape(tf.io.decode_raw(ex['own'], tf.int8), shape=(bsize * bsize,))
  policy = tf.reshape(tf.io.decode_raw(ex['pi'], tf.float32),
                      shape=(bsize * bsize + 1,))
  score = ex['result']  # score from perspective of current player.

  # cast b/c TF hates you.
  board = tf.cast(board, dtype=tf.int32)
  last_moves = tf.cast(last_moves, dtype=tf.int32)
  own = tf.cast(own, dtype=tf.int32)

  # reshape for compatibility with training.
  board = tf.reshape(board, shape=(bsize, bsize))
  last_moves = tf.map_fn(functools.partial(as_loc, bsize=bsize), last_moves)
  own = tf.reshape(own, shape=(bsize, bsize))

  # apply symmetry.
  symmetry = sym.get_random_symmetry()
  board = sym.apply_grid_symmetry(symmetry, board)
  last_moves = tf.map_fn(lambda mv: apply_loc_symmetry(symmetry, mv, bsize),
                         last_moves)
  board_policy = policy[0:bsize * bsize]
  board_policy = tf.reshape(board_policy, shape=(bsize, bsize))
  board_policy = sym.apply_grid_symmetry(symmetry, board_policy)
  board_policy = tf.reshape(board_policy, shape=(bsize * bsize,))
  policy = tf.concat([board_policy, [policy[bsize * bsize]]], axis=0)

  # view ownership from perspective of current player.
  own = tf.cond(color == BLACK_RL, lambda: own, lambda: -own)

  # build input tensor.
  our_stones = tf.cond(color == BLACK_RL, lambda: get_color(board, BLACK_RL),
                       lambda: get_color(board, WHITE_RL))
  opp_stones = tf.cond(color == WHITE_RL, lambda: get_color(board, BLACK_RL),
                       lambda: get_color(board, WHITE_RL))
  input = tf.convert_to_tensor([
      our_stones,
      opp_stones,
      as_one_hot(last_moves[0], bsize=bsize),
      as_one_hot(last_moves[1], bsize=bsize),
      as_one_hot(last_moves[2], bsize=bsize),
      as_one_hot(last_moves[3], bsize=bsize),
      as_one_hot(last_moves[4], bsize=bsize),
  ],
                               dtype=tf.float32)
  input = tf.transpose(input, perm=(1, 2, 0))  # CHW -> HWC
  score_index = tf.cast(score + SCORE_RANGE_MIDPOINT - .5, dtype=tf.int32)
  if score_index < 0:
    score_index = 0
  elif score_index >= SCORE_RANGE:
    score_index = SCORE_RANGE - 1

  score_index = tf.cast([[score_index]], dtype=tf.int32)
  score_one_hot = tf.cast(tf.scatter_nd(score_index, [1.0],
                                        shape=(SCORE_RANGE,)),
                          dtype=tf.float32)
  score -= .5
  return input, [komi], score, score_one_hot, policy, own
