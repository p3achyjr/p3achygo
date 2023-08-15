from __future__ import annotations

import functools
import symmetry as sym
import tensorflow as tf
from constants import *

EX_DESC = {
    'bsize': tf.io.FixedLenFeature([], tf.string),
    'board': tf.io.FixedLenFeature([], tf.string),
    'last_moves': tf.io.FixedLenFeature([], tf.string),
    'stones_atari': tf.io.FixedLenFeature([], tf.string),
    'stones_two_liberties': tf.io.FixedLenFeature([], tf.string),
    'stones_three_liberties': tf.io.FixedLenFeature([], tf.string),
    'color': tf.io.FixedLenFeature([], tf.string),
    'komi': tf.io.FixedLenFeature([], tf.float32),
    'own': tf.io.FixedLenFeature([], tf.string),
    'pi': tf.io.FixedLenFeature([], tf.string),
    'pi_aux': tf.io.FixedLenFeature([], tf.string),
    'score_margin': tf.io.FixedLenFeature([], tf.float32),
    'q30': tf.io.FixedLenFeature([], tf.float32),
    'q100': tf.io.FixedLenFeature([], tf.float32),
    'q200': tf.io.FixedLenFeature([], tf.float32),
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
  shape = (bsize * bsize + 1,)
  if tf.reduce_all(move == non_move):
    return tf.zeros(shape, dtype=tf.float32)

  is_pass = tf.reduce_all(move == pass_move)
  index = bsize * bsize + 1 if is_pass else move[0] * bsize + move[1]
  return tf.cast(tf.scatter_nd(indices=[[index]],
                               updates=tf.constant([1.0]),
                               shape=shape),
                 dtype=tf.float32)


def as_one_hot(move: tf.Tensor, bsize=BOARD_LEN) -> tf.Tensor:
  """Broadcast a move tuple to a one-hot 2D tensor."""
  non_move = tf.constant(NON_MOVE, dtype=tf.int32)
  pass_move = tf.constant(PASS_MOVE, dtype=tf.int32)
  if (tf.reduce_all(move == non_move) or tf.reduce_all(move == pass_move)):
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
  if (tf.reduce_all(loc == NON_MOVE) or tf.reduce_all(loc == PASS_MOVE)):
    return loc

  return sym.apply_loc_symmetry(symmetry, loc, grid_len)


def is_board_move(mv_index: tf.Tensor) -> tf.Tensor:
  if mv_index < 0 or mv_index == PASS_MOVE_ENCODING:
    return False

  return True


def filter_pass(input, komi, score, score_one_hot, policy, own):
  return policy != 361


def expand(tf_example):
  """Expands a tfrecord from cc/recorder/tf_recorder.cc"""
  ex = tf.io.parse_single_example(tf_example, EX_DESC)

  # keep these in sync with cc/recorder/tf_recorder.cc
  bsize = tf.cast(tf.reshape(tf.io.decode_raw(ex['bsize'], tf.uint8), shape=()),
                  dtype=tf.int32)
  board = tf.reshape(tf.io.decode_raw(ex['board'], tf.int8),
                     shape=(bsize * bsize,))
  last_moves = tf.reshape(tf.io.decode_raw(ex['last_moves'], tf.int16),
                          shape=(5,))
  stones_atari = tf.reshape(tf.io.decode_raw(ex['stones_atari'], tf.int8),
                            shape=(bsize * bsize,))
  stones_two_liberties = tf.reshape(tf.io.decode_raw(ex['stones_two_liberties'],
                                                     tf.int8),
                                    shape=(bsize * bsize,))
  stones_three_liberties = tf.reshape(tf.io.decode_raw(
      ex['stones_three_liberties'], tf.int8),
                                      shape=(bsize * bsize,))
  color = tf.reshape(tf.io.decode_raw(ex['color'], tf.int8), shape=())
  komi = ex['komi']
  own = tf.reshape(tf.io.decode_raw(ex['own'], tf.int8), shape=(bsize * bsize,))
  policy = tf.reshape(tf.io.decode_raw(ex['pi'], tf.float32),
                      shape=(bsize * bsize + 1,))
  policy_aux = tf.reshape(tf.io.decode_raw(ex['pi_aux'], tf.int16), shape=())
  score = ex['score_margin']  # score from perspective of current player.
  q30 = ex['q30']
  q100 = ex['q100']
  q200 = ex['q200']

  # cast b/c TF hates you.
  board = tf.cast(board, tf.int32)
  last_moves = tf.cast(last_moves, tf.int32)
  stones_atari = tf.cast(stones_atari, tf.int32)
  stones_two_liberties = tf.cast(stones_two_liberties, tf.int32)
  stones_three_liberties = tf.cast(stones_three_liberties, tf.int32)
  own = tf.cast(own, tf.int32)
  policy_aux = tf.cast(policy_aux, tf.int32)

  # reshape for compatibility with training.
  board = tf.reshape(board, shape=(bsize, bsize))
  last_moves = tf.map_fn(functools.partial(as_loc, bsize=bsize), last_moves)
  stones_atari = tf.reshape(stones_atari, shape=(bsize, bsize))
  stones_two_liberties = tf.reshape(stones_two_liberties, shape=(bsize, bsize))
  stones_three_liberties = tf.reshape(stones_three_liberties,
                                      shape=(bsize, bsize))
  own = tf.reshape(own, shape=(bsize, bsize))

  # apply symmetry.
  symmetry = sym.get_random_symmetry()
  board = sym.apply_grid_symmetry(symmetry, board)
  last_moves = tf.map_fn(lambda mv: apply_loc_symmetry(symmetry, mv, bsize),
                         last_moves)
  stones_atari = sym.apply_grid_symmetry(symmetry, stones_atari)
  stones_two_liberties = sym.apply_grid_symmetry(symmetry, stones_two_liberties)
  stones_three_liberties = sym.apply_grid_symmetry(symmetry,
                                                   stones_three_liberties)
  own = sym.apply_grid_symmetry(symmetry, own)
  board_policy = policy[0:bsize * bsize]
  board_policy = tf.reshape(board_policy, shape=(bsize, bsize))
  board_policy = sym.apply_grid_symmetry(symmetry, board_policy)
  board_policy = tf.reshape(board_policy, shape=(bsize * bsize,))
  policy = tf.concat([board_policy, [policy[bsize * bsize]]], axis=0)

  if is_board_move(tf.cast(policy_aux, tf.int32)):
    policy_aux = as_loc(policy_aux, bsize=bsize)
    policy_aux = sym.apply_loc_symmetry(symmetry, policy_aux, bsize)
    policy_aux = as_index(policy_aux, bsize=bsize)

  # view ownership from perspective of current player.
  own = tf.cond(color == BLACK, lambda: own, lambda: -own)

  # build input tensors.
  black_stones = get_color(board, BLACK)
  white_stones = get_color(board, WHITE)
  black_atari = get_color(stones_atari, BLACK)
  white_atari = get_color(stones_atari, WHITE)
  black_two_liberties = get_color(stones_two_liberties, BLACK)
  white_two_liberties = get_color(stones_two_liberties, WHITE)
  black_three_liberties = get_color(stones_three_liberties, BLACK)
  white_three_liberties = get_color(stones_three_liberties, WHITE)
  our_stones = tf.where(color == BLACK, black_stones, white_stones)
  opp_stones = tf.where(color == BLACK, white_stones, black_stones)
  our_atari = tf.where(color == BLACK, black_atari, white_atari)
  opp_atari = tf.where(color == BLACK, white_atari, black_atari)
  our_two_liberties = tf.where(color == BLACK, black_two_liberties,
                               white_two_liberties)
  opp_two_liberties = tf.where(color == BLACK, white_two_liberties,
                               black_two_liberties)
  our_three_liberties = tf.where(color == BLACK, black_three_liberties,
                                 white_three_liberties)
  opp_three_liberties = tf.where(color == BLACK, white_three_liberties,
                                 black_three_liberties)
  # mask last moves on a small percentage of examples to prevent net from
  # tunnel-visioning on move history.
  mask_last_moves = tf.random.uniform(()) < 0.05
  no_move = tf.zeros((bsize, bsize), dtype=tf.float32)
  input = tf.convert_to_tensor([
      our_stones,
      opp_stones,
      no_move if mask_last_moves else as_one_hot(last_moves[0], bsize=bsize),
      no_move if mask_last_moves else as_one_hot(last_moves[1], bsize=bsize),
      no_move if mask_last_moves else as_one_hot(last_moves[2], bsize=bsize),
      no_move if mask_last_moves else as_one_hot(last_moves[3], bsize=bsize),
      no_move if mask_last_moves else as_one_hot(last_moves[4], bsize=bsize),
      our_atari,
      opp_atari,
      our_two_liberties,
      opp_two_liberties,
      our_three_liberties,
      opp_three_liberties,
  ],
                               dtype=tf.float32)
  input = tf.transpose(input, perm=(1, 2, 0))  # CHW -> HWC

  # wrangle score.
  score = tf.floor(score)
  score_index = tf.cast(score + SCORE_RANGE_MIDPOINT, dtype=tf.int32)
  if score_index < 0:
    score_index = 0
  elif score_index >= SCORE_RANGE:
    score_index = SCORE_RANGE - 1

  score_index = tf.cast([[score_index]], dtype=tf.int32)
  score_one_hot = tf.cast(tf.scatter_nd(score_index, [1.0],
                                        shape=(SCORE_RANGE,)),
                          dtype=tf.float32)

  # gather passes.
  last_move_was_pass = tf.cast(
      tf.map_fn(lambda mv: 1 if tf.reduce_all(mv == PASS_MOVE) else 0,
                last_moves), tf.float32)

  # build global state tensor.
  color_indicator = tf.cast(
      tf.convert_to_tensor([color == BLACK, color == WHITE]), tf.float32)
  input_global_state = tf.concat([color_indicator, last_move_was_pass], axis=0)
  return (input, input_global_state, color, komi, score, score_one_hot, policy,
          policy_aux, own, q30, q100, q200)
