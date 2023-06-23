from __future__ import annotations

import tensorflow as tf

IDENTITY = 0
ROT90 = 1
ROT180 = 2
ROT270 = 3
FLIP = 4
FLIPROT90 = 5
FLIPROT180 = 6
FLIPROT270 = 7

__SYM_MAX = 8


def get_random_symmetry() -> tf.Tensor:
  return tf.random.uniform((), dtype=tf.int32, minval=0, maxval=__SYM_MAX)


def flip(x: tf.Tensor) -> tf.Tensor:
  return tf.reverse(x, axis=[1])


def rotate(x: tf.Tensor, k: int) -> tf.Tensor:
  return tf.experimental.numpy.rot90(x, k=k, axes=(1, 0))


def flip_loc(loc: tf.Tensor, n: int) -> tf.Tensor:
  assert loc.shape == (2,)
  return tf.convert_to_tensor([loc[0], n - loc[1] - 1])


def rotate_loc(loc: tf.Tensor, k: int, n: int) -> tf.Tensor:
  assert loc.shape == (2,)
  if k == 1:
    return tf.convert_to_tensor([loc[1], n - loc[0] - 1])
  elif k == 2:
    return tf.convert_to_tensor([n - loc[0] - 1, n - loc[1] - 1])
  elif k == 3:
    return tf.convert_to_tensor([n - loc[1] - 1, loc[0]])

  return loc


def apply_grid_symmetry(sym: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
  if sym == ROT90:
    return rotate(grid, 1)
  elif sym == ROT180:
    return rotate(grid, 2)
  elif sym == ROT270:
    return rotate(grid, 3)
  elif sym == FLIP:
    return flip(grid)
  elif sym == FLIPROT90:
    return rotate(flip(grid), 1)
  elif sym == FLIPROT180:
    return rotate(flip(grid), 2)
  elif sym == FLIPROT270:
    return rotate(flip(grid), 3)
  else:
    return grid


def apply_loc_symmetry(sym: tf.Tensor, loc: tf.Tensor,
                       grid_len: int) -> tf.Tensor:
  if sym == ROT90:
    return rotate_loc(loc, 1, grid_len)
  elif sym == ROT180:
    return rotate_loc(loc, 2, grid_len)
  elif sym == ROT270:
    return rotate_loc(loc, 3, grid_len)
  elif sym == FLIP:
    return flip_loc(loc, grid_len)
  elif sym == FLIPROT90:
    return rotate_loc(flip_loc(loc, grid_len), 1, grid_len)
  elif sym == FLIPROT180:
    return rotate_loc(flip_loc(loc, grid_len), 2, grid_len)
  elif sym == FLIPROT270:
    return rotate_loc(flip_loc(loc, grid_len), 3, grid_len)
  else:
    return loc
