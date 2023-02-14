'''
Main method by which to yield training examples from games.
'''

from __future__ import annotations

import enum
import itertools
import math
import numpy as np
import tensorflow_datasets as tfds

from absl import logging
from collections import deque
from sgfmill import sgf

from board import GoBoard, GameResult
from datasets.common.constants import *


class GeneratorMode(enum.Enum):
  UNKNOWN = 0,
  ALL = 1,
  SCORED = 2,
  UNSCORED = 3,


class ExampleGenerator:
  '''
  Class containing logic for generating training data from sample games.
  '''

  def __init__(self, root_dir: tfds.core.Path, mode: GeneratorMode):
    self.root_dir = root_dir
    self.mode = mode

  def generate(self):
    paths = deque()
    paths.append(self.root_dir)
    num_examples = 0

    generators_per_sgf = []
    while paths:
      path = paths.popleft()
      if path.is_dir():
        for subpath in path.iterdir():
          paths.append(subpath)

      elif str(path).endswith('.sgf'):
        generators_per_sgf.append(self.generate_examples_from_sgf(path))
        num_examples += 1

    return itertools.chain.from_iterable(generators_per_sgf)

  def generate_examples_from_sgf(self, path: tfds.core.Path):
    """
    Extracts main move line from sgf.
    
    Yields one example per (board_pos, move) tuple in each game.
    """

    def player_from_encoding(encoding: str):
      if encoding.lower() == 'b':
        return GameResult.BLACK
      elif encoding.lower() == 'w':
        return GameResult.WHITE
      else:
        raise ValueError(f'Unknown Color: {encoding}')

    def score_for_player(result: GameResult, player: int):
      if result.is_unknown():
        raise Exception('Unknown Result. Should skip game.')

      player_did_win = result.winner == player
      if player_did_win:
        return RESIGN_WIN_SCORE if result.is_by_resignation() else math.floor(
            result.score_diff)
      else:
        return RESIGN_LOSS_SCORE if result.is_by_resignation() else math.floor(
            -result.score_diff)

    def rot90_points(i, j, k):
      if (i, j) == NON_MOVE or (i, j) == PASS_MOVE:
        return (i, j)

      if k == 0:
        return (i, j)
      elif k == 1:
        return (BOARD_LEN - 1 - j, i)
      elif k == 2:
        return (BOARD_LEN - 1 - i, BOARD_LEN - 1 - j)
      elif k == 3:
        return (j, BOARD_LEN - 1 - i)

    def fliplr_points(i, j):
      if (i, j) == NON_MOVE or (i, j) == PASS_MOVE:
        return (i, j)

      return (i, BOARD_LEN - 1 - j)

    with path.open(mode='rb') as f:
      game = sgf.Sgf_game.from_bytes(f.read())
      main_line = game.get_main_sequence()

    handicap = game.get_handicap()
    if handicap != None and handicap > 0:
      # skip handicap games for now
      return

    result = game.get_root().get('RE')
    result = GameResult.parse_score(result)
    if result.is_unknown():
      # skip game for simplicity when training
      return
    elif self.mode == GeneratorMode.SCORED:
      # only return scored games in this case
      if result.is_by_resignation():
        return
    elif self.mode == GeneratorMode.UNSCORED:
      # only return by-resignation games in this case
      if not result.is_by_resignation():
        return

    komi = game.get_komi()
    board = GoBoard(BOARD_LEN)
    move_num = 0
    move_queue = [NON_MOVE] * NUM_LAST_MOVES
    color_to_move = 0  # 0 for Black, 1 for White
    for node in main_line:
      move = node.get_move()
      if move is None or move[0] is None or move[1] is None:
        continue

      (c, (i, j)) = move
      board_arr = np.array(board.as_black() if c ==
                           'b' else board.as_white()).astype(np.int8)

      # total of 8 unique rotation + reflection combinations
      num_rotations = 4
      year, month = path.parent.parent.name, path.parent.name
      for rot_index in range(num_rotations):
        key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref0'
        b = np.rot90(board_arr, rot_index)
        i_prime, j_prime = rot90_points(i, j, rot_index)

        current_player = player_from_encoding(c)
        current_player_score_diff = score_for_player(result, current_player)
        komi_as_player = komi if current_player == GameResult.WHITE else 0.0

        yield key, {
            'metadata': key,
            'board': b,
            'komi': komi_as_player,
            'color': color_to_move,
            'result': current_player_score_diff,
            'last_moves': [
                rot90_points(i, j, rot_index)
                for (i, j) in move_queue[-NUM_LAST_MOVES:]
            ],
            'policy': (i_prime, j_prime)
        }

      # now reflect and repeat operations
      board_arr = np.fliplr(board_arr)
      i_ref, j_ref = fliplr_points(i, j)
      for rot_index in range(num_rotations):
        key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref1'
        b = np.rot90(board_arr, rot_index)
        i_prime, j_prime = rot90_points(i_ref, j_ref, rot_index)

        yield key, {
            'metadata': key,
            'board': b,
            'komi': komi_as_player,
            'color': color_to_move,
            'result': current_player_score_diff,
            'last_moves': [
                rot90_points(i, j, rot_index)
                for (i, j) in move_queue[-NUM_LAST_MOVES:]
            ],
            'policy': (i_prime, j_prime)
        }

      if c == 'b':
        did_move = board.move_black(i, j)
      elif c == 'w':
        did_move = board.move_white(i, j)
      else:
        raise Exception(f'Unknown Color while parsing sgf {path}')

      if not did_move:
        # something went wrong, return
        logging.warning(f'Sample: {key}')
        logging.warning(f'Failed to move: {i}, {j}')
        logging.warning(f'\n{board.print()}')
        return

      move_num += 1
      move_queue.append((i, j))
      color_to_move ^= 1

    if not result.is_unknown() and not result.is_by_resignation():
      # add two pass moves, since this is a scored game.
      current_player = GameResult.BLACK if color_to_move == 0 else GameResult.WHITE
      current_player_score_diff = score_for_player(result, current_player)
      komi_as_player = komi if current_player == GameResult.WHITE else 0.0

      num_rotations = 4
      for _ in range(2):
        for rot_index in range(num_rotations):
          key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref0'
          b = np.rot90(board_arr, rot_index)

          yield key, {
              'metadata': key,
              'board': b,
              'komi': komi_as_player,
              'color': color_to_move,
              'result': current_player_score_diff,
              'last_moves': [
                  rot90_points(i, j, rot_index)
                  for (i, j) in move_queue[-NUM_LAST_MOVES:]
              ],
              'policy': PASS_MOVE,
          }

        board_arr = np.fliplr(board_arr)
        for rot_index in range(num_rotations):
          key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref1'
          b = np.rot90(board_arr, rot_index)

          yield key, {
              'metadata': key,
              'board': b,
              'komi': komi_as_player,
              'color': color_to_move,
              'result': current_player_score_diff,
              'last_moves': [
                  rot90_points(i, j, rot_index)
                  for (i, j) in move_queue[-NUM_LAST_MOVES:]
              ],
              'policy': PASS_MOVE,
          }

        move_num += 1
        move_queue.append(PASS_MOVE)
        color_to_move ^= 1
