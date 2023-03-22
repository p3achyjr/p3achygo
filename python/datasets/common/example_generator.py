'''
Main method by which to yield training examples from games.
'''

from __future__ import annotations

import enum
import itertools
import math
import numpy as np
import random
import tensorflow_datasets as tfds

from absl import logging
from collections import deque
from sgfmill import sgf

from board import GoBoard, GameResult
from constants import *
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
      if result.is_unknown() or result.is_by_resignation():
        raise Exception('Unknown Score.')

      player_did_win = result.winner == player
      return math.floor(
          result.score_diff if player_did_win else -result.score_diff)

    def score_est_for_resign(move_count: int):

      def bound(lb, ub, x):
        return min(max(x, lb), ub)

      if result.is_unknown() or not result.is_by_resignation():
        raise Exception(
            'Result is Unknown, or getting score est for scored game')

      min_mc, bound_range, bound_scale = 150, 100, 5
      # adjust bound based on move count.
      # move count < 150: no adjustment.
      # uniform scaling up to 250 moves, where at >250, sub 5 from bound.
      bound_adjustment = math.floor(
          bound(0, bound_range, move_count - min_mc) * bound_scale /
          bound_range)
      score_est = random.randint(RESIGN_SCORE_EST_LB - bound_adjustment,
                                 RESIGN_SCORE_EST_UB - bound_adjustment)
      return score_est

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

    try:
      with path.open(mode='rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
        board_size = game.get_root().get_size()
        main_line = game.get_main_sequence()

      handicap = game.get_handicap()
      result = game.get_root().get('RE')
      komi = game.get_komi()
    except:
      return

    if board_size != BOARD_LEN:
      # ignore non 19 x 19 games
      return

    if handicap != None and handicap > 0:
      # skip handicap games for now
      return

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

    board = GoBoard(BOARD_LEN)
    move_num = 0
    move_queue = [NON_MOVE] * NUM_LAST_MOVES
    color_to_move = 0  # 0 for Black, 1 for White
    total_move_count = len(main_line)

    # fill score for resigned game
    if result.is_by_resignation():
      score_est = score_est_for_resign(total_move_count)
      score_black =\
        score_est if result.winner == GameResult.BLACK else -score_est - 1
      score_white =\
        score_est if result.winner == GameResult.WHITE else -score_est - 1
    else:
      score_black = score_for_player(result, GameResult.BLACK)
      score_white = score_for_player(result, GameResult.WHITE)

    year, month = path.parent.parent.name, path.parent.name
    for node in main_line:
      move = node.get_move()
      if move is None or move[0] is None or move[1] is None:
        continue

      (c, (i, j)) = move
      board_arr = board.as_black() if c == 'b' else board.as_white()

      current_player = player_from_encoding(c)
      current_player_score_diff =\
        score_black if current_player == GameResult.BLACK else score_white
      komi_as_player = komi if current_player == GameResult.WHITE else 0.0

      # total of 8 unique rotation + reflection combinations
      num_rotations = 4
      symmetries = random.sample(range(8), REUSE_FACTOR)
      last_moves = move_queue[-NUM_LAST_MOVES:]

      for rot_index in range(num_rotations):
        if rot_index not in symmetries:
          continue
        key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref0'
        b = np.rot90(board_arr, rot_index)
        i_prime, j_prime = rot90_points(i, j, rot_index)

        yield key, {
            'metadata': key,
            'board': b,
            'komi': komi_as_player,
            'color': color_to_move,
            'result': current_player_score_diff,
            'last_moves': [
                rot90_points(i, j, rot_index) for (i, j) in last_moves
            ],
            'policy': (i_prime, j_prime)
        }

      # now reflect and repeat operations
      board_arr = np.fliplr(board_arr)
      last_moves = [fliplr_points(i, j) for (i, j) in last_moves]
      i_ref, j_ref = fliplr_points(i, j)
      for rot_index in range(num_rotations):
        if rot_index + 4 not in symmetries:
          continue
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
                rot90_points(i, j, rot_index) for (i, j) in last_moves
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
      current_player_score_diff =\
        score_black if current_player == GameResult.BLACK else score_white
      komi_as_player = komi if current_player == GameResult.WHITE else 0.0

      num_rotations = 4
      for _ in range(2):
        last_moves = move_queue[-NUM_LAST_MOVES:]
        symmetries = random.sample(range(8), REUSE_FACTOR)
        for rot_index in range(num_rotations):
          if rot_index not in symmetries:
            continue

          key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref0'
          b = np.rot90(board_arr, rot_index)

          yield key, {
              'metadata': key,
              'board': b,
              'komi': komi_as_player,
              'color': color_to_move,
              'result': current_player_score_diff,
              'last_moves': [
                  rot90_points(i, j, rot_index) for (i, j) in last_moves
              ],
              'policy': PASS_MOVE,
          }

        board_arr = np.fliplr(board_arr)
        last_moves = [fliplr_points(i, j) for (i, j) in last_moves]
        for rot_index in range(num_rotations):
          if rot_index + 4 not in symmetries:
            continue
          key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref1'
          b = np.rot90(board_arr, rot_index)

          yield key, {
              'metadata': key,
              'board': b,
              'komi': komi_as_player,
              'color': color_to_move,
              'result': current_player_score_diff,
              'last_moves': [
                  rot90_points(i, j, rot_index) for (i, j) in last_moves
              ],
              'policy': PASS_MOVE,
          }

        move_num += 1
        move_queue.append(PASS_MOVE)
        color_to_move ^= 1
