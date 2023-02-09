"""badukmovies dataset."""

import itertools
import logging
import numpy as np
import resource
import tensorflow_datasets as tfds

from board import GoBoard, GameResult

from collections import deque
from sgfmill import sgf

BOARD_LEN = 19
NUM_LAST_MOVES = 5
SGF_SIZE = 10000

NON_MOVE = (-1, -1)
PASS_MOVE = (BOARD_LEN - 1, BOARD_LEN)

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (low * 4, high))

logger = logging.getLogger('badukmovies_dataset_builder')


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for badukmovies dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Download zip from badukmovies.com
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(badukmovies): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'metadata':
                tfds.features.Text(),
            # 19 x 19 matrix representing board state from perspective of
            # current player
            'board':
                tfds.features.Tensor(shape=(BOARD_LEN, BOARD_LEN),
                                     dtype=np.int8),
            'last_moves':
                tfds.features.Tensor(shape=(NUM_LAST_MOVES, 2), dtype=np.int8),
            # currently encodes tuple indicating next move
            'policy':
                tfds.features.Tensor(shape=(2,), dtype=np.int8),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('board', 'last_moves',
                         'policy'),  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.manual_dir / 'baduk'

    # TODO(badukmovies): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path: tfds.core.Path):
    """Yields examples."""
    # TODO(badukmovies): Yields (key, example) tuples from the dataset
    paths = deque()
    paths.append(path)
    num_examples = 0

    generators_per_sgf = []
    while paths and num_examples < SGF_SIZE:
      path = paths.popleft()
      if path.is_dir():
        for subpath in path.iterdir():
          paths.append(subpath)

      elif str(path).endswith('.sgf'):
        generators_per_sgf.append(self._parse_sgf(path))
        num_examples += 1

    return itertools.chain.from_iterable(generators_per_sgf)

  def _parse_sgf(self, path: tfds.core.Path):
    """
    Extracts main move line from sgf.
    
    Yields one example per (board_pos, move) tuple in each game.
    """

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
    board = GoBoard(BOARD_LEN)
    move_num = 0
    move_queue = [NON_MOVE] * NUM_LAST_MOVES
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

        yield key, {
            'metadata': key,
            'board': b,
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
        logger.warning(f'Sample: {key}')
        logger.warning(f'Failed to move: {i}, {j}')
        logger.warning(f'\n{board.print()}')
        return

      move_num += 1
      move_queue.append((i, j))

    if not result.is_unknown() and not result.is_by_resignation():
      # add two pass moves, since this is a scored game.
      num_rotations = 4
      for _ in range(2):
        for rot_index in range(num_rotations):
          key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref0'
          b = np.rot90(board_arr, rot_index)

          yield key, {
              'metadata': key,
              'board': b,
              'last_moves': [
                  rot90_points(i, j, rot_index)
                  for (i, j) in move_queue[-NUM_LAST_MOVES:]
              ],
              'policy': PASS_MOVE
          }

        board_arr = np.fliplr(board_arr)
        for rot_index in range(num_rotations):
          key = f'y{year}_m{month}_{path.name}_mv{move_num}_rot{rot_index}_ref1'
          b = np.rot90(board_arr, rot_index)

          yield key, {
              'metadata': key,
              'board': b,
              'last_moves': [
                  rot90_points(i, j, rot_index)
                  for (i, j) in move_queue[-NUM_LAST_MOVES:]
              ],
              'policy': PASS_MOVE
          }

        move_num += 1
        move_queue.append(PASS_MOVE)
