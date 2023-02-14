from __future__ import annotations

import bitstring, random
import numpy as np
import tensorflow as tf

BITSTRING_LEN = 128
UINT_MAX = 2**128 - 1

BOARD_LEN = 19
EMPTY = 0
BLACK = 1
WHITE = 2

NON_MOVE = (-1, -1)
PASS_MOVE = (BOARD_LEN - 1, BOARD_LEN)


def is_star_point(board, i, j) -> bool:
  coords = [3, 9, 15]
  # print(i, j, i in coords and j in coords)
  return (i in coords and j in coords)


def char_at(board, i, j):
  if board[i][j] == EMPTY:
    return '+' if is_star_point(board, i, j) else '⋅'
  elif board[i][j] == BLACK:
    return '○'
  else:
    return '●'


def to_char(x):
  if x == EMPTY:
    return '⋅'
  elif x == BLACK:
    return '○'
  else:
    return '●'


class ZobristTable:

  def __init__(self, len: int, piece_count: int) -> None:
    self.len = len
    self.piece_count = piece_count
    self.__zobrist_hash_table = [[
        bitstring.Bits(uint=random.randint(0, UINT_MAX), length=BITSTRING_LEN)
        for _ in range(piece_count)
    ]
                                 for _ in range(len * len)]
    self.__turn_bitstring = bitstring.Bits(uint=random.randint(0, UINT_MAX),
                                           length=BITSTRING_LEN)

  def bitstring_at(self, i, j, piece_index):
    assert (i < self.len and j < self.len and piece_index < self.piece_count)
    return self.__zobrist_hash_table[i * self.len + j][piece_index]

  def turn_bitstring(self):
    return self.__turn_bitstring


class ZobristHash:

  def __init__(self, board, b, zobrist_table: ZobristTable) -> None:
    self.__hash = bitstring.Bits(uint=0, length=BITSTRING_LEN)
    self.__zobrist_table = zobrist_table
    for i in range(b):
      for j in range(b):
        self.__hash ^= self.__zobrist_table.bitstring_at(
            i, j, board.board[i][j])

    self.__hash ^= self.__zobrist_table.turn_bitstring()

  def hash(self):
    return self.__hash

  def recompute_hash(self, transitions: list[(int, int, int, int)]) -> None:
    for (i, j, last_piece, current_piece) in transitions:
      self.__hash ^= self.__zobrist_table.bitstring_at(i, j, last_piece)
      self.__hash ^= self.__zobrist_table.bitstring_at(i, j, current_piece)

    self.__hash ^= self.__zobrist_table.turn_bitstring()


class GoBoard:

  def __init__(self, b=BOARD_LEN) -> None:
    self.len = b
    self.board = [[EMPTY for _ in range(b)] for _ in range(b)]
    self.zobrist_table = ZobristTable(self.len, 3)
    self.zobrist_hash = ZobristHash(self, self.len, self.zobrist_table)
    self.table = set(self.zobrist_hash.hash())

  def move_black(self, i: int, j: int) -> bool:
    return self.move(BLACK, i, j)

  def move_white(self, i: int, j: int) -> bool:
    return self.move(WHITE, i, j)

  def move(self, color: int, i: int, j: int) -> bool:
    if (self.board[i][j] != EMPTY):
      print('Invalid Move! Board Position Not Empty.')
      return False

    # we need to change the color here so the liberty detection works.
    self.board[i][j] = color
    captured = self.get_captured(i, j, BLACK if color == WHITE else WHITE)
    if len(captured) == 0 and self.find_liberties(set([(i, j)]), i, j, color,
                                                  set()) == 0:
      print('Invalid Move! Self atari.')
      self.board[i][j] = EMPTY
      return False

    transitions = [(i, j, EMPTY, color)]
    for i_prime, j_prime in captured:
      transitions.append(
          (i_prime, j_prime, BLACK if color == WHITE else WHITE, EMPTY))

    self.zobrist_hash.recompute_hash(transitions)
    if self.zobrist_hash.hash() in self.table:
      print('Already seen board position!')
      self.board[i][j] = EMPTY
      return False

    self.table.add(self.zobrist_hash.hash())
    for (i, j, _, piece) in transitions:
      self.board[i][j] = piece

    return True

  def get_captured(self, i, j, captured_color) -> set((int, int)):
    # intuition is that we only need to check the 4 spots next to the place just
    # played.
    checked = set()
    captured = set()
    for (i_prime, j_prime) in [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]:
      if (i_prime, j_prime) in checked:
        continue

      if i_prime < 0 or j_prime < 0 or i_prime >= self.len or j_prime >= self.len:
        continue

      if self.board[i_prime][j_prime] != captured_color:
        continue

      group = set([(i_prime, j_prime)])
      # print('-----')
      liberties = self.find_liberties(group, i_prime, j_prime, captured_color,
                                      set())

      checked.update(group)
      if liberties == 0:
        captured.update(group)

      # print('result:', i_prime, j_prime, liberties, group, captured)

    return captured

  def find_liberties(self, group: set((int, int)), i, j, color, visited: set(
      (int, int))) -> None:
    if (i, j) in visited:
      return 0

    if i < 0 or j < 0 or i >= self.len or j >= self.len:
      return 0

    # print(i, j, group, visited, color, self.board[i][j],
    #       self.board[i][j] == EMPTY)

    visited.add((i, j))
    if self.board[i][j] != color:
      return 1 if self.board[i][j] == EMPTY else 0

    group.add((i, j))
    return self.find_liberties(group, i - 1, j, color, visited) + \
           self.find_liberties(group, i + 1, j, color, visited) + \
           self.find_liberties(group, i, j - 1, color, visited) + \
           self.find_liberties(group, i, j + 1, color, visited)

  def as_color(self, color: int) -> list[list[int]]:
    board = [[EMPTY
              for _ in range(len(self.board))]
             for _ in range(len(self.board[0]))]

    for i in range(len(board)):
      for j in range(len(board[0])):
        if self.board[i][j] == color:
          # kind of hacky. Encode "self" player as black
          board[i][j] = BLACK
        elif self.board[i][j] != EMPTY:
          board[i][j] = WHITE

    return board

  def as_black(self):
    return self.as_color(BLACK)

  def as_white(self):
    return self.as_color(WHITE)

  def is_star_point(self, i, j) -> bool:
    coords = [3, 9, 15]
    # print(i, j, i in coords and j in coords)
    return (i in coords and j in coords)

  def char_at(self, i, j):
    if self.board[i][j] == EMPTY:
      return '+' if self.is_star_point(i, j) else '⋅'
    elif self.board[i][j] == BLACK:
      return '○'
    else:
      return '●'

  def print(self):
    for i in range(BOARD_LEN):
      print('{:2d}'.format(i),
            ' '.join([self.char_at(i, j) for j in range(BOARD_LEN)]))

    print('  ', ' '.join(list('ABCDEFGHIJKLMNOPQRS')))

  def print_pretty(self):
    for i in range(BOARD_LEN):
      print('{:2d}'.format(BOARD_LEN - i),
            ' '.join([self.char_at(i, j) for j in range(BOARD_LEN)]))

    print('  ', ' '.join(list('ABCDEFGHIJKLMNOPQRS')))

  @staticmethod
  def to_string(board):
    s = []
    for i in range(len(board)):
      s.append(('{:2d}'.format(i) + ' ' +
                ' '.join([char_at(board, i, j) for j in range(len(board[0]))])))

    s.append('   ' + ' '.join(list('ABCDEFGHIJKLMNOPQRS')))
    return '\n'.join(s)

  @staticmethod
  def move_as_tuple(move: int):
    assert 0 <= move <= BOARD_LEN * BOARD_LEN
    if move == BOARD_LEN * BOARD_LEN:
      return PASS_MOVE

    return (move // BOARD_LEN, move % BOARD_LEN)


class GoBoardTrainingUtils:
  '''
  Various helpers related to training.
  '''

  @staticmethod
  def get_black(board: tf.Tensor) -> tf.Tensor:
    return tf.cast(
        tf.where(tf.math.equal(board, BLACK), board, tf.zeros_like(board)) /
        BLACK,
        dtype=tf.float32)

  @staticmethod
  def get_white(board: tf.Tensor) -> tf.Tensor:
    return tf.cast(
        tf.where(tf.math.equal(board, WHITE), board, tf.zeros_like(board)) /
        WHITE,
        dtype=tf.float32)

  @staticmethod
  def as_one_hot(move: tf.Tensor) -> tf.Tensor:
    non_move = tf.constant(NON_MOVE, dtype=tf.int32)
    pass_move = tf.constant(PASS_MOVE, dtype=tf.int32)
    if tf.reduce_all(move == non_move) or tf.reduce_all(move == pass_move):
      return tf.zeros((BOARD_LEN, BOARD_LEN), dtype=tf.float32)

    return tf.cast(tf.scatter_nd(indices=[move],
                                 updates=tf.constant([1]),
                                 shape=(BOARD_LEN, BOARD_LEN)),
                   dtype=tf.float32)

  @staticmethod
  def as_index(move: tf.Tensor) -> tf.Tensor:
    return tf.cast(move[0] * BOARD_LEN + move[1], dtype=tf.int32)


class GameResult:
  '''Simple class representing game result'''
  UNKNOWN = 0
  BLACK = 1
  WHITE = 2

  def __init__(self, winner=UNKNOWN, score_diff=0, by_resignation=False):
    self.winner = winner
    self.score_diff = score_diff
    self.by_resignation = by_resignation

  def is_unknown(self):
    return self.winner == self.UNKNOWN

  def is_by_resignation(self):
    return self.by_resignation

  @staticmethod
  def unknown():
    return GameResult(winner=GameResult.UNKNOWN,
                      score_diff=0,
                      by_resignation=False)

  @staticmethod
  def parse_score(score: str):
    score = score.lower()
    if '+' not in score:
      return GameResult.unknown()

    tokens = score.split('+')
    if len(tokens) < 2:
      return GameResult.unknown()

    if tokens[0] == 'b':
      winner = GameResult.BLACK
    elif tokens[0] == 'w':
      winner = GameResult.WHITE
    else:
      return GameResult.unknown()

    if tokens[1] == 'r':
      return GameResult(winner=winner, by_resignation=True)

    try:
      point_diff = float(tokens[1])
    except ValueError:
      return GameResult.unknown()

    return GameResult(winner=winner,
                      score_diff=point_diff,
                      by_resignation=False)


def parse_move(move):
  if move is None:
    return
  if len(move) < 2 or len(move) > 3:
    return
  if not move[0] in 'abcdefghijklmnopqrs':
    return
  if not move[1] in '0123456789':
    return

  return (abs(int(move[1:])), ord(move[0]) - ord('a'))
