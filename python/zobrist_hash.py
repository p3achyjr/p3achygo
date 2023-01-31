import bitstring, math, random, string

BOARD_SIZE = 19
BITSTRING_LEN = 128
UINT_MAX = 2**128 - 1

EMPTY = 0
BLACK = 1
WHITE = 2


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

  def __init__(self, b: int) -> None:
    self.len = b
    self.board = [[EMPTY for _ in range(b)] for _ in range(b)]
    self.zobrist_table = ZobristTable(self.len, 3)
    self.zobrist_hash = ZobristHash(self, self.len, self.zobrist_table)
    self.table = set(self.zobrist_hash.hash())

  def move(self, color: int, i: int, j: int) -> bool:
    if (self.board[i][j] != EMPTY):
      print('Invalid Move!')
      return False

    # we need to change the color here so the liberty detection works.
    self.board[i][j] = color
    captured = self.get_captured(i, j, BLACK if color == WHITE else WHITE)
    if len(captured) == 0 and self.find_liberties(set([(i, j)]), i, j, color,
                                                  set()) == 0:
      print('Invalid Move!')
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

  def print(self):
    index = 19
    for row in self.board:
      print('{:2d}'.format(index),
            ' '.join([['⋅', '●', '○'][piece] for piece in row]))
      index -= 1

    print('  ', ' '.join(list('ABCDEFGHIJKLMNOPQRS')))


def parse_move(move):
  if len(move) < 2 or len(move) > 3:
    return
  if not move[0] in 'abcdefghijklmnopqrs':
    return
  if not move[1] in '123456789':
    return

  return (abs(int(move[1:]) - BOARD_SIZE), ord(move[0]) - ord('a'))


if __name__ == '__main__':
  board = GoBoard(BOARD_SIZE)
  turn = 0  # 0 if black, 1 if white
  board.print()
  while True:
    move = input()
    if move == 'pass':
      board.print()
      turn ^= 1
      continue

    i, j = parse_move(move)

    if board.move(2 if turn else 1, i, j):
      turn ^= 1

    board.print()
