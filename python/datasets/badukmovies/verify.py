'''
Janky script to verify training examples.

Please do not use :)
'''

import datasets.badukmovies
import tensorflow_datasets as tfds

from board import GoBoard

NUM_EXAMPLES_TO_PRINT = 10

ds = tfds.load('badukmovies', split='train')

if __name__ == '__main__':
  goban = GoBoard()
  ex_count = 0
  for ex in ds:
    # print(ex['metadata'].numpy())
    if ex_count > NUM_EXAMPLES_TO_PRINT:
      break

    board, last_moves, policy = ex['board'].numpy().tolist(
    ), ex['last_moves'].numpy().tolist(), ex['policy'].numpy().tolist()
    goban.board = board

    print('--------------------')
    print(ex['metadata'].numpy())
    print('last_moves: ', last_moves)
    goban.print()

    if policy[1] == 19:
      print('move: PASS')
      continue

    print('move:', policy[0], 'ABCDEFGHIJKLMNOPQRS'[policy[1]])

    if not goban.move_black(policy[0], policy[1]):
      print(ex['metadata'].numpy())
      print('move: ', policy[0], 'ABCDEFGHIJKLMNOPQRS'[policy[1]])
      goban.print()

    # goban.print()

    ex_count += 1
