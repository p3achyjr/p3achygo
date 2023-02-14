'''
Janky script to verify training examples.

Please do not use :)
'''

from __future__ import annotations

import datasets.badukmovies
import tensorflow_datasets as tfds

from board import GoBoard

NUM_EXAMPLES_TO_PRINT = 20

ds = tfds.load('badukmovies', split='train[99%:100%]')

if __name__ == '__main__':
  goban = GoBoard()
  ex_count = 0
  for ex in ds:
    # print(ex['metadata'].numpy())
    if ex_count > NUM_EXAMPLES_TO_PRINT:
      break

    metadata, board, komi, color, result, last_moves, policy = (
        ex['metadata'], ex['board'].numpy().tolist(), ex['komi'].numpy(),
        ex['color'].numpy(), ex['result'].numpy(),
        ex['last_moves'].numpy().tolist(), ex['policy'].numpy().tolist())
    goban.board = board

    print('--------------------')
    print(f'Metadata: {metadata}')
    print(f'Komi: {komi}')
    print(f'Color: {color}')
    print(f'Result: {result}')
    print(f'Last Moves: {last_moves}')
    goban.print()

    if policy[1] == 19:
      print('Move: PASS')
      continue

    print('Move:', policy[0], 'ABCDEFGHIJKLMNOPQRS'[policy[1]])

    if not goban.move_black(policy[0], policy[1]):
      print(ex['metadata'].numpy())
      print('Move: ', policy[0], 'ABCDEFGHIJKLMNOPQRS'[policy[1]])
      goban.print()

    # goban.print()

    ex_count += 1
