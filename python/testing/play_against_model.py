import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from collections import deque

from board import GoBoard, GoBoardTrainingUtils, parse_move
from model import P3achyGoModel
from model_config import ModelConfig

FLAGS = flags.FLAGS

flags.DEFINE_string('model_ckpt_path', '', 'Path to model checkpoint.')


def play(model):
  '''
  Model is hardcoded to black.
  '''
  board = GoBoard()
  last_moves = deque([
      tf.convert_to_tensor(np.zeros((19, 19), dtype=np.float32))
      for _ in range(5)
  ])

  while True:
    x = tf.convert_to_tensor([
        GoBoardTrainingUtils.get_black(tf.constant(board.board)),
        GoBoardTrainingUtils.get_white(tf.constant(board.board))
    ] + list(last_moves))
    x = tf.expand_dims(x, axis=0)
    x = tf.transpose(x, (0, 2, 3, 1))  # nchw -> nhwc
    move_logits = model(x)
    move_sample = tf.random.categorical(logits=move_logits, num_samples=5)
    move_scalars = [move_sample[0, i].numpy() for i in range(5)]
    moves = [GoBoard.move_as_tuple(move) for move in move_scalars]
    move = moves[0]

    # print('---------------- Last Moves: ----------------')
    # for last_move in last_moves:
    #   print(GoBoard.to_string(last_move.numpy()))
    # print('--------------------------------')

    for move in moves:
      if board.move_black(move[0], move[1]):
        break

    board.print()

    logging.info(f'P3achyGo Moves: {moves}')

    human_move = parse_move(input())
    while human_move is None:
      logging.warning('Invalid Move.')
      human_move = parse_move(input())

    logging.info(f'Your Move: {human_move}')
    board.move_white(human_move[0], human_move[1])
    board.print()

    last_moves.append(GoBoardTrainingUtils.as_one_hot(tf.constant(move)))
    last_moves.append(GoBoardTrainingUtils.as_one_hot(tf.constant(human_move)))
    last_moves.popleft()
    last_moves.popleft()


def main(_):
  if not FLAGS.model_ckpt_path:
    logging.warning('No Checkpoint Path Specified.')
    return

  logging.info(f'Checkpoint Path: {FLAGS.model_ckpt_path}')
  dummy_data = np.random.random((1, 19, 19, 7))
  model = P3achyGoModel.create(ModelConfig.small(), 'p3achy_test')
  model(dummy_data)
  load_status = model.load_weights(FLAGS.model_ckpt_path)

  play(model)


if __name__ == '__main__':
  app.run(main)
