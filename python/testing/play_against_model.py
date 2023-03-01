import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from collections import deque

from board import GoBoard, GoBoardTrainingUtils, parse_move
from constants import *
from model import P3achyGoModel
from model_config import ModelConfig

FLAGS = flags.FLAGS

flags.DEFINE_string('model_ckpt_path', '', 'Path to model checkpoint.')
flags.DEFINE_string('color', 'black',
                    'Color for model to play as (black|white)')


def model_move(model, board, last_moves, game_state, color):
  x = tf.convert_to_tensor([
      GoBoardTrainingUtils.get_white(tf.constant(board.board)),
      GoBoardTrainingUtils.get_black(tf.constant(board.board))
  ])
  # print(x.numpy())
  x = tf.concat([x, list(last_moves)], axis=0)
  x = tf.expand_dims(x, axis=0)
  x = tf.transpose(x, (0, 2, 3, 1))  # nchw -> nhwc
  move_logits, game_outcome, game_ownership, score_logits, gamma = model(
      x, game_state)

  score_sample = tf.random.categorical(logits=score_logits,
                                       num_samples=5).numpy() - 399.5

  chosen_move = None
  logging.info(f'Moves: {tf.nn.softmax(move_logits).numpy()}')
  while True:
    move_sample = tf.random.categorical(logits=move_logits, num_samples=1)
    moves = [GoBoard.move_as_tuple(move_sample[0, 0].numpy())]
    move = moves[0]
    logging.info(f'P3achyGo Considering Move: {move}')
    if board.move(color, move[0], move[1]):
      chosen_move = move
      break

  assert chosen_move != None

  logging.info(
      f'P3achyGo Predicted Outcome: {tf.nn.softmax(game_outcome).numpy()}')
  # logging.info(f'P3achyGo Predicted Ownership: {game_ownership.numpy()}')
  logging.info(f'P3achyGo Predicted Score: {score_sample}')

  return chosen_move


def human_move(board, color):
  logging.info(f'Enter Move: ')
  while True:
    human_move = parse_move(input())
    while human_move is None:
      logging.warning('Invalid Move.')
      human_move = parse_move(input())

    if not board.move(color, human_move[0], human_move[1]):
      continue

    logging.info(f'Your Move: {human_move}')

    return human_move


def play(model, color=BLACK):
  '''
  Play against model.
  '''
  board = GoBoard()
  last_moves = deque([
      tf.convert_to_tensor(np.zeros((19, 19), dtype=np.float32))
      for _ in range(5)
  ])

  komi_state = tf.convert_to_tensor(
      [[0.0]]) if color == BLACK else tf.convert_to_tensor([[0.5]])
  if color == BLACK:
    p0_move = model_move
    p1_move = human_move
  elif color == WHITE:
    p0_move = human_move
    p1_move = model_move
  else:
    raise ValueError(f'Unknown Color: {color}')

  while True:
    # print('--------BLACK--------')
    # print(
    #     GoBoard.to_string(
    #         GoBoardTrainingUtils.get_black(tf.constant(board.board))))
    # print('--------WHITE--------')
    # print(
    #     GoBoard.to_string(
    #         GoBoardTrainingUtils.get_white(tf.constant(board.board))))
    print('---------------- Last Moves: ----------------')
    for move in last_moves:
      print(tf.where(move))
    print('--------------------------------')

    if color == BLACK:
      p0_move = model_move(model, board, last_moves, komi_state, color)
      board.print()
      last_moves.append(GoBoardTrainingUtils.as_one_hot(tf.constant(p0_move)))
      last_moves.popleft()

      p1_move = human_move(board, color ^ 0x3)
      board.print()
      last_moves.append(GoBoardTrainingUtils.as_one_hot(tf.constant(p1_move)))
      last_moves.popleft()
    elif color == WHITE:
      p0_move = human_move(board, color ^ 0x3)
      board.print()
      last_moves.append(GoBoardTrainingUtils.as_one_hot(tf.constant(p0_move)))
      last_moves.popleft()

      p1_move = model_move(model, board, last_moves, komi_state, color)
      board.print()
      last_moves.append(GoBoardTrainingUtils.as_one_hot(tf.constant(p1_move)))
      last_moves.popleft()
    else:
      raise ValueError(f'Unknown Color: {color}')


def main(_):
  if not FLAGS.model_ckpt_path:
    logging.warning('No Checkpoint Path Specified.')
    return

  logging.info(f'Checkpoint Path: {FLAGS.model_ckpt_path}')
  dummy_board = np.random.random((1, 19, 19, 7))
  dummy_game = np.random.random((1, 1))
  model = P3achyGoModel.create(ModelConfig.small(),
                               board_len=BOARD_LEN,
                               num_input_planes=NUM_INPUT_PLANES,
                               num_input_features=NUM_INPUT_FEATURES,
                               name='p3achy_test')
  model(dummy_board, dummy_game)
  load_status = model.load_weights(FLAGS.model_ckpt_path)

  logging.info(load_status)

  if FLAGS.color.lower() == 'black':
    color = BLACK
  elif FLAGS.color.lower() == 'white':
    color = WHITE
  else:
    raise ValueError(f'Unknown Color: {FLAGS.color}')
  play(model, color=color)


if __name__ == '__main__':
  app.run(main)
