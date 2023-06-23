import numpy as np
import tensorflow as tf
import transforms

from absl import app, flags, logging
from collections import deque

from board import GoBoard, parse_move
from constants import *
from model import P3achyGoModel

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Path to SavedModel.')
flags.DEFINE_string('color', 'black',
                    'Color for model to play as (black|white)')


def model_move(model, board, last_moves, game_state, color):
  x = tf.convert_to_tensor([
      transforms.get_white(tf.constant(board.board)),
      transforms.get_black(tf.constant(board.board))
  ])
  print(x.numpy())
  x = tf.concat([x, list(last_moves)], axis=0)
  x = tf.expand_dims(x, axis=0)
  x = tf.transpose(x, (0, 2, 3, 1))  # nchw -> nhwc
  move_logits, _, game_outcome, game_ownership, score_logits, gamma = model(
      x, game_state)

  for i, logit in enumerate(move_logits.numpy()[0]):
    print(f'{i//19, i % 19}', logit)

  score_sample = tf.random.categorical(logits=score_logits,
                                       num_samples=5).numpy() - 399.5

  chosen_move = None
  # logging.info(f'Moves: {tf.nn.softmax(move_logits).numpy()}')
  top_policy_indices = tf.math.top_k(move_logits, k=10).indices
  top_policy_values = tf.math.top_k(move_logits, k=10).values
  logging.info(
      f'Predicted Top 5 Moves: {[GoBoard.move_as_tuple(move.numpy()) for move in top_policy_indices[0]]}'
  )
  logging.info(f'Predicted Top 5 Move Values: {top_policy_values}')
  for move in top_policy_indices[0]:
    print(move)
    move = GoBoard.move_as_tuple(move.numpy())
    logging.info(f'P3achyGo Considering Move: {move}')
    if move == PASS_MOVE or move == PASS_MOVE_RL:
      continue

    if board.move(color, move[0], move[1]):
      chosen_move = move
      break
  # while True:
  #   # move_sample = tf.random.categorical(logits=move_logits, num_samples=1)
  #   # moves = [GoBoard.move_as_tuple(move_sample[0, 0].numpy())]
  #   # move = moves[0]
  #   move_sample = tf.argmax(move_logits, axis=1).numpy()[0]
  #   move = GoBoard.move_as_tuple(move_sample)
  #   if move == PASS_MOVE or move == PASS_MOVE_RL:
  #     continue
  #   logging.info(f'P3achyGo Considering Move: {move}')
  #   if board.move(color, move[0], move[1]):
  #     chosen_move = move
  #     break

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
    #         transforms.get_black(tf.constant(board.board))))
    # print('--------WHITE--------')
    # print(
    #     GoBoard.to_string(
    #         transforms.get_white(tf.constant(board.board))))
    print('---------------- Last Moves: ----------------')
    for move in last_moves:
      print(tf.where(move))
    print('--------------------------------')

    if color == BLACK:
      p0_move = model_move(model, board, last_moves, komi_state, color)
      board.print()
      last_moves.append(transforms.as_one_hot(tf.constant(p0_move)))
      last_moves.popleft()

      p1_move = human_move(board, color ^ 0x3)
      board.print()
      last_moves.append(transforms.as_one_hot(tf.constant(p1_move)))
      last_moves.popleft()
    elif color == WHITE:
      p0_move = human_move(board, color ^ 0x3)
      board.print()
      last_moves.append(transforms.as_one_hot(tf.constant(p0_move)))
      last_moves.popleft()

      p1_move = model_move(model, board, last_moves, komi_state, color)
      board.print()
      last_moves.append(transforms.as_one_hot(tf.constant(p1_move)))
      last_moves.popleft()
    else:
      raise ValueError(f'Unknown Color: {color}')


def main(_):
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return

  logging.info(f'Model Path: {FLAGS.model_path}')
  dummy_board = np.random.random((1, 19, 19, 7))
  dummy_game = np.random.random((1, 1))
  model = tf.keras.models.load_model(
      FLAGS.model_path, custom_objects=P3achyGoModel.custom_objects())
  model(dummy_board, dummy_game)

  if FLAGS.color.lower() == 'black':
    color = BLACK
  elif FLAGS.color.lower() == 'white':
    color = WHITE
  else:
    raise ValueError(f'Unknown Color: {FLAGS.color}')
  play(model, color=color)


if __name__ == '__main__':
  app.run(main)
