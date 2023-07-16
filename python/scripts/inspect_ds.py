import tensorflow as tf
import transforms
import numpy as np

from absl import flags, app

from board import GoBoard
from constants import *
from model import P3achyGoModel

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', '', 'Path to data.')
flags.DEFINE_string('model_path', '', 'Path to model.')


def move(x):
  return 'ABCDEFGHIJKLMNOPQRS'[x % 19], (x // 19).numpy()


def char_at(own_pred, i, j):
  x = own_pred[i, j]
  bounds = [-1.0, -0.5, 0, 0.5, 1.0]
  chars = ['●', '◆', '⋅', '◇', '○']

  deltas = [abs(x - bounds)]
  return chars[np.argmin(deltas)]


def own_pred_to_string(own_pred: tf.Tensor):
  '''`own_pred` is a bsize * bsize grid of reals [-1, 1]'''
  s = []
  for i in range(len(own_pred)):
    s.append(
        ('{:2d}'.format(i) + ' ' +
         ' '.join([char_at(own_pred, i, j) for j in range(len(own_pred[0]))])))

  s.append('   ' + ' '.join(list('ABCDEFGHIJKLMNOPQRS')))
  return '\n'.join(s)


def probs_as_grid_string(probs: tf.Tensor, bsize: int):
  '''`probs` is bsize * bsize + 1'''
  s = []
  for i in range(bsize):
    s.append(('{:2d}'.format(i) + ' ' + ' '.join(
        [f'{(probs[i * bsize + j] * 100):6.3f}' for j in range(bsize)])))

  s.append('    ' + '      '.join(list('ABCDEFGHIJKLMNOPQRS')))
  return '\n'.join(s)


def entropy(probs: tf.Tensor):
  return -tf.reduce_sum(probs * tf.math.log(probs))


def main(_):
  ds_path = FLAGS.data_path
  if not ds_path:
    print('No data path provided')
    return

  model_path = FLAGS.model_path
  if not model_path:
    print('No data path provided')
    return

  ds = tf.data.TFRecordDataset(ds_path, compression_type='ZLIB')
  ds = ds.map(transforms.expand)
  ds = ds.shuffle(1000)
  ds = ds.prefetch(tf.data.AUTOTUNE)

  model = tf.keras.models.load_model(
      model_path, custom_objects=P3achyGoModel.custom_objects())

  for (input_planes, input_global_state, color, komi, score, score_one_hot,
       policy, policy_aux, own, q30, q100, q200) in ds:
    bsize = len(input_planes[0])
    (_, pi_probs, _, outcome_probs, own_pred, _, score_probs, gamma,
     pi_logits_aux, q30_pred, q100_pred,
     q200_pred) = model(tf.expand_dims(input_planes, 0),
                        tf.expand_dims(input_global_state, 0),
                        training=False)
    pi_aux_probs = tf.nn.softmax(pi_logits_aux)

    top_policy_indices = tf.math.top_k(pi_probs[0], k=8).indices
    top_policy_values = tf.math.top_k(pi_probs[0], k=8).values
    top_policy_aux_indices = tf.math.top_k(pi_aux_probs[0], k=8).indices
    top_policy_aux_values = tf.math.top_k(pi_aux_probs[0], k=8).values
    top_score_indices = tf.math.top_k(score_probs[0],
                                      k=20).indices - SCORE_RANGE_MIDPOINT
    top_score_values = tf.math.top_k(score_probs[0], k=20).values

    input_planes = tf.transpose(input_planes, perm=(2, 0, 1))  # HWC -> CHW

    print('-----Board-----')
    print(GoBoard.to_string(BLACK * input_planes[0] + WHITE * input_planes[1]))

    print('-----Policy Pred-----')
    print(probs_as_grid_string(pi_probs[0], bsize))
    print('Pass: ', pi_probs[0][bsize * bsize].numpy())
    print(f'Top 5 Moves:', f'{[move(mv) for mv in top_policy_indices]}',
          f'{top_policy_indices}')
    print(f'Top 5 Move Probs: {top_policy_values}')

    print('-----Policy-----')
    print(tf.math.argmax(policy).numpy(), move(tf.math.argmax(policy)))

    print('-----Policy Entropy-----')
    print(entropy(pi_probs).numpy())

    print('-----Policy Surprise (KL Divergence)')
    print(tf.keras.losses.KLDivergence()(policy, pi_probs).numpy())

    print('-----Policy Aux Pred-----')
    print(probs_as_grid_string(pi_aux_probs[0], bsize))
    print('Pass: ', pi_aux_probs[0][bsize * bsize].numpy())
    print(f'Top 5 Aux Moves:', f'{[move(mv) for mv in top_policy_aux_indices]}',
          f'{top_policy_aux_indices}')
    print(f'Top 5 Aux Move Logits: {top_policy_aux_values}')

    print('-----Policy Aux-----')
    print(policy_aux.numpy(), move(policy_aux))

    print('-----Own Pred-----')
    print(own_pred_to_string(own_pred[0].numpy()))

    print('-----Own-----')
    print(GoBoard.to_string(own))

    print('-----Last Moves-----')
    moves = []
    for i in range(5):
      if input_global_state[2 + i] != 0:
        moves.append('Pass')
        continue

      plane = input_planes[2 + i]
      plane = tf.reshape(plane, (bsize * bsize,))
      mv = tf.argmax(plane)
      moves.append(move(mv))

    print(moves)

    print('-----Score Pred-----')
    print(f'Predicted Scores: {top_score_indices}')
    print(f'Predicted Score Values: {top_score_values}')

    print('-----Score-----')
    print(score)

    print(f'----- Q/Z INFO -----')
    print(f'Predicted Outcome: {outcome_probs},',
          f'Actual Outcome: {1.0 if score >= 0 else 0.0}')
    print(f'q30 Pred: {q30_pred[0]}, Actual: {q30}')
    print(f'q100 Pred: {q100_pred[0]}, Actual: {q100}')
    print(f'q200 Pred: {q200_pred[0]}, Actual: {q200}')
    print()
    print()

    user_input = input("Press enter to view next, or 'stop' to quit: ")
    if user_input.lower() == 'stop':
      break


if __name__ == '__main__':
  app.run(main)
