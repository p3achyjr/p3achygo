import tensorflow as tf
import tf2onnx, onnx
import onnxruntime as ort
import numpy as np
import collections
import transforms
import trt_convert

from absl import app, flags, logging
from pathlib import Path

from model import P3achyGoModel
from constants import *

FLAGS = flags.FLAGS
DUMMY_BATCH_SIZE = 32

flags.DEFINE_string('model_path', '', 'Path to SavedModel.')
flags.DEFINE_string('onnx_name', 'model.onnx', 'Name of ONNX model.')
flags.DEFINE_string('val_ds', '', 'Validation DS, to verify conversion.')


def random_inputs(planes_shape, features_shape):
  return (np.random.random([DUMMY_BATCH_SIZE] + planes_shape).astype(
      np.float32), np.random.random([DUMMY_BATCH_SIZE] + features_shape).astype(
          np.float32))


def update_val_stats(stats, pi_pred, outcome_pred, score_pred, policy, score):
  outcome = score >= 0
  true_move = policy if len(policy.shape) == 1 else np.argmax(policy, axis=1)
  pi_pred = np.argmax(pi_pred, axis=1)
  outcome_pred = np.argmax(outcome_pred, axis=1)
  correct_move = pi_pred == true_move
  correct_outcome = outcome == outcome_pred.astype(np.int32)
  score_pred = np.argmax(score_pred, axis=1) - SCORE_RANGE_MIDPOINT
  score_diff = np.abs(score - score_pred)

  n = pi_pred.size
  stats['num_batches'] += 1
  stats['num_examples'] += n
  stats['correct_moves'] += np.sum(correct_move)
  stats['correct_outcomes'] += np.sum(correct_outcome)
  stats['score_diff'] += np.mean(score_diff)


def main(_):
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return

  model_path = FLAGS.model_path
  logging.info(f'Model Path: {model_path}')

  onnx_path = trt_convert.convert_onnx(onnx_path)
  logging.info(f'Onnx Path: {onnx_path}')

  if FLAGS.val_ds:
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects())
    val_ds = tf.data.TFRecordDataset(FLAGS.val_ds, compression_type='ZLIB')
    val_ds = val_ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(48)
    val_ds = val_ds.take(50)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    scores = (.05 *
              tf.range(-SCORE_RANGE // 2 + .5, SCORE_RANGE // 2 + .5)).numpy()
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    stats_ort, stats_tf = (collections.defaultdict(float),
                           collections.defaultdict(float))
    for (in_board_state, in_global_state, _, _, score, _, policy, _, _, _, _,
         _) in val_ds:
      out_ort = sess.run(
          None, {
              'board_state': in_board_state.numpy(),
              'game_state': in_global_state.numpy(),
              'scores': scores,
          })
      out_tf = model(in_board_state, in_global_state)
      update_val_stats(stats_ort, out_ort[0], out_ort[3], out_ort[6], policy,
                       score)
      update_val_stats(stats_tf, out_tf[0], out_tf[3], out_tf[6], policy, score)

    def stats_str(stats):
      n = stats['num_examples']
      b = stats['num_batches']
      if n == 0:
        n = 1
      if b == 0:
        b = 1
      return "\n".join([
          f'Correct Move Percentage: {stats["correct_moves"] / n}',
          f'Correct Outcome Percentage: {stats["correct_outcomes"] / n}',
          f'Avg Score Diff: {stats["score_diff"] / b}',
      ])

    logging.info(f'\nORT:\n{stats_str(stats_ort)}\n\n' +
                 f'TF:\n{stats_str(stats_tf)}')


if __name__ == '__main__':
  app.run(main)
