import tensorflow as tf
import tf2onnx, onnx
import onnxruntime as ort
import numpy as np
import collections
import transforms

from absl import app, flags, logging
from pathlib import Path
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

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
  onnx_path = str(Path(model_path, '_onnx', FLAGS.onnx_name))
  logging.info(f'Model Path: {model_path}')
  logging.info(f'Onnx Path: {onnx_path}')

  with tf.device("/cpu:0"):
    # Load model and resave without mixed precision.
    tf.keras.mixed_precision.set_global_policy('float32')
    model = tf.keras.models.load_model(
        model_path, custom_objects=P3achyGoModel.custom_objects())
    planes_shape = model.input_planes_shape()
    features_shape = model.input_features_shape()
    model(*[
        tf.convert_to_tensor(x)
        for x in random_inputs(planes_shape, features_shape)
    ])
    model.summary()

    @tf.function
    def model_fn(board_state: tf.Tensor, game_state: tf.Tensor,
                scores: tf.Tensor):
      (pi_logits, pi, outcome_logits, outcome, own, score_logits, score_probs,
      gamma, pi_logits_aux, q30, q100, q200) = model(board_state,
                                                      game_state,
                                                      training=False,
                                                      scores=scores)

      return {
          '00:pi_logits': pi_logits,
          '01:pi': pi,
          '02:outcome_logits': outcome_logits,
          '03:outcome': outcome,
          '04:own': own,
          '05:score_logits': score_logits,
          '06:score_probs': score_probs,
          '07:gamma': gamma,
          '08:pi_logits_aux': pi_logits_aux,
          '09:q30': q30,
          '10:q100': q100,
          '11:q200': q200,
      }

    input_signature = [
        tf.TensorSpec(shape=[None] + model.input_planes_shape(),
                      dtype=tf.float32,
                      name='board_state'),
        tf.TensorSpec(shape=[None] + model.input_features_shape(),
                      dtype=tf.float32,
                      name='game_state'),
        # need this b/c otherwise ONNX loses an edge to the score tensor.
        tf.TensorSpec(shape=(SCORE_RANGE,), dtype=tf.float32, name='scores')
    ]
    onnx_model, _ = tf2onnx.convert.from_function(
        model_fn, input_signature=input_signature)
    print(onnx.printer.to_text(onnx_model))

    Path(model_path, '_onnx').mkdir(exist_ok=True)
    onnx.save(onnx_model, onnx_path)

  if FLAGS.val_ds:
    val_ds = tf.data.TFRecordDataset(FLAGS.val_ds, compression_type='ZLIB')
    val_ds = val_ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(48)
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
