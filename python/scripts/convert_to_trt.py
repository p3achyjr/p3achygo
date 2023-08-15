import trt_convert

from absl import app, flags, logging
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Path to SavedModel.')
flags.DEFINE_string('calib_ds', '', 'Dataset to use for calibration.')
flags.DEFINE_integer('batch_size', 0, 'Conversion Batch Size.')
flags.DEFINE_string(
    'inner_trt_path', '_trt',
    'Path to append to model_path, where we save the TRT model')


def main(_):
  if not FLAGS.calib_ds:
    logging.warning('Please provide --calib_ds file (.tfrecord.zz)')
    return
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return
  if FLAGS.batch_size == 0:
    logging.warning('Please provide --batch_size')
    return

  model_path = FLAGS.model_path
  logging.info(f'Model Path: {model_path}')
  logging.info(f'Chunk: {FLAGS.calib_ds}')

  converter = trt_convert.get_converter(FLAGS.model_path,
                                        FLAGS.calib_ds,
                                        batch_size=FLAGS.batch_size)
  converter.summary()
  converter.save(
      output_saved_model_dir=str(Path(model_path, FLAGS.inner_trt_path)))


if __name__ == '__main__':
  app.run(main)
