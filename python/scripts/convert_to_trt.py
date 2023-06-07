import trt_convert

from absl import app, flags, logging
from pathlib import Path

# Use the same batch size when running self-play.
BATCH_SIZE = 48
NUM_CALIB_BATCHES = 10
FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Path to SavedModel.')
flags.DEFINE_string('chunk', '', 'Which chunk to use.')


def main(_):
  if not FLAGS.chunk:
    logging.warning('Please provide --chunk file (.tfrecord.zz)')
    return
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return

  model_path = FLAGS.model_path
  logging.info(f'Model Path: {model_path}')
  logging.info(f'Chunk: {FLAGS.chunk}')

  converter = trt_convert(FLAGS.model_path, FLAGS.chunk)
  converter.summary()
  converter.save(output_saved_model_dir=str(Path(model_path, '_trt')))


if __name__ == '__main__':
  app.run(main)
