import gcs_utils as gcs
import sys
import rl_loop.model_utils as model_utils
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_enum('model_config', 'small', ['small', 'b24c192', 'b32c256'])
flags.DEFINE_string('model_path', '', 'Path to store model')


def main(_):
  if FLAGS.model_config == '':
    logging.error('No --model_config specified.')
    return
  if FLAGS.model_path == '':
    logging.error('No --model_path specified.')
    return

  model = model_utils.new_model(name=f'p3achygo',
                                model_config=FLAGS.model_config)
  model.summary()
  model.save(FLAGS.model_path)


if __name__ == '__main__':
  sys.stdout.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  sys.stderr.reconfigure(line_buffering=True)  # pytype: disable=attribute-error
  app.run(main)
