import tensorflow as tf
from absl import app, flags, logging
from model import P3achyGoModel, custom_objects_dict_for_serialization

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', '', 'Path to model.')
flags.DEFINE_string('model_save_path', '', 'Path to save new model.')


def main(_):
  if not FLAGS.model_path:
    logging.warning('No Model Path Specified.')
    return

  if not FLAGS.model_save_path:
    logging.warning('No Model Save Path Specified.')
    return

  logging.info(f'Model Path: {FLAGS.model_path}')
  model = tf.keras.models.load_model(
      FLAGS.model_path, custom_objects=custom_objects_dict_for_serialization())

  model.save(FLAGS.model_save_path,
             signatures={
                 'infer_mixed': model.infer_mixed,
                 'infer_float': model.infer_float
             })


if __name__ == '__main__':
  app.run(main)
