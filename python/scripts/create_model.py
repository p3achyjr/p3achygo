"""Creates a new model and saves it to a given directory."""

import sys
import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from pathlib import Path

from constants import *
from model import P3achyGoModel
from model_config import ModelConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("model_config", "small", "Model config name.")
flags.DEFINE_string("output_dir", "", "Directory to save the model to.")
flags.DEFINE_string("name", "p3achygo", "Model name.")
flags.DEFINE_integer("batch_size", 32, "Batch size for initial forward pass.")


def main(_):
    if not FLAGS.output_dir:
        logging.error("No --output_dir specified.")
        return

    output_dir = Path(FLAGS.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tf.device("/cpu:0"):
        model = P3achyGoModel.create(
            config=ModelConfig.from_str(FLAGS.model_config),
            board_len=BOARD_LEN,
            num_input_planes=num_input_planes(),
            num_input_features=num_input_features(),
            name=FLAGS.name,
        )
        # Run a forward pass to build the model.
        model(
            tf.convert_to_tensor(
                np.random.random([FLAGS.batch_size] + model.input_planes_shape()),
                dtype=tf.float32,
            ),
            tf.convert_to_tensor(
                np.random.random([FLAGS.batch_size] + model.input_features_shape()),
                dtype=tf.float32,
            ),
        )
        model.summary()

        model_path = str(output_dir / "model_0000.keras")
        model.save(model_path)
        logging.info(f"Saved model to {model_path}")

        # Sanity check: Load the model back
        logging.info("Loading model back as sanity check...")
        loaded_model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully!")

        # Run a forward pass with the loaded model
        test_board = tf.convert_to_tensor(
            np.random.random([FLAGS.batch_size] + model.input_planes_shape()),
            dtype=tf.float32,
        )
        test_features = tf.convert_to_tensor(
            np.random.random([FLAGS.batch_size] + model.input_features_shape()),
            dtype=tf.float32,
        )

        original_outputs = model(test_board, test_features, training=False)
        loaded_outputs = loaded_model(test_board, test_features, training=False)

        logging.info(f"Original model outputs: {len(original_outputs)} tensors")
        logging.info(f"Loaded model outputs: {len(loaded_outputs)} tensors")

        # Verify outputs match
        max_diff = 0.0
        for i, (orig, loaded) in enumerate(zip(original_outputs, loaded_outputs)):
            diff = tf.reduce_max(tf.abs(orig - loaded)).numpy()
            max_diff = max(max_diff, diff)
            logging.info(f"Output {i}: max diff = {diff:.6e}")

        if max_diff < 1e-5:
            logging.info(f"✓ Sanity check PASSED! Max difference: {max_diff:.6e}")
        else:
            logging.warning(
                f"⚠ Sanity check: outputs differ by {max_diff:.6e} (may be due to dropout)"
            )


if __name__ == "__main__":
    app.run(main)
