"""Build a validation tfrecord chunk from the badukmovies_all TFDS dataset.

Migrated off `tf.data.experimental.TFRecordWriter` (deprecated tf.data sink)
to the standard `tf.io.TFRecordWriter` and a synchronous numpy serialization
loop. transforms.* helpers are now numpy-based; we pull each TFDS example
as numpy via `as_numpy_iterator()` instead of routing through tf.py_function.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import app, flags, logging
from constants import *
from pathlib import Path

import transforms

FLAGS = flags.FLAGS
DATASET = "badukmovies_all"

flags.DEFINE_string("save_dir", "", "Where to save dataset.")


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# Keep in sync with cc/recorder/tf_recorder.cc:MakeTfExample.
def serialize_numpy(ex: dict) -> bytes:
    """Build a serialized tf.train.Example from a numpy-dict TFDS example."""
    board = np.asarray(ex["board"])
    black = transforms.get_color(board, BLACK)
    white = transforms.get_color(board, WHITE)
    board_int = (black * BLACK + white * WHITE).astype(np.int8)

    last_moves = np.asarray(ex["last_moves"], dtype=np.int32)
    assert last_moves.shape == (5, 2)
    last_move_indices = np.array(
        [transforms.as_index(mv) for mv in last_moves], dtype=np.int16
    )

    policy = transforms.as_pi_vec(np.asarray(ex["policy"], dtype=np.int32))
    score = float(ex["result"]) + 0.5
    own = np.zeros((BOARD_LEN, BOARD_LEN), dtype=np.int8)
    color = BLACK  # every position in the SL dataset is black-to-move

    feature = {
        "bsize": _bytes_feature(np.array([BOARD_LEN], dtype=np.uint8).tobytes()),
        "board": _bytes_feature(board_int.tobytes()),
        "last_moves": _bytes_feature(last_move_indices.tobytes()),
        "color": _bytes_feature(np.array([color], dtype=np.int8).tobytes()),
        "komi": _float_feature(float(ex["komi"])),
        "own": _bytes_feature(own.tobytes()),
        "pi": _bytes_feature(policy.astype(np.float32).tobytes()),
        "result": _float_feature(score),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def main(_):
    if not FLAGS.save_dir:
        logging.warning("Please provide --save_dir.")
        return

    save_filename = Path(FLAGS.save_dir, "val.tfrecord.zz")
    chunk_len = 25600

    # Overdraw to prevent large numbers of examples from a single game.
    ds = tfds.load(
        DATASET,
        split=["train[80000000:80500000]"],
        shuffle_files=True,
    )[0]
    ds = ds.shuffle(500000)
    ds = ds.take(chunk_len)

    options = tf.io.TFRecordOptions(compression_type="ZLIB")
    with tf.io.TFRecordWriter(str(save_filename), options=options) as writer:
        for ex in ds.as_numpy_iterator():
            try:
                writer.write(serialize_numpy(ex))
            except Exception as e:
                logging.warning(f"Skipping example: {e}")


if __name__ == "__main__":
    app.run(main)
