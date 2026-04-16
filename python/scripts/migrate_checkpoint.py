"""Migrate a pre-mcts_dist checkpoint to the current architecture.

Between the old and new model, one value-head attribute was renamed and one new
layer was added. Keras 3 keys h5 weights by attribute name, so:

  Old h5 key                          New h5 key
  value_head/outcome_q_biases     ->  value_head/outcome_q_embed   (renamed attr)
  (absent)                        ->  value_head/outcome_mcts_dist  (new layer, random init)

This script:
  1. Reads the model config from the .keras zip (avoids load_model issues).
  2. Constructs a new model and saves its randomly-initialized weights to h5.
  3. Copies all weights from the old h5 into the new h5, mapping the one
     renamed key. The new outcome_mcts_dist layer keeps its random init.
  4. Loads the patched h5 into the new model and saves the result.

Usage:
    python scripts/migrate_checkpoint.py \\
        --input_path /path/to/old_model.keras \\
        --output_path /path/to/migrated_model.keras
"""

import json
import os
import shutil
import tempfile
import zipfile

import h5py
import numpy as np
import keras

from absl import app, flags, logging
from pathlib import Path

from constants import *
from model import P3achyGoModel

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", "", "Path to the old .keras checkpoint.")
flags.DEFINE_string("output_path", "", "Path to save the migrated checkpoint.")

# Maps old h5 path prefix -> new h5 path prefix.
_RENAMES = {
    "layers/value_head/outcome_q_biases": "layers/value_head/outcome_q_embed",
}


def read_model_config(keras_path: str) -> dict:
    """Extract P3achyGoModel's get_config() dict from a .keras zip."""
    with zipfile.ZipFile(keras_path, "r") as zf:
        raw = json.loads(zf.read("config.json"))

    def find_config(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "P3achyGoModel":
                return obj.get("config", {})
            for v in obj.values():
                r = find_config(v)
                if r:
                    return r

    cfg = find_config(raw)
    if cfg is None:
        raise ValueError("Could not find P3achyGoModel config in checkpoint.")
    return cfg


def patch_weights(old_h5_path: str, new_h5_path: str) -> tuple[list, list]:
    """Copy weights from old h5 into new h5, applying rename rules.

    Returns (copied, skipped) key lists.
    """
    with h5py.File(old_h5_path, "r") as old_f:
        old_tensors = {}
        old_f.visititems(
            lambda name, obj: (
                old_tensors.update({name: obj[()]})
                if isinstance(obj, h5py.Dataset)
                else None
            )
        )

    # Build new_key -> old_key lookup, applying renames in reverse.
    def old_key_for(new_key):
        if new_key in old_tensors:
            return new_key
        for old_prefix, new_prefix in _RENAMES.items():
            if new_key.startswith(new_prefix):
                candidate = old_prefix + new_key[len(new_prefix) :]
                if candidate in old_tensors:
                    return candidate
        return None

    copied, skipped = [], []
    with h5py.File(new_h5_path, "r+") as new_f:

        def patch(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            src = old_key_for(name)
            if src is not None:
                old_data = old_tensors[src]
                if old_data.shape == obj.shape:
                    obj[()] = old_data
                    copied.append(name)
                else:
                    logging.warning(
                        f"Shape mismatch '{name}': old={old_data.shape} "
                        f"new={obj.shape} — keeping random init"
                    )
                    skipped.append(name)
            else:
                skipped.append(name)

        new_f.visititems(patch)

    return copied, skipped


def main(_):
    if not FLAGS.input_path or not FLAGS.output_path:
        logging.error("--input_path and --output_path are required.")
        return

    input_path = Path(FLAGS.input_path)
    output_path = Path(FLAGS.output_path)

    logging.info(f"Reading config from {input_path} ...")
    cfg = read_model_config(str(input_path))

    new_model = P3achyGoModel(
        board_len=cfg["board_len"],
        num_input_planes=cfg["num_input_planes"],
        num_input_features=cfg["num_input_features"],
        num_blocks=cfg["num_blocks"],
        num_channels=cfg["num_channels"],
        num_bottleneck_channels=cfg["num_bottleneck_channels"],
        num_head_channels=cfg["num_head_channels"],
        c_val=cfg["c_val"],
        bottleneck_length=cfg["bottleneck_length"],
        conv_size=cfg["conv_size"],
        broadcast_interval=cfg["broadcast_interval"],
        trunk_block_type=cfg["trunk_block_type"],
        generic_arch=cfg.get("generic_arch", False),
        c_l2=cfg.get("c_l2", 1e-4),
        name=cfg["name"],
    )

    dummy_board = np.zeros([1] + new_model.input_planes_shape(), dtype=np.float32)
    dummy_features = np.zeros([1] + new_model.input_features_shape(), dtype=np.float32)
    new_model(dummy_board, dummy_features, training=False)
    logging.info(
        f"New model built: {len(new_model(dummy_board, dummy_features, training=False))} outputs"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        old_h5 = os.path.join(tmpdir, "old.weights.h5")
        new_h5 = os.path.join(tmpdir, "new.weights.h5")

        with zipfile.ZipFile(str(input_path), "r") as zf:
            with zf.open("model.weights.h5") as src, open(old_h5, "wb") as dst:
                shutil.copyfileobj(src, dst)

        new_model.save_weights(new_h5)
        copied, skipped = patch_weights(old_h5, new_h5)
        new_model.load_weights(new_h5)

    logging.info(f"Copied {len(copied)} weight tensors.")
    if skipped:
        logging.info(
            f"{len(skipped)} tensor(s) had no source (random init): "
            + ", ".join(skipped)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(str(output_path))
    logging.info(f"Saved migrated model to {output_path}")


if __name__ == "__main__":
    app.run(main)
