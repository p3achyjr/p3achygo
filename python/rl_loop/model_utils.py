from absl import logging

import tensorflow as tf
import keras
import gcs_utils as gcs
import rl_loop.fs_utils as fs

from model import P3achyGoModel
from model_config import ModelConfig
from constants import *
import proc

from pathlib import Path

NUM_BATCHES_FULL_CHECKPOINT = 1000


def new_model(name: str, model_config="small", optimizer="sgd") -> P3achyGoModel:
    config = ModelConfig.from_str(model_config)
    if optimizer == "muon":
        config.c_l2 = 0
    return P3achyGoModel.create(
        config=config,
        board_len=BOARD_LEN,
        num_input_planes=num_input_planes(),
        num_input_features=num_input_features(),
        name=name,
    )


def swa_avg_weights(weights: list, swa_momentum: float = 0.75) -> list:
    swa_weights = weights[0]
    for i in range(1, len(weights), 1):
        swa_weights = [
            prev_layer_weights * swa_momentum + layer_weights * (1 - swa_momentum)
            for prev_layer_weights, layer_weights in zip(swa_weights, weights[i])
        ]

    return swa_weights


def recompute_bn_statistics(model, ds, num_batches=200):
    """
    Recompute BatchNorm running_mean and running_variance by doing
    forward passes through the ds.

    Note: Running with training=True without a GradientTape context does NOT
    compute or propagate gradients. It only tells BatchNorm layers to update
    their running statistics.
    """

    def _get_all_layers_recursive(layer):
        """Recursively get all layers including nested ones."""
        layers = [layer]
        # Use _layers (private) since custom layers don't expose public .layers
        sublayers = getattr(layer, "_layers", [])
        for sublayer in sublayers:
            layers.extend(_get_all_layers_recursive(sublayer))
        return layers

    def _get_bn_layers(model):
        """Get all BatchNorm layers in the model."""
        all_layers = []
        for layer in model.layers:
            all_layers.extend(_get_all_layers_recursive(layer))
        return [
            layer
            for layer in all_layers
            if isinstance(layer, keras.layers.BatchNormalization)
        ]

    def _log_bn_stats(layer, prefix=""):
        """Log mean/var statistics for a BatchNorm layer."""
        mean = layer.moving_mean.numpy()
        var = layer.moving_variance.numpy()
        print(f"{prefix}BN layer '{layer.name}':")
        print(
            f"  moving_mean  - min: {mean.min():.6f}, max: {mean.max():.6f}, mean: {mean.mean():.6f}"
        )
        print(
            f"  moving_var   - min: {var.min():.6f}, max: {var.max():.6f}, mean: {var.mean():.6f}"
        )

    bn_layers = _get_bn_layers(model)
    first_bn = bn_layers[0] if bn_layers else None
    print(f"Found {len(bn_layers)} BatchNorm layers")

    # Reset BN statistics
    # for layer in bn_layers:
    #     layer.moving_mean.assign(tf.zeros_like(layer.moving_mean))
    #     layer.moving_variance.assign(tf.ones_like(layer.moving_variance))

    if first_bn:
        print("=== Initial BN statistics (after reset) ===")
        _log_bn_stats(first_bn)

    # Forward passes to recompute statistics
    for i, batch in enumerate(ds.take(num_batches)):
        # batch[0] = input (board planes)
        # batch[1] = input_global_state
        input_board = batch[0]
        input_global = batch[1]

        _ = model(input_board, input_global, training=True)

        if (i + 1) % 20 == 0:
            print(
                f"=== recompute_bn_statistics: Processed {i + 1}/{num_batches} batches ==="
            )
            if first_bn:
                _log_bn_stats(first_bn)

    if first_bn:
        print("=== Final BN statistics ===")
        _log_bn_stats(first_bn)


def avg_weights(
    prev_weights: list,
    cur_weights: list,
    num_batches_in_chunk: int,
    swa_momentum: float = 0.4,
) -> list:
    # chunk_ratio = min(1.0,
    #                   float(num_batches_in_chunk) / NUM_BATCHES_FULL_CHECKPOINT)
    # m_swa_new = (1 - SWA_MOMENTUM) * chunk_ratio
    # swa_momentum = 1 - m_swa_new
    print("SWA Momentum:", swa_momentum, "Num Batches: ", num_batches_in_chunk)
    return [
        prev_layer_weights * swa_momentum + layer_weights * (1 - swa_momentum)
        for prev_layer_weights, layer_weights in zip(prev_weights, cur_weights)
    ]


def save_trt_and_upload(
    model: P3achyGoModel,
    calib_ds_path: str,
    local_model_dir: str,
    gen: int,
    run_id: str,
    batch_size: int,
) -> str:
    model_path = save_trt(model, calib_ds_path, local_model_dir, gen, batch_size)
    fs.upload_model(run_id, str(local_model_dir), gen)

    return model_path


def save_trt(
    model: P3achyGoModel,
    calib_ds_path: str,
    local_model_dir: str,
    gen: int,
    batch_size: int,
) -> str:
    """
    Saves model, converts to ONNX, and returns _base_ path of model.
    """
    model_path = save(model, local_model_dir, gen)

    logging.info("Converting to ONNX...")
    cmd = f"python -m python.scripts.convert_to_onnx --model_path={model_path}"
    proc.run_proc(cmd)

    return str(model_path)


def save_onnx_trt(
    model: P3achyGoModel,
    calib_ds_path: str,
    local_model_dir: str,
    gen: int,
    batch_size: int,
    trt_convert_path: str,
) -> str:
    """
    Saves model through ONNX -> TRT path.
    """
    model_path = save(model, local_model_dir, gen)
    logging.info("Converting to ONNX...")
    cmd = f"python -m python.scripts.convert_to_onnx --model_path={model_path} --fp16"
    proc.run_proc(cmd)

    logging.info("Converting to ONNX-TRT...")
    model_p = Path(model_path)
    onnx_path = str(model_p.parent / "_onnx" / (model_p.stem + ".onnx"))
    trt_cmd = (
        f"{trt_convert_path} --onnx_path={onnx_path}"
        + f" --ds_path={calib_ds_path}"
        + f" --batch_size={batch_size}"
    )

    proc.run_proc(trt_cmd)
    return str(model_p.parent / "_onnx" / (model_p.stem + ".trt"))


def save(model: P3achyGoModel, local_model_dir: str, gen: int) -> str:
    """
    Saves model and returns _base_ path of model.
    """
    model_path = Path(local_model_dir, gcs.MODEL_FORMAT.format(gen))
    model.save(str(model_path))

    return str(model_path)
