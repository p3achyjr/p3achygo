from absl import logging

import gcs_utils as gcs
import rl_loop.fs_utils as fs

from model import P3achyGoModel
from model_config import ModelConfig
from constants import *
import proc

from pathlib import Path

NUM_BATCHES_FULL_CHECKPOINT = 1000


def new_model(name: str, model_config="small") -> P3achyGoModel:
    return P3achyGoModel.create(
        config=ModelConfig.from_str(model_config),
        board_len=BOARD_LEN,
        num_input_planes=NUM_INPUT_PLANES,
        num_input_features=NUM_INPUT_FEATURES,
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
