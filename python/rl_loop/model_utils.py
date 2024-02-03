from absl import logging

import gcs_utils as gcs

from model import P3achyGoModel
from model_config import ModelConfig
from constants import *
import trt_convert
import proc

from pathlib import Path

NUM_BATCHES_FULL_CHECKPOINT = 1000


def new_model(name: str, model_config='small') -> P3achyGoModel:
  return P3achyGoModel.create(config=ModelConfig.from_str(model_config),
                              board_len=BOARD_LEN,
                              num_input_planes=NUM_INPUT_PLANES,
                              num_input_features=NUM_INPUT_FEATURES,
                              name=name)


def swa_avg_weights(weights: list, swa_momentum: float = 0.75) -> list:
  swa_weights = weights[0]
  for i in range(1, len(weights), 1):
    swa_weights = [
        prev_layer_weights * swa_momentum + layer_weights * (1 - swa_momentum)
        for prev_layer_weights, layer_weights in zip(swa_weights, weights[i])
    ]

  return swa_weights


def save_trt_and_upload(model: P3achyGoModel, calib_ds_path: str,
                        local_model_dir: str, gen: int, run_id: str,
                        batch_size: int) -> str:
  model_path = save_trt(model, calib_ds_path, local_model_dir, gen, batch_size)
  gcs.upload_model(run_id, str(local_model_dir), gen)

  return model_path


def save_trt(model: P3achyGoModel, calib_ds_path: str, local_model_dir: str,
             gen: int, batch_size: int) -> str:
  '''
  Saves model, converts to TRT, and returns _base_ path of model.
  '''
  model_path = save(model, local_model_dir, gen)

  logging.info('Converting to TensorRT...')
  trt_converter = trt_convert.get_converter(str(model_path), calib_ds_path,
                                            batch_size)
  trt_converter.summary()
  trt_converter.save(output_saved_model_dir=str(Path(model_path, '_trt')))

  return str(model_path)


def save_onnx_trt(model: P3achyGoModel, calib_ds_path: str,
                  local_model_dir: str, gen: int, batch_size: int,
                  trt_convert_path: str) -> str:
  '''
  Saves model through ONNX -> TRT path.
  '''
  model_path = save(model, local_model_dir, gen)
  logging.info('Converting to ONNX...')
  onnx_path = trt_convert.convert_onnx(model_path)

  logging.info('Converting to ONNX-TRT...')
  cmd = (f'{trt_convert_path} --onnx_path={onnx_path}' +
         f' --ds_path={calib_ds_path}' + f' --batch_size={batch_size}')

  proc.run_proc(cmd)
  return str(Path(model_path, '_onnx', 'engine.trt'))


def save(model: P3achyGoModel, local_model_dir: str, gen: int) -> str:
  '''
  Saves model and returns _base_ path of model.
  '''
  model_path = Path(local_model_dir, gcs.MODEL_FORMAT.format(gen))
  model.save(str(model_path))

  return str(model_path)
