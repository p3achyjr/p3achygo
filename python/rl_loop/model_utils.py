from absl import logging

import gcs_utils as gcs

from model import P3achyGoModel
from model_config import ModelConfig
from constants import *
import trt_convert

from pathlib import Path

SWA_MOMENTUM = .75


def new_model(name: str) -> P3achyGoModel:
  return P3achyGoModel.create(config=ModelConfig.small(),
                              board_len=BOARD_LEN,
                              num_input_planes=NUM_INPUT_PLANES,
                              num_input_features=NUM_INPUT_FEATURES,
                              name=name)


def avg_weights(prev_weights: list, cur_weights: list) -> list:
  return [
      prev_layer_weights * SWA_MOMENTUM + layer_weights * (1 - SWA_MOMENTUM)
      for prev_layer_weights, layer_weights in zip(prev_weights, cur_weights)
  ]


def save_trt_and_upload(model: P3achyGoModel, calib_ds_path: str,
                        local_model_dir: str, gen: int, run_id: str) -> str:
  model_path = save_trt(model, calib_ds_path, local_model_dir, gen)
  gcs.upload_model(run_id, str(local_model_dir), gen)

  return model_path


def save_trt(model: P3achyGoModel, calib_ds_path: str, local_model_dir: str,
             gen: int) -> str:
  '''
  Saves model and returns _base_ path of model.
  '''
  model_path = Path(local_model_dir, f'model_{gen}')
  model.save(str(model_path))

  logging.info('Converting to TensorRT...')
  trt_converter = trt_convert.get_converter(str(model_path), calib_ds_path)
  trt_converter.summary()
  trt_converter.save(output_saved_model_dir=str(Path(model_path, '_trt')))

  return str(model_path)
