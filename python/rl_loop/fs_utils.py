from __future__ import annotations

import gcs_utils as gcs

from pathlib import Path
from typing import Tuple, Callable

LOCAL_PATH = ''
MODE = 'gcs'


def _get_most_recent(prefix: str, num_fn: Callable[[str], int],
                     sentinel: int) -> int:
  dir = Path(LOCAL_PATH, prefix)
  most_recent = sentinel
  for blob in dir.iterdir():
    most_recent = max(num_fn(blob.name), most_recent)

  return most_recent


def configure_fs(mode='gcs', local_path=''):
  if mode not in ['local', 'gcs']:
    raise Exception(f'Invalid Mode: {mode}, Must be (local, gcs)')
  if mode == 'local' and not local_path:
    raise Exception('Must pass a local path if using local storage.')

  global MODE
  global LOCAL_PATH
  MODE = mode
  LOCAL_PATH = local_path


def get_most_recent_model(run_id: str) -> int:
  if MODE == 'gcs':
    return gcs.get_most_recent_model(run_id)
  return _get_most_recent(gcs.gcs_models_dir(run_id), gcs._get_model_num, -1)


def get_most_recent_model_cand(run_id: str) -> int:
  if MODE == 'gcs':
    return gcs.get_most_recent_model_cand(run_id)
  return _get_most_recent(gcs.gcs_model_cands_dir(run_id), gcs._get_model_num,
                          -1)


def get_most_recent_chunk(run_id: str) -> int:
  if MODE == 'gcs':
    return gcs.get_most_recent_chunk(run_id)
  return _get_most_recent(gcs.gcs_chunk_dir(run_id), gcs._get_chunk_num, 0)


def upload_chunk(run_id: str, local_chunk_dir: str, gen: int):
  if MODE == 'gcs':
    gcs.upload_chunk(run_id, local_chunk_dir, gen)


def upload_chunk_size(run_id: str, local_chunk_dir: str, gen: int):
  if MODE == 'gcs':
    gcs.upload_chunk_size(run_id, local_chunk_dir, gen)


def remove_local_chunk(local_chunk_dir: str, gen: int):
  if MODE == 'gcs':
    gcs.remove_local_chunk(local_chunk_dir, gen)
  # We should not remove this in case, as the shuffler will need this file.


def upload_model(run_id: str, local_models_dir: str, gen: int):
  if MODE == 'gcs':
    gcs.upload_model(run_id, local_models_dir, gen)


def upload_model_cand(run_id: str, local_models_dir: str, gen: int):
  if MODE == 'gcs':
    gcs.upload_model_cand(run_id, local_models_dir, gen)


def upload_sp_chunk(run_id: str, local_path: Path):
  if MODE == 'gcs':
    gcs.upload_sp_chunk(run_id, local_path)


def upload_sgf(run_id: str, local_path: Path):
  if MODE == 'gcs':
    gcs.upload_sgf(run_id, local_path)


def download_golden_chunk_size(run_id: str, local_chunk_dir: str,
                               gen: int) -> str:
  if MODE == 'gcs':
    return gcs.download_golden_chunk_size(run_id, local_chunk_dir, gen)
  return str(Path(local_chunk_dir, gcs.GOLDEN_CHUNK_SIZE_FORMAT.format(gen)))


def download_golden_chunk(run_id: str, local_chunk_dir: str, gen: int) -> str:
  if MODE == 'gcs':
    return gcs.download_golden_chunk(run_id, local_chunk_dir, gen)
  return str(Path(local_chunk_dir, gcs.GOLDEN_CHUNK_FORMAT.format(gen)))


def download_model(run_id: str, local_models_dir: str, gen: int) -> str:
  if MODE == 'gcs':
    return gcs.download_model(run_id, local_models_dir, gen)
  return str(Path(local_models_dir, gcs.MODEL_FORMAT.format(gen)))


def download_model_cand(run_id: str, local_models_dir: str, gen: int) -> str:
  if MODE == 'gcs':
    return gcs.download_model_cand(run_id, local_models_dir, gen)
  return str(Path(local_models_dir, gcs.MODEL_FORMAT.format(gen)))


def download_val_ds(local_dir: str) -> str:
  return gcs.download_val_ds(local_dir)


def ensure_local_dirs(local_run_dir: str) -> Tuple[Path, Path, Path, Path]:
  local_models_dir = Path(local_run_dir, gcs.MODELS_DIR)
  local_golden_chunk_dir = Path(local_run_dir, gcs.GOLDEN_CHUNK_DIR)
  local_sp_chunk_dir = Path(local_run_dir, gcs.SP_CHUNK_DIR)
  local_sgf_dir = Path(local_run_dir, gcs.SGF_DIR)

  local_models_dir.mkdir(exist_ok=True)
  local_golden_chunk_dir.mkdir(exist_ok=True)
  local_sp_chunk_dir.mkdir(exist_ok=True)
  local_sgf_dir.mkdir(exist_ok=True)

  return (local_models_dir, local_golden_chunk_dir, local_sp_chunk_dir,
          local_sgf_dir)


def file_diff(dir: Path, files: set(Path),
              pat: str) -> Tuple[set(Path), set[Path]]:
  cur_files = set(dir.glob(pat))
  return cur_files, cur_files.difference(files)
