from __future__ import annotations

import shlex, subprocess, re, os

from google.cloud import storage
from pathlib import Path
from typing import Callable

GCS_CLIENT = storage.Client()
GCS_BUCKET = 'p3achygo'

# keep in sync with cc/shuffler/chunk_manager.cc, and GCS file tree structure.
MODELS_DIR = 'models'
MODEL_CANDS_DIR = 'model_cands'
GOLDEN_CHUNK_DIR = 'goldens'
SP_CHUNK_DIR = 'chunks'
SGF_DIR = 'sgf'

GOLDEN_CHUNK_PREFIX = 'chunk'
GOLDEN_CHUNK_FORMAT = GOLDEN_CHUNK_PREFIX + '_{:04d}.tfrecord.zz'
GOLDEN_CHUNK_RE = re.compile(GOLDEN_CHUNK_PREFIX + '_([0-9]+)\.tfrecord\.zz')
GOLDEN_CHUNK_SIZE_FORMAT = GOLDEN_CHUNK_PREFIX + '_{:04d}.size'
GOLDEN_CHUNK_SIZE_RE = re.compile(GOLDEN_CHUNK_PREFIX + '_([0-9]+)\.size')
MODEL_PREFIX = 'model'
MODEL_FORMAT = MODEL_PREFIX + '_{:04d}'
MODEL_RE = re.compile(MODEL_PREFIX + '_([0-9]+)')

# Keep in sync with //cc/shuffler/chunk_info.h
SP_CHUNK_RE = re.compile(
    'gen(\d+)_b(\d+)_g(\d+)_n(\d+)_t(\d+)_(.*)\.tfrecord\.zz')

DONE_PREFIX = 'DONE'
DONE_FILENAME = 'DONE'


def _get_num(blob_path: str, regex: re.Pattern) -> int:
  p, match = Path(blob_path), None
  for part in p.parts:
    match = regex.fullmatch(part)
    if match != None:
      break

  if match is None:
    return -1

  assert len(match.groups()) == 1
  return int(match.group(1))


def _get_model_num(blob_path: str) -> int:
  return _get_num(blob_path, MODEL_RE)


def _get_chunk_num(blob_path: str) -> int:
  return _get_num(blob_path, GOLDEN_CHUNK_RE)


def _get_most_recent(prefix: str, num_fn: Callable[[str], int],
                     sentinel: int) -> int:
  blobs = GCS_CLIENT.list_blobs(GCS_BUCKET, prefix=prefix)
  most_recent = sentinel
  for blob in blobs:
    most_recent = max(num_fn(blob.name), most_recent)

  return most_recent


def _upload_chunk_file(run_id: str, local_chunk_dir: str, chunk_filename: str):
  local_chunk_path = Path(local_chunk_dir, chunk_filename)
  gcs_chunk_path = Path(gcs_chunk_dir(run_id), chunk_filename)
  if not local_chunk_path.exists():
    raise Exception(f'Golden Chunk not found: {str(local_chunk_path)}')

  bucket = GCS_CLIENT.bucket(GCS_BUCKET)
  blob = bucket.blob(str(gcs_chunk_path))
  blob.upload_from_filename(str(local_chunk_path))


def _upload_model(run_id: str, local_models_dir: str, model_dir: str):
  local_dir = Path(local_models_dir, model_dir)
  gcs_dir = Path(gcs_models_dir(run_id), model_dir)
  if not local_dir.exists():
    raise Exception(f'Model not found: {str(local_dir)}')

  cmd = f'gsutil -m rsync -r {str(local_dir)} gs://{GCS_BUCKET}/{str(gcs_dir)}'
  subprocess.run(shlex.split(cmd), check=True)


def _upload_model_cand(run_id: str, local_models_dir: str, model_dir: str):
  local_dir = Path(local_models_dir, model_dir)
  gcs_dir = Path(gcs_model_cands_dir(run_id), model_dir)
  if not local_dir.exists():
    raise Exception(f'Model not found: {str(local_dir)}')

  cmd = f'gsutil -m rsync -r {str(local_dir)} gs://{GCS_BUCKET}/{str(gcs_dir)}'
  subprocess.run(shlex.split(cmd), check=True)


def _download(local_path: str, gcs_path: str):
  bucket = GCS_CLIENT.bucket(GCS_BUCKET)
  blob = bucket.blob(gcs_path)
  blob.download_to_filename(local_path)

  return local_path


def _download_model(run_id: str, local_models_dir: str, model_dir: str) -> str:
  local_dir = Path(local_models_dir, model_dir)
  gcs_dir = Path(gcs_models_dir(run_id), model_dir)

  if not os.path.exists(local_dir):
    os.mkdir(local_dir)

  cmd = f'gsutil -m rsync -r gs://{GCS_BUCKET}/{str(gcs_dir)} {str(local_dir)}'
  subprocess.run(shlex.split(cmd), check=True)

  return str(local_dir)


def _download_model_cand(run_id: str, local_models_dir: str,
                         model_dir: str) -> str:
  local_dir = Path(local_models_dir, model_dir)
  gcs_dir = Path(gcs_model_cands_dir(run_id), model_dir)

  if not os.path.exists(local_dir):
    os.mkdir(local_dir)

  cmd = f'gsutil -m rsync -r gs://{GCS_BUCKET}/{str(gcs_dir)} {str(local_dir)}'
  subprocess.run(shlex.split(cmd), check=True)

  return str(local_dir)


def _download_chunk_file(run_id: str, local_chunk_dir: str,
                         chunk_filename: str) -> str:
  local_chunk_path = Path(local_chunk_dir, chunk_filename)
  gcs_chunk_path = Path(gcs_chunk_dir(run_id), chunk_filename)

  return _download(str(local_chunk_path), str(gcs_chunk_path))


def list_sp_chunks(run_id: str) -> list[str]:
  gcs_dir = gcs_sp_chunk_dir(run_id) + '/'

  blobs = GCS_CLIENT.list_blobs(GCS_BUCKET, prefix=gcs_dir)
  return [f.name for f in filter(lambda f: f.name != gcs_dir, blobs)]


def local_models_dir(model_dir: str) -> str:
  return str(Path(model_dir, MODELS_DIR))


def local_chunk_dir(data_dir: str) -> str:
  return str(Path(data_dir, GOLDEN_CHUNK_DIR))


def gcs_models_dir(run_id: str) -> str:
  return str(Path(run_id, MODELS_DIR))


def gcs_model_cands_dir(run_id: str) -> str:
  return str(Path(run_id, MODEL_CANDS_DIR))


def gcs_chunk_dir(run_id: str) -> str:
  return str(Path(run_id, GOLDEN_CHUNK_DIR))


def gcs_sp_chunk_dir(run_id: str) -> str:
  return str(Path(run_id, SP_CHUNK_DIR))


def gcs_sgf_dir(run_id: str) -> str:
  return str(Path(run_id, SGF_DIR))


def get_most_recent_model(run_id: str) -> int:
  return _get_most_recent(gcs_models_dir(run_id), _get_model_num, -1)


def get_most_recent_model_cand(run_id: str) -> int:
  return _get_most_recent(gcs_model_cands_dir(run_id), _get_model_num, -1)


def get_most_recent_chunk(run_id: str) -> int:
  return _get_most_recent(gcs_chunk_dir(run_id), _get_chunk_num, 0)


def upload_chunk(run_id: str, local_chunk_dir: str, gen: int):
  _upload_chunk_file(run_id, local_chunk_dir, GOLDEN_CHUNK_FORMAT.format(gen))


def upload_chunk_size(run_id: str, local_chunk_dir: str, gen: int):
  _upload_chunk_file(run_id, local_chunk_dir,
                     GOLDEN_CHUNK_SIZE_FORMAT.format(gen))


def upload_model(run_id: str, local_models_dir: str, gen: int):
  _upload_model(run_id, local_models_dir, MODEL_FORMAT.format(gen))


def upload_model_cand(run_id: str, local_models_dir: str, gen: int):
  _upload_model_cand(run_id, local_models_dir, MODEL_FORMAT.format(gen))


def upload_sp_chunk(run_id: str, local_path: Path):
  if not local_path.exists():
    raise Exception(f'Self-Play Chunk not found: {str(local_path)}')

  gcs_path = Path(gcs_sp_chunk_dir(run_id), local_path.name)
  bucket = GCS_CLIENT.bucket(GCS_BUCKET)
  blob = bucket.blob(str(gcs_path))
  blob.upload_from_filename(str(local_path))


def upload_sgf(run_id: str, local_path: Path):
  if not local_path.exists():
    raise Exception(f'SGF not found: {str(local_path)}')

  gcs_path = Path(gcs_sgf_dir(run_id), local_path.name)
  bucket = GCS_CLIENT.bucket(GCS_BUCKET)
  blob = bucket.blob(str(gcs_path))
  blob.upload_from_filename(str(local_path))


def download_golden_chunk_size(run_id: str, local_chunk_dir: str,
                               gen: int) -> str:
  return _download_chunk_file(run_id, local_chunk_dir,
                              GOLDEN_CHUNK_SIZE_FORMAT.format(gen))


def download_golden_chunk(run_id: str, local_chunk_dir: str, gen: int) -> str:
  return _download_chunk_file(run_id, local_chunk_dir,
                              GOLDEN_CHUNK_FORMAT.format(gen))


def download_model(run_id: str, local_models_dir: str, gen: int) -> str:
  return _download_model(run_id, local_models_dir, MODEL_FORMAT.format(gen))


def download_model_cand(run_id: str, local_models_dir: str, gen: int) -> str:
  return _download_model_cand(run_id, local_models_dir,
                              MODEL_FORMAT.format(gen))


def download_val_ds(local_dir: str) -> str:
  local_path = Path(local_dir, 'val.tfrecord.zz')
  return _download(str(local_path), 'common/val.tfrecord.zz')


def is_done(run_id: str) -> bool:
  blobs = GCS_CLIENT.list_blobs(GCS_BUCKET, prefix=Path(run_id, DONE_PREFIX))
  for blob in blobs:
    path = Path(blob.name)
    if path.parent.name == DONE_PREFIX and path.name == DONE_FILENAME:
      return True

  return False


def signal_done(run_id: str):
  path = str(Path(run_id, DONE_PREFIX, DONE_FILENAME))
  blob = GCS_CLIENT.bucket(GCS_BUCKET).blob(path)
  blob.upload_from_string(b'')
