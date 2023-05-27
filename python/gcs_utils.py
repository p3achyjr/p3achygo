from __future__ import annotations

import shlex, subprocess, re

from google.cloud import storage
from pathlib import Path

GCS_CLIENT = storage.Client()
GCS_BUCKET = 'p3achygo'

# keep in sync with cc/shuffler/chunk_manager.cc
MODELS_DIR = 'models'
GOLDEN_CHUNK_DIR = 'goldens'
SGF_DIR = 'sgf'

CHUNK_PREFIX = 'chunk'
CHUNK_FORMAT = CHUNK_PREFIX + '_{}.tfrecord.zz'
CHUNK_RE = re.compile(CHUNK_PREFIX + '_([0-9]+)\.tfrecord\.zz')
MODEL_PREFIX = 'model'
MODEL_FORMAT = MODEL_PREFIX + '_{}'
MODEL_RE = re.compile(MODEL_PREFIX + '_([0-9]+)')


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
  return _get_num(blob_path, CHUNK_RE)


def _upload_chunk(run_id: str, local_chunk_dir: str, chunk_filename: str):
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

  cmd = f'gsutil -m cp -r {str(local_dir)} gs://{GCS_BUCKET}/{str(gcs_dir)}'
  subprocess.run(shlex.split(cmd), check=True)


def _download(local_path: str, gcs_path: str):
  bucket = GCS_CLIENT.bucket(GCS_BUCKET)
  blob = bucket.blob(gcs_path)
  blob.download_to_filename(local_path)

  return local_path


def _download_model(run_id: str, local_models_dir: str, model_dir: str) -> str:
  local_dir = Path(local_models_dir, model_dir)
  gcs_dir = Path(gcs_models_dir(run_id), model_dir)

  cmd = f'gsutil -m cp -r gs://{GCS_BUCKET}/{str(gcs_dir)} {str(local_models_dir)}'
  subprocess.run(shlex.split(cmd), check=True)

  return str(local_dir)


def _download_chunk(run_id: str, local_chunk_dir: str,
                    chunk_filename: str) -> str:
  local_chunk_path = Path(local_chunk_dir, chunk_filename)
  gcs_chunk_path = Path(gcs_chunk_dir(run_id), chunk_filename)

  return _download(str(local_chunk_path), str(gcs_chunk_path))


def local_models_dir(model_dir: str) -> str:
  return str(Path(model_dir, GOLDEN_CHUNK_DIR))


def local_chunk_dir(data_dir: str) -> str:
  return str(Path(data_dir, GOLDEN_CHUNK_DIR))


def gcs_models_dir(run_id: str) -> str:
  return str(Path(run_id, MODELS_DIR))


def gcs_chunk_dir(run_id: str) -> str:
  return str(Path(run_id, GOLDEN_CHUNK_DIR))


def gcs_sgf_dir(run_id: str) -> str:
  return str(Path(run_id, SGF_DIR))


def get_most_recent_model(run_id: str) -> int:
  model_blobs = GCS_CLIENT.list_blobs(GCS_BUCKET, prefix=gcs_models_dir(run_id))
  most_recent = -1
  for blob in model_blobs:
    most_recent = max(_get_model_num(blob.name), most_recent)

  return most_recent


def get_most_recent_chunk(run_id: str) -> int:
  chunk_blobs = GCS_CLIENT.list_blobs(GCS_BUCKET, prefix=gcs_chunk_dir(run_id))
  most_recent = 0
  for blob in chunk_blobs:
    most_recent = max(_get_chunk_num(blob.name), most_recent)

  return most_recent


def upload_chunk(run_id: str, local_chunk_dir: str, gen: int):
  _upload_chunk(run_id, local_chunk_dir, CHUNK_FORMAT.format(gen))


def upload_model(run_id: str, local_models_dir: str, gen: int):
  _upload_model(run_id, local_models_dir, MODEL_FORMAT.format(gen))


def download_golden_chunk(run_id: str, local_chunk_dir: str, gen: int) -> str:
  return _download_chunk(run_id, local_chunk_dir, CHUNK_FORMAT.format(gen))


def download_model(run_id: str, local_models_dir: str, gen: int) -> str:
  return _download_model(run_id, local_models_dir, MODEL_FORMAT.format(gen))
