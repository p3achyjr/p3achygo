from __future__ import annotations

from google.cloud import storage
from pathlib import Path

GCS_CLIENT = storage.Client()
GCS_BUCKET = 'p3achygo'

# keep in sync with cc/shuffler/chunk_manager.cc
MODELS_DIR = 'models'
GOLDEN_CHUNK_DIR = 'goldens'
SGF_DIR = 'sgf'


def _get_model_num(blob_path: str) -> int:
  p, model_name = Path(blob_path), None
  for part in p.parts:
    if part != MODELS_DIR and part.startswith('model'):
      model_name = part

  if model_name is None:
    return -1

  n = ''
  for c in reversed(model_name):
    if not c.isdigit():
      break

    n += c

  return int(n[::-1])


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


def upload_golden_chunk_to_gcs(run_id: str, local_chunk_dir: str, gen: int):
  chunk_filename = f'chunk_{gen}.tfrecord.zz'
  local_chunk_path = Path(local_chunk_dir, chunk_filename)
  gcs_chunk_path = Path(gcs_chunk_dir(run_id), chunk_filename)
  if not local_chunk_path.exists():
    raise Exception(f'Golden Chunk not found: {str(local_chunk_path)}')

  bucket = GCS_CLIENT.bucket(GCS_BUCKET)
  blob = bucket.blob(str(gcs_chunk_path))
  blob.upload_from_filename(str(local_chunk_path))
