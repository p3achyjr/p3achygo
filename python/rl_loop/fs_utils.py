from __future__ import annotations

import gcs_utils as gcs

from pathlib import Path
from typing import Tuple


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
