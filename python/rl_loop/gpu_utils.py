'''GPU utility functions for multi-GPU self-play.'''

from __future__ import annotations

import os


def get_gpu_count() -> int:
  '''Returns the number of available NVIDIA GPUs.'''
  try:
    import pynvml
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return count
  except Exception:
    # Fallback: check CUDA_VISIBLE_DEVICES or assume 1 GPU
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
      return len([d for d in cuda_visible.split(',') if d.strip()])
    return 1
