"""
Backend-aware symbol resolution for training.

All branching happens here, at import time. Implementations live in
backend_tf/ and backend_torch/ and are themselves framework-specific
and clean (no conditional imports in function bodies).

Exported symbols:
  ConvMuon                       — keras-subclass optimizer for trunk/body
  ModelPredictions, GroundTruth  — data containers
  TrainStepResult                — train_step output
  train_step, val_step           — single forward/backward step
  configure_gpu                  — backend-agnostic GPU + mixed-precision setup
  SummaryWriter                  — backend-agnostic TensorBoard scalar writer
"""

import os

_backend = os.environ.get("KERAS_BACKEND", "tensorflow")

if _backend == "torch":
    from backend_torch.train import (
        ModelPredictions,
        GroundTruth,
        TrainStepResult,
        train_step,
        val_step,
    )
    from backend_torch.runtime import configure_gpu, SummaryWriter
elif _backend in ("tensorflow", "tf"):
    from backend_tf.train import (
        ModelPredictions,
        GroundTruth,
        TrainStepResult,
        train_step,
        val_step,
    )
    from backend_tf.runtime import configure_gpu, SummaryWriter
else:
    raise ValueError(f"unsupported KERAS_BACKEND={_backend!r}")

from optimizer import ConvMuon
