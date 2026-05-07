"""Backend-aware LR schedule dispatcher.

Under TF, re-exports the keras-decorated `ConstantLRSchedule` (and the
unused `CyclicLRSchedule` / `CyclicLRDecaySchedule`) from
`backend_tf/lr_schedule.py`. The keras `@register_keras_serializable`
decorator runs at import time, registering `p3achygo>ConstantLRSchedule`
so saved keras checkpoints with embedded schedule instances can be
deserialized.

Under torch, defines a plain Python `ConstantLRSchedule` with the same
call signature — torch's optimizer adapter only needs a callable
returning a float.
"""

from __future__ import annotations

from backend import BACKEND as _backend

if _backend == "tensorflow":
    from backend_tf.lr_schedule import (  # noqa: F401
        ConstantLRSchedule,
        CyclicLRSchedule,
        CyclicLRDecaySchedule,
    )
elif _backend == "torch":

    class ConstantLRSchedule:
        def __init__(self, lr: float):
            self.lr = lr

        def __call__(self, _=None):
            return self.lr

        def info(self) -> str:
            return f"Constant Learning Rate: {self.lr}"

else:
    raise ValueError(f"unsupported backend {_backend!r}")
