from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="p3achygo")
class ConstantLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Constant LR Schedule.
    """

    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, _=None):
        return self.lr

    def info(self) -> str:
        return f"Constant Learning Rate: {self.lr}"

    def get_config(self):
        return {"lr": self.lr}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CyclicLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements cyclic learning rate.

    https://arxiv.org/pdf/1803.09820.pdf
    """

    def __init__(self, min_lr: float, max_lr: float, cycle_len: int):
        self.lr_min = keras.ops.convert_to_tensor(min_lr, dtype="float32")
        self.lr_max = keras.ops.convert_to_tensor(max_lr, dtype="float32")
        self.cycle_len = keras.ops.convert_to_tensor(cycle_len, dtype="int64")
        self.half_cycle_len = keras.ops.convert_to_tensor(
            cycle_len // 2 if cycle_len % 2 == 0 else cycle_len // 2 + 1, dtype="int64"
        )

        self.lr_delta = keras.ops.convert_to_tensor(
            max_lr - min_lr, dtype="float32"
        ) / keras.ops.cast(self.half_cycle_len, dtype="float32")

    def __call__(self, step):
        step %= self.cycle_len
        ninc, ndec = keras.ops.minimum(step, self.half_cycle_len), keras.ops.maximum(
            keras.ops.convert_to_tensor(0, dtype="int64"), step - self.half_cycle_len
        )

        return self.lr_min + self.lr_delta * keras.ops.cast(
            ninc - ndec, dtype="float32"
        )

    def info(self) -> str:
        return (
            f"Cyclic LR. LR Min: {self.lr_min}"
            + f", LR Max: {self.lr_max}"
            + f", Cycle Len: {self.cycle_len}, LR_Delta: {self.lr_delta}"
        )


class CyclicLRDecaySchedule(keras.optimizers.schedules.LearningRateSchedule):
    """
    One-cycle LR schedule with final decay.

    Use with fixed-length dataset.
    """

    def __init__(self, min_lr: float, max_lr: float, cycle_len: int, decay_bound=0.95):
        self._main_cycle_len = keras.ops.convert_to_tensor(
            int(cycle_len * decay_bound), dtype="int64"
        )
        self._half_cycle_len = keras.ops.convert_to_tensor(
            cycle_len // 2 if cycle_len % 2 == 0 else cycle_len // 2 + 1, dtype="int64"
        )
        self._decay_len = (
            keras.ops.convert_to_tensor(cycle_len, dtype="int64") - self._main_cycle_len
        )

        self._lr_min = keras.ops.convert_to_tensor(min_lr, dtype="float32")
        self._lr_max = keras.ops.convert_to_tensor(max_lr, dtype="float32")
        self._lr_delta = keras.ops.convert_to_tensor(
            max_lr - min_lr, dtype="float32"
        ) / keras.ops.cast(self._half_cycle_len, dtype="float32")

        # decay learning rate for end of training cycle
        self._lr_final = self._lr_min * 0.25
        self._lr_decay_delta = (self._lr_min - self._lr_final) / keras.ops.cast(
            self._decay_len, dtype="float32"
        )

    def __call__(self, step):
        cycle_step = keras.ops.cond(
            step < self._main_cycle_len,
            lambda: step % self._main_cycle_len,
            lambda: keras.ops.convert_to_tensor(0, dtype="int64"),
        )
        decay_step = keras.ops.cond(
            step >= self._main_cycle_len,
            lambda: step - self._main_cycle_len,
            lambda: keras.ops.convert_to_tensor(0, dtype="int64"),
        )
        ninc, ndec = keras.ops.minimum(
            cycle_step, self._half_cycle_len
        ), keras.ops.maximum(
            keras.ops.convert_to_tensor(0, dtype="int64"),
            cycle_step - self._half_cycle_len,
        )

        main_delta = self._lr_delta * keras.ops.cast(ninc - ndec, dtype="float32")
        decay_delta = self._lr_decay_delta * keras.ops.cast(decay_step, dtype="float32")

        return self._lr_min + main_delta - decay_delta

    def info(self) -> str:
        return (
            f"Cyclic LR Decay. LR Min: {self._lr_min}"
            + f", LR Max: {self._lr_max}"
            + f", LR Post-Decay: {self._lr_final}"
            + f", Cycle Len: {self._main_cycle_len}, LR_Delta: {self._lr_delta}"
            + f", LR_Decay_Delta: {self._lr_decay_delta}"
        )
