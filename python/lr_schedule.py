from __future__ import annotations

import tensorflow as tf


class ConstantLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''
  Constant LR Schedule.
  '''

  def __init__(self, lr: float):
    self.lr = lr

  def __call__(self, _):
    return self.lr

  def info(self) -> str:
    return (f'Constant Learning Rate: {self.lr}')


class CyclicLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''
  Implements cyclic learning rate.

  https://arxiv.org/pdf/1803.09820.pdf
  '''

  def __init__(self, min_lr: float, max_lr: float, cycle_len: int):
    self.lr_min = tf.constant(min_lr)
    self.lr_max = tf.constant(max_lr)
    self.cycle_len = tf.constant(cycle_len, dtype=tf.int64)
    self.half_cycle_len = tf.constant(cycle_len // 2 if cycle_len %
                                      2 == 0 else cycle_len // 2 + 1,
                                      dtype=tf.int64)

    self.lr_delta = tf.constant(max_lr - min_lr) / tf.cast(self.half_cycle_len,
                                                           dtype=tf.float32)

  def __call__(self, step):
    step %= self.cycle_len
    ninc, ndec = tf.minimum(step, self.half_cycle_len), tf.maximum(
        tf.constant(0, dtype=tf.int64), step - self.half_cycle_len)

    return self.lr_min + self.lr_delta * tf.cast(ninc - ndec, dtype=tf.float32)

  def info(self) -> str:
    return (f'Cyclic LR. LR Min: {self.lr_min}' + f', LR Max: {self.lr_max}' +
            f', Cycle Len: {self.cycle_len}, LR_Delta: {self.lr_delta}')


class CyclicLRDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  '''
  One-cycle LR schedule with final decay.

  Use with fixed-length dataset.
  '''

  def __init__(self,
               min_lr: float,
               max_lr: float,
               cycle_len: int,
               decay_bound=.95):
    self._main_cycle_len = tf.constant(int(cycle_len * decay_bound),
                                       dtype=tf.int64)
    self._half_cycle_len = tf.constant(cycle_len // 2 if cycle_len %
                                       2 == 0 else cycle_len // 2 + 1,
                                       dtype=tf.int64)
    self._decay_len = tf.constant(cycle_len,
                                  dtype=tf.int64) - self._main_cycle_len

    self._lr_min = tf.constant(min_lr)
    self._lr_max = tf.constant(max_lr)
    self._lr_delta = tf.constant(max_lr - min_lr) / tf.cast(
        self._half_cycle_len, dtype=tf.float32)

    # decay learning rate for end of training cycle
    self._lr_final = self._lr_min * .25
    self._lr_decay_delta = (self._lr_min - self._lr_final) / tf.cast(
        self._decay_len, dtype=tf.float32)

  def __call__(self, step):
    cycle_step = tf.cond(step < self._main_cycle_len,
                         lambda: step % self._main_cycle_len,
                         lambda: tf.constant(0, dtype=tf.int64))
    decay_step = tf.cond(step >= self._main_cycle_len,
                         lambda: step - self._main_cycle_len,
                         lambda: tf.constant(0, dtype=tf.int64))
    ninc, ndec = tf.minimum(cycle_step, self._half_cycle_len), tf.maximum(
        tf.constant(0, dtype=tf.int64), cycle_step - self._half_cycle_len)

    main_delta = self._lr_delta * tf.cast(ninc - ndec, dtype=tf.float32)
    decay_delta = self._lr_decay_delta * tf.cast(decay_step, dtype=tf.float32)

    return self._lr_min + main_delta - decay_delta

  def info(self) -> str:
    return (f'Cyclic LR Decay. LR Min: {self._lr_min}' +
            f', LR Max: {self._lr_max}' + f', LR Post-Decay: {self._lr_final}' +
            f', Cycle Len: {self._main_cycle_len}, LR_Delta: {self._lr_delta}' +
            f', LR_Decay_Delta: {self._lr_decay_delta}')
