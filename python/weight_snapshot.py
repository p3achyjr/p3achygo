'''
A mechanism to snapshot weights.
'''
from __future__ import annotations

from typing import List

import tensorflow as tf


class WeightSnapshotManager(object):

  def __init__(self, ss_steps: List[int]):
    self.ss_steps = ss_steps
    self.snapshots = []

  def should_take_snapshot(self, step: int):
    return step in self.ss_steps

  def take_snapshot(self, model: tf.keras.Model):
    self.snapshots.append(model.get_weights())
