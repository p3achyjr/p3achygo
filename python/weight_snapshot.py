"""
A mechanism to snapshot weights.
"""

from __future__ import annotations

from typing import Any, List

import train_shim


class WeightSnapshotManager(object):

    def __init__(self, ss_steps: List[int]):
        self.ss_steps = ss_steps
        self.snapshots = []

    def should_take_snapshot(self, step: int):
        return step in self.ss_steps

    def take_snapshot(self, model: Any):
        self.snapshots.append(train_shim.get_weights(model))
