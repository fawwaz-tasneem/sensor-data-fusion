"""
Constant velocity trajectory: a target moving with fixed velocity.

This is the simplest trajectory and is mainly useful for verification
(KF should achieve essentially optimal performance on it). For more
interesting scenarios see piecewise.py and mountain_pass.py.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateLayout
from sdf.scenarios.base import Trajectory


class ConstantVelocityTrajectory(Trajectory):
    """A target moving with fixed velocity from a given initial state."""

    def __init__(self, initial_state: np.ndarray, layout: StateLayout):
        self.x0 = np.asarray(initial_state, dtype=float)
        self.layout = layout

    def state_at(self, t: float) -> np.ndarray:
        x = self.x0.copy()
        # For each spatial axis, position at time t = position_0 + velocity * t.
        for pos_idx, vel_idx in zip(self.layout.position_idx, self.layout.velocity_idx):
            x[pos_idx] = self.x0[pos_idx] + self.x0[vel_idx] * t
        return x
