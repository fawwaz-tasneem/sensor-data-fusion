"""
A Trajectory is a function: time -> true state vector.

For the minimal example we provide a constant-velocity trajectory. Later
we'll add piecewise trajectories (CV → CT → CV) for IMM scenarios.
"""
#TODO: Separate the abstract base class from its implementations
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sdf.core.state import StateLayout


class Trajectory(ABC):
    layout: StateLayout

    @abstractmethod
    def state_at(self, t: float) -> np.ndarray:
        """Return the true state vector at time t."""


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
