"""
Trajectory: the ground truth function t -> state vector for a target.

A Trajectory does NOT carry process noise; it represents what actually
happens in the world. Filters never see Trajectory directly; they only
see what sensors report.

Implementations should be pure: state_at(t) must be deterministic and
side-effect free, so the same scenario can be replayed and compared
across filters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sdf.core.state import StateLayout


class Trajectory(ABC):
    """Ground truth state as a function of time."""

    layout: StateLayout

    @abstractmethod
    def state_at(self, t: float) -> np.ndarray:
        """
        Return the true state vector at time t.

        The returned vector must be consistent with `self.layout` —
        i.e., layout.position(state_at(t)) must give the true 2D/3D
        position of the target.
        """
