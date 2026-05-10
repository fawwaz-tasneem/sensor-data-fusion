"""
Platform: the carrier of a sensor (a tower, a vehicle, an aircraft).

A Platform exposes position and velocity as a function of time. Stationary
platforms return constants; moving platforms (e.g., AWACS aircraft)
follow a parametric trajectory.

Why velocity matters: GMTI radar's clutter notch depends on the radial
component of *the sensor's own velocity*. Stationary clutter on the
ground is at zero range-rate when the sensor is stationary, but appears
at v_sensor . u_LOS when the sensor is moving — so the clutter notch
shifts in Doppler. This is what enables an airborne GMTI to detect
even slow ground targets that would be lost in clutter from a static
radar.

The dim attribute (2 or 3) is the spatial dimension of the platform's
position. A 2D platform's velocity is also 2D, etc.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Platform(ABC):
    dim: int

    @abstractmethod
    def position_at(self, t: float) -> np.ndarray:
        """Position at time t. Shape: (dim,)."""

    @abstractmethod
    def velocity_at(self, t: float) -> np.ndarray:
        """Velocity at time t. Shape: (dim,)."""


class StationaryPlatform(Platform):
    """A non-moving platform. Velocity is identically zero."""

    def __init__(self, position: np.ndarray):
        position = np.asarray(position, dtype=float)
        if position.ndim != 1 or position.shape[0] not in (2, 3):
            raise ValueError(
                f"position must be a 2- or 3-vector, got shape {position.shape}"
            )
        self._position = position
        self.dim = position.shape[0]

    def position_at(self, t: float) -> np.ndarray:
        return self._position.copy()

    def velocity_at(self, t: float) -> np.ndarray:
        return np.zeros(self.dim)
