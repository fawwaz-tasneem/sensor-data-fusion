"""
Core state representation for the tracking framework.

A StateDistribution is a Gaussian estimate (mean + covariance) at a specific time,
annotated with a StateLayout that tells the rest of the system which indices
correspond to position, velocity, etc. This indirection is what lets a single
sensor implementation work with any motion model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StateLayout:
    """
    Describes the semantic meaning of indices in a state vector.

    Motion models own a StateLayout and expose it. Sensors and visualization
    code use it to extract semantically meaningful slices (position, velocity)
    without hardcoding indices.

    Example for a 2D constant-velocity model with state [x, vx, y, vy]:
        StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

    Example for a 3D constant-velocity model [x, vx, y, vy, z, vz]:
        StateLayout(dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5))
    """

    dim: int  # spatial dimensionality: 2 or 3
    position_idx: tuple[int, ...]
    velocity_idx: tuple[int, ...]
    accel_idx: Optional[tuple[int, ...]] = None
    turn_rate_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.dim}")
        if len(self.position_idx) != self.dim:
            raise ValueError(
                f"position_idx must have {self.dim} entries, got {len(self.position_idx)}"
            )
        if len(self.velocity_idx) != self.dim:
            raise ValueError(
                f"velocity_idx must have {self.dim} entries, got {len(self.velocity_idx)}"
            )

    def position(self, x: np.ndarray) -> np.ndarray:
        """Extract position vector from a state vector."""
        return x[list(self.position_idx)]

    def velocity(self, x: np.ndarray) -> np.ndarray:
        """Extract velocity vector from a state vector."""
        return x[list(self.velocity_idx)]


@dataclass
class StateDistribution:
    """
    Gaussian state estimate: mean + covariance + timestamp + layout.

    This is the universal currency between filters, motion models, and sensors.
    Every filter takes a StateDistribution in and returns one out. The layout
    travels with the state so downstream code never has to guess.
    """

    mean: np.ndarray  # shape (n,)
    covariance: np.ndarray  # shape (n, n)
    timestamp: float
    layout: StateLayout

    def __post_init__(self) -> None:
        # Validate shapes upfront — a corrupt covariance discovered 200 lines
        # into a filter update is an awful debugging experience.
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got shape {self.mean.shape}")
        n = self.mean.shape[0]
        if self.covariance.shape != (n, n):
            raise ValueError(
                f"covariance shape {self.covariance.shape} does not match mean dim {n}"
            )

    @property
    def dim(self) -> int:
        """Dimension of the state vector."""
        return self.mean.shape[0]

    def position(self) -> np.ndarray:
        return self.layout.position(self.mean)

    def velocity(self) -> np.ndarray:
        return self.layout.velocity(self.mean)

    def copy(self) -> "StateDistribution":
        return StateDistribution(
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            timestamp=self.timestamp,
            layout=self.layout,  # frozen dataclass, safe to share
        )
