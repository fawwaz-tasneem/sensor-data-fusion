"""
Constant velocity (CV) motion model.

State layout (2D): [x, vx, y, vy]            — dim 4
State layout (3D): [x, vx, y, vy, z, vz]     — dim 6

Process noise is the standard "discrete white noise acceleration" (DWNA)
model: an acceleration that is constant within a timestep and white
between timesteps, with variance q. This gives the well-known Q matrix
with dt^4/4, dt^3/2, dt^2 entries.

Reference: Bar-Shalom, "Estimation with Applications to Tracking and
Navigation", Section 6.2.2.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateLayout
from sdf.motion_models.base import MotionModel


class ConstantVelocity(MotionModel):
    """Constant velocity model in 2D or 3D."""

    def __init__(self, dim: int = 2, process_noise_std: float = 1.0):
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self._dim = dim
        # process_noise_std is the standard deviation of the white acceleration
        # noise; q = process_noise_std**2 is the variance used in Q.
        self.q = process_noise_std**2
        # In 2D state is [x, vx, y, vy]; in 3D it's [x, vx, y, vy, z, vz].
        # Position lives at indices 0, 2, (4); velocity at 1, 3, (5).
        if dim == 2:
            self.layout = StateLayout(
                dim=2, position_idx=(0, 2), velocity_idx=(1, 3)
            )
        else:
            self.layout = StateLayout(
                dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5)
            )

    def _state_dim(self) -> int:
        return 2 * self._dim

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        # Block-diagonal: each spatial axis evolves independently as
        #   [[1, dt],
        #    [0,  1]]
        block = np.array([[1.0, dt], [0.0, 1.0]])
        F = np.zeros((self.state_dim, self.state_dim))
        for i in range(self._dim):
            F[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
        return F

    def Q(self, dt: float) -> np.ndarray:
        # Per-axis discrete white noise acceleration covariance:
        #   [[dt^4/4, dt^3/2],
        #    [dt^3/2, dt^2  ]] * q
        block = (
            np.array(
                [
                    [dt**4 / 4.0, dt**3 / 2.0],
                    [dt**3 / 2.0, dt**2],
                ]
            )
            * self.q
        )
        Q = np.zeros((self.state_dim, self.state_dim))
        for i in range(self._dim):
            Q[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = block
        return Q
