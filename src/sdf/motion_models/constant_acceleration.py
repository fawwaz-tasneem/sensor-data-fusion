"""
Constant acceleration (CA) motion model.

State layout (2D): [x, vx, ax, y, vy, ay]            — dim 6
State layout (3D): [x, vx, ax, y, vy, ay, z, vz, az] — dim 9

Each spatial axis evolves under constant acceleration:
    x_{k+1}  = x_k + vx_k * dt + ax_k * dt^2 / 2
    vx_{k+1} = vx_k + ax_k * dt
    ax_{k+1} = ax_k

Process noise is the standard "discrete white noise jerk" (DWNJ) model:
a constant white-noise jerk q within each timestep, giving the per-axis
discrete-time Q block (Bar-Shalom Sec. 6.3 / Koch Sec. 4.1):

           [ dt^5/20   dt^4/8    dt^3/6 ]
    Q_a = q [ dt^4/8    dt^3/3    dt^2/2 ]
           [ dt^3/6    dt^2/2    dt     ]

with q = jerk_std**2.

The model is linear, so f(x, dt) = F(dt) @ x and the EKF prediction
collapses to the standard KF prediction.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateLayout
from sdf.motion_models.base import MotionModel


class ConstantAcceleration(MotionModel):
    """Constant acceleration model in 2D or 3D."""

    def __init__(self, dim: int = 2, jerk_std: float = 1.0):
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self._dim = dim
        # q is the variance of the white acceleration noise.
        self.q = jerk_std**2
        # State layout: each axis has 3 components (pos, vel, acc).
        # 2D: [x, vx, ax, y, vy, ay]      -> pos at (0,3), vel at (1,4), acc at (2,5)
        # 3D: [x, vx, ax, y, vy, ay, z, vz, az]
        if dim == 2:
            self.layout = StateLayout(
                dim=2,
                position_idx=(0, 3),
                velocity_idx=(1, 4),
                accel_idx=(2, 5),
            )
        else:
            self.layout = StateLayout(
                dim=3,
                position_idx=(0, 3, 6),
                velocity_idx=(1, 4, 7),
                accel_idx=(2, 5, 8),
            )

    def _state_dim(self) -> int:
        return 3 * self._dim

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        # Per-axis block:
        #   [[1, dt, dt^2/2],
        #    [0,  1, dt    ],
        #    [0,  0,  1    ]]
        block = np.array([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])
        F = np.zeros((self.state_dim, self.state_dim))
        for i in range(self._dim):
            F[3 * i:3 * i + 3, 3 * i:3 * i + 3] = block
        return F

    def Q(self, dt: float) -> np.ndarray:
        # Per-axis DWNJ covariance block.
        block = np.array([
            [dt**5 / 20.0, dt**4 / 8.0,  dt**3 / 6.0],
            [dt**4 / 8.0,  dt**3 / 3.0,  dt**2 / 2.0],
            [dt**3 / 6.0,  dt**2 / 2.0,  dt],
        ]) * self.q
        Q = np.zeros((self.state_dim, self.state_dim))
        for i in range(self._dim):
            Q[3 * i:3 * i + 3, 3 * i:3 * i + 3] = block
        return Q
