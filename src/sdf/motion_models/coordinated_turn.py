"""
Coordinated turn (CT) motion model with known turn rate.

State layout (2D): [x, vx, y, vy]  — dim 4 (same as CV).

Dynamics: the velocity vector rotates at constant angular rate omega in
the (x, y) plane, while position evolves under the rotating velocity.
For omega != 0:

  [x_{k+1} ]   [1   sin(w*dt)/w   0    -(1 - cos(w*dt))/w] [x_k ]
  [vx_{k+1}] = [0   cos(w*dt)     0    -sin(w*dt)        ] [vx_k]
  [y_{k+1} ]   [0   (1-cos(w*dt))/w  1   sin(w*dt)/w    ] [y_k ]
  [vy_{k+1}]   [0   sin(w*dt)     0     cos(w*dt)        ] [vy_k]

For |w * dt| << 1 we use the small-angle limit which collapses to CV
plus second-order corrections; this avoids 0/0 numerical issues.

Although the model is *time-invariant* in omega (constant turn rate)
and *linear* in x, the Jacobian is the same as F (linear). Including F
makes this model usable from the standard EKF without modification.

Process noise is white acceleration intensity q in the (x, y) plane;
in 3D we leave z evolving under CV with the same q. In this 2D
implementation we use the per-axis DWNA covariance block (same as CV).

Reference: Bar-Shalom Sec. 11.7 (constant-turn-rate model).
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateLayout
from sdf.motion_models.base import MotionModel


class CoordinatedTurn(MotionModel):
    """2D coordinated turn with known constant turn rate."""

    def __init__(self, omega: float, process_noise_std: float = 1.0):
        """
        Parameters
        ----------
        omega : float
            Turn rate in rad/s. Positive = counter-clockwise (when viewed
            from +z); negative = clockwise. omega == 0 reduces to CV but
            we use a small-angle expansion to avoid numerical 0/0.
        process_noise_std : float
            Standard deviation of the white acceleration disturbance.
        """
        self._dim = 2
        self.omega = float(omega)
        self.q = process_noise_std**2
        # Use the same CV-shaped layout: [x, vx, y, vy].
        self.layout = StateLayout(
            dim=2, position_idx=(0, 2), velocity_idx=(1, 3)
        )

    def _state_dim(self) -> int:
        return 4

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        w = self.omega
        wdt = w * dt
        # Use the small-angle limit when |wdt| is too small for stable
        # division. The threshold 1e-6 is chosen so that the relative
        # error of the linear approximation is well under 1e-12.
        if abs(wdt) < 1e-6:
            # Series expansion of the exact form to second order in wdt:
            #   sin(wdt)/w     -> dt - (wdt)^2 * dt / 6     ~ dt
            #   (1-cos(wdt))/w -> wdt^2/(2w) = w*dt^2/2     ~ 0
            #   cos(wdt)       -> 1 - (wdt)^2/2             ~ 1
            #   sin(wdt)       -> wdt - (wdt)^3/6           ~ wdt
            return np.array([
                [1.0, dt, 0.0, -0.5 * w * dt**2],
                [0.0, 1.0, 0.0, -wdt],
                [0.0, 0.5 * w * dt**2, 1.0, dt],
                [0.0, wdt, 0.0, 1.0],
            ])
        s = np.sin(wdt)
        c = np.cos(wdt)
        return np.array([
            [1.0, s / w, 0.0, -(1.0 - c) / w],
            [0.0, c, 0.0, -s],
            [0.0, (1.0 - c) / w, 1.0, s / w],
            [0.0, s, 0.0, c],
        ])

    def Q(self, dt: float) -> np.ndarray:
        # Per-axis DWNA covariance, same as CV. Note that this is an
        # approximation in CT context (the exact discretization of
        # process noise under rotating dynamics is more involved, but
        # the per-axis approximation is standard practice when q is
        # small compared to the dynamics).
        block = np.array([
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ]) * self.q
        Q = np.zeros((4, 4))
        Q[0:2, 0:2] = block
        Q[2:4, 2:4] = block
        return Q
