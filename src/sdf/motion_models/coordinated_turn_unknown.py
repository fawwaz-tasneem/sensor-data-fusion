"""
Coordinated turn with unknown turn rate (CT-omega).

State layout: [x, vx, y, vy, omega]   — dim 5

The turn rate omega is now part of the state and is estimated alongside
position and velocity. This is the model that gets used inside an IMM
for "the target is turning, but we don't know how fast".

Dynamics:
    x_{k+1}  = x_k + vx_k * sin(w*dt)/w - vy_k * (1 - cos(w*dt))/w
    vx_{k+1} = vx_k * cos(w*dt) - vy_k * sin(w*dt)
    y_{k+1}  = y_k + vx_k * (1 - cos(w*dt))/w + vy_k * sin(w*dt)/w
    vy_{k+1} = vx_k * sin(w*dt) + vy_k * cos(w*dt)
    w_{k+1}  = w_k                        (constant turn rate model)

This is nonlinear in omega even though it's linear in x, vx, y, vy at
constant omega. The Jacobian therefore has nonzero entries in the
omega column. Process noise on omega itself models slow drift in
turn rate (think: aircraft adjusting bank angle).

Reference: Bar-Shalom Sec. 11.7.4 (CT with random turn rate).
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateDistribution, StateLayout
from sdf.motion_models.base import MotionModel


class CoordinatedTurnUnknown(MotionModel):
    """2D coordinated turn with the turn rate as part of the state."""

    def __init__(
        self,
        process_noise_std: float = 1.0,
        omega_noise_std: float = 0.05,
    ):
        """
        Parameters
        ----------
        process_noise_std : float
            Std-dev of white acceleration disturbance on the (x, y) axes.
        omega_noise_std : float
            Std-dev of the random walk on the turn rate (rad/s per
            unit-time-step). Larger values let omega adjust more freely;
            smaller values pin it to its current estimate.
        """
        self._dim = 2
        self.q = process_noise_std**2
        self.q_omega = omega_noise_std**2
        # Layout includes turn rate as an extra component.
        self.layout = StateLayout(
            dim=2,
            position_idx=(0, 2),
            velocity_idx=(1, 3),
            turn_rate_idx=4,
        )

    def _state_dim(self) -> int:
        return 5

    # The model is nonlinear in omega, so we override f() rather than
    # using the default f = F @ x.
    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        w = float(x[4])
        wdt = w * dt
        x_pos, vx, y_pos, vy = x[0], x[1], x[2], x[3]

        if abs(wdt) < 1e-6:
            # Small-angle expansion (matches CT-known limit at omega -> 0).
            sw_w = dt - (wdt**2) * dt / 6.0      # sin(wdt)/w
            omw_w = w * dt**2 / 2.0               # (1 - cos(wdt))/w
            c = 1.0 - 0.5 * wdt**2
            s = wdt - (wdt**3) / 6.0
        else:
            sw_w = np.sin(wdt) / w
            omw_w = (1.0 - np.cos(wdt)) / w
            c = np.cos(wdt)
            s = np.sin(wdt)

        return np.array([
            x_pos + vx * sw_w - vy * omw_w,
            vx * c - vy * s,
            y_pos + vx * omw_w + vy * sw_w,
            vx * s + vy * c,
            w,
        ])

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        w = float(x[4])
        wdt = w * dt
        vx, vy = x[1], x[3]

        if abs(wdt) < 1e-6:
            # Small-angle limits.
            sw_w = dt - (wdt**2) * dt / 6.0
            omw_w = w * dt**2 / 2.0
            c = 1.0 - 0.5 * wdt**2
            s = wdt - (wdt**3) / 6.0
            # Derivatives of sw_w and omw_w w.r.t. w. Computed from
            # series so the limit at w=0 is well-defined.
            dsw_dw = -dt**3 * w / 3.0     # ~0 at w=0
            domw_dw = dt**2 / 2.0          # exactly dt^2/2 at w=0
        else:
            sw_w = np.sin(wdt) / w
            omw_w = (1.0 - np.cos(wdt)) / w
            c = np.cos(wdt)
            s = np.sin(wdt)
            # d(sin(wdt)/w)/dw = (dt cos(wdt) w - sin(wdt)) / w^2
            #                  = dt * cos(wdt) / w - sin(wdt) / w^2
            dsw_dw = (dt * c) / w - s / (w * w)
            # d((1-cos(wdt))/w)/dw = (dt sin(wdt) w - (1-cos(wdt))) / w^2
            #                      = dt sin(wdt) / w - (1-cos(wdt))/w^2
            domw_dw = (dt * s) / w - (1.0 - c) / (w * w)

        F = np.zeros((5, 5))
        # Row 0: x
        F[0, 0] = 1.0
        F[0, 1] = sw_w
        F[0, 3] = -omw_w
        F[0, 4] = vx * dsw_dw - vy * domw_dw
        # Row 1: vx
        F[1, 1] = c
        F[1, 3] = -s
        F[1, 4] = -vx * dt * s - vy * dt * c
        # Row 2: y
        F[2, 1] = omw_w
        F[2, 2] = 1.0
        F[2, 3] = sw_w
        F[2, 4] = vx * domw_dw + vy * dsw_dw
        # Row 3: vy
        F[3, 1] = s
        F[3, 3] = c
        F[3, 4] = vx * dt * c - vy * dt * s
        # Row 4: omega is constant.
        F[4, 4] = 1.0
        return F

    def Q(self, dt: float) -> np.ndarray:
        # Per-axis DWNA on (x, y) plus a small random-walk noise on omega.
        block = np.array([
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2],
        ]) * self.q
        Q = np.zeros((5, 5))
        Q[0:2, 0:2] = block
        Q[2:4, 2:4] = block
        Q[4, 4] = self.q_omega * dt
        return Q
