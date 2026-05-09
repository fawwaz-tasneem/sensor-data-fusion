"""
MountainPassTrajectory: a 3D analytic trajectory of a vehicle moving along
a winding mountain pass road.

Motion is parametrized by:
  * v_kmh:   forward speed (along x), in km/h.
  * length:  characteristic length scale of the road, in meters.
             (The slow z-oscillation has period 2 * length; the y-
             oscillation has period length / 2 — i.e., 4 lateral wiggles
             per up/down cycle.)
  * y_amp:   amplitude of the lateral (y) sinusoid, in meters.
  * z_amp:   amplitude of the vertical (z) sinusoid, in meters.
  * y_cycles_per_length:  how many y-oscillations occur per `length` along x.
                          Default 4 reproduces the original course where
                          k_y = 4 pi v / length.
  * z_cycles_per_length:  same for z. Default 1, so k_z = pi v / length.

Position equations (matches the user-provided code when defaults are used):
    x(t) = v * t
    y(t) = y_amp * sin(k_y * t),   k_y = (y_cycles_per_length * pi * v) / length
    z(t) = z_amp * sin(k_z * t),   k_z = (z_cycles_per_length * pi * v) / length

Because the motion is analytic, we can return both position AND velocity
exactly. This is important: a filter's `initial_state` should sit on the
true state, otherwise convergence behavior depends on the initial offset
rather than the algorithm being tested.

Note on the choice of CV-shaped state layout: the trajectory itself is
NOT constant velocity — the y and z velocities oscillate. But we still
expose the state in CV layout `[x, vx, y, vy, z, vz]` because (a) all
existing filters in this framework use that layout, and (b) the model
mismatch between truth (sinusoidal) and filter assumption (CV) is
exactly the kind of stress test that motivates EKF and IMM.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateLayout
from sdf.scenarios.base import Trajectory


class MountainPassTrajectory(Trajectory):
    def __init__(
        self,
        v_kmh: float = 20.0,
        length: float = 10_000.0,
        y_amp: float = 1_000.0,
        z_amp: float = 1_000.0,
        y_cycles_per_length: float = 4.0,
        z_cycles_per_length: float = 1.0,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
    ):
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if v_kmh <= 0:
            raise ValueError(f"v_kmh must be positive, got {v_kmh}")

        # Convert km/h to m/s once, up front. Storing the SI value avoids a
        # whole class of unit bugs later — every method below sees only m/s.
        self.v = v_kmh / 3.6

        self.length = length
        self.y_amp = y_amp
        self.z_amp = z_amp
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        # Precompute angular frequencies so state_at is fast and the formulas
        # in state_at read like the math.
        self.ky = (y_cycles_per_length * np.pi * self.v) / length
        self.kz = (z_cycles_per_length * np.pi * self.v) / length

        # 3D CV-shaped layout: state vector [x, vx, y, vy, z, vz].
        self.layout = StateLayout(
            dim=3,
            position_idx=(0, 2, 4),
            velocity_idx=(1, 3, 5),
        )

    # ----- Convenience accessors ---------------------------------------

    def position_at(self, t: float) -> np.ndarray:
        """3-vector of true position at time t."""
        x = self.x0 + self.v * t
        y = self.y0 + self.y_amp * np.sin(self.ky * t)
        z = self.z0 + self.z_amp * np.sin(self.kz * t)
        return np.array([x, y, z])

    def velocity_at(self, t: float) -> np.ndarray:
        """
        3-vector of true velocity at time t — the analytic derivative of
        position. Exposing this lets filters and tests start from the
        true initial state instead of guessing.
        """
        vx = self.v
        vy = self.y_amp * self.ky * np.cos(self.ky * t)
        vz = self.z_amp * self.kz * np.cos(self.kz * t)
        return np.array([vx, vy, vz])

    # ----- Trajectory interface ----------------------------------------

    def state_at(self, t: float) -> np.ndarray:
        """Full 6-vector [x, vx, y, vy, z, vz] at time t."""
        p = self.position_at(t)
        v = self.velocity_at(t)
        return np.array([p[0], v[0], p[1], v[1], p[2], v[2]])
