"""
RadarSensor: a stationary sensor that measures range and bearing(s) to a
target's position relative to the sensor.

Measurement vectors:
  2D: (range, bearing)               — bearing is azimuth in the xy-plane
  3D: (range, azimuth, elevation)    — azimuth in xy-plane; elevation off it

Conventions:
  range r       = || target_position - sensor_position ||_2
  azimuth phi   = atan2(dy, dx)        in [-pi, pi]
  elevation th  = atan2(dz, sqrt(dx^2 + dy^2))   in [-pi/2, pi/2]

The Jacobian H = dh/dx is sparse: it has nonzero columns only at the
target's position indices. We compute it via the standard polar
coordinate derivatives — see the comments inside H() for the full
expressions and a derivation reference.

Bearing wrap-around: at small ranges, atan2 changes rapidly with small
position changes. Bearings near pi can flip sign across timesteps. We
override innovation() to wrap angle differences into [-pi, pi]; without
this, the EKF update produces wild swings whenever the target crosses
the +pi/-pi line.

References:
  Bar-Shalom, "Estimation with Applications to Tracking and Navigation",
  Section 1.5.4 (polar measurements).
  Koch, "Tracking and Sensor Data Fusion", Section 3.4.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateLayout
from sdf.sensors.base import Sensor
from sdf.sensors.occlusion import OcclusionModel


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into [-pi, pi]. Works for scalars and arrays."""
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


class RadarSensor(Sensor):
    """Range/bearing(/elevation) sensor at a fixed position."""

    def __init__(
        self,
        sensor_id: str,
        position: np.ndarray,
        range_std: float = 10.0,
        bearing_std: float = 1e-2,  # ~0.6 degrees
        elevation_std: Optional[float] = None,  # required in 3D
        detection_prob: float = 1.0,
        occlusion_model: Optional[OcclusionModel] = None,
    ):
        position = np.asarray(position, dtype=float)
        if position.shape not in ((2,), (3,)):
            raise ValueError(
                f"position must be a 2- or 3-vector, got shape {position.shape}"
            )
        self._dim = position.shape[0]
        if self._dim == 3 and elevation_std is None:
            raise ValueError("elevation_std must be provided for a 3D radar")

        self.sensor_id = sensor_id
        self.position = position
        self.detection_prob = detection_prob
        self.occlusion_model = occlusion_model

        # Build R from the per-component stds. We store stds in case
        # downstream code (e.g., visualization or NEES tests) wants to
        # report them.
        self.range_std = range_std
        self.bearing_std = bearing_std
        self.elevation_std = elevation_std

        if self._dim == 2:
            self.R = np.diag([range_std**2, bearing_std**2])
        else:
            self.R = np.diag([range_std**2, bearing_std**2, elevation_std**2])

    @property
    def measurement_dim(self) -> int:
        return self._dim

    # ----- Measurement function and Jacobian ---------------------------

    def h(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        """Noiseless measurement: (range, bearing) or (range, az, el)."""
        if layout.dim != self._dim:
            raise ValueError(
                f"Sensor is {self._dim}D but layout is {layout.dim}D"
            )
        d = layout.position(x) - self.position  # delta in world coords
        r = np.linalg.norm(d)
        # Guard against the degenerate "target at sensor" case. The
        # measurement function is undefined there; in practice this only
        # happens during testing or with a buggy initial state.
        if r < 1e-9:
            raise ValueError(
                f"Target essentially at sensor position (r={r:.2e}); "
                "measurement is undefined"
            )
        if self._dim == 2:
            azimuth = np.arctan2(d[1], d[0])
            return np.array([r, azimuth])
        # 3D
        azimuth = np.arctan2(d[1], d[0])
        ground_range = np.hypot(d[0], d[1])
        elevation = np.arctan2(d[2], ground_range)
        return np.array([r, azimuth, elevation])

    def H(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        """
        Jacobian of h at x. Shape (m, n) where n = state_dim, m = measurement_dim.

        Only columns at layout.position_idx are nonzero — h depends only on
        position, not velocity. We derive the position-block of the Jacobian
        analytically and scatter it into the correct columns.

        2D derivation (let d = [dx, dy], r = sqrt(dx^2 + dy^2)):
            d(r)/d(dx)        =  dx / r
            d(r)/d(dy)        =  dy / r
            d(phi)/d(dx)      = -dy / r^2
            d(phi)/d(dy)      =  dx / r^2

        3D adds elevation th = atan2(dz, rg) where rg = sqrt(dx^2+dy^2):
            d(r)/d(dz)        =  dz / r
            d(phi)/d(dz)      =  0
            d(th)/d(dx)       = -dx*dz / (r^2 * rg)
            d(th)/d(dy)       = -dy*dz / (r^2 * rg)
            d(th)/d(dz)       =  rg   /  r^2
        """
        d = layout.position(x) - self.position
        r = np.linalg.norm(d)
        if r < 1e-9:
            raise ValueError(
                f"Target essentially at sensor (r={r:.2e}); Jacobian undefined"
            )

        n = x.shape[0]

        if self._dim == 2:
            dx, dy = d[0], d[1]
            r2 = r * r
            # Position-block: rows = (range, bearing), cols = (px, py).
            block = np.array(
                [
                    [dx / r, dy / r],
                    [-dy / r2, dx / r2],
                ]
            )
            H = np.zeros((2, n))
            for i, idx in enumerate(layout.position_idx):
                H[:, idx] = block[:, i]
            return H

        # 3D
        dx, dy, dz = d[0], d[1], d[2]
        r2 = r * r
        rg = np.hypot(dx, dy)
        # Same guard as above: a target directly above/below the sensor
        # has rg = 0 and elevation = +/- pi/2, but the Jacobian's
        # elevation rows divide by rg.
        if rg < 1e-9:
            raise ValueError(
                "Target directly above/below sensor; elevation Jacobian undefined"
            )
        # Position-block: rows = (range, az, el), cols = (px, py, pz).
        block = np.array(
            [
                [dx / r, dy / r, dz / r],
                [-dy / (rg * rg), dx / (rg * rg), 0.0],
                [-dx * dz / (r2 * rg), -dy * dz / (r2 * rg), rg / r2],
            ]
        )
        H = np.zeros((3, n))
        for i, idx in enumerate(layout.position_idx):
            H[:, idx] = block[:, i]
        return H

    # ----- Angle-aware innovation --------------------------------------

    def innovation(
        self, measurement_value: np.ndarray, predicted_measurement: np.ndarray
    ) -> np.ndarray:
        """Wrap bearing (and elevation) differences into [-pi, pi]."""
        diff = measurement_value - predicted_measurement
        # Range component is just a difference; bearing/elevation must wrap.
        if self._dim == 2:
            diff[1] = wrap_to_pi(diff[1])
        else:
            diff[1] = wrap_to_pi(diff[1])
            diff[2] = wrap_to_pi(diff[2])
        return diff

    # ----- Override measure to wrap noisy angle into valid range -------

    def measure(
        self,
        true_state: np.ndarray,
        layout: StateLayout,
        t: float,
        rng: np.random.Generator,
    ) -> Optional[Measurement]:
        if not self.is_detected(true_state, layout, rng):
            return None
        z_true = self.h(true_state, layout)
        noise = rng.multivariate_normal(np.zeros(self.measurement_dim), self.R)
        z = z_true + noise
        # Wrap angular components after adding noise. Range does not wrap.
        # Note: a negative range from extreme noise is unphysical but
        # would still produce a valid update; we intentionally leave it
        # alone rather than clip, so that downstream tests can detect
        # an unrealistically high range_std.
        if self._dim == 2:
            z[1] = wrap_to_pi(z[1])
        else:
            z[1] = wrap_to_pi(z[1])
            z[2] = wrap_to_pi(z[2])
        return Measurement(
            value=z,
            timestamp=t,
            sensor_id=self.sensor_id,
            R=self.R.copy(),
        )
