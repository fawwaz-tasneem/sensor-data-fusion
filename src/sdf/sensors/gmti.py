"""
GMTIRadarSensor: Ground Moving Target Indicator radar.

Extends RadarSensor with a range-rate (Doppler) measurement, which is
what distinguishes GMTI from generic radar (Koch, Sec. 7.2 / Lecture 6,
pages 4-8).

Measurement vectors:
  2D: (range, bearing, range_rate)
  3D: (range, azimuth, elevation, range_rate)

Range-rate is the radial component of the target's velocity relative to
the sensor:
    dot{r} = (p - p_sensor) . v / r = u_hat . v
where u_hat is the unit line-of-sight from sensor to target.

Because range-rate depends on velocity, the Jacobian H now has nonzero
columns at velocity indices in addition to position indices.

Also implements a state-dependent detection probability through the
DopplerBlindnessOcclusion model (see sdf.sensors.doppler_occlusion):
targets with |range-rate| below the sensor's MDV (Minimum Detectable
Velocity) are likely undetected, modelling the GMTI clutter notch.
This is critical for the "stopping target" scenario from the lecture.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateLayout
from sdf.sensors.occlusion import OcclusionModel
from sdf.sensors.radar import RadarSensor, wrap_to_pi


class GMTIRadarSensor(RadarSensor):
    """Range / bearing / (elevation) / range-rate sensor."""

    def __init__(
        self,
        sensor_id: str,
        position: np.ndarray,
        range_std: float = 10.0,
        bearing_std: float = 1e-2,
        elevation_std: Optional[float] = None,
        range_rate_std: float = 0.5,
        detection_prob: float = 1.0,
        occlusion_model: Optional[OcclusionModel] = None,
    ):
        # We deliberately *don't* call RadarSensor.__init__ for R because
        # GMTI's R is one row/col larger. We do everything else manually
        # to avoid a fragile partial-init from the parent.
        position = np.asarray(position, dtype=float)
        if position.shape not in ((2,), (3,)):
            raise ValueError(
                f"position must be a 2- or 3-vector, got shape {position.shape}"
            )
        self._dim = position.shape[0]
        if self._dim == 3 and elevation_std is None:
            raise ValueError("elevation_std must be provided for a 3D GMTI radar")

        self.sensor_id = sensor_id
        self.position = position
        self.detection_prob = detection_prob
        self.occlusion_model = occlusion_model

        self.range_std = range_std
        self.bearing_std = bearing_std
        self.elevation_std = elevation_std
        self.range_rate_std = range_rate_std

        if self._dim == 2:
            self.R = np.diag([range_std**2, bearing_std**2, range_rate_std**2])
        else:
            assert elevation_std is not None  # narrowed by guard above
            self.R = np.diag(
                [range_std**2, bearing_std**2, elevation_std**2, range_rate_std**2]
            )

    @property
    def measurement_dim(self) -> int:
        # range, bearing(s), range_rate
        return self._dim + 1

    # ----- Measurement function and Jacobian ---------------------------

    def h(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        """
        Returns:
          2D: (r, az, dot{r})
          3D: (r, az, el, dot{r})
        """
        if layout.dim != self._dim:
            raise ValueError(
                f"GMTI is {self._dim}D but layout is {layout.dim}D"
            )

        # Reuse RadarSensor.h logic by computing position-only measurement,
        # then append range-rate.
        position = layout.position(x)
        velocity = layout.velocity(x)
        d = position - self.position
        r = np.linalg.norm(d)
        if r < 1e-9:
            raise ValueError(
                f"Target essentially at sensor (r={r:.2e}); GMTI undefined"
            )
        u_hat = d / r
        range_rate = float(u_hat @ velocity)

        if self._dim == 2:
            azimuth = np.arctan2(d[1], d[0])
            return np.array([r, azimuth, range_rate])
        # 3D
        azimuth = np.arctan2(d[1], d[0])
        ground_range = np.hypot(d[0], d[1])
        elevation = np.arctan2(d[2], ground_range)
        return np.array([r, azimuth, elevation, range_rate])

    def H(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        """
        Jacobian. Position-block is identical to RadarSensor's H (range,
        bearing[, elevation] only depend on position). Range-rate adds a
        new row that depends on BOTH position AND velocity.

        For range-rate dot{r} = u_hat . v with u_hat = d/r:
          d(dot{r}) / d(p) = (v - dot{r} * u_hat) / r  (a row vector of length dim)
          d(dot{r}) / d(v) = u_hat                     (a row vector of length dim)
        """
        position = layout.position(x)
        velocity = layout.velocity(x)
        d = position - self.position
        r = np.linalg.norm(d)
        if r < 1e-9:
            raise ValueError(
                f"Target essentially at sensor (r={r:.2e}); GMTI Jacobian undefined"
            )

        n_state = x.shape[0]

        # Reuse the radar position-block by calling parent's H. The base
        # class returns shape (dim, n_state) with only position columns set.
        H_radar = super().H(x, layout)  # shape (dim, n_state)

        # Build the range-rate row.
        u_hat = d / r
        range_rate = float(u_hat @ velocity)
        # Position-block of the range-rate row: (v - dot{r} * u_hat) / r
        rr_pos_block = (velocity - range_rate * u_hat) / r  # length dim
        # Velocity-block of the range-rate row: u_hat
        rr_vel_block = u_hat  # length dim

        rr_row = np.zeros(n_state)
        for i, idx in enumerate(layout.position_idx):
            rr_row[idx] = rr_pos_block[i]
        for i, idx in enumerate(layout.velocity_idx):
            rr_row[idx] = rr_vel_block[i]

        # Stack: radar rows on top, range-rate row at the bottom.
        H = np.vstack([H_radar, rr_row[np.newaxis, :]])
        return H

    # ----- Innovation: only angles wrap; range-rate doesn't -----------

    def innovation(
        self, measurement_value: np.ndarray, predicted_measurement: np.ndarray
    ) -> np.ndarray:
        diff = measurement_value - predicted_measurement
        if self._dim == 2:
            # (range, bearing, range_rate) -> wrap index 1 only
            diff[1] = wrap_to_pi(diff[1])
        else:
            # (range, az, el, range_rate) -> wrap indices 1, 2 only
            diff[1] = wrap_to_pi(diff[1])
            diff[2] = wrap_to_pi(diff[2])
        return diff

    # ----- measure(): wrap angle components after adding noise --------

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
        if self._dim == 2:
            z[1] = wrap_to_pi(z[1])
        else:
            z[1] = wrap_to_pi(z[1])
            z[2] = wrap_to_pi(z[2])
        return Measurement(
            value=z, timestamp=t, sensor_id=self.sensor_id, R=self.R.copy()
        )
