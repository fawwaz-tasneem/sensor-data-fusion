"""
GMTIRadarSensor: Ground Moving Target Indicator radar.

Extends RadarSensor with a range-rate (Doppler) measurement, which is
what distinguishes GMTI from generic radar.

Measurement vectors:
  2D: (range, bearing, range_rate)
  3D: (range, azimuth, elevation, range_rate)

Range-rate is the radial component of the target's velocity *relative to
the sensor*:
    dot{r} = u_hat . (v_target - v_sensor)
where u_hat is the unit line-of-sight from sensor to target. For a
stationary sensor v_sensor = 0 and this collapses to u_hat . v_target.

Because range-rate depends on velocity, the Jacobian H has nonzero
columns at velocity indices in addition to position indices. The
sensor velocity is treated as a *known* parameter (not part of the
state vector), so it does not enter the Jacobian.

Moving sensors:
  Pass a Platform via the `platform` argument. The sensor's
  set_time(t) method updates self.position and self.velocity from the
  platform; if a DopplerBlindnessOcclusion is attached, its
  sensor_position and sensor_velocity are also synced. Stationary
  sensors keep self.velocity at zero.

State-dependent detection:
  Combine with DopplerBlindnessOcclusion (see sdf.sensors.doppler_occlusion)
  to model the GMTI clutter notch. Targets whose world-frame radial
  velocity is below the MDV are likely undetected.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateLayout
from sdf.scenarios.platform import Platform
from sdf.sensors.doppler_occlusion import DopplerBlindnessOcclusion
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
        platform: Optional[Platform] = None,
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
        self.velocity = np.zeros(self._dim)  # zero unless platform updates it
        self.detection_prob = detection_prob
        self.occlusion_model = occlusion_model
        self.platform = platform

        self.range_std = range_std
        self.bearing_std = bearing_std
        self.elevation_std = elevation_std
        self.range_rate_std = range_rate_std

        if self._dim == 2:
            self.R = np.diag([range_std**2, bearing_std**2, range_rate_std**2])
        else:
            assert elevation_std is not None
            self.R = np.diag(
                [range_std**2, bearing_std**2, elevation_std**2, range_rate_std**2]
            )

        # If a platform was provided, initialize position/velocity from it
        # at t=0. (Subsequent set_time calls will keep these in sync.)
        if self.platform is not None:
            if self.platform.dim != self._dim:
                raise ValueError(
                    f"platform dim {self.platform.dim} != sensor dim {self._dim}"
                )
            self.set_time(0.0)

    def set_time(self, t: float) -> None:
        """Update sensor position and velocity from the platform at time t."""
        if self.platform is None:
            return
        self.position = self.platform.position_at(t)
        self.velocity = self.platform.velocity_at(t)
        # Keep DopplerBlindnessOcclusion in sync.
        if isinstance(self.occlusion_model, DopplerBlindnessOcclusion):
            self.occlusion_model.sensor_position = self.position
            self.occlusion_model.sensor_velocity = self.velocity

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

        For a moving sensor, dot{r} = u_hat . (v_target - v_sensor). For a
        stationary sensor v_sensor = 0 and this collapses to u_hat . v_target.
        """
        if layout.dim != self._dim:
            raise ValueError(
                f"GMTI is {self._dim}D but layout is {layout.dim}D"
            )

        position = layout.position(x)
        velocity = layout.velocity(x)
        d = position - self.position
        r = np.linalg.norm(d)
        if r < 1e-9:
            raise ValueError(
                f"Target essentially at sensor (r={r:.2e}); GMTI undefined"
            )
        u_hat = d / r
        # Range-rate of the target relative to the sensor.
        relative_velocity = velocity - self.velocity
        range_rate = float(u_hat @ relative_velocity)

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
        new row that depends on BOTH position AND velocity. Sensor velocity
        is treated as a known parameter (not part of state x), so it does
        not appear in the Jacobian directly — but it does shift the
        effective relative velocity used in the position-derivative.

        For range-rate dot{r} = u_hat . (v_target - v_sensor) with u_hat = d/r:
          d(dot{r}) / d(p_target) = ((v_t - v_s) - dot{r} * u_hat) / r
          d(dot{r}) / d(v_target) = u_hat
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
        H_radar = super().H(x, layout)

        # Build the range-rate row using the relative velocity.
        u_hat = d / r
        relative_velocity = velocity - self.velocity
        range_rate = float(u_hat @ relative_velocity)
        rr_pos_block = (relative_velocity - range_rate * u_hat) / r
        rr_vel_block = u_hat

        rr_row = np.zeros(n_state)
        for i, idx in enumerate(layout.position_idx):
            rr_row[idx] = rr_pos_block[i]
        for i, idx in enumerate(layout.velocity_idx):
            rr_row[idx] = rr_vel_block[i]

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
        # Sync platform pose (position + velocity, and any Doppler occlusion)
        # to time t before measuring. Without this a GMTI on a moving platform
        # stays frozen at its t=0 pose and range-rate is computed wrong.
        self.set_time(t)
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
