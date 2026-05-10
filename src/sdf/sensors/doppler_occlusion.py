"""
DopplerBlindnessOcclusion: state-dependent detection probability for GMTI.

GMTI radar suppresses returns whose Doppler shift falls within the
"clutter notch" centered on the main-lobe-clutter range-rate. For a
stationary sensor and stationary ground clutter, the clutter notch
is centered at zero range-rate; for a moving sensor, the clutter
notch is centered at v_sensor . u_LOS — which depends on the LOS to
each target. So in general dot{r}_mlc is a function of the target
position.

The lecture (page 7) gives the approximation:

    P_D(r, phi, dot{r}) approx P_d * [1 - 2*pi*mdv * N(0; h_n(x_k), mdv^2)]

where:
    h_n(x_k) = dot{r}_k - dot{r}_mlc(x_k)
    mdv      = Minimum Detectable Velocity (sensor parameter, m/s)

When |dot{r}| << mdv, the gaussian factor is at its peak (1/(sqrt(2*pi)*mdv))
and the bracket evaluates to ~ 1 - sqrt(2*pi). For mdv finite this can
go negative, which the textbook handles in the full mixture formulation;
for our single-hypothesis tracker we clip to a P_D floor so that
detection probability stays in [pd_floor, 1].

We model this as occlusion: the probability of being "occluded" (no
detection) is 1 - P_D / P_d (so that the sensor's own detection_prob
still applies on top, e.g., if you want a 0.95 baseline).

For moving GMTI sensors, sensor_position and sensor_velocity must be
updated externally each timestep (typically by the GMTI sensor's
set_time() method). Both attributes are mutable so the moving sensor
can keep them in sync with its platform.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.state import StateLayout
from sdf.sensors.occlusion import OcclusionModel


class DopplerBlindnessOcclusion(OcclusionModel):
    """
    Probabilistic occlusion model for GMTI Doppler blindness.

    For a stationary sensor, sensor_velocity stays at zero and the clutter
    notch sits at dot{r} = 0. For a moving sensor, set sensor_velocity to
    the platform's velocity each scan; the clutter notch then shifts
    along the LOS and pure ground clutter at relative velocity zero
    appears at v_sensor . u_LOS in Doppler.
    """

    def __init__(
        self,
        sensor_position: np.ndarray,
        mdv: float,
        sensor_velocity: Optional[np.ndarray] = None,
        pd_floor: float = 0.05,
    ):
        """
        Parameters
        ----------
        sensor_position : (dim,) array
            Position of the GMTI sensor at the current scan.
        mdv : float
            Minimum Detectable Velocity (m/s).
        sensor_velocity : (dim,) array, optional
            Velocity of the GMTI sensor at the current scan. Default
            zero (stationary sensor).
        pd_floor : float
            Minimum detection probability inside the clutter notch
            (clip floor for the Koch approximation).
        """
        self.sensor_position = np.asarray(sensor_position, dtype=float)
        if self.sensor_position.ndim != 1 or self.sensor_position.shape[0] not in (2, 3):
            raise ValueError("sensor_position must be a 2- or 3-vector")
        if mdv <= 0:
            raise ValueError(f"mdv must be positive, got {mdv}")
        self.mdv = mdv
        if sensor_velocity is None:
            self.sensor_velocity = np.zeros(self.sensor_position.shape[0])
        else:
            self.sensor_velocity = np.asarray(sensor_velocity, dtype=float)
            if self.sensor_velocity.shape != self.sensor_position.shape:
                raise ValueError(
                    "sensor_velocity and sensor_position must have same shape"
                )
        self.pd_floor = pd_floor

    def detection_factor(
        self, target_state: np.ndarray, layout: StateLayout
    ) -> float:
        """
        Multiplier on the sensor's baseline P_d, in [pd_floor, 1].

        Returns the *factor* that downscales detection probability when
        the target is in (or near) the clutter notch. Public so it can
        be used by tests, visualization, and likelihood functions.
        """
        position = layout.position(target_state)
        velocity = layout.velocity(target_state)
        d = position - self.sensor_position
        r = np.linalg.norm(d)
        if r < 1e-9:
            return 1.0
        u_hat = d / r

        # Target range-rate: relative velocity along LOS.
        #   dot{r}_target = u_hat . (v_target - v_sensor)
        # But the lecture's range-rate measurement is u_hat . v_target
        # (target velocity along LOS, with sensor stationary). For a
        # moving sensor, the *measured* range-rate is u_hat . (v_target
        # - v_sensor); the clutter (stationary ground) appears at
        # u_hat . (0 - v_sensor) = -u_hat . v_sensor.
        # h_n is the difference between measured target range-rate and
        # the clutter range-rate, both in the sensor's frame:
        #   h_n = u_hat . (v_target - v_sensor) - (- u_hat . v_sensor)
        #       = u_hat . v_target
        # So h_n actually equals the target's velocity along LOS in the
        # *world* frame, regardless of sensor motion. This is the
        # physical content of the clutter notch: a target whose world-
        # frame radial velocity is zero (stationary or purely tangential)
        # is indistinguishable from clutter, regardless of where the
        # sensor is or how fast it's moving.
        h_n = float(u_hat @ velocity)

        gaussian = np.exp(-(h_n**2) / (2 * self.mdv**2)) / (
            np.sqrt(2 * np.pi) * self.mdv
        )
        factor = 1.0 - 2.0 * np.pi * self.mdv * gaussian
        return float(np.clip(factor, self.pd_floor, 1.0))

    def is_occluded(
        self,
        target_state: np.ndarray,
        layout: StateLayout,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        """
        Probabilistic occlusion: with probability (1 - factor) the target
        is occluded (i.e., not detected this scan).
        """
        factor = self.detection_factor(target_state, layout)
        if rng is None:
            return factor < 0.5
        return rng.random() > factor
