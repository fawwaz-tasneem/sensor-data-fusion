"""
DopplerBlindnessOcclusion: state-dependent detection probability for GMTI.

GMTI radar suppresses returns whose Doppler shift falls within the
"clutter notch" centered on the main-lobe-clutter range-rate. For a
stationary clutter field on flat ground (and a stationary sensor),
the clutter notch is centered at zero range-rate; in general it's
centered at dot{r}_mlc(x_k) determined by sensor/platform motion.

Koch's lecture (page 7) gives the approximation:

    P_D(r, phi, dot{r}) approx P_d * [1 - 2*pi*mdv * N(0; h_n(x_k), mdv^2)]

where:
    h_n(x_k) = dot{r}_k - dot{r}_mlc(x_k)
    mdv      = Minimum Detectable Velocity (sensor parameter, m/s)

For our (stationary) sensor with a stationary clutter field:
    dot{r}_mlc = 0,   so   h_n = dot{r}_k

When |dot{r}| << mdv, the gaussian factor is at its peak (1/(sqrt(2*pi)*mdv))
and the bracket evaluates to ~ 1 - sqrt(2*pi). For mdv finite this can
go negative, which Koch handles in the full mixture formulation; for our
single-hypothesis tracker we clip to a P_D floor so that detection
probability stays in [0, P_d].

We model this as occlusion: the probability of being "occluded" (no
detection) is 1 - P_D / P_d (so that the sensor's own detection_prob
still applies on top, e.g., if you want a 0.95 baseline).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.state import StateLayout
from sdf.sensors.occlusion import OcclusionModel


class DopplerBlindnessOcclusion(OcclusionModel):
    """Probabilistic occlusion model for GMTI Doppler blindness."""

    def __init__(
        self,
        sensor_position: np.ndarray,
        mdv: float,
        clutter_range_rate: float = 0.0,
        pd_floor: float = 0.05,
    ):
        """
        Parameters
        ----------
        sensor_position : (dim,) array
            Position of the GMTI sensor.
        mdv : float
            Minimum Detectable Velocity (m/s). Targets whose range-rate
            relative to clutter is much less than this are likely missed.
        clutter_range_rate : float
            Range-rate of the main-lobe clutter (zero for stationary sensor
            and stationary ground clutter; nonzero for airborne platforms).
        pd_floor : float
            Minimum detection probability inside the clutter notch.
            Prevents P_D from going to zero or negative due to the
            approximation; physically there is always some chance of
            detection (e.g., target SNR fluctuations).
        """
        self.sensor_position = np.asarray(sensor_position, dtype=float)
        if self.sensor_position.ndim != 1 or self.sensor_position.shape[0] not in (2, 3):
            raise ValueError("sensor_position must be a 2- or 3-vector")
        if mdv <= 0:
            raise ValueError(f"mdv must be positive, got {mdv}")
        self.mdv = mdv
        self.clutter_range_rate = clutter_range_rate
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
            # Degenerate; treat as detected.
            return 1.0
        u_hat = d / r
        range_rate = float(u_hat @ velocity)
        h_n = range_rate - self.clutter_range_rate

        # Koch's approximation: P_D / P_d = 1 - 2*pi*mdv*N(0; h_n, mdv^2).
        # N(0; h_n, mdv^2) = exp(-h_n^2 / (2*mdv^2)) / (sqrt(2*pi)*mdv)
        gaussian = np.exp(-(h_n**2) / (2 * self.mdv**2)) / (
            np.sqrt(2 * np.pi) * self.mdv
        )
        factor = 1.0 - 2.0 * np.pi * self.mdv * gaussian
        # Clip to [pd_floor, 1].
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
        # Probability of occlusion is 1 - factor.
        if rng is None:
            # Deterministic fallback: occluded iff factor < 0.5.
            return factor < 0.5
        return rng.random() > factor
