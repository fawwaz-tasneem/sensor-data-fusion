"""
Sensor: produces noisy, possibly-missed measurements of a target's state.

The Sensor base class provides:
  - h(x):     measurement function (deterministic, noiseless)
  - H(x):     Jacobian of h at x, used by EKF
  - measure(true_state, t, rng):  full measurement pipeline including
              detection probability, occlusion, and measurement noise.
  - set_time(t):  hook for moving sensors to update their position before
                  any subsequent h, H, or measure calls.

Occlusion is delegated to a pluggable OcclusionModel so that terrain
shadowing, GMTI MDV blind zones, and angular blind sectors can be
composed independently.

Stationary vs moving sensors:
  Stationary sensors keep `self.position` as a constant. Moving sensors
  override set_time(t) to update self.position from a Platform object.
  Filters and the simulation engine should call set_time(t) before any
  sensor query so that h, H, and measure see the correct sensor pose.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateLayout
from sdf.sensors.occlusion import OcclusionModel


class Sensor(ABC):
    sensor_id: str
    R: np.ndarray
    detection_prob: float
    occlusion_model: Optional[OcclusionModel]

    @property
    @abstractmethod
    def measurement_dim(self) -> int: ...

    @abstractmethod
    def h(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        """Measurement function. Returns the noiseless measurement of state x."""

    @abstractmethod
    def H(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        """Jacobian of h at x. Shape: (m, n)."""

    def set_time(self, t: float) -> None:
        """
        Update sensor's pose for time t. Default is no-op (stationary
        sensor). Moving sensors override to update self.position (and
        any other time-varying attributes) from a Platform.
        """
        return

    def innovation(
        self, measurement_value: np.ndarray, predicted_measurement: np.ndarray
    ) -> np.ndarray:
        """
        Compute z - z_pred in measurement space.

        Default is plain subtraction. Sensors with angular components
        (radar bearing, GMTI bearing) override this to wrap angle
        differences into [-pi, pi]. This is critical: a naive subtraction
        across the +pi/-pi branch cut produces a ~6.28 rad innovation when
        the true difference is small, which destroys the EKF update.
        """
        return measurement_value - predicted_measurement

    def is_detected(
        self, true_state: np.ndarray, layout: StateLayout, rng: np.random.Generator
    ) -> bool:
        """Decide whether this measurement is produced (vs. missed/occluded)."""
        if self.occlusion_model is not None and self.occlusion_model.is_occluded(
            true_state, layout, rng
        ):
            return False
        return rng.random() < self.detection_prob

    def measure(
        self,
        true_state: np.ndarray,
        layout: StateLayout,
        t: float,
        rng: np.random.Generator,
    ) -> Optional[Measurement]:
        """
        Generate a measurement of the true state, or None if the target is
        not detected at this timestep.
        """
        # Sync the sensor's pose with the current time before measuring.
        self.set_time(t)
        if not self.is_detected(true_state, layout, rng):
            return None
        z_true = self.h(true_state, layout)
        noise = rng.multivariate_normal(np.zeros(self.measurement_dim), self.R)
        return Measurement(
            value=z_true + noise,
            timestamp=t,
            sensor_id=self.sensor_id,
            R=self.R.copy(),
        )
