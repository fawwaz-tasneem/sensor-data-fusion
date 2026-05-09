"""
Measurement: a single observation produced by a sensor at a point in time.

Carries its own R (measurement noise covariance) so that a filter doesn't
have to look it up from the sensor — useful when a track is updated with
measurements from heterogeneous sensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    value: np.ndarray  # shape (m,) — the actual measured vector
    timestamp: float
    sensor_id: str
    R: np.ndarray  # shape (m, m) — measurement noise covariance

    def __post_init__(self) -> None:
        if self.value.ndim != 1:
            raise ValueError(f"value must be 1D, got shape {self.value.shape}")
        m = self.value.shape[0]
        if self.R.shape != (m, m):
            raise ValueError(
                f"R shape {self.R.shape} does not match value dim {m}"
            )

    @property
    def dim(self) -> int:
        return self.value.shape[0]
