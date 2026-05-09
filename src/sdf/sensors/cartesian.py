"""
CartesianPositionSensor: a "toy" sensor that directly observes the target's
position in Cartesian coordinates. Used for unit tests and the minimal
end-to-end example because it makes the linear KF case trivially correct.

For real applications, use radar (range/bearing) or GMTI sensors instead.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.state import StateLayout
from sdf.sensors.base import Sensor
from sdf.sensors.occlusion import OcclusionModel


class CartesianPositionSensor(Sensor):
    """
    Linear sensor that observes position directly.

    h(x) = position(x), so H is just a selector matrix that picks out the
    position rows. This makes the KF update step linear and exact.
    """

    def __init__(
        self,
        sensor_id: str,
        dim: int,
        noise_std: float = 1.0,
        detection_prob: float = 1.0,
        occlusion_model: Optional[OcclusionModel] = None,
    ):
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.sensor_id = sensor_id
        self._dim = dim
        # Isotropic Gaussian noise on each axis.
        self.R = (noise_std**2) * np.eye(dim)
        self.detection_prob = detection_prob
        self.occlusion_model = occlusion_model

    @property
    def measurement_dim(self) -> int:
        return self._dim

    def h(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        return layout.position(x)

    def H(self, x: np.ndarray, layout: StateLayout) -> np.ndarray:
        # Selector matrix: row i picks out position component i from x.
        H = np.zeros((self._dim, x.shape[0]))
        for i, idx in enumerate(layout.position_idx):
            H[i, idx] = 1.0
        return H
