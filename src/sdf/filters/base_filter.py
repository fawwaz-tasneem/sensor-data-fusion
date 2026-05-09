"""
Filter: estimates a target's state over time, given a motion model and
incoming measurements.

The base class provides predict/update/step. step() handles the common
case where you may or may not have a measurement at this timestep
(e.g., due to occlusion or a missed detection): predict to t, then
update only if a measurement was provided. This gives "track coasting"
through gaps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from sdf.core.measurement import Measurement
from sdf.core.state import StateDistribution
from sdf.motion_models.base import MotionModel
from sdf.sensors.base import Sensor


class Filter(ABC):
    motion_model: MotionModel
    state: StateDistribution

    @abstractmethod
    def predict(self, t: float) -> StateDistribution: ...

    @abstractmethod
    def update(self, measurement: Measurement, sensor: Sensor) -> StateDistribution: ...

    def step(
        self, t: float, measurement: Optional[Measurement], sensor: Optional[Sensor]
    ) -> StateDistribution:
        """
        Convenience: predict to time t, then update if a measurement is given.
        Returns the resulting StateDistribution.
        """
        self.predict(t)
        if measurement is not None:
            if sensor is None:
                raise ValueError("sensor must be provided when measurement is not None")
            self.update(measurement, sensor)
        return self.state
