"""
SimulationEngine: orchestrates a single-target tracking scenario.

At each timestep:
    1. Get true state from the trajectory.
    2. Each sensor produces a measurement (or None if missed/occluded).
    3. The filter steps forward, ingesting any measurements.
    4. Truth, measurements, and estimates are logged for later analysis.

The result is a SimulationResult containing parallel time series of
ground truth states, measurements (with sensor IDs), and filter
estimates. This is the single source of truth for plots and metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateDistribution
from sdf.core.track import Track
from sdf.filters.base import Filter
from sdf.scenarios.trajectory import Trajectory
from sdf.sensors.base import Sensor


@dataclass
class SimulationResult:
    times: np.ndarray  # shape (T,)
    truths: np.ndarray  # shape (T, n) — true full state at each step
    track: Track  # filter estimates over time
    # Measurements per timestep, per sensor. Outer list length T,
    # inner dict maps sensor_id to Measurement (absent if missed).
    measurements: list[dict[str, Measurement]] = field(default_factory=list)


class SimulationEngine:
    def __init__(
        self,
        trajectory: Trajectory,
        sensors: list[Sensor],
        filter: Filter,
        dt: float,
        duration: float,
        seed: Optional[int] = None,
    ):
        self.trajectory = trajectory
        self.sensors = sensors
        self.filter = filter
        self.dt = dt
        self.duration = duration
        self.rng = np.random.default_rng(seed)

    def run(self) -> SimulationResult:
        n_steps = int(round(self.duration / self.dt)) + 1
        times = np.arange(n_steps) * self.dt
        truths = np.array([self.trajectory.state_at(t) for t in times])

        track = Track(track_id="target_1")
        # The filter's initial state is at t=0, so log it directly.
        track.append(self.filter.state.copy())

        measurements_per_step: list[dict[str, Measurement]] = [{}]  # nothing at t=0

        # For every step after t=0, generate measurements and run filter.step.
        for k in range(1, n_steps):
            t = times[k]
            true_state = truths[k]

            # Each sensor independently decides whether it detects the target.
            step_meas: dict[str, Measurement] = {}
            for sensor in self.sensors:
                m = sensor.measure(true_state, self.trajectory.layout, t, self.rng)
                if m is not None:
                    step_meas[sensor.sensor_id] = m
            measurements_per_step.append(step_meas)

            # Apply measurements to the filter. For the minimal example we
            # assume one sensor — if there are multiple, we update sequentially.
            if len(step_meas) == 0:
                # No detections this step — predict only.
                self.filter.step(t, None, None)
            else:
                # Predict to t with the first measurement, then sequential updates.
                first = True
                for sensor in self.sensors:
                    m = step_meas.get(sensor.sensor_id)
                    if m is None:
                        continue
                    if first:
                        self.filter.step(t, m, sensor)
                        first = False
                    else:
                        # Already at time t; just update with this sensor.
                        self.filter.update(m, sensor)
            track.append(self.filter.state.copy())

        return SimulationResult(
            times=times,
            truths=truths,
            track=track,
            measurements=measurements_per_step,
        )
