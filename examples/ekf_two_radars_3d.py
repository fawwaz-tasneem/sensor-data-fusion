"""
EKF tracking a 3D constant-velocity target with two stationary radars.

This is the natural follow-up to examples/minimal_kf_2d.py: still
constant velocity motion, but observed through two nonlinear
(range, azimuth, elevation) sensors instead of a Cartesian one. The
two radars are at (0, 10000, 100) and (10000, 0, 100), placed
symmetrically off-axis to give well-conditioned geometry.

Things to look for when you run it:
  * Initial position error is large (we deliberately offset the prior).
  * Steady-state error settles well below either radar's range_std,
    because two radars improve cross-range observability over a single one.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateDistribution
from sdf.filters import ExtendedKalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.scenarios import ConstantVelocityTrajectory
from sdf.sensors import RadarSensor


def main() -> None:
    # 3D constant-velocity target moving across a 10 km x 10 km region
    # so both radars see good-geometry returns.
    cv = ConstantVelocity(dim=3, process_noise_std=0.5)
    layout = cv.layout
    # state vector layout: [x, vx, y, vy, z, vz]
    true_initial = np.array([1000.0, 10.0, 1000.0, 5.0, 200.0, 0.0])
    trajectory = ConstantVelocityTrajectory(true_initial, layout)

    radar_a = RadarSensor(
        sensor_id="radar_a",
        position=np.array([0.0, 10_000.0, 100.0]),
        range_std=20.0,
        bearing_std=2e-3,
        elevation_std=2e-3,
        detection_prob=1.0,
    )
    radar_b = RadarSensor(
        sensor_id="radar_b",
        position=np.array([10_000.0, 0.0, 100.0]),
        range_std=20.0,
        bearing_std=2e-3,
        elevation_std=2e-3,
        detection_prob=1.0,
    )
    sensors = [radar_a, radar_b]

    # Filter prior — deliberately offset from truth so we can see convergence.
    initial_estimate = StateDistribution(
        mean=np.array([1050.0, 8.0, 950.0, 6.0, 220.0, 0.5]),
        covariance=np.diag([400.0, 25.0, 400.0, 25.0, 400.0, 25.0]),
        timestamp=0.0,
        layout=layout,
    )
    ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=initial_estimate)

    dt = 0.1
    duration = 40.0
    times = np.arange(0.0, duration + dt, dt)
    rng = np.random.default_rng(42)

    truths, track = [], []
    for t in times:
        x_true = trajectory.state_at(t)
        truths.append(x_true)
        if t == 0.0:
            track.append(ekf.state.mean.copy())
            continue
        ekf.predict(t)
        for sensor in sensors:
            m = sensor.measure(x_true, layout, t, rng)
            if m is not None:
                ekf.update(m, sensor)
        track.append(ekf.state.mean.copy())

    truths_a = np.array(truths)
    track_a = np.array(track)
    truth_pos = truths_a[:, list(layout.position_idx)]
    est_pos = track_a[:, list(layout.position_idx)]
    pos_err = np.linalg.norm(est_pos - truth_pos, axis=1)

    print(f"EKF + 2 radars (3D): {len(times)} steps over {times[-1]:.1f} s")
    print(f"Initial position error:        {pos_err[0]:.2f} m")
    print(f"Final position error:          {pos_err[-1]:.2f} m")
    print(f"Mean position error after t=10s: {pos_err[times >= 10.0].mean():.2f} m")
    print(f"Each radar's range std:        {radar_a.range_std:.2f} m")


if __name__ == "__main__":
    main()
