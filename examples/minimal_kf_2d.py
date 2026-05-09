"""
Minimal end-to-end example.

Scenario:
  - 2D constant-velocity target starting at (0, 0) with velocity (10, 5) m/s.
  - One Cartesian position sensor with isotropic noise std = 2.0 m.
  - Standard linear Kalman filter with a CV motion model.
  - Run for 30 seconds at 10 Hz.

What this validates end-to-end:
  StateDistribution / StateLayout  → MotionModel.predict
  Sensor.measure (detection + noise)
  KalmanFilter.predict / update    → Track logging
  SimulationEngine wiring
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateDistribution
from sdf.filters.kalman import KalmanFilter
from sdf.motion_models.constant_velocity import ConstantVelocity
from sdf.scenarios.constant_velocity import ConstantVelocityTrajectory
from sdf.sensors.cartesian import CartesianPositionSensor
from sdf.simulation.engine import SimulationEngine


def main() -> None:
    # 1. Define ground truth.
    motion_model = ConstantVelocity(dim=2, process_noise_std=0.5)
    layout = motion_model.layout
    true_initial = np.array([0.0, 10.0, 0.0, 5.0])  # x, vx, y, vy
    trajectory = ConstantVelocityTrajectory(true_initial, layout)

    # 2. Define sensor.
    sensor = CartesianPositionSensor(
        sensor_id="cart_1", dim=2, noise_std=2.0, detection_prob=1.0
    )

    # 3. Define filter with deliberately-wrong initial estimate so we can
    #    see it converge to truth. Initial covariance is loose.
    initial_estimate = StateDistribution(
        mean=np.array([5.0, 8.0, -3.0, 6.0]),  # offset from truth
        covariance=np.diag([100.0, 25.0, 100.0, 25.0]),
        timestamp=0.0,
        layout=layout,
    )
    kf = KalmanFilter(motion_model=motion_model, initial_state=initial_estimate)

    # 4. Run simulation.
    engine = SimulationEngine(
        trajectory=trajectory,
        sensors=[sensor],
        filter=kf,
        dt=0.1,
        duration=30.0,
        seed=42,  # reproducible
    )
    result = engine.run()

    # 5. Report basic metrics.
    truth_positions = result.truths[:, [layout.position_idx[0], layout.position_idx[1]]]
    est_positions = result.track.positions()
    pos_err = np.linalg.norm(est_positions - truth_positions, axis=1)

    print(f"Simulation: {len(result.times)} steps over {result.times[-1]:.1f} s")
    print(f"Initial position error: {pos_err[0]:.2f} m")
    print(f"Final position error:   {pos_err[-1]:.2f} m")
    print(f"Mean position error after t=5s: {pos_err[result.times >= 5.0].mean():.2f} m")
    print(f"Sensor noise std was: 2.00 m  (filter should beat this when fused over time)")


if __name__ == "__main__":
    main()
