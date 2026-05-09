"""
EKF tracking a 2D constant-velocity target with a radar sensor.

This is the natural follow-up to examples/minimal_kf_2d.py: same target
motion, but observed through a nonlinear (range, bearing) sensor instead
of a Cartesian one. The whole point: exercise the nonlinear measurement
path of the EKF.

Things to look for when you run it:
  * Initial position error is large (we deliberately offset the prior).
  * Steady-state error settles well below the radar range_std.
  * The estimate doesn't blow up when bearing is near +-pi (the wrap-
    around case): we fly the target through the +y axis where bearing
    transitions from positive to slightly larger positive but stays
    finite-difference well-behaved.
"""
from __future__ import annotations

import numpy as np

from sdf.core.state import StateDistribution
from sdf.filters import ExtendedKalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.scenarios import ConstantVelocityTrajectory
from sdf.sensors import RadarSensor
from sdf.simulation import SimulationEngine


def main() -> None:
    # Ground truth.
    cv = ConstantVelocity(dim=2, process_noise_std=0.1)
    layout = cv.layout
    # Place the target far enough from the radar that the Jacobian is
    # well-conditioned and so the linearization is reasonable.
    true_initial = np.array([1000.0, 10.0, 500.0, 5.0])
    trajectory = ConstantVelocityTrajectory(true_initial, layout)

    # Radar at origin.
    radar = RadarSensor(
        sensor_id="radar_1",
        position=np.array([0.0, 0.0]),
        range_std=10.0,
        bearing_std=1e-3,  # ~0.06 degrees
        detection_prob=1.0,
    )

    # Filter prior — deliberately offset from truth so we can see convergence.
    initial_estimate = StateDistribution(
        mean=np.array([1010.0, 8.0, 490.0, 6.0]),
        covariance=np.diag([400.0, 25.0, 400.0, 25.0]),
        timestamp=0.0,
        layout=layout,
    )
    ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=initial_estimate)

    engine = SimulationEngine(
        trajectory=trajectory,
        sensors=[radar],
        filter=ekf,
        dt=0.1,
        duration=40.0,
        seed=42,
    )
    result = engine.run()

    truth_positions = result.truths[:, [layout.position_idx[0], layout.position_idx[1]]]
    est_positions = result.track.positions()
    pos_err = np.linalg.norm(est_positions - truth_positions, axis=1)

    print(f"EKF + Radar: {len(result.times)} steps over {result.times[-1]:.1f} s")
    print(f"Initial position error:        {pos_err[0]:.2f} m")
    print(f"Final position error:          {pos_err[-1]:.2f} m")
    print(f"Mean position error after t=10s: {pos_err[result.times >= 10.0].mean():.2f} m")
    print(f"Radar range std was:           {radar.range_std:.2f} m")


if __name__ == "__main__":
    main()
