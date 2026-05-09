"""
Tests for the simulation engine.

The engine is a thin coordinator. We test:
  - Reproducibility: same seed → same result.
  - Track length matches number of timesteps.
  - With detection_prob=0, no measurements are produced and the filter
    still advances through prediction (track coasting).
"""
import numpy as np

from sdf.core import StateDistribution
from sdf.filters import KalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.scenarios import ConstantVelocityTrajectory
from sdf.sensors import CartesianPositionSensor
from sdf.simulation import SimulationEngine


def _build_engine(seed=0, detection_prob=1.0):
    cv = ConstantVelocity(dim=2, process_noise_std=0.5)
    traj = ConstantVelocityTrajectory(
        np.array([0.0, 10.0, 0.0, 5.0]), cv.layout
    )
    sensor = CartesianPositionSensor(
        sensor_id="s1", dim=2, noise_std=2.0, detection_prob=detection_prob
    )
    init = StateDistribution(
        mean=np.array([5.0, 8.0, -3.0, 6.0]),
        covariance=np.diag([100.0, 25.0, 100.0, 25.0]),
        timestamp=0.0,
        layout=cv.layout,
    )
    kf = KalmanFilter(motion_model=cv, initial_state=init)
    return SimulationEngine(
        trajectory=traj, sensors=[sensor], filter=kf, dt=0.1, duration=5.0, seed=seed
    )


class TestSimulationEngine:
    def test_track_length_matches_steps(self):
        engine = _build_engine(seed=0)
        result = engine.run()
        # 5 seconds at 0.1s = 51 steps including t=0.
        assert len(result.track) == 51
        assert result.times.shape == (51,)
        assert result.truths.shape == (51, 4)

    def test_reproducible_with_same_seed(self):
        r1 = _build_engine(seed=42).run()
        r2 = _build_engine(seed=42).run()
        np.testing.assert_array_equal(r1.track.positions(), r2.track.positions())

    def test_different_seeds_give_different_results(self):
        r1 = _build_engine(seed=1).run()
        r2 = _build_engine(seed=2).run()
        # At least one position should differ.
        assert not np.allclose(r1.track.positions(), r2.track.positions())

    def test_no_detections_still_advances_filter(self):
        # With detection_prob=0, no measurements happen but the filter must
        # still produce predictions at every step (track coasting).
        engine = _build_engine(seed=0, detection_prob=0.0)
        result = engine.run()
        assert len(result.track) == 51
        # Every measurements dict should be empty.
        for step_meas in result.measurements:
            assert step_meas == {}
        # Without measurements, covariance should grow monotonically.
        traces = [np.trace(s.covariance) for s in result.track.history]
        for a, b in zip(traces[:-1], traces[1:]):
            assert b >= a
