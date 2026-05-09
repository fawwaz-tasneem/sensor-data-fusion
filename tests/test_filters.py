"""
Tests for the Kalman filter.

The classic correctness checks for a KF:
  1. Predict-only: covariance grows monotonically (process noise adds uncertainty).
  2. Update with a perfect (zero-noise) measurement at the same place as the
     mean leaves the mean unchanged but shrinks the covariance.
  3. Convergence: with a known-correct model and many measurements, the
     filter's estimate gets arbitrarily close to the truth in steady state.
  4. Covariance stays symmetric and positive-semidefinite throughout.
  5. Predicting backwards in time raises an error.
"""
import numpy as np
import pytest

from sdf.core import Measurement, StateDistribution
from sdf.filters import KalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.sensors import CartesianPositionSensor


def _make_kf_2d(initial_mean, initial_cov, process_noise_std=0.1):
    cv = ConstantVelocity(dim=2, process_noise_std=process_noise_std)
    init = StateDistribution(
        mean=np.asarray(initial_mean, dtype=float),
        covariance=np.asarray(initial_cov, dtype=float),
        timestamp=0.0,
        layout=cv.layout,
    )
    return KalmanFilter(motion_model=cv, initial_state=init), cv


class TestKalmanFilterBasics:
    def test_predict_only_grows_covariance(self):
        kf, _ = _make_kf_2d([0, 1, 0, 1], np.eye(4) * 0.1)
        initial_trace = np.trace(kf.state.covariance)
        kf.predict(t=1.0)
        assert np.trace(kf.state.covariance) > initial_trace

    def test_predict_advances_timestamp(self):
        kf, _ = _make_kf_2d([0, 1, 0, 1], np.eye(4) * 0.1)
        kf.predict(t=2.5)
        assert kf.state.timestamp == 2.5

    def test_predict_backwards_raises(self):
        kf, _ = _make_kf_2d([0, 1, 0, 1], np.eye(4) * 0.1)
        kf.predict(t=1.0)
        with pytest.raises(ValueError, match="Cannot predict backwards"):
            kf.predict(t=0.5)

    def test_update_shrinks_covariance(self):
        kf, _ = _make_kf_2d([0, 0, 0, 0], np.eye(4))
        sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=0.1, detection_prob=1.0
        )
        # Predict to t=1 first so update happens at the same time as measurement.
        kf.predict(t=1.0)
        cov_before = kf.state.covariance.copy()
        m = Measurement(
            value=np.array([0.0, 0.0]),
            timestamp=1.0,
            sensor_id="s1",
            R=np.eye(2) * 0.01,
        )
        kf.update(m, sensor)
        # Position diagonals should shrink.
        for i in (0, 2):
            assert kf.state.covariance[i, i] < cov_before[i, i]

    def test_covariance_stays_symmetric(self):
        kf, _ = _make_kf_2d([0, 1, 0, 1], np.eye(4))
        sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=1.0, detection_prob=1.0
        )
        rng = np.random.default_rng(0)
        for k in range(1, 50):
            t = k * 0.1
            kf.predict(t)
            m = sensor.measure(
                np.array([10.0 * t, 10.0, 5.0 * t, 5.0]),
                kf.state.layout,
                t,
                rng,
            )
            kf.update(m, sensor)
            # Joseph form should preserve symmetry.
            np.testing.assert_allclose(
                kf.state.covariance,
                kf.state.covariance.T,
                atol=1e-10,
            )

    def test_covariance_stays_positive_semidefinite(self):
        kf, _ = _make_kf_2d([0, 1, 0, 1], np.eye(4))
        sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=1.0, detection_prob=1.0
        )
        rng = np.random.default_rng(0)
        for k in range(1, 50):
            t = k * 0.1
            kf.predict(t)
            m = sensor.measure(
                np.array([10.0 * t, 10.0, 5.0 * t, 5.0]),
                kf.state.layout,
                t,
                rng,
            )
            kf.update(m, sensor)
            eigvals = np.linalg.eigvalsh(kf.state.covariance)
            assert eigvals.min() >= -1e-10


class TestKalmanFilterConvergence:
    def test_filter_beats_sensor_noise(self):
        """
        With many measurements of a CV target whose model exactly matches
        truth, the filter's steady-state position error should be much
        smaller than a single sensor measurement's noise std.
        """
        kf, cv = _make_kf_2d(
            initial_mean=[5.0, 8.0, -3.0, 6.0],  # offset from truth
            initial_cov=np.diag([100.0, 25.0, 100.0, 25.0]),
            process_noise_std=0.1,
        )
        sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=2.0, detection_prob=1.0
        )
        rng = np.random.default_rng(42)

        true_init = np.array([0.0, 10.0, 0.0, 5.0])
        dt = 0.1
        errors = []
        for k in range(1, 301):  # 30 seconds
            t = k * dt
            x_true = np.array(
                [true_init[0] + true_init[1] * t,
                 true_init[1],
                 true_init[2] + true_init[3] * t,
                 true_init[3]]
            )
            kf.predict(t)
            m = sensor.measure(x_true, kf.state.layout, t, rng)
            kf.update(m, sensor)
            est_pos = kf.state.position()
            true_pos = x_true[[0, 2]]
            errors.append(np.linalg.norm(est_pos - true_pos))

        steady_state_error = np.mean(errors[100:])  # after t=10s
        # Should be well under the 2.0 m sensor noise.
        assert steady_state_error < 1.0
