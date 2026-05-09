"""
Tests for the EKF.

Cornerstone tests:
  1. EKF on a linear system (CV motion + Cartesian sensor) reproduces KF
     behavior exactly. If this fails, our EKF implementation has diverged
     from the KF in the linear case — a serious bug.
  2. EKF on a nonlinear sensor (radar) converges below sensor range noise
     when given many measurements.
  3. Covariance stays symmetric and PSD across many radar updates.
"""
import numpy as np
import pytest

from sdf.core import Measurement, StateDistribution
from sdf.filters import ExtendedKalmanFilter, KalmanFilter
from sdf.motion_models import ConstantVelocity
from sdf.sensors import CartesianPositionSensor, RadarSensor


def _initial_state_2d(layout, mean):
    return StateDistribution(
        mean=np.asarray(mean, dtype=float),
        covariance=np.diag([100.0, 25.0, 100.0, 25.0]),
        timestamp=0.0,
        layout=layout,
    )


class TestEKFEqualsKFOnLinearSystem:
    """If EKF != KF for a fully linear problem, EKF is broken."""

    def test_linear_system_agrees(self):
        cv = ConstantVelocity(dim=2, process_noise_std=0.5)
        sensor = CartesianPositionSensor(
            sensor_id="s1", dim=2, noise_std=2.0, detection_prob=1.0
        )

        init = _initial_state_2d(cv.layout, [5.0, 8.0, -3.0, 6.0])
        kf = KalmanFilter(motion_model=cv, initial_state=init)
        ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=init.copy())

        rng = np.random.default_rng(123)
        true = np.array([0.0, 10.0, 0.0, 5.0])
        for k in range(1, 100):
            t = k * 0.1
            x_true = np.array(
                [true[0] + true[1] * t, true[1], true[2] + true[3] * t, true[3]]
            )
            # Use the SAME measurement for both filters so any divergence
            # is purely an algorithmic difference.
            m = sensor.measure(x_true, cv.layout, t, rng)
            kf.predict(t)
            kf.update(m, sensor)
            ekf.predict(t)
            ekf.update(m, sensor)

        np.testing.assert_allclose(kf.state.mean, ekf.state.mean, atol=1e-10)
        np.testing.assert_allclose(
            kf.state.covariance, ekf.state.covariance, atol=1e-10
        )


class TestEKFWithRadar:
    def test_converges_below_range_noise(self):
        cv = ConstantVelocity(dim=2, process_noise_std=0.1)
        # Radar at origin; target moving along (10 m/s, 5 m/s) starting at (1000, 500).
        # Avoid placing the target near the sensor where the Jacobian is poorly conditioned.
        sensor = RadarSensor(
            sensor_id="r1",
            position=np.array([0.0, 0.0]),
            range_std=10.0,
            bearing_std=1e-3,
            detection_prob=1.0,
        )

        init = StateDistribution(
            mean=np.array([1010.0, 8.0, 490.0, 6.0]),  # offset from truth
            covariance=np.diag([400.0, 25.0, 400.0, 25.0]),
            timestamp=0.0,
            layout=cv.layout,
        )
        ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=init)

        rng = np.random.default_rng(42)
        true_init = np.array([1000.0, 10.0, 500.0, 5.0])
        errors = []
        for k in range(1, 401):  # 40 seconds
            t = k * 0.1
            x_true = np.array(
                [true_init[0] + true_init[1] * t,
                 true_init[1],
                 true_init[2] + true_init[3] * t,
                 true_init[3]]
            )
            m = sensor.measure(x_true, cv.layout, t, rng)
            ekf.predict(t)
            ekf.update(m, sensor)
            est_pos = ekf.state.position()
            true_pos = x_true[[0, 2]]
            errors.append(np.linalg.norm(est_pos - true_pos))

        steady = np.mean(errors[200:])
        # Steady state position error should be smaller than the per-measurement
        # range std (10 m), since we are fusing many measurements.
        assert steady < 5.0, f"steady-state error {steady:.2f} m exceeds 5 m"

    def test_covariance_invariants(self):
        cv = ConstantVelocity(dim=2, process_noise_std=0.1)
        sensor = RadarSensor(
            sensor_id="r1",
            position=np.array([0.0, 0.0]),
            range_std=10.0,
            bearing_std=1e-3,
        )
        init = StateDistribution(
            mean=np.array([1000.0, 10.0, 500.0, 5.0]),
            covariance=np.diag([400.0, 25.0, 400.0, 25.0]),
            timestamp=0.0,
            layout=cv.layout,
        )
        ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=init)
        rng = np.random.default_rng(0)
        for k in range(1, 200):
            t = k * 0.1
            x_true = np.array([1000.0 + 10 * t, 10.0, 500.0 + 5 * t, 5.0])
            m = sensor.measure(x_true, cv.layout, t, rng)
            ekf.predict(t)
            ekf.update(m, sensor)
            P = ekf.state.covariance
            np.testing.assert_allclose(P, P.T, atol=1e-10)
            assert np.linalg.eigvalsh(P).min() >= -1e-9


def test_ekf_predict_backwards_raises():
    cv = ConstantVelocity(dim=2)
    init = _initial_state_2d(cv.layout, [0, 1, 0, 1])
    ekf = ExtendedKalmanFilter(motion_model=cv, initial_state=init)
    ekf.predict(t=1.0)
    with pytest.raises(ValueError, match="Cannot predict backwards"):
        ekf.predict(t=0.5)
