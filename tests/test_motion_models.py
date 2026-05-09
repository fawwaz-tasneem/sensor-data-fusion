"""
Tests for motion models.

Two key invariants we test:
  1. Numerical correctness: F and Q have the closed-form values we expect.
  2. Predict moves a deterministic state along the expected straight line
     (since CV has no acceleration, position evolves linearly).
"""
import numpy as np
import pytest

from sdf.core import StateDistribution
from sdf.motion_models import ConstantVelocity


class TestConstantVelocity2D:
    def setup_method(self):
        self.cv = ConstantVelocity(dim=2, process_noise_std=1.0)

    def test_state_dim_is_4(self):
        assert self.cv.state_dim == 4

    def test_F_for_dt_1(self):
        F = self.cv.F(np.zeros(4), dt=1.0)
        expected = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ]
        )
        np.testing.assert_array_equal(F, expected)

    def test_Q_is_symmetric(self):
        Q = self.cv.Q(dt=0.5)
        np.testing.assert_allclose(Q, Q.T)

    def test_Q_is_positive_semidefinite(self):
        Q = self.cv.Q(dt=0.5)
        # All eigenvalues should be >= 0 (allowing tiny numerical noise).
        eigvals = np.linalg.eigvalsh(Q)
        assert eigvals.min() >= -1e-12

    def test_predict_moves_position_linearly(self):
        # Start at origin moving at 10 m/s in x and 5 m/s in y.
        state = StateDistribution(
            mean=np.array([0.0, 10.0, 0.0, 5.0]),
            covariance=np.eye(4) * 0.01,
            timestamp=0.0,
            layout=self.cv.layout,
        )
        predicted = self.cv.predict(state, dt=2.0)
        # After 2 seconds, position should be (20, 10), velocity unchanged.
        np.testing.assert_allclose(predicted.mean, [20.0, 10.0, 10.0, 5.0])
        assert predicted.timestamp == 2.0

    def test_predict_grows_uncertainty(self):
        state = StateDistribution(
            mean=np.array([0.0, 10.0, 0.0, 5.0]),
            covariance=np.eye(4) * 0.01,
            timestamp=0.0,
            layout=self.cv.layout,
        )
        predicted = self.cv.predict(state, dt=1.0)
        # Trace of covariance must increase (process noise adds uncertainty).
        assert np.trace(predicted.covariance) > np.trace(state.covariance)


class TestConstantVelocity3D:
    def test_state_dim_is_6(self):
        cv = ConstantVelocity(dim=3, process_noise_std=1.0)
        assert cv.state_dim == 6

    def test_predict_moves_position_in_3d(self):
        cv = ConstantVelocity(dim=3, process_noise_std=0.1)
        state = StateDistribution(
            mean=np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0]),
            covariance=np.eye(6) * 0.01,
            timestamp=0.0,
            layout=cv.layout,
        )
        predicted = cv.predict(state, dt=4.0)
        # Positions: 0+1*4=4, 0+2*4=8, 0+3*4=12.
        np.testing.assert_allclose(
            predicted.mean, [4.0, 1.0, 8.0, 2.0, 12.0, 3.0]
        )


def test_invalid_dim_raises():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        ConstantVelocity(dim=4)
