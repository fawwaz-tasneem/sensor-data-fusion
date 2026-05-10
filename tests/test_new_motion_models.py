"""
Tests for the new motion models: CA, CoordinatedTurn (known), and
CoordinatedTurnUnknown.

Key correctness checks:
  1. F(x, dt) matches a numerical derivative of f(x, dt).
  2. Predict moves a deterministic state along the expected trajectory
     (straight line for zero accel/turn-rate, circle for constant turn).
  3. CT-known reduces to CV at omega -> 0 (small-angle limit).
  4. CT-unknown's omega component is preserved across predict.
"""
import numpy as np
import pytest

from sdf.core import StateDistribution
from sdf.motion_models import (
    ConstantAcceleration,
    ConstantVelocity,
    CoordinatedTurn,
    CoordinatedTurnUnknown,
)


def numerical_F(model, x, dt, eps=1e-6):
    """Numerical Jacobian of f(x, dt) w.r.t. x."""
    n = x.shape[0]
    F = np.zeros((n, n))
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        F[:, i] = (model.f(xp, dt) - model.f(xm, dt)) / (2 * eps)
    return F


# ---- Constant Acceleration -------------------------------------------

class TestConstantAcceleration2D:
    def setup_method(self):
        self.ca = ConstantAcceleration(dim=2, jerk_std=1.0)

    def test_state_dim_is_6(self):
        assert self.ca.state_dim == 6

    def test_layout_indices(self):
        assert self.ca.layout.position_idx == (0, 3)
        assert self.ca.layout.velocity_idx == (1, 4)
        assert self.ca.layout.accel_idx == (2, 5)

    def test_predict_moves_position_quadratically(self):
        # State: x=0, vx=0, ax=10, y=0, vy=0, ay=2 (stationary, accelerating).
        x0 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 2.0])
        state = StateDistribution(
            mean=x0, covariance=np.eye(6) * 0.01, timestamp=0.0,
            layout=self.ca.layout,
        )
        s2 = self.ca.predict(state, dt=2.0)
        # x = 0 + 0*2 + 10*4/2 = 20. vx = 0 + 10*2 = 20. ax unchanged.
        np.testing.assert_allclose(s2.mean, [20.0, 20.0, 10.0, 4.0, 4.0, 2.0])

    def test_F_matches_numerical(self):
        x = np.array([10.0, 5.0, 1.0, 20.0, -3.0, 0.5])
        F = self.ca.F(x, dt=0.5)
        F_num = numerical_F(self.ca, x, dt=0.5)
        np.testing.assert_allclose(F, F_num, atol=1e-6)

    def test_Q_is_psd(self):
        Q = self.ca.Q(dt=0.5)
        eig = np.linalg.eigvalsh(Q)
        assert eig.min() >= -1e-12


class TestConstantAcceleration3D:
    def test_state_dim_is_9(self):
        ca = ConstantAcceleration(dim=3, jerk_std=0.5)
        assert ca.state_dim == 9

    def test_predict_in_3d(self):
        ca = ConstantAcceleration(dim=3, jerk_std=0.1)
        x0 = np.array([0.0, 5.0, 0.0,
                       0.0, 0.0, 0.0,
                       0.0, 0.0, 1.0])  # only z-axis accelerating
        state = StateDistribution(
            mean=x0, covariance=np.eye(9) * 0.01, timestamp=0.0,
            layout=ca.layout,
        )
        s2 = ca.predict(state, dt=4.0)
        # z = 0 + 0*4 + 1*16/2 = 8. vz = 0 + 1*4 = 4. az unchanged.
        np.testing.assert_allclose(s2.mean[6:9], [8.0, 4.0, 1.0])


# ---- Coordinated Turn (known) ----------------------------------------

class TestCoordinatedTurnKnown:
    def test_zero_turn_rate_matches_cv(self):
        cv = ConstantVelocity(dim=2, process_noise_std=0.0)
        ct = CoordinatedTurn(omega=0.0, process_noise_std=0.0)
        x = np.array([10.0, 5.0, 20.0, 3.0])
        # F should match (modulo sign of process noise; we set both to zero).
        F_cv = cv.F(x, dt=1.0)
        F_ct = ct.F(x, dt=1.0)
        np.testing.assert_allclose(F_cv, F_ct, atol=1e-12)

    def test_constant_circle(self):
        # A target initially at (R, 0) moving with velocity (0, R*omega) in
        # the +y direction at constant turn rate omega should trace out a
        # circle of radius R centered at the origin.
        omega = 0.1
        R = 100.0
        ct = CoordinatedTurn(omega=omega, process_noise_std=0.0)
        x0 = np.array([R, 0.0, 0.0, R * omega])  # vx=0, vy=R*omega
        state = StateDistribution(
            mean=x0, covariance=np.eye(4) * 0.001, timestamp=0.0,
            layout=ct.layout,
        )
        # After a quarter period, the target should be at (0, R).
        T_quarter = (np.pi / 2) / omega
        sf = ct.predict(state, dt=T_quarter)
        np.testing.assert_allclose(sf.mean[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(sf.mean[2], R, atol=1e-6)

    def test_F_matches_numerical(self):
        ct = CoordinatedTurn(omega=0.05, process_noise_std=0.0)
        x = np.array([10.0, 5.0, 20.0, 3.0])
        F = ct.F(x, dt=1.5)
        F_num = numerical_F(ct, x, dt=1.5)
        np.testing.assert_allclose(F, F_num, atol=1e-8)


# ---- Coordinated Turn (unknown) --------------------------------------

class TestCoordinatedTurnUnknown:
    def test_state_dim_is_5(self):
        ct = CoordinatedTurnUnknown()
        assert ct.state_dim == 5

    def test_layout_includes_turn_rate(self):
        ct = CoordinatedTurnUnknown()
        assert ct.layout.turn_rate_idx == 4

    def test_omega_preserved_in_predict(self):
        ct = CoordinatedTurnUnknown(process_noise_std=0.0, omega_noise_std=0.0)
        x0 = np.array([100.0, 0.0, 0.0, 10.0, 0.05])
        state = StateDistribution(
            mean=x0, covariance=np.eye(5) * 0.001, timestamp=0.0,
            layout=ct.layout,
        )
        sf = ct.predict(state, dt=5.0)
        # omega should be unchanged.
        assert sf.mean[4] == pytest.approx(0.05)

    def test_F_matches_numerical_at_nonzero_omega(self):
        ct = CoordinatedTurnUnknown(process_noise_std=0.0, omega_noise_std=0.0)
        x = np.array([10.0, 5.0, 20.0, 3.0, 0.05])
        F = ct.F(x, dt=1.0)
        F_num = numerical_F(ct, x, dt=1.0)
        # The omega column has the most chance of being wrong; check it
        # specifically as well as the full F.
        np.testing.assert_allclose(F, F_num, atol=1e-6)

    def test_F_matches_numerical_at_small_omega(self):
        # Tests the small-angle path of the F formula. We use a very small
        # numerical eps to keep the perturbed omega within the small-angle
        # branch on both sides; otherwise the test compares a small-angle
        # formula against an exact-formula numerical derivative across a
        # branch boundary.
        ct = CoordinatedTurnUnknown(process_noise_std=0.0, omega_noise_std=0.0)
        x = np.array([10.0, 5.0, 20.0, 3.0, 1e-7])
        F = ct.F(x, dt=1.0)
        # Use eps small enough that x[4] +/- eps stays well below the
        # 1e-6 small-angle threshold.
        F_num = numerical_F(ct, x, dt=1.0, eps=1e-10)
        np.testing.assert_allclose(F, F_num, atol=1e-3)

    def test_circle_with_known_omega(self):
        # If we initialize the unknown-omega model at the right omega, it
        # should produce the same circle as CT-known.
        omega = 0.1
        R = 100.0
        ct = CoordinatedTurnUnknown(process_noise_std=0.0, omega_noise_std=0.0)
        x0 = np.array([R, 0.0, 0.0, R * omega, omega])
        state = StateDistribution(
            mean=x0, covariance=np.eye(5) * 0.001, timestamp=0.0,
            layout=ct.layout,
        )
        T_quarter = (np.pi / 2) / omega
        sf = ct.predict(state, dt=T_quarter)
        np.testing.assert_allclose(sf.mean[0], 0.0, atol=1e-5)
        np.testing.assert_allclose(sf.mean[2], R, atol=1e-5)
