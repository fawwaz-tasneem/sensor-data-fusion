"""
Tests for MountainPassTrajectory.

We verify:
  1. Position equations match the original parametrization with default args.
  2. The analytical velocity equals the numerical derivative of position.
  3. Trajectory is reproducible — state_at(t) is pure.
  4. Custom parameters propagate correctly (no hidden hardcoding).
"""
import numpy as np

from sdf.scenarios import MountainPassTrajectory


class TestDefaultsMatchOriginal:
    """At default args this must match the user's original code."""

    def setup_method(self):
        self.traj = MountainPassTrajectory(
            v_kmh=20.0,
            length=10_000.0,
            y_amp=1_000.0,
            z_amp=1_000.0,
            y_cycles_per_length=4.0,
            z_cycles_per_length=1.0,
        )

    def test_speed_in_si(self):
        assert self.traj.v == 20.0 / 3.6

    def test_x_grows_linearly(self):
        for t in [0.0, 10.0, 100.0]:
            assert self.traj.position_at(t)[0] == self.traj.v * t

    def test_y_oscillation_period(self):
        # From the original: ky = 4 pi v / length.
        v = 20.0 / 3.6
        length = 10_000.0
        expected_ky = 4 * np.pi * v / length
        # Position at t and at t + 2*pi/ky must agree in y.
        period = 2 * np.pi / expected_ky
        y0 = self.traj.position_at(0.0)[1]
        y_period = self.traj.position_at(period)[1]
        assert abs(y_period - y0) < 1e-6

    def test_y_amplitude(self):
        # Quarter-period after 0 puts sin at 1, so y = y_amp.
        period = 2 * np.pi / self.traj.ky
        y_quarter = self.traj.position_at(period / 4)[1]
        assert abs(y_quarter - 1_000.0) < 1e-6


class TestAnalyticVelocityMatchesNumericalDerivative:
    def test_velocity_central_difference(self):
        traj = MountainPassTrajectory()
        h = 1e-3
        for t in [0.0, 50.0, 200.0, 1000.0]:
            p_plus = traj.position_at(t + h)
            p_minus = traj.position_at(t - h)
            v_numeric = (p_plus - p_minus) / (2 * h)
            v_analytic = traj.velocity_at(t)
            np.testing.assert_allclose(v_analytic, v_numeric, atol=1e-5)


class TestStateAtIsPure:
    def test_repeated_calls_give_same_result(self):
        traj = MountainPassTrajectory()
        a = traj.state_at(123.45)
        b = traj.state_at(123.45)
        np.testing.assert_array_equal(a, b)


class TestCustomParameters:
    def test_higher_speed_increases_x(self):
        slow = MountainPassTrajectory(v_kmh=10.0)
        fast = MountainPassTrajectory(v_kmh=40.0)
        assert fast.position_at(100.0)[0] > slow.position_at(100.0)[0]

    def test_offsets_apply(self):
        traj = MountainPassTrajectory(x0=500.0, y0=200.0, z0=50.0)
        p0 = traj.position_at(0.0)
        np.testing.assert_allclose(p0, [500.0, 200.0, 50.0])

    def test_cycles_per_length_changes_frequency(self):
        a = MountainPassTrajectory(y_cycles_per_length=1.0)
        b = MountainPassTrajectory(y_cycles_per_length=8.0)
        assert b.ky > a.ky


class TestStateLayout:
    def test_layout_is_3d_cv_shaped(self):
        traj = MountainPassTrajectory()
        assert traj.layout.dim == 3
        assert traj.layout.position_idx == (0, 2, 4)
        assert traj.layout.velocity_idx == (1, 3, 5)

    def test_state_at_matches_layout(self):
        traj = MountainPassTrajectory()
        s = traj.state_at(50.0)
        # Position from the layout must equal position_at directly.
        np.testing.assert_array_equal(traj.layout.position(s), traj.position_at(50.0))
        np.testing.assert_array_equal(traj.layout.velocity(s), traj.velocity_at(50.0))
