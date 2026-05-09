"""
Tests for RadarSensor.

We verify:
  1. h() produces the geometrically correct range and bearing(s).
  2. H() matches a numerical Jacobian of h() to high precision.
  3. innovation() correctly wraps bearings across the +-pi branch.
  4. measure() output respects the angular range after noise.
  5. Errors are raised for degenerate geometries.
"""
import numpy as np
import pytest

from sdf.core.state import StateLayout
from sdf.sensors.radar import RadarSensor, wrap_to_pi


def numerical_jacobian(h_func, x, eps=1e-6):
    """Central-difference Jacobian for testing analytical H."""
    z0 = h_func(x)
    n = x.shape[0]
    m = z0.shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        J[:, i] = (h_func(x_plus) - h_func(x_minus)) / (2 * eps)
    return J


class TestRadar2D:
    def setup_method(self):
        self.layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        self.sensor = RadarSensor(
            sensor_id="r1",
            position=np.array([0.0, 0.0]),
            range_std=10.0,
            bearing_std=1e-3,
        )

    def test_range_at_known_position(self):
        # Target at (3, 0, 4, 0) in [x, vx, y, vy] format → range = 5.
        x = np.array([3.0, 0.0, 4.0, 0.0])
        z = self.sensor.h(x, self.layout)
        assert z[0] == pytest.approx(5.0)

    def test_bearing_in_each_quadrant(self):
        # +x axis: bearing 0
        z = self.sensor.h(np.array([10.0, 0.0, 0.0, 0.0]), self.layout)
        assert z[1] == pytest.approx(0.0, abs=1e-9)
        # +y axis: bearing pi/2
        z = self.sensor.h(np.array([0.0, 0.0, 10.0, 0.0]), self.layout)
        assert z[1] == pytest.approx(np.pi / 2, abs=1e-9)
        # -x axis: bearing pi (or -pi; atan2 returns +pi here)
        z = self.sensor.h(np.array([-10.0, 0.0, 0.0, 0.0]), self.layout)
        assert abs(wrap_to_pi(z[1] - np.pi)) < 1e-9

    def test_jacobian_matches_numerical(self):
        # Generic point not aligned to axes.
        x = np.array([100.0, 5.0, 50.0, -3.0])
        H_analytic = self.sensor.H(x, self.layout)

        def h_only(state):
            return self.sensor.h(state, self.layout)

        H_numeric = numerical_jacobian(h_only, x)
        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_jacobian_only_depends_on_position(self):
        # Velocity columns should be exactly zero.
        x = np.array([100.0, 5.0, 50.0, -3.0])
        H = self.sensor.H(x, self.layout)
        np.testing.assert_array_equal(H[:, [1, 3]], 0)

    def test_target_at_sensor_raises(self):
        x = np.array([0.0, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="essentially at sensor"):
            self.sensor.h(x, self.layout)


class TestRadarInnovation:
    def setup_method(self):
        self.sensor = RadarSensor(
            sensor_id="r1", position=np.array([0.0, 0.0])
        )

    def test_innovation_wraps_across_branch_cut(self):
        # measurement bearing = -pi + 0.01, predicted = pi - 0.01.
        # Naive subtraction gives ~ -2*pi + 0.02 ≈ -6.26; correct is ~0.02.
        z = np.array([100.0, -np.pi + 0.01])
        z_pred = np.array([100.0, np.pi - 0.01])
        y = self.sensor.innovation(z, z_pred)
        assert y[0] == pytest.approx(0.0)
        assert abs(y[1]) < 0.05  # not 6.26


class TestRadar3D:
    def setup_method(self):
        self.layout = StateLayout(
            dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5)
        )
        self.sensor = RadarSensor(
            sensor_id="r1",
            position=np.array([0.0, 0.0, 0.0]),
            range_std=10.0,
            bearing_std=1e-3,
            elevation_std=1e-3,
        )

    def test_h_3d_at_known_geometry(self):
        # Target at (3, 4, 0): range 5, az = atan2(4,3), el = 0.
        x = np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        z = self.sensor.h(x, self.layout)
        assert z[0] == pytest.approx(5.0)
        assert z[1] == pytest.approx(np.arctan2(4, 3))
        assert z[2] == pytest.approx(0.0, abs=1e-12)

    def test_h_3d_with_elevation(self):
        # Target at (1, 0, sqrt(3)): r = 2, az = 0, el = pi/3 (60 deg).
        x = np.array([1.0, 0.0, 0.0, 0.0, np.sqrt(3.0), 0.0])
        z = self.sensor.h(x, self.layout)
        assert z[0] == pytest.approx(2.0)
        assert z[2] == pytest.approx(np.pi / 3)

    def test_jacobian_3d_matches_numerical(self):
        x = np.array([100.0, 5.0, 50.0, -3.0, 30.0, 1.0])
        H_analytic = self.sensor.H(x, self.layout)

        def h_only(state):
            return self.sensor.h(state, self.layout)

        H_numeric = numerical_jacobian(h_only, x)
        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)


class TestMeasureWrapsAngles:
    def test_noisy_bearing_stays_in_range(self):
        sensor = RadarSensor(
            sensor_id="r1",
            position=np.array([0.0, 0.0]),
            range_std=1.0,
            bearing_std=10.0,  # huge noise to force wrap
        )
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        rng = np.random.default_rng(0)
        x = np.array([100.0, 0.0, 0.0, 0.0])
        for _ in range(200):
            m = sensor.measure(x, layout, t=0.0, rng=rng)
            assert -np.pi <= m.value[1] <= np.pi
