"""
Tests for GMTI radar sensor.

We verify:
  1. h() returns range, bearing(s), AND range-rate.
  2. Range-rate equals the radial component of velocity.
  3. Jacobian matches numerical reference (including velocity columns).
  4. Stationary target gives zero range-rate.
  5. Innovation wraps angle components but not range or range-rate.
"""
import numpy as np
import pytest

from sdf.core.state import StateLayout
from sdf.sensors import GMTIRadarSensor


def numerical_jacobian(h_func, x, eps=1e-6):
    z0 = h_func(x)
    n = x.shape[0]
    m = z0.shape[0]
    J = np.zeros((m, n))
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        J[:, i] = (h_func(xp) - h_func(xm)) / (2 * eps)
    return J


class TestGMTI2D:
    def setup_method(self):
        self.layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        self.sensor = GMTIRadarSensor(
            sensor_id="g1",
            position=np.array([0.0, 0.0]),
            range_std=10.0,
            bearing_std=1e-3,
            range_rate_std=0.5,
        )

    def test_measurement_dim_is_3(self):
        assert self.sensor.measurement_dim == 3

    def test_stationary_target_has_zero_range_rate(self):
        x = np.array([100.0, 0.0, 100.0, 0.0])  # at rest
        z = self.sensor.h(x, self.layout)
        assert z[2] == pytest.approx(0.0, abs=1e-12)

    def test_target_moving_radially_outward(self):
        # Target at (3, 4), velocity (3, 4) — so range = 5, dot{r} = 5.
        x = np.array([3.0, 3.0, 4.0, 4.0])  # [x, vx, y, vy]
        z = self.sensor.h(x, self.layout)
        assert z[0] == pytest.approx(5.0)
        assert z[2] == pytest.approx(5.0)

    def test_target_moving_tangentially_has_zero_range_rate(self):
        # Target at (10, 0), velocity perpendicular to LOS = (0, 5).
        x = np.array([10.0, 0.0, 0.0, 5.0])
        z = self.sensor.h(x, self.layout)
        assert z[2] == pytest.approx(0.0, abs=1e-12)

    def test_jacobian_matches_numerical(self):
        x = np.array([100.0, 5.0, 50.0, -3.0])
        H_analytic = self.sensor.H(x, self.layout)

        def h_only(state):
            return self.sensor.h(state, self.layout)

        H_numeric = numerical_jacobian(h_only, x)
        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_jacobian_has_velocity_columns_for_range_rate_row(self):
        x = np.array([100.0, 5.0, 50.0, -3.0])
        H = self.sensor.H(x, self.layout)
        # H shape (3, 4). Range and bearing rows depend only on position
        # (cols 0, 2). Range-rate row depends on both position and velocity.
        rr_row = H[2]
        # Velocity columns for range-rate row should be nonzero.
        assert abs(rr_row[1]) > 1e-9 or abs(rr_row[3]) > 1e-9

    def test_innovation_wraps_only_angles(self):
        # Innovation between two GMTI measurements differing only in
        # bearing across the +-pi branch should wrap; range and range-rate
        # differences should pass through unchanged.
        z = np.array([100.0, -np.pi + 0.01, 5.0])
        z_pred = np.array([99.0, np.pi - 0.01, -3.0])
        y = self.sensor.innovation(z, z_pred)
        assert y[0] == pytest.approx(1.0)
        assert abs(y[1]) < 0.05  # wrapped
        assert y[2] == pytest.approx(8.0)


class TestGMTI3D:
    def test_jacobian_3d_matches_numerical(self):
        layout = StateLayout(
            dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5)
        )
        sensor = GMTIRadarSensor(
            sensor_id="g1",
            position=np.array([0.0, 0.0, 0.0]),
            range_std=10.0,
            bearing_std=1e-3,
            elevation_std=1e-3,
            range_rate_std=0.5,
        )
        x = np.array([100.0, 5.0, 50.0, -3.0, 30.0, 1.0])
        H_analytic = sensor.H(x, layout)

        def h_only(state):
            return sensor.h(state, layout)

        H_numeric = numerical_jacobian(h_only, x)
        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_3d_measurement_dim_is_4(self):
        sensor = GMTIRadarSensor(
            sensor_id="g1",
            position=np.array([0.0, 0.0, 0.0]),
            elevation_std=1e-3,
        )
        assert sensor.measurement_dim == 4
