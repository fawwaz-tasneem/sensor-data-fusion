"""
Tests for AzimuthOnlyRadarSensor (range + azimuth, no elevation).

Verifies:
  1. measurement_dim is 2 (no elevation row).
  2. h() returns slant range and azimuth.
  3. The analytic Jacobian matches a numerical reference, and the azimuth
     row's z-column is exactly zero (azimuth carries no height info).
  4. Innovation wraps azimuth but not range.
"""
import numpy as np

from sdf.core.state import StateLayout
from sdf.sensors import AzimuthOnlyRadarSensor
from sdf.sensors.radar import wrap_to_pi


def _numerical_jacobian(h_func, x, eps=1e-6):
    z0 = h_func(x)
    J = np.zeros((z0.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        xp = x.copy(); xm = x.copy()
        xp[i] += eps; xm[i] -= eps
        J[:, i] = (h_func(xp) - h_func(xm)) / (2 * eps)
    return J


class TestAzimuthOnlyRadar:
    def setup_method(self):
        self.layout = StateLayout(dim=3, position_idx=(0, 2, 4),
                                  velocity_idx=(1, 3, 5))
        self.sensor = AzimuthOnlyRadarSensor(
            sensor_id="az", position=np.array([0.0, 10_000.0, 100.0]),
            range_std=80.0, bearing_std=8e-3,
        )
        # state [x, vx, y, vy, z, vz]
        self.x = np.array([3000.0, 30.0, 2000.0, -10.0, 500.0, 1.0])

    def test_measurement_dim_is_2(self):
        assert self.sensor.measurement_dim == 2

    def test_h_is_ground_range_and_azimuth(self):
        z = self.sensor.h(self.x, self.layout)
        d = np.array([3000.0, 2000.0, 500.0]) - np.array([0.0, 10_000.0, 100.0])
        # Ground range = horizontal distance only (independent of dz).
        assert np.isclose(z[0], np.hypot(d[0], d[1]))
        assert np.isclose(z[1], np.arctan2(d[1], d[0]))

    def test_measurement_is_independent_of_height(self):
        # Changing the target's z must not change either output.
        x_high = self.x.copy()
        x_high[4] += 5000.0  # move z by 5 km
        np.testing.assert_allclose(
            self.sensor.h(self.x, self.layout),
            self.sensor.h(x_high, self.layout),
        )

    def test_jacobian_matches_numerical(self):
        H = self.sensor.H(self.x, self.layout)

        def h_only(x):
            return self.sensor.h(x, self.layout)

        np.testing.assert_allclose(H, _numerical_jacobian(h_only, self.x),
                                   atol=1e-5)

    def test_no_z_dependence_in_jacobian(self):
        # Both rows' z column (state index 4) must be exactly 0 — altitude is
        # unobservable from a ground-range / azimuth radar.
        H = self.sensor.H(self.x, self.layout)
        assert H[0, 4] == 0.0  # ground-range row
        assert H[1, 4] == 0.0  # azimuth row

    def test_innovation_wraps_azimuth_only(self):
        z = np.array([1000.0, 3.10])
        zp = np.array([1000.0, -3.10])
        diff = self.sensor.innovation(z, zp)
        assert diff[0] == 0.0  # range: plain difference
        assert abs(diff[1]) < 0.1  # azimuth wrapped, not ~6.2
        assert np.isclose(diff[1], wrap_to_pi(6.20))
