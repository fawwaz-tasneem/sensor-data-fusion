"""
Tests for GMTI sensor on a moving platform.

Verify that:
  1. set_time(t) updates self.position and self.velocity from the platform.
  2. h() correctly subtracts sensor velocity for the range-rate component.
  3. Jacobian matches numerical reference even with non-zero sensor velocity.
  4. DopplerBlindnessOcclusion attached to a moving GMTI gets its
     sensor_position and sensor_velocity synced via set_time.
"""
import numpy as np
import pytest

from sdf.core.state import StateLayout
from sdf.scenarios.awacs import StraightFlight
from sdf.sensors import DopplerBlindnessOcclusion, GMTIRadarSensor


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


class TestMovingGMTI:
    def setup_method(self):
        self.layout = StateLayout(
            dim=3, position_idx=(0, 2, 4), velocity_idx=(1, 3, 5)
        )
        self.platform = StraightFlight(
            start_position=np.array([0.0, 0.0, 1000.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
        )
        self.sensor = GMTIRadarSensor(
            sensor_id="awacs_gmti",
            position=self.platform.position_at(0.0),
            range_std=10.0,
            bearing_std=1e-3,
            elevation_std=1e-3,
            range_rate_std=0.5,
            platform=self.platform,
        )

    def test_set_time_updates_position_and_velocity(self):
        self.sensor.set_time(10.0)
        np.testing.assert_array_equal(self.sensor.position, [1000.0, 0.0, 1000.0])
        np.testing.assert_array_equal(self.sensor.velocity, [100.0, 0.0, 0.0])

    def test_h_subtracts_sensor_velocity(self):
        # Place the sensor at origin (move platform back to t=0).
        self.sensor.set_time(0.0)
        # Target far ahead at (10000, 0, 0) with zero velocity.
        # Sensor at (0,0,1000) moving at +100 m/s in x. Relative velocity
        # (target - sensor) is (-100, 0, 0); LOS u_hat ~ (10000, 0, -1000)/r.
        # range-rate = u_hat . (-100, 0, 0) ~ -100 * 10000/sqrt(10000^2+1000^2)
        x_target_at_rest = np.array([10000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        z = self.sensor.h(x_target_at_rest, self.layout)
        # Range-rate should be negative (target appears to recede in sensor frame).
        assert z[3] < 0
        # Magnitude approximately 100 * 10000/r where r = sqrt(10000^2+1000^2)
        r_expected = np.sqrt(10000**2 + 1000**2)
        rr_expected = -100.0 * 10000.0 / r_expected
        assert z[3] == pytest.approx(rr_expected, rel=1e-6)

    def test_jacobian_matches_numerical_with_moving_sensor(self):
        self.sensor.set_time(5.0)
        x = np.array([5000.0, 30.0, 2000.0, -10.0, 100.0, 5.0])
        H_analytic = self.sensor.H(x, self.layout)

        def h_only(state):
            return self.sensor.h(state, self.layout)

        H_numeric = numerical_jacobian(h_only, x)
        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-4)

    def test_doppler_blindness_synced(self):
        blindness = DopplerBlindnessOcclusion(
            sensor_position=self.platform.position_at(0.0),
            mdv=2.0,
        )
        sensor = GMTIRadarSensor(
            sensor_id="g",
            position=self.platform.position_at(0.0),
            range_std=10.0,
            bearing_std=1e-3,
            elevation_std=1e-3,
            range_rate_std=0.5,
            occlusion_model=blindness,
            platform=self.platform,
        )
        sensor.set_time(20.0)
        # The occlusion's sensor position and velocity must now match the
        # platform at t=20.
        np.testing.assert_array_equal(blindness.sensor_position, [2000.0, 0.0, 1000.0])
        np.testing.assert_array_equal(blindness.sensor_velocity, [100.0, 0.0, 0.0])

    def test_set_time_noop_for_stationary(self):
        # GMTI without a platform should leave position/velocity unchanged.
        gmti = GMTIRadarSensor(
            sensor_id="g",
            position=np.array([100.0, 200.0, 50.0]),
            range_std=10.0,
            bearing_std=1e-3,
            elevation_std=1e-3,
            range_rate_std=0.5,
        )
        original_pos = gmti.position.copy()
        gmti.set_time(123.0)
        np.testing.assert_array_equal(gmti.position, original_pos)
        np.testing.assert_array_equal(gmti.velocity, [0.0, 0.0, 0.0])


def test_platform_dim_mismatch_raises():
    platform_2d = StraightFlight(
        start_position=np.array([0.0, 0.0]),
        velocity=np.array([10.0, 0.0]),
    )
    with pytest.raises(ValueError, match="platform dim"):
        GMTIRadarSensor(
            sensor_id="g",
            position=np.array([0.0, 0.0, 1000.0]),
            elevation_std=1e-3,
            platform=platform_2d,
        )
