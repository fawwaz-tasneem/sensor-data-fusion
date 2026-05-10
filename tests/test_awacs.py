"""
Tests for AWACS-style flight platforms.

We verify that each pattern returns the right position and velocity, and
that the analytic velocity matches a numerical derivative of position
(this is the strongest correctness check for parametric trajectories).
"""
import numpy as np
import pytest

from sdf.scenarios.awacs import CircleFlight, RacetrackFlight, StraightFlight
from sdf.scenarios.platform import StationaryPlatform


def numerical_velocity(platform, t, h=1e-3):
    return (platform.position_at(t + h) - platform.position_at(t - h)) / (2 * h)


class TestStationaryPlatform:
    def test_position_constant(self):
        p = StationaryPlatform(np.array([1.0, 2.0, 3.0]))
        for t in [0.0, 5.0, 100.0]:
            np.testing.assert_array_equal(p.position_at(t), [1.0, 2.0, 3.0])

    def test_velocity_zero(self):
        p = StationaryPlatform(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(p.velocity_at(7.5), [0.0, 0.0])


class TestStraightFlight:
    def test_position_evolves_linearly(self):
        f = StraightFlight(
            start_position=np.array([0.0, 0.0, 100.0]),
            velocity=np.array([10.0, 5.0, 0.0]),
        )
        np.testing.assert_array_equal(f.position_at(0.0), [0.0, 0.0, 100.0])
        np.testing.assert_array_equal(f.position_at(10.0), [100.0, 50.0, 100.0])

    def test_velocity_constant(self):
        f = StraightFlight(
            start_position=np.array([0.0, 0.0]),
            velocity=np.array([3.0, 4.0]),
        )
        np.testing.assert_array_equal(f.velocity_at(7.5), [3.0, 4.0])

    def test_analytic_velocity_matches_numerical(self):
        f = StraightFlight(
            start_position=np.array([10.0, 20.0, 30.0]),
            velocity=np.array([7.0, -2.0, 0.5]),
        )
        for t in [0.0, 5.0, 100.0]:
            np.testing.assert_allclose(f.velocity_at(t), numerical_velocity(f, t),
                                        atol=1e-8)


class TestCircleFlight:
    def test_period_brings_us_back(self):
        f = CircleFlight(
            center=np.array([0.0, 0.0, 100.0]),
            radius=1000.0,
            speed=100.0,
        )
        # Period is 2*pi*r / speed.
        T = 2 * np.pi * 1000.0 / 100.0
        np.testing.assert_allclose(f.position_at(0.0), f.position_at(T), atol=1e-6)

    def test_speed_is_constant(self):
        f = CircleFlight(
            center=np.array([0.0, 0.0]),
            radius=500.0,
            speed=50.0,
        )
        for t in np.linspace(0, 100, 11):
            assert np.linalg.norm(f.velocity_at(t)) == pytest.approx(50.0)

    def test_altitude_held_in_3d(self):
        f = CircleFlight(
            center=np.array([0.0, 0.0, 250.0]),
            radius=1000.0,
            speed=80.0,
        )
        for t in np.linspace(0, 200, 11):
            assert f.position_at(t)[2] == pytest.approx(250.0)
            assert f.velocity_at(t)[2] == pytest.approx(0.0)

    def test_analytic_velocity_matches_numerical(self):
        f = CircleFlight(
            center=np.array([100.0, 200.0, 50.0]),
            radius=1500.0,
            speed=70.0,
        )
        for t in [0.0, 30.0, 200.0]:
            np.testing.assert_allclose(f.velocity_at(t), numerical_velocity(f, t),
                                        atol=1e-3)


class TestRacetrackFlight:
    def test_position_loops(self):
        f = RacetrackFlight(
            center=np.array([0.0, 0.0, 100.0]),
            leg_length=2000.0,
            radius=500.0,
            speed=100.0,
        )
        # Total path length = 2 * 2000 + 2 * pi * 500 = 4000 + ~3141.6 = ~7141.6
        # Period at speed 100 m/s = ~71.4 s.
        T = (2 * 2000.0 + 2 * np.pi * 500.0) / 100.0
        np.testing.assert_allclose(f.position_at(0.0), f.position_at(T), atol=1e-3)

    def test_speed_is_constant(self):
        f = RacetrackFlight(
            center=np.array([0.0, 0.0]),
            leg_length=1500.0,
            radius=400.0,
            speed=80.0,
        )
        for t in np.linspace(0, 50, 21):
            assert np.linalg.norm(f.velocity_at(t)) == pytest.approx(80.0, rel=1e-9)

    def test_starts_on_top_leg(self):
        f = RacetrackFlight(
            center=np.array([0.0, 0.0, 200.0]),
            leg_length=2000.0,
            radius=500.0,
            speed=100.0,
        )
        # At t=0 with phase=0, we're at the start of the top straight leg:
        # position (-leg/2, +radius). Center at origin so x=-1000, y=+500.
        np.testing.assert_allclose(f.position_at(0.0), [-1000.0, 500.0, 200.0])
        # Velocity should be +x.
        np.testing.assert_allclose(f.velocity_at(0.0), [100.0, 0.0, 0.0])

    def test_analytic_velocity_matches_numerical_along_loop(self):
        f = RacetrackFlight(
            center=np.array([0.0, 0.0]),
            leg_length=2000.0,
            radius=500.0,
            speed=100.0,
        )
        # Sample times spread across all 4 segments of the loop, avoiding
        # the discontinuities at segment boundaries (where velocity is
        # continuous but our piecewise formula has a kink in the
        # parametrization).
        for t in [5.0, 15.0, 30.0, 45.0, 55.0, 65.0]:
            np.testing.assert_allclose(f.velocity_at(t), numerical_velocity(f, t),
                                        atol=1e-3)


def test_circle_rejects_bad_args():
    with pytest.raises(ValueError, match="radius"):
        CircleFlight(center=np.array([0.0, 0.0]), radius=-1.0, speed=10.0)
    with pytest.raises(ValueError, match="speed"):
        CircleFlight(center=np.array([0.0, 0.0]), radius=100.0, speed=0.0)


def test_straight_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="!="):
        StraightFlight(
            start_position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 1.0, 1.0]),
        )
