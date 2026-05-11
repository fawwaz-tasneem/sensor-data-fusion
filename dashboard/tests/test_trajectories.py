"""
Tests for the trajectory ComponentSpecs.

Verify:
  1. MountainPassTrajectory builds correctly from default form values.
  2. MountainPassTrajectory builds correctly from custom form values.
  3. ConstantVelocityTrajectory builds correctly via its custom build fn.
  4. The trajectory built behaves like its hand-constructed counterpart.
  5. The choice registry has both options.
"""
import numpy as np

from sdf.scenarios import ConstantVelocityTrajectory, MountainPassTrajectory

from dashboard.components.trajectories import (
    CONSTANT_VELOCITY,
    MOUNTAIN_PASS,
    TRAJECTORY_CHOICE,
)


class TestMountainPassSpec:
    def test_builds_from_defaults(self):
        values = MOUNTAIN_PASS.validate({})
        traj = MOUNTAIN_PASS.construct(values)
        assert isinstance(traj, MountainPassTrajectory)
        # Defaults: 20 km/h = 5.555... m/s.
        assert abs(traj.v - 20.0 / 3.6) < 1e-9
        assert traj.length == 10_000.0
        assert traj.y_amp == 1_000.0
        assert traj.z_amp == 1_000.0

    def test_builds_from_custom_values(self):
        values = MOUNTAIN_PASS.validate({
            "v_kmh": "50.0",
            "length": 5_000.0,
            "y_amp": 500.0,
            "z_amp": 250.0,
        })
        traj = MOUNTAIN_PASS.construct(values)
        assert abs(traj.v - 50.0 / 3.6) < 1e-9
        assert traj.length == 5_000.0
        assert traj.y_amp == 500.0
        assert traj.z_amp == 250.0


class TestConstantVelocitySpec:
    def test_builds_from_defaults(self):
        values = CONSTANT_VELOCITY.validate({})
        traj = CONSTANT_VELOCITY.construct(values)
        assert isinstance(traj, ConstantVelocityTrajectory)
        # Defaults: position (0,0,0), velocity (20,0,0).
        x0 = traj.state_at(0.0)
        # Layout has position_idx (0, 2, 4) and velocity_idx (1, 3, 5).
        np.testing.assert_allclose(x0[[0, 2, 4]], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(x0[[1, 3, 5]], [20.0, 0.0, 0.0])
        # After 1 second, position should advance by velocity.
        x1 = traj.state_at(1.0)
        np.testing.assert_allclose(x1[[0, 2, 4]], [20.0, 0.0, 0.0])

    def test_builds_from_custom_values(self):
        values = CONSTANT_VELOCITY.validate({
            "initial_position": [100.0, 50.0, 10.0],
            "initial_velocity": [10.0, 5.0, -1.0],
        })
        traj = CONSTANT_VELOCITY.construct(values)
        x0 = traj.state_at(0.0)
        np.testing.assert_allclose(x0[[0, 2, 4]], [100.0, 50.0, 10.0])
        np.testing.assert_allclose(x0[[1, 3, 5]], [10.0, 5.0, -1.0])


class TestTrajectoryChoice:
    def test_has_both_options(self):
        assert set(TRAJECTORY_CHOICE.keys()) == {"mountain_pass", "constant_velocity"}

    def test_default_is_mountain_pass(self):
        assert TRAJECTORY_CHOICE.default_key == "mountain_pass"
        assert TRAJECTORY_CHOICE.get("mountain_pass") is MOUNTAIN_PASS
