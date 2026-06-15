"""
Smoke tests for the simulation runner.

These tests exercise the full pipeline:
  defaults from each ComponentSpec → assembled config dict → runner →
  SimulationResult with sensible arrays.

We don't assert on tracking quality (that's the framework's tests' job);
we verify the runner produces well-shaped output and doesn't crash on
the canonical scenarios.
"""
import numpy as np
import pytest

from dashboard.components import (
    FILTER_CHOICE,
    MOTION_MODEL_CHOICE,
    OCCLUSION_CHOICE,
    ROAD_MAP,
    SENSOR_LIST,
    TRAJECTORY_CHOICE,
)
from dashboard.simulation import SimulationResult, run_simulation


def _default_config(
    trajectory_key="mountain_pass",
    motion_model_key="cv",
    filter_key="ekf",
    occlusion_key="none",
    road_map_enabled=False,
):
    """Build a minimal valid config dict from spec defaults."""
    traj_spec = TRAJECTORY_CHOICE.get(trajectory_key)
    mm_spec = MOTION_MODEL_CHOICE.get(motion_model_key)
    occ_spec = OCCLUSION_CHOICE.get(occlusion_key)
    flt_spec = FILTER_CHOICE.get(filter_key)

    return {
        "trajectory": {"type": trajectory_key,
                       "params": traj_spec.validate({})},
        "motion_model": {"type": motion_model_key,
                         "params": mm_spec.validate({})},
        "sensor_list": SENSOR_LIST.validate(SENSOR_LIST.defaults()),
        "occlusion": {"type": occlusion_key,
                      "params": occ_spec.validate({})},
        "filter": {"type": filter_key,
                   "params": flt_spec.validate({})},
        "road_map": {"enabled": road_map_enabled,
                     "params": ROAD_MAP.validate({})},
        "sim": {"seed": 42, "dt": 2.0},
    }


class TestSimulationRunner:
    def test_default_mountain_pass_ekf(self):
        config = _default_config()
        result = run_simulation(config)
        assert isinstance(result, SimulationResult)
        T = len(result.times)
        assert T > 100  # 1800s / 2s = 900 steps approximately
        assert result.truth_positions.shape == (T, 3)
        assert result.estimate_positions.shape == (T, 3)
        assert result.sensor_positions.shape == (T, 2, 3)
        assert result.sensor_detected.shape == (T, 2)
        # No road map → no road nodes.
        assert result.road_nodes is None

    def test_with_road_map(self):
        config = _default_config(road_map_enabled=True)
        result = run_simulation(config)
        assert result.road_nodes is not None
        assert result.road_nodes.shape == (30, 3)  # default n_nodes=30

    def test_road_map_enables_road_aiding(self):
        # Enabling the road map IS the road-aided switch: the estimate should
        # track well (the fictitious cross-track measurement is fused).
        config = _default_config(filter_key="ekf", road_map_enabled=True)
        result = run_simulation(config)
        err = np.linalg.norm(
            result.truth_positions - result.estimate_positions, axis=1
        )
        assert np.median(err[10:]) < 200.0

    def test_road_map_keeps_estimate_on_road_through_tunnel(self):
        # With the road map enabled, road-aiding is on, so through a tunnel the
        # estimate stays ON the road (small cross-track) rather than coasting
        # off it. Without the road map there is no constraint, so it wanders.
        from dashboard.simulation import _build_road_map, _build_trajectory

        cfg_on = _default_config(occlusion_key="tunnel", road_map_enabled=True)
        road = _build_road_map(_build_trajectory(cfg_on), cfg_on)
        on = run_simulation(cfg_on)
        gap = ~on.sensor_detected.any(axis=1)
        ct = np.array([road.closest_segment(p[:3])[2]
                       for p in on.estimate_positions])
        assert ct[gap].max() < 100.0

    def test_tunnel_occlusion_produces_blackouts(self):
        config = _default_config(occlusion_key="tunnel",
                                 road_map_enabled=True)
        result = run_simulation(config)
        # Some scans should have zero detections across both sensors.
        n_blackouts = int((result.sensor_detected.sum(axis=1) == 0).sum())
        assert n_blackouts > 0
        # Tunnel wireframe segments should be present.
        assert result.tunnel_segments is not None
        assert len(result.tunnel_segments) > 0

    def test_tunnel_without_road_map_runs_and_shows_failure(self):
        # A tunnel needs the road *geometry* to position itself, which the
        # runner builds on demand — but with the road map (road-aiding) OFF,
        # the filter is NOT constrained, so through the tunnel it coasts off
        # the road. The run must succeed (so the failure is visible), not error.
        from dashboard.simulation import _build_road_map, _build_trajectory

        config = _default_config(occlusion_key="tunnel",
                                 road_map_enabled=False)
        result = run_simulation(config)
        # The tunnel produced a blackout, and the geometry was built to draw it.
        assert (result.sensor_detected.sum(axis=1) == 0).sum() > 0
        assert result.tunnel_segments is not None
        # Road-aiding is OFF, so the estimate wanders off the road in the gap.
        road = _build_road_map(_build_trajectory(config), config)
        gap = ~result.sensor_detected.any(axis=1)
        ct = np.array([road.closest_segment(p[:3])[2]
                       for p in result.estimate_positions])
        assert ct[gap].max() > 100.0


class TestGmtiPlatformAndClutter:
    """Moving GMTI platforms and ground-truth clutter-notch reporting."""

    def _gmti_entry(self, platform="stationary", **extra):
        params = {
            "sensor_id": "gmti_a", "position": [0.0, 0.0, 1000.0],
            "range_std": 80.0, "bearing_std": 8e-3, "elevation_std": 8e-3,
            "range_rate_std": 0.5, "detection_prob": 1.0, "platform": platform,
            "platform_velocity": [150.0, 0.0, 0.0], "platform_speed": 150.0,
            "platform_radius": 5000.0, "platform_leg_length": 10000.0,
        }
        params.update(extra)
        return [{"type": "gmti", "params": params}]

    def _config(self, sensor_list, occlusion_key="none"):
        config = _default_config(occlusion_key=occlusion_key)
        config["sensor_list"] = SENSOR_LIST.validate(sensor_list)
        return config

    def test_moving_gmti_position_advances(self):
        # A circle-flight GMTI must change position over time (the platform is
        # actually driven through set_time during measurement).
        result = run_simulation(self._config(self._gmti_entry("circle")))
        first = result.sensor_positions[0, 0]
        last = result.sensor_positions[-1, 0]
        assert np.linalg.norm(last - first) > 1000.0

    def test_stationary_gmti_does_not_move(self):
        result = run_simulation(self._config(self._gmti_entry("stationary")))
        spread = result.sensor_positions[:, 0].std(axis=0)
        assert np.allclose(spread, 0.0)

    def test_clutter_factor_reported_and_predicts_misses(self):
        # With a Doppler clutter notch, the ground-truth factor must be stored,
        # and detections must be far more likely when clear than in the notch.
        result = run_simulation(
            self._config(self._gmti_entry("racetrack"), occlusion_key="doppler"))
        assert result.clutter_factor is not None
        cf = result.clutter_factor[:, 0]
        det = result.sensor_detected[:, 0]
        assert not np.all(np.isnan(cf))
        clear, notch = cf > 0.9, cf < 0.3
        # Only assert if both regimes actually occur in this run.
        if clear.any() and notch.any():
            assert det[clear].mean() > det[notch].mean() + 0.5

    def test_no_clutter_factor_without_doppler(self):
        result = run_simulation(self._config(self._gmti_entry("stationary")))
        assert result.clutter_factor is None
