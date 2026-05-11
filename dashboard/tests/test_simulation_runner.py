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

    def test_road_aided_ekf_requires_road_map(self):
        import pytest
        config = _default_config(filter_key="road_aided_ekf",
                                 road_map_enabled=False)
        with pytest.raises(ValueError, match="road map"):
            run_simulation(config)

    def test_road_aided_ekf_with_road_map(self):
        config = _default_config(filter_key="road_aided_ekf",
                                 road_map_enabled=True)
        result = run_simulation(config)
        # Truth and estimate should be close-ish (within tens of metres
        # average for a default-noise mountain-pass run).
        err = np.linalg.norm(
            result.truth_positions - result.estimate_positions, axis=1
        )
        assert np.median(err[10:]) < 200.0

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

    def test_tunnel_without_road_map_falls_back_to_none(self):
        # User asks for tunnel but doesn't enable road map. Runner should
        # silently disable rather than crash.
        config = _default_config(occlusion_key="tunnel",
                                 road_map_enabled=False)
        result = run_simulation(config)
        # All scans should be detected (no occlusion model applied).
        n_blackouts = int((result.sensor_detected.sum(axis=1) == 0).sum())
        # The first step has no measurements by convention, but after that
        # both radars should always fire.
        assert n_blackouts <= 1
