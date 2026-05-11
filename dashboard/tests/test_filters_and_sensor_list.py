"""
Smoke tests for filter specs and the SensorListSpec.

Filters don't construct directly from defaults (they need cross-component
dependencies), so we only verify the spec's defaults are well-formed.
The SensorListSpec, by contrast, *does* round-trip end-to-end —
defaults → validate → build → working sensor objects.
"""
import numpy as np

from sdf.sensors import RadarSensor

from dashboard.components import FILTER_CHOICE, SENSOR_LIST


class TestFilterSpecs:
    def test_all_choices_present(self):
        assert set(FILTER_CHOICE.keys()) == {"kf", "ekf", "road_aided_ekf", "imm"}

    def test_kf_defaults(self):
        spec = FILTER_CHOICE.get("kf")
        d = spec.defaults()
        assert "prior_position_offset" in d
        assert "prior_position_sigma" in d
        assert "prior_velocity_sigma" in d

    def test_imm_has_subfilter_params(self):
        spec = FILTER_CHOICE.get("imm")
        d = spec.defaults()
        # All three sub-models' noises plus the TPM diagonal.
        for required in ("cv_process_noise_std", "ca_jerk_std",
                         "ct_omega", "ct_process_noise_std", "tpm_self_prob"):
            assert required in d, f"missing {required} in IMM defaults"
        # TPM self-probability should be in [0.5, 1.0).
        assert 0.5 <= d["tpm_self_prob"] < 1.0


class TestSensorListSpec:
    def test_default_is_two_radars(self):
        defaults = SENSOR_LIST.defaults()
        assert len(defaults) == 2
        assert all(e["type"] == "radar" for e in defaults)
        # Both at z=100 m, one at y=10000, the other at x=10000.
        positions = [e["params"]["position"] for e in defaults]
        assert positions[0] == [0.0, 10_000.0, 100.0]
        assert positions[1] == [10_000.0, 0.0, 100.0]

    def test_validate_passes_for_defaults(self):
        defaults = SENSOR_LIST.defaults()
        validated = SENSOR_LIST.validate(defaults)
        # Validation should round-trip the structure.
        assert len(validated) == 2
        for entry in validated:
            assert "type" in entry and "params" in entry

    def test_build_returns_sensor_objects(self):
        defaults = SENSOR_LIST.defaults()
        sensors = SENSOR_LIST.build(defaults)
        assert len(sensors) == 2
        for s in sensors:
            assert isinstance(s, RadarSensor)
            assert isinstance(s.position, np.ndarray)

    def test_build_empty_list(self):
        assert SENSOR_LIST.build([]) == []

    def test_validate_rejects_malformed_entry(self):
        import pytest
        with pytest.raises(ValueError, match="type"):
            SENSOR_LIST.validate([{"params": {}}])  # missing 'type'

    def test_validate_rejects_unknown_type(self):
        import pytest
        with pytest.raises(KeyError):
            SENSOR_LIST.validate([{"type": "lidar_xyz", "params": {}}])
