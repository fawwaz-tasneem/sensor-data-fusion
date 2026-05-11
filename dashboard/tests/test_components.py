"""
Smoke tests for the remaining component specs.

Per the agreed lighter testing policy: each spec is verified to build a
working framework object from its declared defaults. Per-parameter
validation paths are covered by the schema-layer tests; per-spec
parametrised testing is deferred.
"""
import numpy as np

from sdf.motion_models import (
    ConstantAcceleration,
    ConstantVelocity,
    CoordinatedTurn,
    CoordinatedTurnUnknown,
)
from sdf.scenarios.awacs import CircleFlight, RacetrackFlight, StraightFlight
from sdf.scenarios.platform import StationaryPlatform
from sdf.sensors import (
    CartesianPositionSensor,
    DopplerBlindnessOcclusion,
    GMTIRadarSensor,
    RadarSensor,
)

from dashboard.components import (
    MOTION_MODEL_CHOICE,
    OCCLUSION_CHOICE,
    PLATFORM_CHOICE,
    ROAD_MAP,
    SENSOR_CHOICE,
)


def _build_defaults(spec):
    """Validate empty form values (filling in defaults) and construct."""
    return spec.construct(spec.validate({}))


class TestMotionModelSpecs:
    def test_cv_builds(self):
        m = _build_defaults(MOTION_MODEL_CHOICE.get("cv"))
        assert isinstance(m, ConstantVelocity)

    def test_ca_builds(self):
        m = _build_defaults(MOTION_MODEL_CHOICE.get("ca"))
        assert isinstance(m, ConstantAcceleration)

    def test_ct_known_builds(self):
        m = _build_defaults(MOTION_MODEL_CHOICE.get("ct_known"))
        assert isinstance(m, CoordinatedTurn)

    def test_ct_unknown_builds(self):
        m = _build_defaults(MOTION_MODEL_CHOICE.get("ct_unknown"))
        assert isinstance(m, CoordinatedTurnUnknown)


class TestSensorSpecs:
    def test_cartesian_builds(self):
        s = _build_defaults(SENSOR_CHOICE.get("cartesian"))
        assert isinstance(s, CartesianPositionSensor)

    def test_radar_builds(self):
        s = _build_defaults(SENSOR_CHOICE.get("radar"))
        assert isinstance(s, RadarSensor)
        # Position should arrive as ndarray, not list.
        assert isinstance(s.position, np.ndarray)

    def test_gmti_builds(self):
        s = _build_defaults(SENSOR_CHOICE.get("gmti"))
        assert isinstance(s, GMTIRadarSensor)


class TestOcclusionSpecs:
    def test_none_builds(self):
        result = _build_defaults(OCCLUSION_CHOICE.get("none"))
        assert result is None

    def test_doppler_builds(self):
        # DopplerBlindnessOcclusion needs sensor_position; spec exposes only
        # mdv and pd_floor, so a direct build from defaults will miss the
        # required arg. That's expected: the runner supplies it. Here we
        # just verify the spec defaults dict is well-formed.
        spec = OCCLUSION_CHOICE.get("doppler")
        defaults = spec.defaults()
        assert defaults == {"mdv": 3.0, "pd_floor": 0.05}

    def test_tunnel_defaults(self):
        # TunnelOcclusion needs road and resolved l_in / l_out (m, not
        # fractions). Spec exposes l_in_frac, l_out_frac, radius. Runner
        # combines those with the road map. Verify the fraction defaults
        # are sane.
        spec = OCCLUSION_CHOICE.get("tunnel")
        defaults = spec.defaults()
        assert 0.0 <= defaults["l_in_frac"] < defaults["l_out_frac"] <= 1.0
        assert defaults["radius"] > 0.0


class TestPlatformSpecs:
    def test_stationary_builds(self):
        p = _build_defaults(PLATFORM_CHOICE.get("stationary"))
        assert isinstance(p, StationaryPlatform)

    def test_straight_builds(self):
        p = _build_defaults(PLATFORM_CHOICE.get("straight"))
        assert isinstance(p, StraightFlight)

    def test_circle_builds(self):
        p = _build_defaults(PLATFORM_CHOICE.get("circle"))
        assert isinstance(p, CircleFlight)

    def test_racetrack_builds(self):
        p = _build_defaults(PLATFORM_CHOICE.get("racetrack"))
        assert isinstance(p, RacetrackFlight)


class TestRoadMapSpec:
    def test_road_map_defaults(self):
        # Like Tunnel, RoadMap construction depends on the trajectory
        # (the runner samples it). Verify the spec's own defaults are sane.
        defaults = ROAD_MAP.defaults()
        assert defaults["n_nodes"] >= 5
        assert defaults["sigma_nodes"] > 0.0
