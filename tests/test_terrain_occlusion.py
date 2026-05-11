"""
Tests for TerrainOcclusion / TunnelOcclusion.

Verify:
  1. Constructor validation (range ordering, positive radius, in-road range).
  2. A target before the tunnel entry is not occluded.
  3. A target inside the tunnel range AND inside the radius is occluded.
  4. A target inside the tunnel range but outside the radius is NOT
     occluded (it's above the structure / off the road).
  5. A target past the tunnel exit is not occluded.
  6. The deterministic test is independent of rng (same answer for any
     rng, including None).
  7. Tunnel works on a curved (multi-segment) road, not just a straight one.
  8. Composition: TunnelOcclusion stacks under CompositeOcclusion with
     DopplerBlindnessOcclusion — either model can trigger occlusion.
  9. Arc-length anchoring: the same tunnel definition produces consistent
     results when the road is rotated in the plane (sanity check that
     we don't accidentally rely on x-coordinates).
 10. Sensor integration: a CartesianPositionSensor with a tunnel returns
     None for measurements taken inside the tunnel.
"""
import numpy as np
import pytest

from sdf.core.state import StateLayout
from sdf.scenarios import PolygonalRoadMap
from sdf.sensors import (
    CartesianPositionSensor,
    CompositeOcclusion,
    DopplerBlindnessOcclusion,
    TerrainOcclusion,
    TunnelOcclusion,
)


def _straight_road(length: float = 1000.0, n: int = 11) -> PolygonalRoadMap:
    """A 2D straight road along the x-axis from x=0 to x=length."""
    xs = np.linspace(0.0, length, n)
    nodes = np.column_stack([xs, np.zeros_like(xs)])
    return PolygonalRoadMap(nodes=nodes, sigma_nodes=1.0)


def _curved_road(radius: float = 500.0, n: int = 21) -> PolygonalRoadMap:
    """A 2D quarter-circle arc of given radius, centred at the origin."""
    theta = np.linspace(0.0, np.pi / 2, n)
    nodes = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    # Explicit arc length: a true circular arc has length = radius * dtheta.
    arc = np.concatenate([[0.0], np.cumsum(np.full(n - 1, radius * (np.pi / 2) / (n - 1)))])
    return PolygonalRoadMap(nodes=nodes, arc_lengths=arc, sigma_nodes=1.0)


class TestTunnelOcclusionValidation:
    def setup_method(self):
        self.road = _straight_road()

    def test_invalid_range_ordering(self):
        with pytest.raises(ValueError, match="must exceed"):
            TunnelOcclusion(self.road, l_in=200.0, l_out=100.0, radius=10.0)

    def test_zero_radius(self):
        with pytest.raises(ValueError, match="radius"):
            TunnelOcclusion(self.road, l_in=100.0, l_out=200.0, radius=0.0)

    def test_range_outside_road(self):
        with pytest.raises(ValueError, match="outside road"):
            TunnelOcclusion(self.road, l_in=-10.0, l_out=200.0, radius=10.0)
        with pytest.raises(ValueError, match="outside road"):
            TunnelOcclusion(self.road, l_in=100.0, l_out=20_000.0, radius=10.0)

    def test_is_terrain_occlusion(self):
        # Sanity: TunnelOcclusion is-a TerrainOcclusion, so downstream code
        # can dispatch on the category.
        tunnel = TunnelOcclusion(self.road, l_in=100.0, l_out=200.0, radius=10.0)
        assert isinstance(tunnel, TerrainOcclusion)


class TestTunnelOcclusionGeometry:
    def setup_method(self):
        self.road = _straight_road(length=1000.0, n=11)
        self.tunnel = TunnelOcclusion(
            self.road, l_in=400.0, l_out=600.0, radius=10.0
        )
        self.layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

    def _state(self, x: float, y: float) -> np.ndarray:
        return np.array([x, 0.0, y, 0.0])

    def test_before_tunnel_not_occluded(self):
        assert not self.tunnel.is_occluded(self._state(100.0, 0.0), self.layout)

    def test_inside_tunnel_on_road_occluded(self):
        # At x=500, y=0: well inside the [400, 600] range, on the road.
        assert self.tunnel.is_occluded(self._state(500.0, 0.0), self.layout)

    def test_inside_range_but_above_tube_not_occluded(self):
        # Inside x-range but 20m off the road, with radius 10 — outside the tube.
        assert not self.tunnel.is_occluded(self._state(500.0, 20.0), self.layout)

    def test_past_tunnel_not_occluded(self):
        assert not self.tunnel.is_occluded(self._state(700.0, 0.0), self.layout)

    def test_at_tunnel_boundary_inclusive(self):
        # Entry and exit points themselves should be occluded (inclusive range).
        assert self.tunnel.is_occluded(self._state(400.0, 0.0), self.layout)
        assert self.tunnel.is_occluded(self._state(600.0, 0.0), self.layout)


class TestTunnelOcclusionRngIndependence:
    def test_deterministic_across_rngs(self):
        road = _straight_road()
        tunnel = TunnelOcclusion(road, l_in=300.0, l_out=500.0, radius=10.0)
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        x = np.array([400.0, 0.0, 0.0, 0.0])

        results = {
            tunnel.is_occluded(x, layout, rng=None),
            tunnel.is_occluded(x, layout, rng=np.random.default_rng(0)),
            tunnel.is_occluded(x, layout, rng=np.random.default_rng(123)),
        }
        # All three branches must collapse to the same boolean.
        assert results == {True}


class TestTunnelOcclusionCurvedRoad:
    def test_inside_curve(self):
        road = _curved_road(radius=500.0, n=21)
        # Tunnel covers the middle of the arc — arc-length range that
        # straddles the pi/4 point of the quarter circle.
        total = road.total_length()
        tunnel = TunnelOcclusion(
            road, l_in=total * 0.4, l_out=total * 0.6, radius=15.0
        )
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

        # A target at the midpoint of the arc, on the road.
        theta_mid = np.pi / 4
        x = 500.0 * np.cos(theta_mid)
        y = 500.0 * np.sin(theta_mid)
        state = np.array([x, 0.0, y, 0.0])
        assert tunnel.is_occluded(state, layout)

    def test_outside_curve(self):
        road = _curved_road(radius=500.0, n=21)
        total = road.total_length()
        tunnel = TunnelOcclusion(
            road, l_in=total * 0.4, l_out=total * 0.6, radius=15.0
        )
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

        # A target at the start of the arc, before the tunnel range.
        state = np.array([500.0, 0.0, 0.0, 0.0])
        assert not tunnel.is_occluded(state, layout)


class TestTunnelOcclusionInvariance:
    def test_rotation_invariance(self):
        """
        The same arc-length tunnel definition on two rotated copies of
        the same road must give the same answer for rotated copies of
        the same target. This is a sanity check that arc-length anchoring
        decouples the tunnel from world-frame x/y coordinates.
        """
        road_a = _straight_road(length=1000.0, n=11)
        # Rotate the nodes 30 degrees about the origin.
        theta = np.pi / 6
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        road_b = PolygonalRoadMap(nodes=road_a.nodes @ rot.T, sigma_nodes=1.0)

        tunnel_a = TunnelOcclusion(road_a, l_in=400.0, l_out=600.0, radius=10.0)
        tunnel_b = TunnelOcclusion(road_b, l_in=400.0, l_out=600.0, radius=10.0)

        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        pos_a = np.array([500.0, 0.0])
        pos_b = rot @ pos_a
        state_a = np.array([pos_a[0], 0.0, pos_a[1], 0.0])
        state_b = np.array([pos_b[0], 0.0, pos_b[1], 0.0])

        assert tunnel_a.is_occluded(state_a, layout)
        assert tunnel_b.is_occluded(state_b, layout)


class TestTunnelOcclusionComposition:
    def test_composite_with_doppler(self):
        """
        A CompositeOcclusion that ORs a TunnelOcclusion and a
        DopplerBlindnessOcclusion should fire if either model would.
        """
        road = _straight_road()
        tunnel = TunnelOcclusion(road, l_in=400.0, l_out=600.0, radius=10.0)
        doppler = DopplerBlindnessOcclusion(
            sensor_position=np.array([0.0, -1000.0]), mdv=3.0, pd_floor=0.05
        )
        composite = CompositeOcclusion([tunnel, doppler])
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))

        # Inside the tunnel: tunnel triggers regardless of Doppler.
        in_tunnel = np.array([500.0, 30.0, 0.0, 0.0])
        assert composite.is_occluded(in_tunnel, layout, rng=np.random.default_rng(0))

        # Outside tunnel, moving radially (fast). Doppler factor near 1,
        # so composite should not occlude — but the result is rng-dependent
        # since Doppler is probabilistic. Use many draws and require *some*
        # of them to be non-occluded.
        outside = np.array([100.0, 100.0, 100.0, 0.0])
        results = [
            composite.is_occluded(outside, layout, rng=np.random.default_rng(k))
            for k in range(20)
        ]
        assert any(not r for r in results)


class TestTunnelOcclusionSensorIntegration:
    def test_cartesian_sensor_with_tunnel_returns_none_inside(self):
        """
        End-to-end: a CartesianPositionSensor with a TunnelOcclusion
        produces None for measurements inside the tunnel, and a real
        measurement outside it.
        """
        road = _straight_road(length=1000.0, n=11)
        tunnel = TunnelOcclusion(road, l_in=400.0, l_out=600.0, radius=10.0)
        sensor = CartesianPositionSensor(
            sensor_id="cart",
            dim=2,
            noise_std=2.0,
            detection_prob=1.0,
            occlusion_model=tunnel,
        )
        layout = StateLayout(dim=2, position_idx=(0, 2), velocity_idx=(1, 3))
        rng = np.random.default_rng(42)

        inside = np.array([500.0, 0.0, 0.0, 0.0])
        outside = np.array([100.0, 0.0, 0.0, 0.0])

        assert sensor.measure(inside, layout, t=1.0, rng=rng) is None
        m = sensor.measure(outside, layout, t=1.0, rng=rng)
        assert m is not None
        assert m.sensor_id == "cart"
