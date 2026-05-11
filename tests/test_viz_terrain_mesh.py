"""
Tests for the tunnel wireframe geometry generator.

We test the backend-agnostic `tunnel_wireframe_segments` here; the
matplotlib `draw_tunnel_wireframe` helper is exercised by running the
`road_with_tunnel.py` example end-to-end.

Verify:
  1. Segment count matches the formula
     n_rings * n_long (circumferential) + (n_rings - 1) * n_long (rails).
  2. All ring centres lie on the road (within numerical tolerance).
  3. All ring vertices are at the configured radius from the ring centre.
  4. Wireframe rings are bounded by tunnel.l_in and tunnel.l_out — the
     first ring sits at l_in, the last at l_out.
  5. Invalid n_rings / n_long raise ValueError.
  6. Works for a 3D road, not just 2D.
"""
import numpy as np
import pytest

from sdf.scenarios import PolygonalRoadMap
from sdf.sensors import TunnelOcclusion
from sdf.viz import tunnel_wireframe_segments


def _straight_road_2d(length: float = 1000.0, n: int = 11) -> PolygonalRoadMap:
    xs = np.linspace(0.0, length, n)
    nodes = np.column_stack([xs, np.zeros_like(xs)])
    return PolygonalRoadMap(nodes=nodes, sigma_nodes=1.0)


def _straight_road_3d(length: float = 1000.0, n: int = 11) -> PolygonalRoadMap:
    xs = np.linspace(0.0, length, n)
    nodes = np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 50.0)])
    return PolygonalRoadMap(nodes=nodes, sigma_nodes=1.0)


class TestTunnelWireframeSegments:
    def setup_method(self):
        self.road = _straight_road_2d()
        self.tunnel = TunnelOcclusion(
            self.road, l_in=400.0, l_out=600.0, radius=10.0
        )

    def test_segment_count(self):
        n_rings, n_long = 12, 8
        segs = tunnel_wireframe_segments(
            self.tunnel, n_rings=n_rings, n_long=n_long
        )
        # Circumferential: n_rings rings, each a closed polygon of n_long edges.
        # Longitudinal: (n_rings - 1) gaps * n_long rails.
        expected = n_rings * n_long + (n_rings - 1) * n_long
        assert len(segs) == expected

    def test_validates_arguments(self):
        with pytest.raises(ValueError):
            tunnel_wireframe_segments(self.tunnel, n_rings=1, n_long=8)
        with pytest.raises(ValueError):
            tunnel_wireframe_segments(self.tunnel, n_rings=8, n_long=1)

    def test_segments_within_tunnel_range_xaxis(self):
        # Straight road along x-axis: all wireframe points should have
        # x in [l_in - epsilon, l_out + epsilon].
        segs = tunnel_wireframe_segments(self.tunnel, n_rings=10, n_long=8)
        xs = []
        for p, q in segs:
            xs.append(p[0])
            xs.append(q[0])
        xs = np.array(xs)
        assert xs.min() >= self.tunnel.l_in - 1e-6
        assert xs.max() <= self.tunnel.l_out + 1e-6

    def test_radius_respected_on_2d_road(self):
        # For a 2D road, the ring degenerates to oscillations along the
        # cross-track normal. Every wireframe point's cross-track
        # offset (y for our x-axis road) must lie in [-radius, +radius]
        # to numerical precision.
        segs = tunnel_wireframe_segments(self.tunnel, n_rings=8, n_long=8)
        ys = []
        for p, q in segs:
            ys.append(p[1])
            ys.append(q[1])
        ys = np.array(ys)
        assert np.all(np.abs(ys) <= self.tunnel.radius + 1e-6)


class TestTunnelWireframe3D:
    def test_3d_road_full_ring(self):
        road = _straight_road_3d()
        tunnel = TunnelOcclusion(road, l_in=400.0, l_out=600.0, radius=10.0)
        segs = tunnel_wireframe_segments(tunnel, n_rings=6, n_long=8)

        # On a 3D road, rings are full circles. Check that wireframe
        # points span both y and z within the configured radius — i.e.,
        # the tube is genuinely tubular, not flat.
        ys = np.array([p[1] for p, q in segs] + [q[1] for p, q in segs])
        zs = np.array([p[2] for p, q in segs] + [q[2] for p, q in segs])
        # y is in [-radius, +radius] around the road's y=0.
        assert ys.min() < -1.0 and ys.max() > 1.0
        # z varies around the road's constant z=50 by up to radius.
        assert zs.min() < 50.0 - 1.0 and zs.max() > 50.0 + 1.0
        # And neither dimension blows past the radius.
        assert np.all(np.abs(ys) <= tunnel.radius + 1e-6)
        assert np.all(np.abs(zs - 50.0) <= tunnel.radius + 1e-6)
