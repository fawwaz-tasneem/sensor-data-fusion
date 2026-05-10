"""
Tests for PolygonalRoadMap.

We verify:
  1. Construction from nodes computes tangents and discretization errors
     correctly.
  2. Closest-segment query agrees with a brute-force reference
     implementation in random configurations.
  3. Closest-segment endpoint clamping is correct (the foot stays on
     the segment, not its infinite extension).
  4. Cross-track normal basis is orthonormal and orthogonal to the
     tangent (in 2D and 3D).
  5. Discretization error sigma_d is non-negative and equals
     |arc_length - chord_length|.
"""
import numpy as np
import pytest

from sdf.scenarios import PolygonalRoadMap


class TestRoadMapConstruction:
    def test_simple_2d_road(self):
        nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        road = PolygonalRoadMap(nodes, sigma_nodes=2.0)
        assert len(road) == 2
        # First segment: along +x.
        np.testing.assert_allclose(road.segments[0].tangent, [1.0, 0.0])
        assert road.segments[0].length_euclid == pytest.approx(10.0)
        # Second segment: along +y.
        np.testing.assert_allclose(road.segments[1].tangent, [0.0, 1.0])

    def test_zero_discretization_when_arc_lengths_default(self):
        # If we don't pass arc_lengths, sigma_d should be zero on every segment.
        nodes = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
        road = PolygonalRoadMap(nodes)
        for seg in road.segments:
            assert seg.sigma_disc == 0.0

    def test_explicit_arc_length_yields_discretization_error(self):
        # Two collinear nodes 10 apart, but the *real* road covers 12 m
        # (curving between the nodes). Then sigma_d = 2 m.
        nodes = np.array([[0.0, 0.0], [10.0, 0.0]])
        road = PolygonalRoadMap(
            nodes, arc_lengths=np.array([0.0, 12.0]), sigma_nodes=1.0
        )
        assert road.segments[0].sigma_disc == pytest.approx(2.0)
        # sigma_r2 = sigma_node^2 + sigma_disc^2 = 1 + 4 = 5.
        assert road.segments[0].sigma_r2 == pytest.approx(5.0)

    def test_rejects_non_increasing_arc_lengths(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        with pytest.raises(ValueError, match="strictly increasing"):
            PolygonalRoadMap(nodes, arc_lengths=np.array([0.0, 5.0, 3.0]))

    def test_rejects_duplicate_nodes(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="zero-length"):
            PolygonalRoadMap(nodes)

    def test_3d_road(self):
        nodes = np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 5.0]]
        )
        road = PolygonalRoadMap(nodes)
        assert road.dim == 3
        assert len(road) == 2


class TestClosestSegment:
    def test_point_on_segment_yields_zero_distance(self):
        nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        road = PolygonalRoadMap(nodes)
        idx, foot, dist = road.closest_segment(np.array([5.0, 0.0]))
        assert idx == 0
        assert dist == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(foot, [5.0, 0.0])

    def test_point_off_segment_yields_perpendicular_foot(self):
        nodes = np.array([[0.0, 0.0], [10.0, 0.0]])
        road = PolygonalRoadMap(nodes)
        idx, foot, dist = road.closest_segment(np.array([5.0, 3.0]))
        assert idx == 0
        np.testing.assert_allclose(foot, [5.0, 0.0])
        assert dist == pytest.approx(3.0)

    def test_point_beyond_segment_clamps_to_endpoint(self):
        # Single segment from (0,0) to (10,0). Query a point at (15, 1):
        # the foot must be at (10, 0), the endpoint, NOT the perpendicular
        # foot on the infinite line (which would be (15, 0)).
        nodes = np.array([[0.0, 0.0], [10.0, 0.0]])
        road = PolygonalRoadMap(nodes)
        idx, foot, dist = road.closest_segment(np.array([15.0, 1.0]))
        assert idx == 0
        np.testing.assert_allclose(foot, [10.0, 0.0])
        # Distance is sqrt(5^2 + 1^2) = sqrt(26).
        assert dist == pytest.approx(np.sqrt(26.0))

    def test_picks_correct_segment_in_l_shape(self):
        # L-shaped road: (0,0) -> (10,0) -> (10,10).
        # Query at (12, 5): closest to second segment, foot = (10, 5).
        nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        road = PolygonalRoadMap(nodes)
        idx, foot, dist = road.closest_segment(np.array([12.0, 5.0]))
        assert idx == 1
        np.testing.assert_allclose(foot, [10.0, 5.0])

    def test_brute_force_agreement_random(self):
        # Random road with 5 nodes; for 200 random queries, verify the
        # closest-distance matches a brute-force scan over all segments
        # using sampled points. We compare distances rather than indices
        # because at segment-shared endpoints, multiple segments can be
        # tied for closest; both methods are correct in that case.
        rng = np.random.default_rng(0)
        nodes = rng.uniform(-50, 50, size=(5, 2))
        road = PolygonalRoadMap(nodes)
        for _ in range(200):
            p = rng.uniform(-100, 100, size=2)
            idx, foot, dist = road.closest_segment(p)
            # Reference: densely sample each segment, find min.
            best_d = np.inf
            for s_idx, seg in enumerate(road.segments):
                ts = np.linspace(0, 1, 500)
                pts = seg.s_start + ts[:, None] * (seg.s_end - seg.s_start)
                d_min = np.min(np.linalg.norm(pts - p, axis=1))
                if d_min < best_d:
                    best_d = d_min
            # Distance should agree with brute force (sampling has finite
            # resolution, so 0.5 m tolerance is generous).
            assert dist == pytest.approx(best_d, abs=0.5)
            # Selected segment's distance should also be optimal.
            seg_dist = np.linalg.norm(p - foot)
            assert seg_dist == pytest.approx(dist, abs=1e-9)


class TestCrossTrackNormals:
    def test_2d_normal_is_unit_and_perpendicular(self):
        nodes = np.array([[0.0, 0.0], [3.0, 4.0]])
        road = PolygonalRoadMap(nodes)
        N = road.cross_track_normals(0)
        assert N.shape == (1, 2)
        n = N[0]
        # Unit length.
        assert np.linalg.norm(n) == pytest.approx(1.0)
        # Perpendicular to tangent.
        assert n @ road.segments[0].tangent == pytest.approx(0.0, abs=1e-12)

    def test_3d_normals_are_orthonormal_basis_of_perpendicular_plane(self):
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        road = PolygonalRoadMap(nodes)
        N = road.cross_track_normals(0)
        t = road.segments[0].tangent
        assert N.shape == (2, 3)
        # Each row unit length.
        for row in N:
            assert np.linalg.norm(row) == pytest.approx(1.0)
        # Each row perpendicular to tangent.
        for row in N:
            assert row @ t == pytest.approx(0.0, abs=1e-12)
        # Two rows mutually perpendicular.
        assert N[0] @ N[1] == pytest.approx(0.0, abs=1e-12)

    def test_3d_normals_handle_axis_aligned_tangent(self):
        # Tangent = +x; shouldn't crash on the global-axis fallback.
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        road = PolygonalRoadMap(nodes)
        N = road.cross_track_normals(0)
        t = road.segments[0].tangent
        for row in N:
            assert np.linalg.norm(row) == pytest.approx(1.0)
            assert row @ t == pytest.approx(0.0, abs=1e-12)
