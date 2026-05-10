"""
PolygonalRoadMap: Koch's polygonal road model (Tracking and Sensor Data
Fusion, Sec. 9.1.1).

A road is a continuous 3D curve R*(l) parametrized by arc length l. We
approximate it by a polygonal curve R(l) defined by node vectors s_m
and tangent vectors t_m:

    R(l) = sum_m [ s_m + (l - l_m) * t_m ] * chi_m(l)

where chi_m(l) is the indicator of segment m.

Each segment m carries:
  * s_m, s_{m+1}     : Cartesian endpoints (positions of the nodes)
  * t_m              : unit tangent (s_{m+1} - s_m) / ||s_{m+1} - s_m||
  * lambda_m         : arc-length distance covered on segment m
                       (in general, lambda_m >= ||s_{m+1} - s_m|| because
                       the real road may curve between nodes)
  * sigma_m          : 1-sigma uncertainty of node position s_m (mapping)
  * sigma_d_m        : 1-sigma discretization error
                       sigma_d_m = | lambda_m - ||s_{m+1} - s_m|| |
  * sigma_r_m^2      : total cross-track variance = sigma_m^2 + sigma_d_m^2
                       (Koch eq. on page 24 / Sec. 9.1.1)

Critically, the map is independent of the ground truth. Build a
PolygonalRoadMap from surveyed nodes (or a coarse sampling of any
function), and the discretization error sigma_d_m emerges naturally
from the gap between straight-line distance ||s_{m+1} - s_m|| and the
arc length lambda_m that a vehicle would actually travel.

Closest-segment query: given a predicted Cartesian position p, find
the segment m* whose perpendicular projection of p onto the segment
line minimizes Euclidean distance, with the projection clamped to
the segment's endpoints. This is the "single-hypothesis" approximation
of the full Bayesian segment-marginalization (§9.1.2 of Koch's book).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoadSegment:
    """A single edge of the polygonal road, with Koch's error parameters."""

    s_start: np.ndarray  # node at the segment's start
    s_end: np.ndarray  # node at the segment's end
    tangent: np.ndarray  # unit vector along the segment
    length_euclid: float  # ||s_end - s_start||
    arc_length: float  # lambda_m, may exceed length_euclid
    sigma_node: float  # mapping uncertainty of the nodes
    sigma_disc: float  # discretization error |lambda - length_euclid|

    @property
    def sigma_r2(self) -> float:
        """Total cross-track variance for this segment."""
        return self.sigma_node**2 + self.sigma_disc**2


class PolygonalRoadMap:
    """A polygonal approximation of a road, with Koch's error model."""

    def __init__(
        self,
        nodes: np.ndarray,
        arc_lengths: np.ndarray | None = None,
        sigma_nodes: np.ndarray | float = 1.0,
    ):
        """
        Build a polygonal road from a sequence of node positions.

        Parameters
        ----------
        nodes : (N, dim) array
            The node positions s_0, s_1, ..., s_{N-1}. dim is 2 or 3.
        arc_lengths : (N,) array, optional
            Cumulative arc length at each node (l_0, l_1, ..., l_{N-1}).
            If None, defaults to cumulative Euclidean distance — i.e.,
            zero discretization error. Provide an explicit array when
            you know the real road is longer than the polygonal chord
            (e.g., the polygonal map under-samples a curve).
        sigma_nodes : scalar or (N,) array
            Mapping accuracy of each node. Scalar broadcasts to all nodes.
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError(
                f"nodes must be (N, 2) or (N, 3), got shape {nodes.shape}"
            )
        if nodes.shape[0] < 2:
            raise ValueError(f"need >= 2 nodes, got {nodes.shape[0]}")

        self.nodes = nodes
        self.dim = nodes.shape[1]

        # Vectorized: chord lengths between consecutive nodes.
        diffs = np.diff(nodes, axis=0)
        chord_lengths = np.linalg.norm(diffs, axis=1)
        if np.any(chord_lengths < 1e-9):
            raise ValueError("road has zero-length segment(s); duplicate nodes?")

        # Default cumulative arc length is the cumulative chord length
        # (i.e., assume the polygonal approximation is exact).
        if arc_lengths is None:
            arc_lengths = np.concatenate([[0.0], np.cumsum(chord_lengths)])
        else:
            arc_lengths = np.asarray(arc_lengths, dtype=float)
            if arc_lengths.shape != (nodes.shape[0],):
                raise ValueError(
                    f"arc_lengths shape {arc_lengths.shape} != ({nodes.shape[0]},)"
                )
            if not np.all(np.diff(arc_lengths) > 0):
                raise ValueError("arc_lengths must be strictly increasing")

        if np.isscalar(sigma_nodes):
            sigma_nodes = np.full(nodes.shape[0], float(sigma_nodes))
        else:
            sigma_nodes = np.asarray(sigma_nodes, dtype=float)
            if sigma_nodes.shape != (nodes.shape[0],):
                raise ValueError(
                    f"sigma_nodes shape {sigma_nodes.shape} != ({nodes.shape[0]},)"
                )

        self.arc_lengths = arc_lengths
        self.sigma_nodes = sigma_nodes

        # Build segment objects up-front.
        self.segments: list[RoadSegment] = []
        for i in range(nodes.shape[0] - 1):
            seg_arc = arc_lengths[i + 1] - arc_lengths[i]
            chord = chord_lengths[i]
            # The discretization error is what's left over: the part of the
            # real road's length that the chord doesn't capture.
            sigma_d = abs(seg_arc - chord)
            tangent = diffs[i] / chord
            # Average the two adjacent nodes' uncertainties for this segment
            # (the lecture's R_m on page 15 is per-node; for cross-track
            # likelihood we need a per-segment value, and the average is
            # a defensible aggregation).
            sigma_node = 0.5 * (sigma_nodes[i] + sigma_nodes[i + 1])
            self.segments.append(
                RoadSegment(
                    s_start=nodes[i].copy(),
                    s_end=nodes[i + 1].copy(),
                    tangent=tangent,
                    length_euclid=chord,
                    arc_length=seg_arc,
                    sigma_node=sigma_node,
                    sigma_disc=sigma_d,
                )
            )

    # ----- Geometry helpers --------------------------------------------

    def closest_segment(
        self, position: np.ndarray
    ) -> tuple[int, np.ndarray, float]:
        """
        Find the segment whose nearest point is closest to `position`.

        Returns
        -------
        idx : int
            Index of the closest segment.
        foot : (dim,) array
            The closest point on that segment (perpendicular foot, clamped
            to the segment's endpoints).
        dist : float
            Euclidean distance from position to foot.
        """
        position = np.asarray(position, dtype=float)
        if position.shape != (self.dim,):
            raise ValueError(
                f"position has shape {position.shape}, expected ({self.dim},)"
            )

        best_idx = -1
        best_dist2 = np.inf
        best_foot = np.zeros(self.dim)
        for i, seg in enumerate(self.segments):
            foot = self._project_onto_segment(position, seg)
            d2 = np.sum((position - foot) ** 2)
            if d2 < best_dist2:
                best_dist2 = d2
                best_idx = i
                best_foot = foot
        return best_idx, best_foot, float(np.sqrt(best_dist2))

    @staticmethod
    def _project_onto_segment(p: np.ndarray, seg: RoadSegment) -> np.ndarray:
        """Perpendicular projection of p onto segment, clamped to endpoints."""
        # Parametric foot: s_start + u * (s_end - s_start), u in [0, 1].
        v = seg.s_end - seg.s_start
        # u = (p - s_start) . v / ||v||^2; here ||v|| = length_euclid.
        u = np.dot(p - seg.s_start, v) / (seg.length_euclid**2)
        u = float(np.clip(u, 0.0, 1.0))
        return seg.s_start + u * v

    def cross_track_normals(self, segment_idx: int) -> np.ndarray:
        """
        Return an orthonormal basis of the cross-track subspace for
        segment `segment_idx`.

        For a 2D road this is a single unit vector perpendicular to the
        tangent; for a 3D road it is two unit vectors spanning the plane
        perpendicular to the tangent.

        Shape: (dim - 1, dim).
        """
        t = self.segments[segment_idx].tangent
        if self.dim == 2:
            # Rotate tangent by 90 degrees: (tx, ty) -> (-ty, tx).
            n = np.array([-t[1], t[0]])
            return n.reshape(1, 2)

        # 3D: pick any vector not parallel to t, then Gram-Schmidt twice.
        # Choose the global axis least aligned with t to avoid degeneracy.
        global_axes = np.eye(3)
        # Indices ordered by abs(dot product) with t — least aligned first.
        order = np.argsort(np.abs(global_axes @ t))
        ref = global_axes[order[0]]
        n1 = ref - (ref @ t) * t
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(t, n1)
        # n2 is already unit since t and n1 are orthonormal.
        return np.stack([n1, n2], axis=0)

    # ----- Convenience for plotting / inspection ------------------------

    def total_length(self) -> float:
        """Total arc length of the road."""
        return float(self.arc_lengths[-1] - self.arc_lengths[0])

    def __len__(self) -> int:
        return len(self.segments)
