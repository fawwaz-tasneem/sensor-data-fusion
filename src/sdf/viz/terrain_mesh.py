"""
Tunnel wireframe rendering.

Two functions:

  * `tunnel_wireframe_segments(tunnel, ...)` — backend-agnostic. Returns
    line segments (pairs of 3D points) that, when drawn, form a tubular
    wireframe along the occluded portion of the road. The Dash dashboard
    will use this directly and render with Plotly.

  * `draw_tunnel_wireframe(ax, tunnel, ...)` — matplotlib convenience.
    Draws the segments onto a 3D matplotlib axis. For 2D axes, drops
    the z coordinate.

The wireframe consists of:
  * `n_rings` circular cross-sections perpendicular to the road tangent,
    spaced along the tunnel's arc-length range, and
  * `n_long` longitudinal "rails" connecting corresponding points on
    successive rings.

The road tangent at each ring is taken from whichever segment of the
underlying `PolygonalRoadMap` the ring sits on. For 2D roads (no
elevation) the tube lies in a vertical plane; for 3D roads it uses the
existing `cross_track_normals` from the road map to build a plane
perpendicular to the tangent.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from sdf.scenarios.road_map import PolygonalRoadMap
from sdf.sensors.terrain_occlusion import TunnelOcclusion


# A wireframe segment: ((x0,y0,z0), (x1,y1,z1)).
LineSegment3D = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


def _segment_index_at_arc_length(road: PolygonalRoadMap, l: float) -> int:
    """Find the segment whose arc-length range contains l."""
    # arc_lengths is strictly increasing, length N (N-1 segments).
    # The segment containing l is the largest i with arc_lengths[i] <= l.
    idx = int(np.searchsorted(road.arc_lengths, l, side="right") - 1)
    return int(np.clip(idx, 0, len(road) - 1))


def _point_at_arc_length(road: PolygonalRoadMap, l: float) -> np.ndarray:
    """Position on the road at arc length `l`."""
    i = _segment_index_at_arc_length(road, l)
    seg = road.segments[i]
    seg_start_arc = road.arc_lengths[i]
    # Fraction of *this segment's* arc length covered by l.
    fraction = (l - seg_start_arc) / seg.arc_length
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return seg.s_start + fraction * (seg.s_end - seg.s_start)


def _ring(
    centre: np.ndarray,
    normals: np.ndarray,
    radius: float,
    n_long: int,
) -> np.ndarray:
    """
    A ring of `n_long` points around `centre`, in the plane spanned by
    `normals` (shape (dim-1, dim)), with radius `radius`.

    For 2D roads (`normals.shape == (1, 2)`), the "ring" degenerates to
    two diametrically opposite points (the tunnel walls). For 3D roads
    (`normals.shape == (2, 3)`), it's a full circle.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_long, endpoint=False)
    if normals.shape[0] == 1:
        # 2D road: ring is just two points (positive and negative offset).
        # Use cos and sin of two angles 180 deg apart for symmetry.
        n = normals[0]
        # Lift to 3D with z=0 so the segments dict is uniformly 3D.
        c = np.array([centre[0], centre[1], 0.0])
        n3 = np.array([n[0], n[1], 0.0])
        # Place n_long points evenly around — for a 2D road most fall
        # at the same two locations, which the caller can dedupe if it
        # cares. Keeping the same count means longitudinal rails match
        # up between rings.
        pts = np.zeros((n_long, 3))
        for k, ang in enumerate(angles):
            pts[k] = c + radius * np.cos(ang) * n3
        return pts

    # 3D road: full circle in the plane spanned by normals[0], normals[1].
    n1, n2 = normals[0], normals[1]
    pts = np.zeros((n_long, 3))
    for k, ang in enumerate(angles):
        pts[k] = centre + radius * (np.cos(ang) * n1 + np.sin(ang) * n2)
    return pts


def tunnel_wireframe_segments(
    tunnel: TunnelOcclusion,
    n_rings: int = 12,
    n_long: int = 12,
) -> list[LineSegment3D]:
    """
    Generate wireframe line segments for a tunnel.

    Returns
    -------
    list of ((x,y,z), (x,y,z)) pairs. Suitable for Plotly Scatter3d (in
    "lines" mode, separated by Nones) or matplotlib 3D `plot`.
    """
    road = tunnel.road
    if n_rings < 2 or n_long < 2:
        raise ValueError("n_rings and n_long must each be >= 2")

    arc_positions = np.linspace(tunnel.l_in, tunnel.l_out, n_rings)
    rings: list[np.ndarray] = []
    for l in arc_positions:
        centre = _point_at_arc_length(road, l)
        seg_idx = _segment_index_at_arc_length(road, l)
        normals = road.cross_track_normals(seg_idx)
        rings.append(_ring(centre, normals, tunnel.radius, n_long))

    segments: list[LineSegment3D] = []

    # Circumferential edges (each ring forms a closed polygon).
    for ring in rings:
        for k in range(n_long):
            p = ring[k]
            q = ring[(k + 1) % n_long]
            segments.append(((p[0], p[1], p[2]), (q[0], q[1], q[2])))

    # Longitudinal rails (corresponding points across consecutive rings).
    for i in range(n_rings - 1):
        for k in range(n_long):
            p = rings[i][k]
            q = rings[i + 1][k]
            segments.append(((p[0], p[1], p[2]), (q[0], q[1], q[2])))

    return segments


def draw_tunnel_wireframe(
    ax,
    tunnel: TunnelOcclusion,
    n_rings: int = 12,
    n_long: int = 12,
    color: str = "gray",
    alpha: float = 0.4,
    linewidth: float = 0.6,
) -> None:
    """
    Draw a tunnel wireframe on a matplotlib axis.

    Works with both 2D and 3D axes. For 2D axes (no `plot3D` method),
    z coordinates are discarded, producing the outline of the tunnel
    projected onto the xy plane.
    """
    segs = tunnel_wireframe_segments(tunnel, n_rings=n_rings, n_long=n_long)

    is_3d = hasattr(ax, "plot3D") or getattr(ax, "name", "") == "3d"

    if is_3d:
        for p, q in segs:
            ax.plot(
                [p[0], q[0]],
                [p[1], q[1]],
                [p[2], q[2]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
    else:
        for p, q in segs:
            ax.plot(
                [p[0], q[0]],
                [p[1], q[1]],
                color=color,
                alpha=alpha,
                linewidth=linewidth,
            )
