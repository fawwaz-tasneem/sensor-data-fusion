"""
TerrainOcclusion: deterministic occlusion from world geometry.

This module groups occlusion models whose answer depends only on the
target's position relative to fixed world features (terrain, structures,
line-of-sight obstructions) rather than on sensor parameters like the
Doppler clutter notch. The shared base class `TerrainOcclusion` exists
to label this category in the type system; future subclasses such as
`RidgeOcclusion` or `LineOfSightOcclusion` will sit alongside
`TunnelOcclusion`.

TunnelOcclusion is the first concrete case: a target inside a tube along
a specified arc-length range of a `PolygonalRoadMap` is hidden from any
sensor that carries this occlusion model. The framework's existing
sensor pipeline does the rest — `Sensor.is_detected` calls
`occlusion_model.is_occluded` before sampling detection, so the sensor
simply produces no measurement while the target is inside the tunnel,
and the road-aided EKF coasts through the gap on motion model + road
constraint alone.

Arc-length anchoring (rather than, say, an x-coordinate range) means the
same `TunnelOcclusion` definition works for straight roads, curved
mountain passes, and 3D winding routes without modification.
"""
from __future__ import annotations

from abc import ABC
from typing import Optional

import numpy as np

from sdf.core.state import StateLayout
from sdf.scenarios.road_map import PolygonalRoadMap
from sdf.sensors.occlusion import OcclusionModel


class TerrainOcclusion(OcclusionModel, ABC):
    """
    Abstract base for occlusion models defined by world geometry.

    Inherits `is_occluded` from `OcclusionModel`. Subclasses must
    implement it. This class exists to label the category — terrain
    shadowing is conceptually distinct from sensor-parameter occlusion
    like `DopplerBlindnessOcclusion` even though both implement the
    same ABC, and downstream code (especially visualization) sometimes
    wants to query "is there terrain occlusion present?" without
    enumerating every possible subclass.
    """

    pass


class TunnelOcclusion(TerrainOcclusion):
    """
    A target inside a tube along a road segment range is fully occluded.

    The tunnel is defined by:
      * the road it follows (any `PolygonalRoadMap`),
      * an arc-length entry/exit `[l_in, l_out]` along that road, and
      * a cross-track radius `radius` (the half-thickness of the tube).

    A target's position is mapped onto the road via `closest_segment`;
    the foot of that projection has an arc length along the road. The
    target is occluded iff the foot's arc length lies in `[l_in, l_out]`
    AND the cross-track distance from the foot is at most `radius`.

    This deterministic test is run regardless of `rng`. The `rng`
    parameter exists in the ABC for probabilistic subclasses (like
    `DopplerBlindnessOcclusion`) and is unused here.

    Parameters
    ----------
    road : PolygonalRoadMap
        The road the tunnel runs along.
    l_in : float
        Arc length at which the tunnel's interior begins.
    l_out : float
        Arc length at which the tunnel's interior ends. Must exceed l_in.
    radius : float
        Cross-track radius of the tunnel tube, in metres. A target whose
        perpendicular distance from the road exceeds this is *not*
        considered inside the tunnel (it's outside the structure).
    """

    def __init__(
        self,
        road: PolygonalRoadMap,
        l_in: float,
        l_out: float,
        radius: float,
    ):
        if l_out <= l_in:
            raise ValueError(
                f"l_out ({l_out}) must exceed l_in ({l_in})"
            )
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        total = road.total_length()
        if l_in < 0 or l_out > total + 1e-9:
            raise ValueError(
                f"tunnel range [{l_in}, {l_out}] outside road arc-length "
                f"range [0, {total}]"
            )

        self.road = road
        self.l_in = float(l_in)
        self.l_out = float(l_out)
        self.radius = float(radius)

    def _arc_length_at_foot(self, segment_idx: int, foot: np.ndarray) -> float:
        """
        Arc-length coordinate of `foot` on segment `segment_idx`.

        The foot is by construction on the segment, so its arc length
        equals the start-of-segment arc length plus the fraction of the
        segment's *true* arc length covered. We measure the fraction by
        Euclidean distance along the chord (foot is between the chord's
        endpoints), then scale by the segment's actual arc length —
        which may exceed the chord length when the polygonal map
        under-samples a curve.
        """
        seg = self.road.segments[segment_idx]
        offset = np.linalg.norm(foot - seg.s_start)
        # `offset / seg.length_euclid` is the fractional position along
        # the chord, in [0, 1]. Multiply by seg.arc_length (which may be
        # > seg.length_euclid) to get the arc-length distance from the
        # segment's start node.
        fraction = offset / seg.length_euclid
        return float(self.road.arc_lengths[segment_idx] + fraction * seg.arc_length)

    def is_occluded(
        self,
        target_state: np.ndarray,
        layout: StateLayout,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        position = layout.position(target_state)
        seg_idx, foot, cross_track = self.road.closest_segment(position)
        if cross_track > self.radius:
            return False
        l = self._arc_length_at_foot(seg_idx, foot)
        return self.l_in <= l <= self.l_out
