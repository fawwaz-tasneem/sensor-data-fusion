"""
Road-aided Extended Kalman Filter.

Implements Koch's road-map-aided tracking (Tracking and Sensor Data
Fusion, Sec. 9.1) as a *fictitious measurement*: when the sensors give
us nothing, we fall back on the road map to keep the estimate from
drifting.

The fictitious measurement is the **projected point on the road**. We
take the predicted position, find the closest discretized segment, and
project onto that segment's line — i.e. "how far along the segment the
target is". That projected foot is fused as a full position measurement,
but with an *oriented* (anisotropic) covariance expressed in the
road-aligned frame:

  * low variance cross-track   (sigma_r^2 = sigma_m^2 + sigma_d^2) —
    we are confident the target is laterally on the road, and
  * high variance along-track  (sigma_long^2 = (k * sigma_r)^2, k >> 1) —
    we make almost no claim about where along the segment it is.

So the update pins the estimate to the road manifold without pretending
to know the longitudinal position. (The classic cross-track-only
pseudo-measurement is the limit k -> inf of this; using a finite, large
k keeps it as a single well-conditioned position update and lets the
road endpoints gently bound the longitudinal position too.)

This is a single-hypothesis approximation of the lecture's full
Bayesian formulation: at each step we hard-assign the track to the
closest segment rather than maintaining a Gaussian mixture over all
segments.

Usage pattern (road map as an occlusion fallback):
    f = RoadAidedExtendedKalmanFilter(motion_model, init_state, road_map)
    for each timestep t:
        f.predict(t)
        measurement = sensor.measure(...)      # may be None (occluded)
        if measurement is not None:
            f.update(measurement, sensor)
        else:
            f.update_with_road()               # no reading -> fall back

The intended policy is to apply the road update *only* on steps where no
sensor returned a measurement: the sensors do the tracking, and the road
takes over to coast the estimate across a blackout. (A caller may still
apply it every step to demonstrate the bare cross-track mechanism, but
that is not the operational default.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateDistribution
from sdf.filters.extended_kalman import ExtendedKalmanFilter
from sdf.motion_models.base import MotionModel
from sdf.scenarios.road_map import PolygonalRoadMap
from sdf.sensors.base import Sensor


# Default ratio of along-track to cross-track 1-sigma for the fictitious
# road measurement: sigma_long = LONGITUDINAL_FACTOR * sigma_cross. Large
# enough that the update is, in practice, cross-track only, but finite so the
# point update stays a single well-conditioned position measurement.
LONGITUDINAL_FACTOR = 100.0


def road_cross_track_update(
    state: StateDistribution,
    road_map: PolygonalRoadMap,
    longitudinal_factor: float = LONGITUDINAL_FACTOR,
) -> tuple[StateDistribution, int, np.ndarray]:
    """
    Apply one road-map fictitious-measurement update to ``state``.

    This is the standalone engine behind the road-aided filter. It asserts
    "the target is on the nearest road segment" by projecting the predicted
    position onto that segment's line (the *foot* — i.e. how far along the
    segment the target is) and fusing that foot as a full position
    measurement, with an *oriented* covariance in the road-aligned frame:

      * cross-track variance  sigma_r^2 = sigma_mapping^2 + sigma_disc^2
        (low — we are confident the target is laterally on the road), and
      * along-track variance  (longitudinal_factor * sigma_r)^2
        (high — we make almost no claim about longitudinal position).

    Because the foot is the perpendicular projection, the innovation
    ``foot - position`` lies in the cross-track subspace, so the update pins
    the estimate to the road manifold without dragging it along-track (away
    from segment endpoints). The classic cross-track-only pseudo-measurement
    is the ``longitudinal_factor -> inf`` limit of this.

    Pulling this out as a free function lets *any* filter (not just
    RoadAidedExtendedKalmanFilter) fuse the road map — e.g. the dashboard
    applies it to a plain EKF on steps where every sensor missed.

    Returns the updated state plus the chosen segment index and projection
    foot (handy for visualization / association diagnostics).
    """
    layout = state.layout
    x_pred = state.mean
    P_pred = state.covariance

    # 1. Closest road segment to the predicted position, and the foot of the
    #    perpendicular projection onto it (clamped to the segment endpoints).
    position = layout.position(x_pred)
    seg_idx, foot, _ = road_map.closest_segment(position)
    seg = road_map.segments[seg_idx]

    # 2. Measurement model: the foot is a direct (linear) position observation.
    #    H selects the position components from the full state.
    n_state = x_pred.shape[0]
    H = np.zeros((layout.dim, n_state))
    for i, idx in enumerate(layout.position_idx):
        H[i, idx] = 1.0

    z = foot
    z_pred = position
    y = z - z_pred  # lies in the cross-track subspace (perpendicular foot)

    # 3. Oriented measurement covariance in the road-aligned frame:
    #    high along the tangent, low across it. {t} u rows(N) is orthonormal,
    #    so R is symmetric positive-definite with eigenvalues sigma_long^2
    #    (once) and sigma_r^2 (dim - 1 times).
    t = seg.tangent
    N = road_map.cross_track_normals(seg_idx)  # (dim - 1, dim)
    sigma_cross2 = seg.sigma_r2
    sigma_long2 = (longitudinal_factor**2) * sigma_cross2
    R = sigma_long2 * np.outer(t, t) + sigma_cross2 * (N.T @ N)

    # 4. Standard (linear) Kalman update — position is linear in the state.
    S = H @ P_pred @ H.T + R
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
    x_new = x_pred + K @ y
    I = np.eye(n_state)
    IKH = I - K @ H
    P_new = IKH @ P_pred @ IKH.T + K @ R @ K.T

    new_state = StateDistribution(
        mean=x_new,
        covariance=P_new,
        timestamp=state.timestamp,
        layout=layout,
    )
    return new_state, seg_idx, foot


class RoadAidedExtendedKalmanFilter(ExtendedKalmanFilter):
    """EKF with the ability to incorporate road-map information."""

    def __init__(
        self,
        motion_model: MotionModel,
        initial_state: StateDistribution,
        road_map: PolygonalRoadMap,
    ):
        super().__init__(motion_model=motion_model, initial_state=initial_state)
        if road_map.dim != initial_state.layout.dim:
            raise ValueError(
                f"road_map.dim ({road_map.dim}) != layout.dim "
                f"({initial_state.layout.dim})"
            )
        self.road_map = road_map
        # Track which segment was used most recently — useful for visualization
        # and for tests that check association behavior.
        self.last_segment_idx: Optional[int] = None
        self.last_foot: Optional[np.ndarray] = None

    # ----- Road-map fictitious measurement update ----------------------

    def update_with_road(
        self, longitudinal_factor: float = LONGITUDINAL_FACTOR
    ) -> StateDistribution:
        """
        Apply a road-map "fictitious measurement" update.

        Procedure (Koch §9.1.1, single-hypothesis approximation):
          1. Find the closest segment to the predicted position and project
             onto its line to get the foot ("how far along the segment").
          2. Fuse the foot as a full position measurement with an oriented
             covariance: low cross-track (sigma_r^2), high along-track
             ((longitudinal_factor * sigma_r)^2).
          3. Apply a standard (linear) Kalman update.

        Intended to be called only on steps where no sensor returned a
        measurement (the occlusion fallback); see the module docstring.

        The actual math lives in the module-level ``road_cross_track_update``
        so that other filters can reuse it; this method just applies it to
        ``self.state`` and records the chosen segment for visualization.
        """
        new_state, seg_idx, foot = road_cross_track_update(
            self.state, self.road_map, longitudinal_factor
        )
        self.last_segment_idx = seg_idx
        self.last_foot = foot.copy()
        self.state = new_state
        return self.state

    # ----- Convenience -------------------------------------------------

    def step_with_road(
        self,
        t: float,
        measurement: Optional[Measurement],
        sensor: Optional[Sensor],
        apply_road: bool = True,
    ) -> StateDistribution:
        """
        One full step: predict to t, then either fuse the sensor measurement
        or — when there is no measurement — fall back on the road map.

        ``apply_road`` gates the fallback (set it False to coast on pure
        motion-model prediction through a gap instead). The road update is
        *not* applied on top of a live sensor measurement; the sensors do the
        tracking, the road only takes over when they return nothing.
        """
        self.predict(t)
        if measurement is not None:
            if sensor is None:
                raise ValueError("sensor required when measurement is given")
            self.update(measurement, sensor)
        elif apply_road:
            self.update_with_road()
        return self.state
