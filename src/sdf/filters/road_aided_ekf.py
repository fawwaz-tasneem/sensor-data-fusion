"""
Road-aided Extended Kalman Filter.

Implements Koch's road-map-aided tracking (Tracking and Sensor Data
Fusion, Sec. 9.1). The road map is incorporated as a *fictitious*
scalar (or 2-vector in 3D) measurement: the cross-track displacement
of the predicted position from the closest road segment, with
"measured value" zero and variance sigma_r^2 = sigma_m^2 + sigma_d^2.

This is a single-hypothesis approximation of the lecture's full
Bayesian formulation: at each step we hard-assign the track to the
closest segment rather than maintaining a Gaussian mixture over all
segments. This is what's used in §9.1.3's quantitative comparison
when "road" is enabled but not "MHT".

Usage pattern:
    f = RoadAidedExtendedKalmanFilter(motion_model, init_state, road_map)
    for each timestep t:
        f.predict(t)
        if a sensor measurement is available:
            f.update(measurement, sensor)
        if the track should be road-constrained:
            f.update_with_road()        # apply the fictitious measurement

Whether to apply the road update at every step (constant constraint)
or only when sensor data is missing (occlusion fallback) is up to the
caller. Constant application is the most common choice and matches
the lecture's formulation.
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


def road_cross_track_update(
    state: StateDistribution, road_map: PolygonalRoadMap
) -> tuple[StateDistribution, int, np.ndarray]:
    """
    Apply one road-map fictitious cross-track measurement to ``state``.

    This is the standalone engine behind the road-aided filter: it asserts
    "the target is on the nearest road segment" by constructing a fictitious
    measurement of the cross-track displacement (perpendicular to the road),
    with measured value zero and variance sigma_r^2 = sigma_mapping^2 +
    sigma_discretization^2. Only the cross-track directions are constrained;
    the along-track (tangent) direction is left free, so the update pins the
    estimate to the road manifold without claiming to know where along it the
    target is. Pulling this out as a free function lets *any* filter (not just
    RoadAidedExtendedKalmanFilter) fuse the road map — e.g. the dashboard
    applies it to a plain EKF whenever the road map is enabled.

    Returns the updated state plus the chosen segment index and projection
    foot (handy for visualization / association diagnostics).
    """
    layout = state.layout
    x_pred = state.mean
    P_pred = state.covariance

    # 1. Closest road segment to the predicted position.
    position = layout.position(x_pred)
    seg_idx, foot, _ = road_map.closest_segment(position)
    seg = road_map.segments[seg_idx]

    # 2. Cross-track normal basis, shape (dim - 1, dim).
    N = road_map.cross_track_normals(seg_idx)

    # 3. h(x) = N (p(x) - foot); dh/dx = N J_p where J_p selects positions.
    n_state = x_pred.shape[0]
    J_p = np.zeros((layout.dim, n_state))
    for i, idx in enumerate(layout.position_idx):
        J_p[i, idx] = 1.0
    H = N @ J_p

    z_pred = N @ (position - foot)
    z = np.zeros(layout.dim - 1)
    R = seg.sigma_r2 * np.eye(layout.dim - 1)

    # 4. Standard (linear) Kalman update — cross-track is linear in position.
    y = z - z_pred
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

    def update_with_road(self) -> StateDistribution:
        """
        Apply a road-map "fictitious measurement" update.

        Procedure (Koch §9.1.1, single-hypothesis approximation):
          1. Find the closest segment to the predicted position.
          2. Construct the cross-track normal basis N (1 row in 2D, 2 in 3D).
          3. Form the measurement function h(x) = N (p(x) - foot), with
             "measured" value 0 and noise covariance sigma_r^2 * I.
          4. Apply a standard EKF update.

        The measurement is *scalar* in 2D and *2-dimensional* in 3D, so
        the update is numerically clean (no near-singular S).

        The actual math lives in the module-level ``road_cross_track_update``
        so that other filters can reuse it; this method just applies it to
        ``self.state`` and records the chosen segment for visualization.
        """
        new_state, seg_idx, foot = road_cross_track_update(
            self.state, self.road_map
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
        One full step: predict to t, optional sensor update, optional road
        update. Provided for convenience when running with a simulation
        engine that doesn't know about road updates.
        """
        self.predict(t)
        if measurement is not None:
            if sensor is None:
                raise ValueError("sensor required when measurement is given")
            self.update(measurement, sensor)
        if apply_road:
            self.update_with_road()
        return self.state
