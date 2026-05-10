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
        """
        layout = self.state.layout
        x_pred = self.state.mean
        P_pred = self.state.covariance

        # 1. Segment selection.
        position = layout.position(x_pred)
        seg_idx, foot, _ = self.road_map.closest_segment(position)
        seg = self.road_map.segments[seg_idx]
        self.last_segment_idx = seg_idx
        self.last_foot = foot.copy()

        # 2. Cross-track normal basis. Shape: (dim - 1, dim).
        N = self.road_map.cross_track_normals(seg_idx)

        # 3. Build H. h(x) = N (p(x) - foot) where p(x) selects positions
        #    out of x. dh/dx = N * J_p, where J_p is the selector matrix
        #    that picks out the position indices from x.
        n_state = x_pred.shape[0]
        J_p = np.zeros((layout.dim, n_state))
        for i, idx in enumerate(layout.position_idx):
            J_p[i, idx] = 1.0
        H = N @ J_p  # shape (dim - 1, n_state)

        # Predicted measurement is the cross-track displacement of the
        # predicted position from the segment foot.
        z_pred = N @ (position - foot)
        # The "fictitious" measurement is exactly zero: we assert the
        # vehicle is on the road.
        z = np.zeros(layout.dim - 1)

        # Total cross-track variance (mapping + discretization).
        R = seg.sigma_r2 * np.eye(layout.dim - 1)

        # 4. Standard EKF update equations. We don't go through sensor.update
        #    because there's no Sensor object — the road is internal.
        y = z - z_pred  # plain subtraction is fine: cross-track is linear
        S = H @ P_pred @ H.T + R
        # Solve form, not inv.
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
        x_new = x_pred + K @ y
        I = np.eye(n_state)
        IKH = I - K @ H
        P_new = IKH @ P_pred @ IKH.T + K @ R @ K.T

        self.state = StateDistribution(
            mean=x_new,
            covariance=P_new,
            timestamp=self.state.timestamp,
            layout=layout,
        )
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
