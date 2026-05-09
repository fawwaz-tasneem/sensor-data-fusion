"""
Standard linear Kalman filter.

For a linear system:
    x_{k+1} = F x_k + w_k,        w_k ~ N(0, Q)
    z_k     = H x_k + v_k,        v_k ~ N(0, R)

Predict:
    x_pred = F x
    P_pred = F P F^T + Q

Update (with measurement z):
    y = z - H x_pred                         (innovation)
    S = H P_pred H^T + R                     (innovation covariance)
    K = P_pred H^T S^{-1}                    (Kalman gain)
    x = x_pred + K y
    P = (I - K H) P_pred

We use the Joseph form for the covariance update because it preserves
symmetry and positive-definiteness even with floating-point error.
"""
from __future__ import annotations

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateDistribution
from sdf.filters.base import Filter
from sdf.motion_models.base import MotionModel
from sdf.sensors.base import Sensor


class KalmanFilter(Filter):
    def __init__(self, motion_model: MotionModel, initial_state: StateDistribution):
        self.motion_model = motion_model
        self.state = initial_state

    def predict(self, t: float) -> StateDistribution:
        dt = t - self.state.timestamp
        if dt < 0:
            raise ValueError(
                f"Cannot predict backwards: current t={self.state.timestamp}, "
                f"requested t={t}"
            )
        if dt == 0:
            return self.state
        self.state = self.motion_model.predict(self.state, dt)
        return self.state

    def update(self, measurement: Measurement, sensor: Sensor) -> StateDistribution:
        # Predict the measurement and its covariance.
        x_pred = self.state.mean
        P_pred = self.state.covariance
        H = sensor.H(x_pred, self.state.layout)
        z_pred = sensor.h(x_pred, self.state.layout)

        # Innovation and its covariance.
        y = measurement.value - z_pred
        S = H @ P_pred @ H.T + measurement.R

        # Kalman gain via solving S^T K^T = (P H^T)^T, which is more numerically
        # stable than computing inv(S) explicitly.
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T

        # State update.
        x_new = x_pred + K @ y

        # Joseph-form covariance update: P_new = (I - KH) P (I - KH)^T + K R K^T.
        # More expensive than the simple form but numerically robust.
        I = np.eye(x_pred.shape[0])
        IKH = I - K @ H
        P_new = IKH @ P_pred @ IKH.T + K @ measurement.R @ K.T

        self.state = StateDistribution(
            mean=x_new,
            covariance=P_new,
            timestamp=measurement.timestamp,
            layout=self.state.layout,
        )
        return self.state
