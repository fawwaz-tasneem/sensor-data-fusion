"""
Extended Kalman Filter (EKF).

Differences from the linear KF:
  * Prediction can use a nonlinear motion model. The mean propagates
    through the true f(x, dt); the covariance propagates through
    F P F^T + Q where F is the Jacobian of f at x. Both responsibilities
    sit inside motion_model.predict(), so this method is identical to
    the KF's.
  * Update can use a nonlinear measurement function. The predicted
    measurement is sensor.h(x_pred); the innovation covariance uses
    sensor.H(x_pred, layout) — the Jacobian of h at the predicted mean.

For a linear motion model + linear sensor, the EKF equations reduce
exactly to the KF equations. We exploit that: the EKF code is the KF
code with `H` re-evaluated at the predicted mean each update. We do
NOT inherit from KalmanFilter — duplicating ~30 lines is clearer than
hiding the difference behind subclassing.

Equations:
    Predict:  handled by motion_model.predict() — see motion_models/base.py
    Update (with measurement z at the same time as the current state):
        z_pred = h(x_pred)
        H      = dh/dx | x = x_pred
        y      = z - z_pred              (innovation)
        S      = H P_pred H^T + R        (innovation covariance)
        K      = P_pred H^T S^{-1}       (Kalman gain)
        x      = x_pred + K y
        P      = (I - K H) P (I - K H)^T + K R K^T   (Joseph form)
"""
from __future__ import annotations

import numpy as np

from sdf.core.measurement import Measurement
from sdf.core.state import StateDistribution
from sdf.filters.base import Filter
from sdf.motion_models.base import MotionModel
from sdf.sensors.base import Sensor


class ExtendedKalmanFilter(Filter):
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
        # motion_model.predict propagates the mean through f(x, dt) and the
        # covariance through F P F^T + Q where F is dF/dx at x. For linear
        # models this reduces to KF prediction; for nonlinear models this
        # is the EKF prediction. Same call site, both worlds.
        self.state = self.motion_model.predict(self.state, dt)
        return self.state

    def update(self, measurement: Measurement, sensor: Sensor) -> StateDistribution:
        x_pred = self.state.mean
        P_pred = self.state.covariance

        # Linearize the measurement function at the predicted mean. For a
        # linear sensor this returns a constant matrix; for a nonlinear
        # sensor (radar, GMTI) this is the Jacobian dh/dx evaluated at x_pred.
        H = sensor.H(x_pred, self.state.layout)

        # Predicted measurement uses the FULL nonlinear h, not H @ x.
        # This distinction is what makes it an EKF and not a "linearized KF
        # with bias" — the prediction is unbiased to first order.
        z_pred = sensor.h(x_pred, self.state.layout)

        # Innovation and its covariance. Use sensor.innovation() rather than
        # plain subtraction so that sensors with angular components (radar
        # bearing, GMTI bearing) can wrap the angle difference correctly.
        y = sensor.innovation(measurement.value, z_pred)
        S = H @ P_pred @ H.T + measurement.R

        # Kalman gain via solve, not inv — same numerical reasoning as in KF.
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T

        # State update.
        x_new = x_pred + K @ y

        # Joseph-form covariance update for symmetry preservation.
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
