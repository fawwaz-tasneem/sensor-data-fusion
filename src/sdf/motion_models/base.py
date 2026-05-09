"""
MotionModel: defines how a target's state evolves between timesteps.

The same MotionModel object serves linear filters (KF — uses F directly) and
nonlinear filters (EKF/UKF — call f, then F gives the Jacobian at the
linearization point). For linear models, f(x, dt) = F(dt) @ x.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sdf.core.state import StateDistribution, StateLayout


class MotionModel(ABC):
    """Base class for state transition models."""

    layout: StateLayout

    @property
    def state_dim(self) -> int:
        """Total dimensionality of the state vector this model uses."""
        return self._state_dim()

    @abstractmethod
    def _state_dim(self) -> int: ...

    @abstractmethod
    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        State transition matrix (or its Jacobian for nonlinear models),
        evaluated at x with timestep dt. Shape: (n, n).
        """

    @abstractmethod
    def Q(self, dt: float) -> np.ndarray:
        """Process noise covariance for a step of dt seconds. Shape: (n, n)."""

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Nonlinear state transition. Default implementation is linear: F @ x.
        Override for genuinely nonlinear models (e.g., coordinated turn).
        """
        return self.F(x, dt) @ x

    def predict(self, state: StateDistribution, dt: float) -> StateDistribution:
        """
        Predict the state forward by dt. This is the standard EKF predict step:
            mean_pred = f(mean, dt)
            cov_pred  = F P F^T + Q
        For linear models this collapses to the standard KF prediction.
        """
        F = self.F(state.mean, dt)
        Q = self.Q(dt)
        new_mean = self.f(state.mean, dt)
        new_cov = F @ state.covariance @ F.T + Q
        return StateDistribution(
            mean=new_mean,
            covariance=new_cov,
            timestamp=state.timestamp + dt,
            layout=state.layout,
        )
