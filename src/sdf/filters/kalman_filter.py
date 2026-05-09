import numpy as np
from .base_filter import BaseFilter

class KalmanFilter(BaseFilter):
    def predict(self, F, D):
        """Time Update Phase"""
        # x_k|k-1 = F * x_k-1|k-1
        self.x = F @ self.x
        # P_k|k-1 = F * P * F.T + D
        self.P = F @ self.P @ F.T + D
        return self.x, self.P

    def update(self, z, H, R):
        """Measurement Update Phase"""
        # Innovation: nu = z - H * x
        nu = z - (H @ self.x)
        # Innovation Covariance: S = H * P * H.T + R
        S = H @ self.P @ H.T + R
        # Kalman Gain: W = P * H.T * inv(S)
        W = self.P @ self.H_T_helper(H) @ np.linalg.inv(S)
        
        # Posterior Update
        self.x = self.x + (W @ nu)
        self.P = self.P - (W @ S @ W.T)
        return self.x, self.P

    def H_T_helper(self, H):
        return H.T