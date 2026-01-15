import numpy as np

class RadarSensor:
    """
    Simulates a Radar measuring Range and Azimuth.
    Note: Standard KF expects linear H. This sensor provides nonlinear z.
    """
    def __init__(self, x_s, y_s, z_s, sigma_r=10.0, sigma_phi=0.1):
        self.pos_s = np.array([x_s, y_s, z_s]) # Radar position
        self.sigma_r = sigma_r                 # 10m
        self.sigma_phi = np.radians(sigma_phi) # Convert 0.1 deg to rad
        
        # Measurement Noise Covariance R
        self.R = np.diag([self.sigma_r**2, self.sigma_phi**2])

    def measure(self, target_pos):
        """
        Calculates range and azimuth from radar to target.
        target_pos: [x, y, z]
        """
        tx, ty, tz = target_pos.flatten()
        sx, sy, sz = self.pos_s
        
        # Range equation
        # range = sqrt((x-xs)^2 + (y-ys)^2 + (z-zs)^2 - zs^2) 
        # Note: -zs^2 is specific to the provided slide's geometry.
        rng = np.sqrt((tx-sx)**2 + (ty-sy)**2 + (tz-sz)**2 - sz**2)
        
        # Azimuth equation
        # phi = arctan((yk-ys)/(xk-xs))
        phi = np.arctan2((ty-sy), (tx-sx))
        
        # Add normally distributed noise
        z = np.array([[rng], [phi]]) + np.random.normal(0, 1, (2, 1)) * np.array([[self.sigma_r], [self.sigma_phi]])
        return z