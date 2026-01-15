import numpy as np

class MountainPassRoad:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.v = 20.0 / 3.6  # 20 km/h to m/s
        self.ax = 10000.0    # 10 km
        self.ay = 1000.0     # 1 km
        self.az = 1000.0     # 1 km

    def get_state(self, t):
        """Calculates r(t) based on"""
        x = self.v * t
        y = self.ay * np.sin((4 * np.pi * self.v / self.ax) * t)
        z = self.az * np.sin((np.pi * self.v / self.ax) * t)
        return np.array([[x], [y], [z]])